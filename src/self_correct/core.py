"""Core module for self-correct-agent: Chain-of-Verification anti-hallucination wrapper.

This library implements the Chain-of-Verification (CoVe) methodology
described by Dhuliawala et al. (2023) [1] for reducing hallucinations
in LLM-generated text. It also draws on the claim-level decomposition
approach from FActScore (Min et al., 2023) [2].

References
----------
[1] Dhuliawala, S. et al. (2023). "Chain-of-Verification Reduces
    Hallucination in Large Language Models." arXiv:2309.11495.
[2] Min, S. et al. (2023). "FActScore: Fine-grained Atomic Evaluation
    of Factual Precision in Long Form Text Generation." arXiv:2305.14251.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Published USD rates per 1,000,000 tokens, as (prompt, completion).
#: Approximate; providers change these, so treat them as an estimate.
MODEL_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}


def model_pricing(model: str) -> Optional[tuple]:
    """Return (prompt, completion) rates per 1M tokens for ``model``.

    Deployed model names usually carry a dated or versioned suffix, such as
    "gpt-4o-mini-2024-07-18", so an exact match is tried first and then the
    longest matching known prefix. "gpt-4-turbo-preview" therefore resolves to
    gpt-4-turbo rather than to gpt-4.

    Returns None when nothing matches. Callers must report that as unknown
    rather than substituting a default: quoting one model's rates for another
    is worse than admitting the rate isn't known.
    """

    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    candidates = [name for name in MODEL_PRICING if model.startswith(name)]
    if not candidates:
        return None
    return MODEL_PRICING[max(candidates, key=len)]



class _CallBudget:
    """Thread-safe counter that caps LLM calls within one pipeline run.

    Claim verifications can run concurrently, so acquiring a call slot has
    to be atomic: a budget of N must never admit more than N calls even when
    several threads try at once.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._used = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Consume one call slot, returning False once the budget is spent."""
        with self._lock:
            if self._used >= self._limit:
                return False
            self._used += 1
            return True


#: Severity labels a flagged claim can carry.
VALID_SEVERITIES = ("critical", "major", "minor")

#: Ordered ``(regex, severity)`` rules applied to a flagged claim's critique;
#: the first match wins and anything unmatched is ``minor``. Critical
#: patterns assert the claim is actually wrong, major patterns flag claims
#: that cannot be established either way.
DEFAULT_SEVERITY_RULES = (
    (
        r"\bfalse\b|\bincorrect\b|\binaccurate\b|contradict\w*"
        r"|fabricat\w*|misleading",
        "critical",
    ),
    (
        r"\bunverifiable\b|cannot be (?:verified|confirmed)|\bdisputed\b"
        r"|\boutdated\b|\bunsupported\b|\bno evidence\b",
        "major",
    ),
)


def classify_severity(
    text: str, rules: Optional[List[Tuple[str, str]]] = None
) -> str:
    """Classify a critique (or flagged-claim message) as a severity label.

    Rules are ``(regex, severity)`` pairs scanned case-insensitively in
    order; the first matching rule decides. Text matching nothing falls
    back to ``minor``, so a custom rule set only needs to describe its
    exceptions.
    """
    if rules is None:
        rules = DEFAULT_SEVERITY_RULES
    for pattern, severity in rules:
        if severity not in VALID_SEVERITIES:
            raise ValueError(
                f"unknown severity {severity!r}; expected one of "
                + ", ".join(VALID_SEVERITIES)
            )
        if re.search(pattern, text or "", re.IGNORECASE):
            return severity
    return "minor"


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Tracks token consumption across all pipeline phases."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        with self._lock:
            return self.prompt_tokens + self.completion_tokens

    def add(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Atomically add token counts from a single LLM call."""
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

    def estimate_cost_for_model(self, model: str) -> Optional[float]:
        """Estimate cost in USD using the published rates for ``model``.

        Returns None for a model with no entry in MODEL_PRICING, so callers can
        say "unknown" rather than quoting a number derived from the wrong
        model's rates.
        """

        rates = model_pricing(model)
        if rates is None:
            return None
        prompt_per_1m, completion_per_1m = rates
        return (
            (self.prompt_tokens / 1_000_000) * prompt_per_1m
            + (self.completion_tokens / 1_000_000) * completion_per_1m
        )

    def estimate_cost(
        self,
        prompt_cost_per_1k: float = 0.005,
        completion_cost_per_1k: float = 0.015,
    ) -> float:
        """
        Estimate the USD cost of the tokens consumed.

        Parameters
        ----------
        prompt_cost_per_1k : float
            Cost per 1,000 prompt tokens (default: GPT-4o pricing).
        completion_cost_per_1k : float
            Cost per 1,000 completion tokens (default: GPT-4o pricing).

        Returns
        -------
        float
            Estimated cost in USD.
        """
        return (
            (self.prompt_tokens / 1000) * prompt_cost_per_1k
            + (self.completion_tokens / 1000) * completion_cost_per_1k
        )


@dataclass
class AntiHallucinationResponse:
    """Response object returned by AntiHallucinator.generate()."""

    content: str
    hallucinations_caught: List[str] = field(default_factory=list)
    verification_log: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    elapsed_seconds: float = 0.0
    phase_timings: Dict[str, float] = field(default_factory=dict)

    def evaluate(self, policy: "VerificationPolicy") -> "VerificationDecision":
        """Evaluate this response against a reusable release policy."""

        return policy.evaluate(self)

    def hallucination_density(self, per_words: int = 100) -> float:
        """Score how densely hallucinations occur per chunk of response text.

        The density is the number of hallucinated claims caught per
        ``per_words`` words of the final response, so shorter answers are not
        unfairly penalised for a single mistake. An empty response scores
        zero.
        """
        if per_words <= 0:
            raise ValueError("per_words must be positive")
        words = len(self.content.split())
        if not words:
            return 0.0
        return len(self.hallucinations_caught) / words * per_words

    def claim_summary(self) -> Dict[str, int]:
        """Summarise verification verdicts from the claim log.

        Returns counts for verified and flagged claims plus how many were
        checked against external evidence. Log entries that carry phase
        metadata rather than a per-claim verdict are ignored.
        """
        verdicts = [entry for entry in self.verification_log if "is_valid" in entry]
        return {
            "total_claims": len(verdicts),
            "verified_claims": sum(bool(entry.get("is_valid")) for entry in verdicts),
            "flagged_claims": sum(not bool(entry.get("is_valid")) for entry in verdicts),
            "evidence_claims": sum(bool(entry.get("evidence_used")) for entry in verdicts),
        }

    def severity_summary(
        self, rules: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, int]:
        """Count flagged claims by severity class.

        Every verdict entry whose claim was rejected contributes its
        critique text to :func:`classify_severity`; phase entries and
        verified claims are ignored. Pass ``rules`` to override the
        default taxonomy.
        """
        counts = {severity: 0 for severity in VALID_SEVERITIES}
        for entry in self.verification_log:
            if entry.get("is_valid") is not False:
                continue
            severity = classify_severity(str(entry.get("critique", "")), rules)
            counts[severity] += 1
        return counts

    def budget_report(self) -> Dict[str, Any]:
        """Report which claims or phases an exhausted call budget skipped.

        Log entries produced when the LLM call budget runs out carry a
        ``skipped_by_budget`` marker; phase-level halts use a ``phase``
        entry instead. Both are collected here so callers can see what was
        left unverified without parsing the log themselves.
        """
        skipped_claims = [
            str(entry["claim"])
            for entry in self.verification_log
            if entry.get("skipped_by_budget") and entry.get("claim")
        ]
        skipped_phases = [
            str(entry["phase"])
            for entry in self.verification_log
            if entry.get("skipped_by_budget") and entry.get("phase")
        ]
        extraction_halted = any(
            entry.get("phase") == "budget" for entry in self.verification_log
        )
        return {
            "exhausted": bool(skipped_claims or skipped_phases or extraction_halted),
            "skipped_claims": skipped_claims,
            "skipped_phases": skipped_phases,
        }

    def evidence_summary(self) -> Dict[str, Any]:
        """Summarize traceable external sources used across claim checks."""

        sources = [
            source
            for entry in self.verification_log
            for source in entry.get("evidence_sources", [])
            if isinstance(source, dict)
        ]
        urls = {str(source["url"]) for source in sources if source.get("url")}
        tools = sorted({str(source["tool"]) for source in sources if source.get("tool")})
        return {
            "source_count": len(sources),
            "unique_url_count": len(urls),
            "tools": tools,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the response to a plain dictionary."""
        return {
            "content": self.content,
            "hallucinations_caught": self.hallucinations_caught,
            "verification_log": self.verification_log,
            "hallucination_density": round(self.hallucination_density(), 3),
            "claim_summary": self.claim_summary(),
            "budget": self.budget_report(),
            "severity_summary": self.severity_summary(),
            "evidence_summary": self.evidence_summary(),
            "token_usage": {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens,
            },
            "phase_timings": {
                name: round(seconds, 3) for name, seconds in self.phase_timings.items()
            },
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """
        Serialize the response to a JSON string.

        Parameters
        ----------
        indent : int
            Pretty-print indentation level.
        ensure_ascii : bool
            If True, non-ASCII characters are escaped.

        Returns
        -------
        str
            JSON string representation.
        """
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)

    def to_markdown(self, include_log: bool = False) -> str:
        """
        Format the response as a human-readable Markdown report.

        Parameters
        ----------
        include_log : bool
            If True, include the full verification log in the output.

        Returns
        -------
        str
            Markdown-formatted report.
        """
        lines: List[str] = []
        lines.append("# Self-Correct Agent Report")
        lines.append("")
        lines.append(f"- **Tokens used**: {self.token_usage.total_tokens}")
        lines.append(f"- **Duration**: {self.elapsed_seconds:.2f}s")
        if self.phase_timings:
            breakdown = ", ".join(
                f"{name} {seconds:.2f}s"
                for name, seconds in self.phase_timings.items()
            )
            lines.append(f"- **Phase timings**: {breakdown}")
        lines.append(f"- **Hallucinations caught**: {len(self.hallucinations_caught)}")
        lines.append(
            f"- **Hallucination density**: {self.hallucination_density():.2f} "
            "per 100 words"
        )
        report = self.budget_report()
        if report["exhausted"]:
            lines.append(
                f"- **LLM call budget**: exhausted; "
                f"{len(report['skipped_claims'])} claim(s) left unverified"
            )
        lines.append("")

        if self.hallucinations_caught:
            lines.append("## Flagged Claims")
            lines.append("")
            for i, h in enumerate(self.hallucinations_caught, 1):
                lines.append(f"{i}. [{classify_severity(h)}] {h}")
                lines.append("")

        lines.append("## Final Output")
        lines.append("")
        lines.append(self.content)
        lines.append("")

        if include_log and self.verification_log:
            lines.append("## Verification Log")
            lines.append("")
            for entry in self.verification_log:
                claim = entry.get("claim", "N/A")
                valid = "\u2713" if entry.get("is_valid") else "\u2717"
                cached = " (cached)" if entry.get("cached") else ""
                lines.append(f"- {valid} **{claim}**{cached}")
                critique = entry.get("critique", "")
                if critique:
                    lines.append(f"  - *Critique*: {critique[:200]}")
                lines.append("")

        return "\n".join(lines)


@dataclass(frozen=True)
class VerificationDecision:
    """Outcome of applying a verification policy to a response."""

    passed: bool
    verified_ratio: float
    total_claims: int
    verified_claims: int
    flagged_claims: int
    evidence_ratio: float
    evidence_claims: int
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "verified_ratio": self.verified_ratio,
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "flagged_claims": self.flagged_claims,
            "evidence_ratio": self.evidence_ratio,
            "evidence_claims": self.evidence_claims,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class VerificationPolicy:
    """Quality gate for deciding whether a verified response may ship."""

    min_verified_ratio: float = 1.0
    min_verified_claims: int = 0
    max_flagged_claims: int = 0
    require_claims: bool = False
    min_evidence_ratio: float = 0.0
    min_evidence_claims: int = 0
    max_hallucination_density: float | None = None
    max_critical_claims: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationPolicy":
        if not isinstance(data, dict):
            raise ValueError("verification policy must be a JSON object")
        allowed = {
            "min_verified_ratio",
            "min_verified_claims",
            "max_flagged_claims",
            "require_claims",
            "min_evidence_ratio",
            "min_evidence_claims",
            "max_hallucination_density",
            "max_critical_claims",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown verification policy fields: {', '.join(unknown)}")
        return cls(
            min_verified_ratio=float(data.get("min_verified_ratio", 1.0)),
            min_verified_claims=int(data.get("min_verified_claims", 0)),
            max_flagged_claims=int(data.get("max_flagged_claims", 0)),
            require_claims=bool(data.get("require_claims", False)),
            min_evidence_ratio=float(data.get("min_evidence_ratio", 0.0)),
            min_evidence_claims=int(data.get("min_evidence_claims", 0)),
            max_hallucination_density=(
                float(data["max_hallucination_density"])
                if data.get("max_hallucination_density") is not None
                else None
            ),
            max_critical_claims=int(data.get("max_critical_claims", 0)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "VerificationPolicy":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read verification policy '{source}': {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def starter_template(cls) -> Dict[str, Any]:
        """Return a starter policy JSON object covering every supported field.

        The values form a working first policy: a verified-ratio floor of
        90%, two tolerated flagged claims, one tolerated critical claim, and
        a hallucination-density ceiling. Every dataclass field appears so the
        template doubles as a schema reference and parses cleanly through
        :meth:`from_dict`.
        """
        template: Dict[str, Any] = {
            "min_verified_ratio": 0.9,
            "min_verified_claims": 0,
            "max_flagged_claims": 2,
            "require_claims": True,
            "min_evidence_ratio": 0.0,
            "min_evidence_claims": 0,
            "max_hallucination_density": 3.0,
            "max_critical_claims": 1,
        }
        missing = [field.name for field in fields(cls) if field.name not in template]
        if missing:
            raise RuntimeError(
                "starter policy template is missing fields: " + ", ".join(missing)
            )
        return template

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_verified_ratio <= 1.0:
            raise ValueError("min_verified_ratio must be between 0 and 1")
        if self.min_verified_claims < 0:
            raise ValueError("min_verified_claims must be non-negative")
        if self.max_flagged_claims < 0:
            raise ValueError("max_flagged_claims must be non-negative")
        if not 0.0 <= self.min_evidence_ratio <= 1.0:
            raise ValueError("min_evidence_ratio must be between 0 and 1")
        if self.min_evidence_claims < 0:
            raise ValueError("min_evidence_claims must be non-negative")
        if (
            self.max_hallucination_density is not None
            and self.max_hallucination_density < 0
        ):
            raise ValueError("max_hallucination_density must be non-negative")
        if self.max_critical_claims < 0:
            raise ValueError("max_critical_claims must be non-negative")

    def evaluate(self, response: AntiHallucinationResponse) -> VerificationDecision:
        total = len(response.verification_log)
        verified = sum(bool(entry.get("is_valid")) for entry in response.verification_log)
        evidence_claims = sum(
            bool(entry.get("evidence_used")) for entry in response.verification_log
        )
        flagged = len(response.hallucinations_caught)
        critical = sum(
            classify_severity(str(entry.get("critique", ""))) == "critical"
            for entry in response.verification_log
            if entry.get("is_valid") is False
        )
        ratio = verified / total if total else 1.0
        evidence_ratio = evidence_claims / total if total else 1.0
        reasons: List[str] = []
        if self.require_claims and total == 0:
            reasons.append("no factual claims were verified")
        if ratio < self.min_verified_ratio:
            reasons.append(
                f"verified ratio {ratio:.1%} is below {self.min_verified_ratio:.1%}"
            )
        if verified < self.min_verified_claims:
            reasons.append(
                f"verified claims {verified} are below {self.min_verified_claims}"
            )
        if evidence_ratio < self.min_evidence_ratio:
            reasons.append(
                f"evidence ratio {evidence_ratio:.1%} is below "
                f"{self.min_evidence_ratio:.1%}"
            )
        if evidence_claims < self.min_evidence_claims:
            reasons.append(
                f"evidence-backed claims {evidence_claims} are below {self.min_evidence_claims}"
            )
        if flagged > self.max_flagged_claims:
            reasons.append(
                f"flagged claims {flagged} exceed {self.max_flagged_claims}"
            )
        if critical > self.max_critical_claims:
            reasons.append(
                f"critical claims {critical} exceed {self.max_critical_claims}"
            )
        if (
            self.max_hallucination_density is not None
        ):
            density_method = getattr(response, "hallucination_density", None)
            if callable(density_method):
                density = float(density_method())
            else:
                words = len(str(getattr(response, "content", "")).split())
                density = len(response.hallucinations_caught) / words * 100 if words else 0.0
            if density > self.max_hallucination_density:
                reasons.append(
                    f"hallucination density {density:.2f} exceeds "
                    f"{self.max_hallucination_density:.2f} per 100 words"
                )
        return VerificationDecision(
            passed=not reasons,
            verified_ratio=ratio,
            total_claims=total,
            verified_claims=verified,
            flagged_claims=flagged,
            evidence_ratio=evidence_ratio,
            evidence_claims=evidence_claims,
            reasons=reasons,
        )


def load_layered_policy(
    paths: "Sequence[str | Path]",
) -> Tuple["VerificationPolicy", List[str]]:
    """Load one verification policy layered over another.

    Files are merged left to right and later files win: a field set in an
    override file replaces the base value, while fields the override omits
    keep the base value. The merged mapping is validated once, so unknown
    or invalid values are rejected exactly as a single-file policy would be.

    Returns the merged policy plus one human-readable note per field whose
    value actually changed between layers, in application order.
    """

    merged: Dict[str, Any] = {}
    conflicts: List[str] = []
    for path in paths:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read verification policy '{source}': {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"verification policy '{source}' must be a JSON object")
        for key, value in payload.items():
            if key in merged and merged[key] != value:
                conflicts.append(
                    f"{key}: {merged[key]!r} overridden with {value!r} by '{source.name}'"
                )
        merged.update(payload)
    return VerificationPolicy.from_dict(merged), conflicts


class ContentCheck(ABC):
    """Interface for user-defined checks applied to generated content.

    A check inspects the pipeline's final text and returns one message per
    violation. Findings are folded into the verification log as flagged
    verdicts, so release policies, summaries, and exit-code gates treat
    them exactly like rejected factual claims.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used in reports and log entries."""

    @abstractmethod
    def check(self, content: str) -> List[str]:
        """Return a message per violation found in ``content``."""


class RegexContentCheck(ContentCheck):
    """Flag content matching a regular expression."""

    #: Upper bound on reported matches so one hot pattern cannot flood the log.
    MAX_FINDINGS = 10

    def __init__(self, name: str, pattern: str, flags: int = 0) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("content check name must be a non-empty string")
        self._name = name
        self._regex = re.compile(pattern, flags)

    @property
    def name(self) -> str:
        return self._name

    def check(self, content: str) -> List[str]:
        return [
            match.group(0)
            for match in self._regex.finditer(content or "")
            ][: self.MAX_FINDINGS]


#: Flag names accepted in a checks file, mapped onto ``re`` constants.
_REGEX_FLAG_NAMES = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "ASCII": re.ASCII,
}


def load_content_checks(path: str | Path) -> List[ContentCheck]:
    """Build content checks from a JSON definition file.

    The file contains ``{"checks": [...]}``; every entry needs a ``type``
    (only ``regex`` ships built-in), a non-empty ``name``, and for regex
    checks a ``pattern`` plus optional ``flags`` drawn from
    :data:`_REGEX_FLAG_NAMES`.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read content checks '{source}': {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("checks"), list):
        raise ValueError("content checks file must contain a 'checks' list")

    checks: List[ContentCheck] = []
    for entry in payload["checks"]:
        if not isinstance(entry, dict):
            raise ValueError("every content check must be a JSON object")
        kind = entry.get("type", "regex")
        if kind != "regex":
            raise ValueError(f"unknown content check type: {kind!r}")
        flags = 0
        for flag_name in entry.get("flags") or []:
            if flag_name not in _REGEX_FLAG_NAMES:
                raise ValueError(f"unknown regex flag: {flag_name!r}")
            flags |= _REGEX_FLAG_NAMES[flag_name]
        checks.append(
            RegexContentCheck(
                name=entry.get("name"),
                pattern=entry.get("pattern"),
                flags=flags,
            )
        )
    return checks


# ------------------------------------------------------------------
# LRU Claim Cache
# ------------------------------------------------------------------

class _ClaimCache:
    """Thread-safe LRU cache for verified claims."""

    def __init__(self, max_size: int = 256, ttl: Optional[float] = None) -> None:
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._stored_at: Dict[str, float] = {}
        self._max_size = max_size
        #: Seconds an entry stays valid. None means entries never expire.
        self._ttl = ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._expirations = 0
        self._evictions = 0

    @staticmethod
    def _key(claim: str, scope: Optional[str] = None) -> str:
        """Hash a claim and optional verification scope to create a key."""
        normalized_claim = claim.strip().lower()
        material = normalized_claim if scope is None else f"{scope.strip()}\0{normalized_claim}"
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def _is_expired(self, key: str, now: float) -> bool:
        if self._ttl is None:
            return False
        return (now - self._stored_at.get(key, 0.0)) >= self._ttl

    def get(self, claim: str, scope: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return a cached result, or None if absent or expired."""
        key = self._key(claim, scope)
        now = time.time()
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            if self._is_expired(key, now):
                # A stale answer is worse than no answer: verification results
                # describe the world, and the world moves.
                del self._cache[key]
                self._stored_at.pop(key, None)
                self._expirations += 1
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def put(
        self, claim: str, result: Dict[str, Any], scope: Optional[str] = None
    ) -> None:
        """Store a verification result."""
        key = self._key(claim, scope)
        with self._lock:
            self._cache[key] = result
            self._stored_at[key] = time.time()
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_size:
                evicted, _ = self._cache.popitem(last=False)
                self._stored_at.pop(evicted, None)
                self._evictions += 1

    @property
    def ttl(self) -> Optional[float]:
        """Seconds an entry stays valid, or None when entries never expire."""
        return self._ttl

    def stats(self) -> Dict[str, Any]:
        """Counters describing how the cache has behaved this session."""
        with self._lock:
            looked_up = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / looked_up) if looked_up else 0.0,
                "expirations": self._expirations,
                "evictions": self._evictions,
            }

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        with self._lock:
            return len(self._cache)

    @property
    def max_size(self) -> int:
        """Configured maximum number of cached entries."""
        return self._max_size

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def export_data(self) -> Dict[str, Any]:
        """Return a versioned, JSON-serializable cache snapshot."""

        with self._lock:
            return {
                "schema_version": 1,
                "entries": [
                    {
                        "key": key,
                        "stored_at": self._stored_at[key],
                        "result": result,
                    }
                    for key, result in self._cache.items()
                ],
            }

    def import_data(self, payload: Dict[str, Any]) -> int:
        """Merge a cache snapshot and return the number of usable entries."""

        if payload.get("schema_version") != 1 or not isinstance(
            payload.get("entries"), list
        ):
            raise ValueError("unsupported claim cache snapshot")
        now = time.time()
        accepted = 0
        with self._lock:
            for entry in payload["entries"]:
                if not isinstance(entry, dict):
                    raise ValueError("claim cache entries must be objects")
                key = entry.get("key")
                stored_at = entry.get("stored_at")
                result = entry.get("result")
                if (
                    not isinstance(key, str)
                    or len(key) != 16
                    or not isinstance(stored_at, (int, float))
                    or not isinstance(result, dict)
                ):
                    raise ValueError("claim cache entry has invalid fields")
                if self._ttl is not None and now - float(stored_at) >= self._ttl:
                    continue
                self._cache[key] = result
                self._stored_at[key] = float(stored_at)
                self._cache.move_to_end(key)
                accepted += 1
            while len(self._cache) > self._max_size:
                evicted, _ = self._cache.popitem(last=False)
                self._stored_at.pop(evicted, None)
        return accepted


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------

class AntiHallucinator:
    """Wrapper around an LLM client that auto-corrects hallucinations.

    Implements the Chain-of-Verification (CoVe) pipeline [1] with
    optional tool-assisted fact-checking and LRU claim caching.

    References
    ----------
    [1] Dhuliawala et al. (2023), arXiv:2309.11495.
    """

    _CLAIM_PATTERNS = [
        re.compile(r"^\s*(\d+)\.\s+(.+)"),
        re.compile(r"^\s*(\d+)\)\s+(.+)"),
        re.compile(r"^\s*-\s+(.+)"),
        re.compile(r"^\s*\*\s+(.+)"),
    ]

    _EXTRACTION_PROMPT = (
        "Extract a numbered list of all discrete factual claims made "
        "in the provided text. Focus on claims containing entities, "
        "dates, numbers, or specific assertions. Output ONLY the "
        "numbered list, nothing else."
    )

    def __init__(
        self,
        client: Any,
        strictness: float = 1.0,
        tools: Optional[List[Any]] = None,
        cache_size: int = 256,
        cache_ttl: Optional[float] = None,
        draft_system_prompt: str = "You are a helpful assistant.",
        extraction_prompt: str = _EXTRACTION_PROMPT,
        critique_prompt: Optional[str] = None,
        correction_prompt: str = (
            "You are a strict editor. Rewrite the provided draft to "
            "completely eliminate any factual claims that were flagged. "
            "You may remove sentences or rewrite them. "
            "Do NOT add new factual claims."
        ),
        max_retries: int = 0,
        retry_backoff: float = 0.0,
        max_evidence_results: int = 3,
        max_llm_calls: Optional[int] = None,
        content_checks: Optional[List[ContentCheck]] = None,
        model_draft: Optional[str] = None,
        model_extract: Optional[str] = None,
        model_verify: Optional[str] = None,
        model_correct: Optional[str] = None,
    ) -> None:
        """
        Initialize the AntiHallucinator wrapper.

        Parameters
        ----------
        client : Any
            The wrapped LLM client (e.g., openai.OpenAI()).
        strictness : float
            0.0 = passthrough, 0.5 = light critique, 1.0 = strict + tools.
        tools : Optional[List[Tool]]
            Verification tools (e.g., DuckDuckGoSearchTool).
        cache_size : int
            Max entries in the LRU claim cache.
        max_retries : int
            Number of additional attempts after a failed provider call.
        retry_backoff : float
            Initial delay between retries; delays double for each attempt.
        max_llm_calls : Optional[int]
            Hard cap on LLM API calls for one run. Once spent, remaining
            claims are logged as skipped instead of verified; cached claim
            lookups do not consume the budget.
        model_draft : Optional[str]
            Model for drafting phase (falls back to generate() model arg).
        model_extract : Optional[str]
            Model for fact extraction phase (falls back to generate() model arg).
        model_verify : Optional[str]
            Model for claim verification phase (falls back to generate() model arg).
        model_correct : Optional[str]
            Model for correction phase (falls back to generate() model arg).
        """
        if isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not math.isfinite(retry_backoff) or retry_backoff < 0:
            raise ValueError("retry_backoff must be a finite non-negative number")
        if isinstance(max_evidence_results, bool) or max_evidence_results < 1:
            raise ValueError("max_evidence_results must be a positive integer")
        if max_llm_calls is not None and (
            isinstance(max_llm_calls, bool) or not isinstance(max_llm_calls, int) or max_llm_calls < 1
        ):
            raise ValueError("max_llm_calls must be a positive integer or None")
        if content_checks is not None and not all(
            isinstance(check, ContentCheck) for check in content_checks
        ):
            raise ValueError("content_checks must contain ContentCheck instances")
        self.client = client
        self.strictness = max(0.0, min(1.0, strictness))
        self.tools = tools or []
        self._cache = _ClaimCache(max_size=cache_size, ttl=cache_ttl)
        self._draft_system_prompt = draft_system_prompt
        self._extraction_prompt = extraction_prompt
        self._critique_prompt = critique_prompt
        self._correction_prompt = correction_prompt
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_evidence_results = max_evidence_results
        self.max_llm_calls = max_llm_calls
        self.content_checks: List[ContentCheck] = list(content_checks or [])
        self._model_draft = model_draft
        self._model_extract = model_extract
        self._model_verify = model_verify
        self._model_correct = model_correct

    @property
    def cache(self) -> _ClaimCache:
        """Access the claim verification cache."""
        return self._cache

    @property
    def cache_size(self) -> int:
        """Current number of cached claims."""
        return self._cache.size

    @property
    def cache_max_size(self) -> int:
        """Configured maximum cache size."""
        return self._cache.max_size

    def clear_cache(self) -> None:
        """Remove all cached claim verification results."""
        self._cache.clear()

    def save_cache(self, path: str | Path) -> Path:
        """Persist verified claims so later processes can reuse them."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self._cache.export_data(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    def load_cache(self, path: str | Path) -> int:
        """Load usable verified claims from a cache snapshot."""

        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read claim cache '{source}': {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("claim cache snapshot must contain a JSON object")
        return self._cache.import_data(payload)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(
        self, model: str, system_prompt: str, user_prompt: str,
        usage: Optional[TokenUsage] = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Invoke the underlying LLM provider.

        Parameters
        ----------
        usage : Optional[TokenUsage]
            If provided, token counts are accumulated here.

        Returns
        -------
        str
            The model's text response. Guaranteed non-None.
        """
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.chat.completions.create(**kwargs)
                    break
                except AttributeError:
                    raise
                except Exception:
                    if attempt >= self.max_retries:
                        raise
                    delay = self.retry_backoff * (2**attempt)
                    if delay:
                        time.sleep(delay)
            # Accumulate token usage if the response provides it
            if usage is not None:
                resp_usage = getattr(response, "usage", None)
                if resp_usage is not None:
                    usage.add(
                        prompt_tokens=getattr(resp_usage, "prompt_tokens", 0),
                        completion_tokens=getattr(resp_usage, "completion_tokens", 0),
                    )

            content = response.choices[0].message.content
            if content is None:
                logger.warning("LLM returned None content.")
                return ""
            return content
        except AttributeError:
            raise ValueError(
                "Client does not have an OpenAI-compatible "
                "`.chat.completions.create()` interface."
            )
        except Exception as exc:
            raise RuntimeError(
                f"LLM API call failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _parse_claims(self, claims_text: str) -> List[str]:
        """Parse a numbered/bulleted list of claims from LLM output."""
        if not claims_text or not claims_text.strip():
            return []
        claims: List[str] = []
        for line in claims_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            matched = False
            for pattern in self._CLAIM_PATTERNS:
                m = pattern.match(line)
                if m:
                    claims.append(m.group(m.lastindex).strip())
                    matched = True
                    break
            if not matched and len(line) > 10:
                claims.append(line)
        return claims

    def _build_critique_prompt(self) -> str:
        """Build the critique system prompt based on strictness."""
        if self._critique_prompt is not None:
            return self._critique_prompt
        if self.strictness >= 0.8:
            return (
                "Critique the following factual claim with strict empirical "
                "skepticism. If it is absolutely, provably true based on "
                "well-established facts, output 'VERIFIED: True'. "
                "If it is false, inaccurate, misleading, or unverifiable, "
                "output 'VERIFIED: False' and explain why."
            )
        return (
            "Briefly check the following claim for obvious factual errors. "
            "If the claim is broadly correct, output 'VERIFIED: True'. "
            "Only output 'VERIFIED: False' if the claim is clearly wrong."
        )

    def _cache_scope(self, model: str, critique_prompt: str, use_tools: bool) -> str:
        """Return the settings that make a claim verification reusable."""

        tool_names = sorted(
            str(getattr(tool, "name", type(tool).__name__)) for tool in self.tools
        )
        return json.dumps(
            {
                "model": model,
                "critique_prompt": critique_prompt,
                "strictness": self.strictness,
                "tools": tool_names if use_tools else [],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _search_evidence(
        self, claim: str, max_results: int = 3
    ) -> tuple[str, List[Dict[str, str]]]:
        """Search tools and return prompt-ready evidence plus source metadata."""
        evidence_parts: List[str] = []
        evidence_sources: List[Dict[str, str]] = []
        for tool in self.tools:
            try:
                results = tool.search(claim, max_results=max_results)
                for r in results:
                    evidence_parts.append(
                        f"- [{r.title}]({r.url}): {r.snippet}"
                    )
                    evidence_sources.append(
                        {
                            "title": str(r.title),
                            "url": str(r.url),
                            "tool": str(getattr(tool, "name", type(tool).__name__)),
                        }
                    )
            except Exception as exc:
                logger.warning(
                    "Tool '%s' failed for claim '%s': %s",
                    getattr(tool, "name", "unknown"),
                    claim[:50],
                    exc,
                )
        return "\n".join(evidence_parts), evidence_sources

    def _verify_single_claim(
        self,
        claim: str,
        model: str,
        critique_prompt: str,
        use_tools: bool,
        usage: TokenUsage,
        max_tokens: int | None = None,
        cache_scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify a single claim, checking the cache first.

        Returns
        -------
        Dict[str, Any]
            Verification result dict with keys: claim, is_valid,
            critique, evidence_used, cached.
        """
        # Check cache first
        cached = self._cache.get(claim, scope=cache_scope)
        if cached is not None:
            return {**cached, "cached": True}

        evidence_context = ""
        evidence_sources: List[Dict[str, str]] = []
        if use_tools:
            evidence_context, evidence_sources = self._search_evidence(
                claim, max_results=self.max_evidence_results
            )

        user_msg = f"Claim: {claim}"
        if evidence_context:
            user_msg += (
                f"\n\nThe following search results were found as evidence. "
                f"Use them to decide if the claim is true or false:\n"
                f"{evidence_context}"
            )

        critique = self._call_llm(
            model=model,
            system_prompt=critique_prompt,
            user_prompt=user_msg,
            usage=usage,
            max_tokens=max_tokens,
        )

        is_valid = "VERIFIED: True" in critique
        result = {
            "claim": claim,
            "is_valid": is_valid,
            "critique": critique,
            "evidence_used": bool(evidence_context),
            "evidence_sources": evidence_sources,
            "cached": False,
        }

        # Store in cache
        self._cache.put(
            claim,
            {
                "claim": claim,
                "is_valid": is_valid,
                "critique": critique,
                "evidence_used": bool(evidence_context),
                "evidence_sources": evidence_sources,
            },
            scope=cache_scope,
        )

        return result

    def _apply_content_checks(self, response: AntiHallucinationResponse) -> None:
        """Fold content-check findings into a response as flagged verdicts.

        A check that raises is skipped with a warning: an auxiliary check
        must not destroy an otherwise completed verification.
        """
        for check in self.content_checks:
            try:
                findings = check.check(response.content)
            except Exception as exc:
                logger.warning(
                    "Content check '%s' failed: %s",
                    getattr(check, "name", type(check).__name__),
                    exc,
                )
                continue
            for finding in findings:
                message = str(finding)
                response.verification_log.append({
                    "claim": f"content:{check.name}",
                    "is_valid": False,
                    "critique": message,
                    "content_check": check.name,
                })
                response.hallucinations_caught.append(
                    f"Content check '{check.name}' flagged: {message}"
                )

    # ------------------------------------------------------------------
    # Public API: Synchronous
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int | None = None,
        model_draft: Optional[str] = None,
        model_extract: Optional[str] = None,
        model_verify: Optional[str] = None,
        model_correct: Optional[str] = None,
    ) -> AntiHallucinationResponse:
        """
        Generate text with automatic hallucination detection and correction.

        This is the synchronous entry point. For parallel claim verification,
        use `generate_async()` instead. When `strictness > 0` and content
        checks are configured, every returned response has been audited by
        them first.

        Parameters
        ----------
        model : str
            Default model name for all phases.
        prompt : str
            The prompt to process.
        max_tokens : Optional[int]
            Maximum tokens per LLM response.
        model_draft : Optional[str]
            Override model for drafting phase.
        model_extract : Optional[str]
            Override model for fact extraction phase.
        model_verify : Optional[str]
            Override model for claim verification phase.
        model_correct : Optional[str]
            Override model for correction phase.
        """
        # Use instance defaults if not overridden
        draft = model_draft or self._model_draft or model
        extract = model_extract or self._model_extract or model
        verify = model_verify or self._model_verify or model
        correct = model_correct or self._model_correct or model

        response = self._generate_sync(model, prompt, max_tokens,
                                       model_draft=draft, model_extract=extract,
                                       model_verify=verify, model_correct=correct)
        if self.strictness > 0.0:
            self._apply_content_checks(response)
        return response

    def _generate_sync(
        self, model: str, prompt: str, max_tokens: int | None = None,
        model_draft: Optional[str] = None, model_extract: Optional[str] = None,
        model_verify: Optional[str] = None, model_correct: Optional[str] = None
    ) -> AntiHallucinationResponse:
        start = time.monotonic()
        usage = TokenUsage()
        budget = _CallBudget(self.max_llm_calls) if self.max_llm_calls else None
        phase_timings: Dict[str, float] = {}

        # Phase 1: Drafting
        if budget is not None and not budget.try_acquire():
            raise RuntimeError("LLM call budget exhausted before drafting")
        phase_started = time.monotonic()
        draft = self._call_llm(
            model=model_draft,
            system_prompt=self._draft_system_prompt,
            user_prompt=prompt,
            usage=usage,
            max_tokens=max_tokens,
        )
        phase_timings["drafting"] = time.monotonic() - phase_started

        if self.strictness == 0.0:
            return AntiHallucinationResponse(
                content=draft,
                verification_log=[{"phase": "bypassed", "reason": "strictness=0.0"}],
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
                phase_timings=phase_timings,
            )

        # Phase 2: Fact Extraction
        if budget is not None and not budget.try_acquire():
            return AntiHallucinationResponse(
                content=draft,
                verification_log=[{
                    "phase": "budget",
                    "skipped_by_budget": True,
                    "detail": "extraction skipped: LLM call budget reached",
                }],
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
                phase_timings=phase_timings,
            )
        phase_started = time.monotonic()
        claims_text = self._call_llm(
            model=model_extract,
            system_prompt=self._extraction_prompt,
            user_prompt=f"Text to analyze:\n\n{draft}",
            usage=usage,
            max_tokens=max_tokens,
        )
        phase_timings["extraction"] = time.monotonic() - phase_started

        claims = self._parse_claims(claims_text)

        if not claims:
            logger.warning("No claims extracted. Returning draft with warning.")
            return AntiHallucinationResponse(
                content=draft,
                verification_log=[{
                    "phase": "extraction",
                    "warning": "No claims extracted.",
                    "raw_extraction": claims_text,
                }],
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
                phase_timings=phase_timings,
            )

        # Phase 3: Critique / Verification
        critique_prompt = self._build_critique_prompt()
        use_tools = self.strictness >= 0.8 and len(self.tools) > 0
        cache_scope = self._cache_scope(model_verify, critique_prompt, use_tools)

        verification_log: List[Dict[str, Any]] = []
        hallucinations_caught: List[str] = []

        phase_started = time.monotonic()
        for claim in claims:
            if budget is not None and not budget.try_acquire():
                verification_log.append({
                    "claim": claim,
                    "skipped_by_budget": True,
                    "critique": "not verified: LLM call budget reached",
                })
                continue
            result = self._verify_single_claim(
                claim,
                model_verify,
                critique_prompt,
                use_tools,
                usage,
                max_tokens=max_tokens,
                cache_scope=cache_scope,
            )
            verification_log.append(result)
            if not result["is_valid"]:
                hallucinations_caught.append(
                    f"Claim '{claim}' flagged: {result['critique']}"
                )

        phase_timings["verification"] = time.monotonic() - phase_started

        # Phase 4: Correction
        if not hallucinations_caught:
            return AntiHallucinationResponse(
                content=draft,
                verification_log=verification_log,
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
                phase_timings=phase_timings,
            )

        if budget is not None and not budget.try_acquire():
            return AntiHallucinationResponse(
                content=draft,
                hallucinations_caught=hallucinations_caught,
                verification_log=verification_log
                + [{"phase": "correction", "skipped_by_budget": True}],
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
                phase_timings=phase_timings,
            )

        phase_started = time.monotonic()
        final_content = self._call_llm(
            model=model_correct,
            system_prompt=self._correction_prompt,
            user_prompt=(
                f"Original Draft:\n{draft}\n\n"
                f"Critiques to address:\n"
                + "\n\n".join(hallucinations_caught)
            ),
            usage=usage,
            max_tokens=max_tokens,
        )

        phase_timings["correction"] = time.monotonic() - phase_started

        return AntiHallucinationResponse(
            content=final_content,
            hallucinations_caught=hallucinations_caught,
            verification_log=verification_log,
            token_usage=usage,
            elapsed_seconds=time.monotonic() - start,
            phase_timings=phase_timings,
        )

    def generate_many(
        self,
        model: str,
        prompts: List[str],
        max_tokens: int | None = None,
    ) -> List[AntiHallucinationResponse]:
        """Generate verified responses for a batch of prompts in input order.

        Calls are deliberately ordered so callers can replay a batch with
        deterministic provider usage and still benefit from the claim cache.
        Use ``generate_async`` per prompt when lower latency matters more than
        that reproducibility.
        """
        if isinstance(prompts, (str, bytes)):
            raise TypeError("prompts must be a sequence of non-empty strings")
        prompt_list = list(prompts)
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompt_list):
            raise ValueError("every prompt must be a non-empty string")
        return [
            self.generate(model=model, prompt=prompt, max_tokens=max_tokens)
            for prompt in prompt_list
        ]

    async def generate_many_async(
        self,
        model: str,
        prompts: List[str],
        *,
        max_concurrency: int | None = None,
        prompt_concurrency: int | None = None,
    ) -> List[AntiHallucinationResponse]:
        """Verify a batch of prompts concurrently while preserving input order."""

        if isinstance(prompts, (str, bytes)):
            raise TypeError("prompts must be a sequence of non-empty strings")
        prompt_list = list(prompts)
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompt_list):
            raise ValueError("every prompt must be a non-empty string")
        if prompt_concurrency is not None and (
            isinstance(prompt_concurrency, bool) or prompt_concurrency < 1
        ):
            raise ValueError("prompt_concurrency must be a positive integer or None")
        semaphore = asyncio.Semaphore(prompt_concurrency) if prompt_concurrency else None

        async def _generate(prompt: str) -> AntiHallucinationResponse:
            if semaphore is None:
                return await self.generate_async(
                    model, prompt, max_concurrency=max_concurrency
                )
            async with semaphore:
                return await self.generate_async(
                    model, prompt, max_concurrency=max_concurrency
                )

        return list(await asyncio.gather(*[_generate(prompt) for prompt in prompt_list]))

    # ------------------------------------------------------------------
    # Public API: Asynchronous (parallel claim verification)
    # ------------------------------------------------------------------

    async def generate_async(
        self,
        model: str,
        prompt: str,
        *,
        max_concurrency: int | None = None,
        model_draft: Optional[str] = None,
        model_extract: Optional[str] = None,
        model_verify: Optional[str] = None,
        model_correct: Optional[str] = None,
    ) -> AntiHallucinationResponse:
        """
        Async version of generate() that verifies claims in parallel.

        Claims are verified concurrently using asyncio, significantly
        reducing total latency when many claims are extracted.  Set
        `max_concurrency` to protect a provider from bursts or rate limits.

        Usage::

            import asyncio
            result = asyncio.run(safe_client.generate_async(model, prompt))

        Parameters
        ----------
        model : str
            Default model name for all phases.
        prompt : str
            The prompt to process.
        max_concurrency : Optional[int]
            Maximum concurrent claim verifications.
        model_draft : Optional[str]
            Override model for drafting phase.
        model_extract : Optional[str]
            Override model for fact extraction phase.
        model_verify : Optional[str]
            Override model for claim verification phase.
        model_correct : Optional[str]
            Override model for correction phase.
        """
        # Use instance defaults if not overridden
        draft = model_draft or self._model_draft or model
        extract = model_extract or self._model_extract or model
        verify = model_verify or self._model_verify or model
        correct = model_correct or self._model_correct or model

        response = await self._generate_async_impl(
            model, prompt, max_concurrency=max_concurrency,
            model_draft=draft, model_extract=extract,
            model_verify=verify, model_correct=correct
        )
        if self.strictness > 0.0:
            self._apply_content_checks(response)
        return response

    async def _generate_async_impl(
        self, model: str, prompt: str, *, max_concurrency: int | None = None,
        model_draft: Optional[str] = None, model_extract: Optional[str] = None,
        model_verify: Optional[str] = None, model_correct: Optional[str] = None
    ) -> AntiHallucinationResponse:
        if max_concurrency is not None and (
            isinstance(max_concurrency, bool) or max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer or None")

        start = time.monotonic()
        usage = TokenUsage()
        budget = _CallBudget(self.max_llm_calls) if self.max_llm_calls else None
        phase_timings: Dict[str, float] = {}

        # Phase 1 & 2 are sequential (need the draft before extraction)
        if budget is not None and not budget.try_acquire():
            raise RuntimeError("LLM call budget exhausted before drafting")
        phase_started = time.monotonic()
        draft = await asyncio.to_thread(
            self._call_llm, model_draft, self._draft_system_prompt, prompt, usage
        )
        phase_timings["drafting"] = time.monotonic() - phase_started

        if self.strictness == 0.0:
            return AntiHallucinationResponse(
                content=draft,
                verification_log=[{"phase": "bypassed", "reason": "strictness=0.0"}],
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
                phase_timings=phase_timings,
            )

        if budget is not None and not budget.try_acquire():
            return AntiHallucinationResponse(
                content=draft,
                verification_log=[{
                    "phase": "budget",
                    "skipped_by_budget": True,
                    "detail": "extraction skipped: LLM call budget reached",
                }],
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
                phase_timings=phase_timings,
            )

        phase_started = time.monotonic()
        claims_text = await asyncio.to_thread(
            self._call_llm,
            model_extract,
            self._extraction_prompt,
            f"Text to analyze:\n\n{draft}",
            usage,
        )
        phase_timings["extraction"] = time.monotonic() - phase_started

        claims = self._parse_claims(claims_text)

        if not claims:
            return AntiHallucinationResponse(
                content=draft,
                verification_log=[{
                    "phase": "extraction",
                    "warning": "No claims extracted.",
                    "raw_extraction": claims_text,
                }],
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
                phase_timings=phase_timings,
            )

        # Phase 3: Parallel claim verification
        critique_prompt = self._build_critique_prompt()
        use_tools = self.strictness >= 0.8 and len(self.tools) > 0
        cache_scope = self._cache_scope(model_verify, critique_prompt, use_tools)
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        def _skip(claim: str) -> Dict[str, Any]:
            return {
                "claim": claim,
                "skipped_by_budget": True,
                "critique": "not verified: LLM call budget reached",
            }

        async def _verify(claim: str) -> Dict[str, Any]:
            if budget is not None and not budget.try_acquire():
                return _skip(claim)
            if semaphore is None:
                return await asyncio.to_thread(
                    self._verify_single_claim,
                    claim,
                    model_verify,
                    critique_prompt,
                    use_tools,
                    usage,
                    cache_scope=cache_scope,
                )
            async with semaphore:
                return await asyncio.to_thread(
                    self._verify_single_claim,
                    claim,
                    model_verify,
                    critique_prompt,
                    use_tools,
                    usage,
                    cache_scope=cache_scope,
                )

        phase_started = time.monotonic()
        verification_results = await asyncio.gather(
            *[_verify(c) for c in claims]
        )

        phase_timings["verification"] = time.monotonic() - phase_started
        verification_log = list(verification_results)
        hallucinations_caught = [
            f"Claim '{r['claim']}' flagged: {r['critique']}"
            for r in verification_log
            if r.get("is_valid") is False
        ]

        # Phase 4: Correction
        if not hallucinations_caught:
            return AntiHallucinationResponse(
                content=draft,
                verification_log=verification_log,
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
                phase_timings=phase_timings,
            )

        if budget is not None and not budget.try_acquire():
            return AntiHallucinationResponse(
                content=draft,
                hallucinations_caught=hallucinations_caught,
                verification_log=verification_log
                + [{"phase": "correction", "skipped_by_budget": True}],
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
                phase_timings=phase_timings,
            )

        phase_started = time.monotonic()
        final_content = await asyncio.to_thread(
            self._call_llm,
            model_correct,
            self._correction_prompt,
            (
                f"Original Draft:\n{draft}\n\n"
                f"Critiques to address:\n"
                + "\n\n".join(hallucinations_caught)
            ),
            usage,
        )

        phase_timings["correction"] = time.monotonic() - phase_started

        return AntiHallucinationResponse(
            content=final_content,
            hallucinations_caught=hallucinations_caught,
            verification_log=verification_log,
            token_usage=usage,
            elapsed_seconds=time.monotonic() - start,
            phase_timings=phase_timings,
        )