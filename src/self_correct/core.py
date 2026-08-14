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

import asyncio
import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the response to a plain dictionary."""
        return {
            "content": self.content,
            "hallucinations_caught": self.hallucinations_caught,
            "verification_log": self.verification_log,
            "token_usage": {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens,
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
        lines.append(f"- **Hallucinations caught**: {len(self.hallucinations_caught)}")
        lines.append("")

        if self.hallucinations_caught:
            lines.append("## Flagged Claims")
            lines.append("")
            for i, h in enumerate(self.hallucinations_caught, 1):
                lines.append(f"{i}. {h}")
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
    def _key(claim: str) -> str:
        """Hash the claim text to create a cache key."""
        return hashlib.sha256(claim.strip().lower().encode()).hexdigest()[:16]

    def _is_expired(self, key: str, now: float) -> bool:
        if self._ttl is None:
            return False
        return (now - self._stored_at.get(key, 0.0)) >= self._ttl

    def get(self, claim: str) -> Optional[Dict[str, Any]]:
        """Return a cached result, or None if absent or expired."""
        key = self._key(claim)
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

    def put(self, claim: str, result: Dict[str, Any]) -> None:
        """Store a verification result."""
        key = self._key(claim)
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
        """
        self.client = client
        self.strictness = max(0.0, min(1.0, strictness))
        self.tools = tools or []
        self._cache = _ClaimCache(max_size=cache_size, ttl=cache_ttl)
        self._draft_system_prompt = draft_system_prompt
        self._extraction_prompt = extraction_prompt
        self._critique_prompt = critique_prompt
        self._correction_prompt = correction_prompt

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
            response = self.client.chat.completions.create(**kwargs)
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

    def _search_evidence(self, claim: str, max_results: int = 3) -> str:
        """Search all registered tools for evidence related to a claim."""
        evidence_parts: List[str] = []
        for tool in self.tools:
            try:
                results = tool.search(claim, max_results=max_results)
                for r in results:
                    evidence_parts.append(
                        f"- [{r.title}]({r.url}): {r.snippet}"
                    )
            except Exception as exc:
                logger.warning(
                    "Tool '%s' failed for claim '%s': %s",
                    getattr(tool, "name", "unknown"),
                    claim[:50],
                    exc,
                )
        return "\n".join(evidence_parts)

    def _verify_single_claim(
        self,
        claim: str,
        model: str,
        critique_prompt: str,
        use_tools: bool,
        usage: TokenUsage,
        max_tokens: int | None = None,
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
        cached = self._cache.get(claim)
        if cached is not None:
            return {**cached, "cached": True}

        evidence_context = ""
        if use_tools:
            evidence_context = self._search_evidence(claim)

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
            "cached": False,
        }

        # Store in cache
        self._cache.put(claim, {
            "claim": claim,
            "is_valid": is_valid,
            "critique": critique,
            "evidence_used": bool(evidence_context),
        })

        return result

    # ------------------------------------------------------------------
    # Public API: Synchronous
    # ------------------------------------------------------------------

    def generate(self, model: str, prompt: str, max_tokens: int | None = None) -> AntiHallucinationResponse:
        """
        Generate text with automatic hallucination detection and correction.

        This is the synchronous entry point. For parallel claim verification,
        use ``generate_async()`` instead.
        """
        start = time.monotonic()
        usage = TokenUsage()

        # Phase 1: Drafting
        draft = self._call_llm(
            model=model,
            system_prompt=self._draft_system_prompt,
            user_prompt=prompt,
            usage=usage,
            max_tokens=max_tokens,
        )

        if self.strictness == 0.0:
            return AntiHallucinationResponse(
                content=draft,
                verification_log=[{"phase": "bypassed", "reason": "strictness=0.0"}],
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
            )

        # Phase 2: Fact Extraction
        claims_text = self._call_llm(
            model=model,
            system_prompt=self._extraction_prompt,
            user_prompt=f"Text to analyze:\n\n{draft}",
            usage=usage,
            max_tokens=max_tokens,
        )

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
            )

        # Phase 3: Critique / Verification
        critique_prompt = self._build_critique_prompt()
        use_tools = self.strictness >= 0.8 and len(self.tools) > 0

        verification_log: List[Dict[str, Any]] = []
        hallucinations_caught: List[str] = []

        for claim in claims:
            result = self._verify_single_claim(
                claim, model, critique_prompt, use_tools, usage,
                max_tokens=max_tokens,
            )
            verification_log.append(result)
            if not result["is_valid"]:
                hallucinations_caught.append(
                    f"Claim '{claim}' flagged: {result['critique']}"
                )

        # Phase 4: Correction
        if not hallucinations_caught:
            return AntiHallucinationResponse(
                content=draft,
                verification_log=verification_log,
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
            )

        final_content = self._call_llm(
            model=model,
            system_prompt=self._correction_prompt,
            user_prompt=(
                f"Original Draft:\n{draft}\n\n"
                f"Critiques to address:\n"
                + "\n\n".join(hallucinations_caught)
            ),
            usage=usage,
            max_tokens=max_tokens,
        )

        return AntiHallucinationResponse(
            content=final_content,
            hallucinations_caught=hallucinations_caught,
            verification_log=verification_log,
            token_usage=usage,
            elapsed_seconds=time.monotonic() - start,
        )

    # ------------------------------------------------------------------
    # Public API: Asynchronous (parallel claim verification)
    # ------------------------------------------------------------------

    async def generate_async(
        self, model: str, prompt: str
    ) -> AntiHallucinationResponse:
        """
        Async version of generate() that verifies claims in parallel.

        All claims are verified concurrently using asyncio, significantly
        reducing total latency when many claims are extracted.

        Usage::

            import asyncio
            result = asyncio.run(safe_client.generate_async(model, prompt))
        """
        start = time.monotonic()
        usage = TokenUsage()

        # Phase 1 & 2 are sequential (need the draft before extraction)
        draft = await asyncio.to_thread(
            self._call_llm, model, self._draft_system_prompt, prompt, usage
        )

        if self.strictness == 0.0:
            return AntiHallucinationResponse(
                content=draft,
                verification_log=[{"phase": "bypassed", "reason": "strictness=0.0"}],
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
            )

        claims_text = await asyncio.to_thread(
            self._call_llm,
            model,
            self._extraction_prompt,
            f"Text to analyze:\n\n{draft}",
            usage,
        )

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
            )

        # Phase 3: Parallel claim verification
        critique_prompt = self._build_critique_prompt()
        use_tools = self.strictness >= 0.8 and len(self.tools) > 0

        async def _verify(claim: str) -> Dict[str, Any]:
            return await asyncio.to_thread(
                self._verify_single_claim,
                claim, model, critique_prompt, use_tools, usage,
            )

        verification_results = await asyncio.gather(
            *[_verify(c) for c in claims]
        )

        verification_log = list(verification_results)
        hallucinations_caught = [
            f"Claim '{r['claim']}' flagged: {r['critique']}"
            for r in verification_log
            if not r["is_valid"]
        ]

        # Phase 4: Correction
        if not hallucinations_caught:
            return AntiHallucinationResponse(
                content=draft,
                verification_log=verification_log,
                token_usage=usage,
                elapsed_seconds=time.monotonic() - start,
            )

        final_content = await asyncio.to_thread(
            self._call_llm,
            model,
            self._correction_prompt,
            (
                f"Original Draft:\n{draft}\n\n"
                f"Critiques to address:\n"
                + "\n\n".join(hallucinations_caught)
            ),
            usage,
        )

        return AntiHallucinationResponse(
            content=final_content,
            hallucinations_caught=hallucinations_caught,
            verification_log=verification_log,
            token_usage=usage,
            elapsed_seconds=time.monotonic() - start,
        )
