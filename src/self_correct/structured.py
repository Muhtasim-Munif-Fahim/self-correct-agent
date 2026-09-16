"""Structured LLM output for claim extraction and verification results.

OpenAI-compatible clients can return JSON via JSON mode
(``response_format={"type": "json_object"}``) or via tool/function calling.
This module holds the request schemas, response parsers, and the typed result
objects returned by :meth:`AntiHallucinationResponse.to_structured`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Modes accepted by ``AntiHallucinator(structured_output=...)``.
STRUCTURED_MODES = ("json", "function")

EXTRACT_FUNCTION_NAME = "extract_claims"
VERIFY_FUNCTION_NAME = "verify_claim"

EXTRACT_CLAIMS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Discrete factual claims extracted from the text.",
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

CLAIM_VERDICT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_valid": {
            "type": "boolean",
            "description": "True when the claim is factually supported.",
        },
        "critique": {
            "type": "string",
            "description": "Short explanation of the verdict.",
        },
    },
    "required": ["is_valid", "critique"],
    "additionalProperties": False,
}

EXTRACT_JSON_INSTRUCTION = (
    " Return a JSON object with a 'claims' array of strings, "
    "one discrete factual claim per element. Output JSON only."
)

VERIFY_JSON_INSTRUCTION = (
    " Return a JSON object with boolean 'is_valid' and string 'critique'. "
    "Set is_valid to true only when the claim is factually supported. "
    "Output JSON only."
)

_UNSUPPORTED_HINTS = (
    "response_format",
    "json_object",
    "json_schema",
    "tool_choice",
    "unknown parameter",
    "unrecognized request argument",
    "unexpected keyword",
    "invalid parameter",
    "extra fields not permitted",
)


def normalize_structured_mode(value: Any) -> Optional[str]:
    """Coerce constructor/CLI values to ``'json'``, ``'function'``, or ``None``."""

    if value is None or value is False:
        return None
    if value is True:
        return "json"
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in ("", "off", "none", "false", "0"):
            return None
        if mode in ("on", "true", "1", "json", "json_object", "json-mode", "json_mode"):
            return "json"
        if mode in ("function", "functions", "tool", "tools", "function_call"):
            return "function"
        raise ValueError(
            f"unknown structured_output mode {value!r}; expected 'json' or 'function'"
        )
    raise ValueError("structured_output must be False, True, 'json', or 'function'")


def append_json_instruction(prompt: str, kind: str) -> str:
    """Ensure a JSON-mode prompt actually asks for JSON."""

    suffix = EXTRACT_JSON_INSTRUCTION if kind == "extract" else VERIFY_JSON_INSTRUCTION
    text = prompt or ""
    if "json" in text.lower():
        return text
    return text.rstrip() + suffix


def function_calling_kwargs(name: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build OpenAI tool-calling kwargs that force one function invocation."""

    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": name}},
    }


def request_kwargs(mode: str, kind: str) -> Dict[str, Any]:
    """Extra ``chat.completions.create`` kwargs for a structured pipeline phase."""

    if mode == "json":
        return {"response_format": {"type": "json_object"}}
    if mode == "function":
        if kind == "extract":
            return function_calling_kwargs(
                EXTRACT_FUNCTION_NAME,
                "Extract discrete factual claims from the draft.",
                EXTRACT_CLAIMS_SCHEMA,
            )
        return function_calling_kwargs(
            VERIFY_FUNCTION_NAME,
            "Verify a factual claim and return a structured verdict.",
            CLAIM_VERDICT_SCHEMA,
        )
    raise ValueError(f"unknown structured output mode: {mode!r}")


def is_unsupported_structured_error(exc: BaseException) -> bool:
    """Return True when a provider rejected JSON mode or tool-calling kwargs."""

    if isinstance(exc, TypeError):
        return True
    text = str(exc).lower()
    return any(hint in text for hint in _UNSUPPORTED_HINTS)


def parse_json_text(text: Any) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from model text, including fenced Markdown blocks."""

    if not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _function_arguments(obj: Any) -> Optional[str]:
    """Read a JSON-arguments string from a tool call or function_call object."""

    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return None
    if isinstance(obj, dict):
        nested = obj.get("function") if isinstance(obj.get("function"), dict) else {}
        arguments = obj.get("arguments") or nested.get("arguments")
        return arguments if isinstance(arguments, str) else None
    arguments = getattr(obj, "arguments", None)
    if isinstance(arguments, str):
        return arguments
    function = getattr(obj, "function", None)
    if function is None or function is obj:
        return None
    nested = function.get("arguments") if isinstance(function, dict) else getattr(
        function, "arguments", None
    )
    return nested if isinstance(nested, str) else None


def completion_text(response: Any) -> str:
    """Return ``message.content`` as text, or an empty string when absent."""

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def payload_from_response(response: Any) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from JSON-mode content or a function/tool call."""

    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return None

    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, (list, tuple)):
        for call in tool_calls:
            parsed = parse_json_text(_function_arguments(call))
            if parsed is not None:
                return parsed

    parsed = parse_json_text(_function_arguments(getattr(message, "function_call", None)))
    if parsed is not None:
        return parsed
    return parse_json_text(getattr(message, "content", None))


def parse_extracted_claims(payload: Dict[str, Any]) -> Optional[List[str]]:
    """Validate an extraction payload; return None when the schema does not match."""

    claims = payload.get("claims")
    if not isinstance(claims, list):
        return None
    return [str(item).strip() for item in claims if str(item).strip()]


def parse_claim_verdict(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate a claim-verification payload into ``is_valid`` and ``critique``."""

    if "is_valid" in payload:
        raw = payload["is_valid"]
    elif "verified" in payload:
        raw = payload["verified"]
    else:
        return None
    if isinstance(raw, str):
        is_valid = raw.strip().lower() in ("true", "yes", "1")
    else:
        is_valid = bool(raw)
    critique = payload.get("critique") or payload.get("reason") or payload.get("explanation") or ""
    return {"is_valid": is_valid, "critique": str(critique)}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


@dataclass
class StructuredClaim:
    """One claim verdict in a typed verification result."""

    claim: str
    is_valid: Optional[bool] = None
    critique: str = ""
    evidence_used: bool = False
    evidence_sources: List[Dict[str, str]] = field(default_factory=list)
    cached: bool = False
    skipped_by_budget: bool = False
    phase: Optional[str] = None
    structured: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredClaim":
        if not isinstance(data, dict):
            raise ValueError("structured claim must be a JSON object")
        is_valid = data.get("is_valid")
        if is_valid is not None and not isinstance(is_valid, bool):
            is_valid = _as_bool(is_valid)
        sources = [
            {str(key): str(value) for key, value in source.items()}
            for source in (data.get("evidence_sources") or [])
            if isinstance(source, dict)
        ]
        phase = data.get("phase")
        return cls(
            claim=str(data.get("claim", "")),
            is_valid=is_valid,
            critique=str(data.get("critique", "") or ""),
            evidence_used=_as_bool(data.get("evidence_used", False)),
            evidence_sources=sources,
            cached=_as_bool(data.get("cached", False)),
            skipped_by_budget=_as_bool(data.get("skipped_by_budget", False)),
            phase=str(phase) if phase is not None else None,
            structured=_as_bool(data.get("structured", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "claim": self.claim,
            "is_valid": self.is_valid,
            "critique": self.critique,
            "evidence_used": self.evidence_used,
            "evidence_sources": list(self.evidence_sources),
            "cached": self.cached,
            "skipped_by_budget": self.skipped_by_budget,
            "structured": self.structured,
        }
        if self.phase is not None:
            payload["phase"] = self.phase
        return payload


@dataclass
class StructuredVerificationResult:
    """Typed verification result: claims, summaries, and token usage.

    This is a validated view of :class:`AntiHallucinationResponse` suitable for
    JSON APIs and Pydantic ``model_validate`` (see :meth:`to_pydantic`).
    """

    content: str
    claims: List[StructuredClaim] = field(default_factory=list)
    hallucinations_caught: List[str] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    phase_timings: Dict[str, float] = field(default_factory=dict)
    token_usage_by_phase: Dict[str, Dict[str, int]] = field(default_factory=dict)
    claim_summary: Dict[str, int] = field(default_factory=dict)
    severity_summary: Dict[str, int] = field(default_factory=dict)
    hallucination_density_report: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, response: Any) -> "StructuredVerificationResult":
        """Build a typed result from an :class:`AntiHallucinationResponse`."""

        claims = [
            StructuredClaim.from_dict(entry)
            for entry in (getattr(response, "verification_log", None) or [])
            if isinstance(entry, dict) and ("claim" in entry or "is_valid" in entry)
        ]
        usage = getattr(response, "token_usage", None)
        summary = getattr(response, "claim_summary", None)
        severity = getattr(response, "severity_summary", None)
        density = getattr(response, "hallucination_density_report", None)
        return cls(
            content=str(getattr(response, "content", "") or ""),
            claims=claims,
            hallucinations_caught=[
                str(item) for item in (getattr(response, "hallucinations_caught", None) or [])
            ],
            token_usage={
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0) if usage else 0,
            },
            elapsed_seconds=float(getattr(response, "elapsed_seconds", 0.0) or 0.0),
            phase_timings={
                str(key): float(value)
                for key, value in (getattr(response, "phase_timings", None) or {}).items()
                if isinstance(value, (int, float))
            },
            token_usage_by_phase=dict(getattr(response, "token_usage_by_phase", None) or {}),
            claim_summary=dict(summary()) if callable(summary) else {},
            severity_summary=dict(severity()) if callable(severity) else {},
            hallucination_density_report=dict(density()) if callable(density) else {},
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredVerificationResult":
        if not isinstance(data, dict):
            raise ValueError("structured verification result must be a JSON object")
        raw_claims = data.get("claims") or []
        if not isinstance(raw_claims, list):
            raise ValueError("claims must be a list")
        raw_usage = data.get("token_usage") or {}
        if not isinstance(raw_usage, dict):
            raise ValueError("token_usage must be a JSON object")
        return cls(
            content=str(data.get("content", "")),
            claims=[
                StructuredClaim.from_dict(item)
                for item in raw_claims
                if isinstance(item, dict)
            ],
            hallucinations_caught=[
                str(item) for item in (data.get("hallucinations_caught") or [])
            ],
            token_usage={
                "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(raw_usage.get("total_tokens", 0) or 0),
            },
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0) or 0.0),
            phase_timings={
                str(key): float(value)
                for key, value in (data.get("phase_timings") or {}).items()
                if isinstance(value, (int, float))
            },
            token_usage_by_phase=dict(data.get("token_usage_by_phase") or {}),
            claim_summary=dict(data.get("claim_summary") or {}),
            severity_summary=dict(data.get("severity_summary") or {}),
            hallucination_density_report=dict(
                data.get("hallucination_density_report") or {}
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "claims": [claim.to_dict() for claim in self.claims],
            "hallucinations_caught": list(self.hallucinations_caught),
            "token_usage": dict(self.token_usage),
            "elapsed_seconds": self.elapsed_seconds,
            "phase_timings": dict(self.phase_timings),
            "token_usage_by_phase": dict(self.token_usage_by_phase),
            "claim_summary": dict(self.claim_summary),
            "severity_summary": dict(self.severity_summary),
            "hallucination_density_report": dict(self.hallucination_density_report),
        }

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)

    def to_pydantic(self) -> Any:
        """Validate this result as a Pydantic model (v1 or v2).

        ``pydantic`` is already pulled in by the OpenAI client dependency.
        """

        try:
            from pydantic import BaseModel, Field
        except ImportError as exc:
            raise ImportError(
                "pydantic is required for StructuredVerificationResult.to_pydantic()"
            ) from exc

        class StructuredClaimModel(BaseModel):
            claim: str
            is_valid: Optional[bool] = None
            critique: str = ""
            evidence_used: bool = False
            evidence_sources: List[Dict[str, str]] = Field(default_factory=list)
            cached: bool = False
            skipped_by_budget: bool = False
            phase: Optional[str] = None
            structured: bool = False

        class StructuredVerificationResultModel(BaseModel):
            content: str
            claims: List[StructuredClaimModel]
            hallucinations_caught: List[str]
            token_usage: Dict[str, int]
            elapsed_seconds: float
            phase_timings: Dict[str, float] = Field(default_factory=dict)
            token_usage_by_phase: Dict[str, Dict[str, int]] = Field(default_factory=dict)
            claim_summary: Dict[str, int] = Field(default_factory=dict)
            severity_summary: Dict[str, int] = Field(default_factory=dict)
            hallucination_density_report: Dict[str, Any] = Field(default_factory=dict)

        data = self.to_dict()
        if hasattr(StructuredVerificationResultModel, "model_validate"):
            return StructuredVerificationResultModel.model_validate(data)
        return StructuredVerificationResultModel.parse_obj(data)
