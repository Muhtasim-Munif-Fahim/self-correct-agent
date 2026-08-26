"""Mask sensitive spans before reports and logs are persisted.

Verification reports can quote whatever the model was given, so API keys,
tokens, or internal hostnames typed into a prompt would otherwise be
written verbatim into ``--output`` files and printed reports. A redactor
runs configurable regular expressions over the rendered report and
replaces each matched span with a fixed placeholder before anything is
written.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: Placeholder written in place of every matched span unless a rule says
#: otherwise.
DEFAULT_REPLACEMENT = "[REDACTED]"

#: Flag names accepted in a redaction file, mapped onto ``re`` constants.
_REGEX_FLAG_NAMES = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "ASCII": re.ASCII,
}


class SecretRedactor:
    """Apply compiled redaction rules to text."""

    def __init__(
        self,
        rules: List[Tuple[Any, str]],
        default_replacement: str = DEFAULT_REPLACEMENT,
    ) -> None:
        if not isinstance(default_replacement, str) or not default_replacement:
            raise ValueError("default replacement must be a non-empty string")
        self._rules: List[Tuple[re.Pattern, str]] = []
        for index, (pattern, replacement) in enumerate(rules):
            if not isinstance(replacement, str) or not replacement:
                raise ValueError(
                    f"redaction rule {index} replacement must be a non-empty string"
                )
            if pattern.search("") is not None:
                raise ValueError(
                    f"redaction rule {index} pattern must not match an empty span"
                )
            self._rules.append((pattern, replacement))

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def redact(self, text: str) -> str:
        """Return ``text`` with every matched span replaced."""

        for pattern, replacement in self._rules:
            text = pattern.sub(replacement, text)
        return text


def load_redaction_rules(path: str | Path) -> SecretRedactor:
    """Build a :class:`SecretRedactor` from a JSON definition file.

    The file contains a ``rules`` list; every entry needs a non-empty
    ``name`` and ``pattern``, plus an optional per-rule ``replacement``
    and ``flags`` drawn from :data:`_REGEX_FLAG_NAMES`. An optional
    top-level ``replacement`` overrides the built-in default for rules
    that do not name their own. Patterns that can match the empty string
    are rejected: they would shred the surrounding text instead of
    masking identifiable spans.
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read redaction rules '{source}': {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
        raise ValueError("redaction file must contain a 'rules' list")

    default_replacement = payload.get("replacement", DEFAULT_REPLACEMENT)
    rules: List[Tuple[Any, str]] = []
    for entry in payload["rules"]:
        if not isinstance(entry, dict):
            raise ValueError("every redaction rule must be a JSON object")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("redaction rule name must be a non-empty string")
        raw_pattern = entry.get("pattern")
        if not isinstance(raw_pattern, str) or not raw_pattern:
            raise ValueError(f"redaction rule '{name}' needs a non-empty pattern")
        flags = 0
        for flag_name in entry.get("flags") or []:
            if flag_name not in _REGEX_FLAG_NAMES:
                raise ValueError(f"unknown regex flag: {flag_name!r}")
            flags |= _REGEX_FLAG_NAMES[flag_name]
        try:
            pattern = re.compile(raw_pattern, flags)
        except re.error as exc:
            raise ValueError(f"redaction rule '{name}' has an invalid pattern: {exc}") from exc
        replacement = entry.get("replacement", default_replacement)
        if not isinstance(replacement, str) or not replacement:
            raise ValueError(f"redaction rule '{name}' replacement must be a non-empty string")
        rules.append((pattern, replacement))
    return SecretRedactor(rules, default_replacement=default_replacement)