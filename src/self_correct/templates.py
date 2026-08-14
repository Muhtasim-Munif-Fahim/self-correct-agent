"""Prompt templates for common verification tasks.

Chain-of-Verification pays off most when the prompt asks for something
checkable. These templates encode that: each one pushes the model toward
discrete, attributable claims rather than prose that cannot be verified
claim by claim.

Built-ins live here. Users can add their own as ``.txt`` files in the
templates directory; a user template shadows a built-in of the same name, so a
built-in can be overridden without editing the package.
"""

from __future__ import annotations

import os
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Environment variable overriding where user templates are read from.
TEMPLATE_DIR_ENV = "SELF_CORRECT_TEMPLATES"

#: name -> (one-line description, template body)
BUILTIN_TEMPLATES: Dict[str, Tuple[str, str]] = {
    "factual-summary": (
        "Summarise a topic as discrete, checkable statements",
        "Summarise $topic.\n\n"
        "State each fact as its own sentence. Include dates, quantities and "
        "proper names wherever they apply. Do not include statements you "
        "cannot support.",
    ),
    "technical-explanation": (
        "Explain a mechanism for a technical reader",
        "Explain how $topic works, for a reader who is technical but new to "
        "this area.\n\n"
        "Describe the mechanism step by step. Name the components involved "
        "and what each one does. Where a figure or limit applies, give it "
        "explicitly rather than describing it as large or small.",
    ),
    "research-answer": (
        "Answer a question with the evidence separated from the conclusion",
        "Answer this question: $question\n\n"
        "Give the answer first in one sentence. Then list the evidence "
        "supporting it, one item per line. Finish with anything that would "
        "change the answer if it turned out to be false.",
    ),
    "comparison": (
        "Compare two things on stated criteria",
        "Compare $a and $b.\n\n"
        "Use these criteria: $criteria. Address every criterion for both. "
        "Where they are equivalent, say so rather than manufacturing a "
        "difference.",
    ),
    "timeline": (
        "Lay out events in order with dates",
        "Give a timeline of $topic.\n\n"
        "One event per line, earliest first, each beginning with its date. "
        "Where a date is disputed or approximate, mark it as such.",
    ),
}


def template_dir() -> Path:
    """Return the directory user templates are read from."""

    override = os.environ.get(TEMPLATE_DIR_ENV)
    if override:
        return Path(override)
    return Path.home() / ".self-correct" / "templates"


def user_templates() -> Dict[str, str]:
    """Return user-defined templates as name -> body."""

    directory = template_dir()
    found: Dict[str, str] = {}
    try:
        paths = sorted(directory.glob("*.txt"))
    except OSError:
        return found
    for path in paths:
        try:
            found[path.stem] = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return found


def list_templates() -> List[Tuple[str, str, str]]:
    """Return (name, source, description) for every available template.

    A user template shadows a built-in of the same name.
    """

    custom = user_templates()
    rows: List[Tuple[str, str, str]] = []
    for name, (description, _) in BUILTIN_TEMPLATES.items():
        if name in custom:
            rows.append((name, "user (overrides built-in)", _first_line(custom[name])))
        else:
            rows.append((name, "built-in", description))
    for name, body in custom.items():
        if name not in BUILTIN_TEMPLATES:
            rows.append((name, "user", _first_line(body)))
    return sorted(rows, key=lambda row: row[0])


def _first_line(body: str) -> str:
    line = body.strip().splitlines()[0] if body.strip() else ""
    return line if len(line) <= 60 else line[:57] + "..."


def get_template(name: str) -> Optional[str]:
    """Return a template body by name, or None if there is no such template."""

    custom = user_templates()
    if name in custom:
        return custom[name]
    if name in BUILTIN_TEMPLATES:
        return BUILTIN_TEMPLATES[name][1]
    return None


def placeholders(body: str) -> List[str]:
    """Return the $variable names a template expects, in order of appearance."""

    seen: List[str] = []
    for match in re.finditer(r"\$(\w+)|\$\{(\w+)\}", body):
        name = match.group(1) or match.group(2)
        if name not in seen:
            seen.append(name)
    return seen


def render(body: str, values: Dict[str, str]) -> Tuple[Optional[str], List[str]]:
    """Fill a template. Returns (rendered, missing_placeholder_names).

    Rendering is refused when a placeholder has no value, rather than leaving
    a literal "$topic" in a prompt that then gets sent to a model.
    """

    missing = [name for name in placeholders(body) if name not in values]
    if missing:
        return None, missing
    return string.Template(body).safe_substitute(values), []
