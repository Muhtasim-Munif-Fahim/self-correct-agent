"""JUnit XML export of verification results for CI dashboards."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Mapping


def _result_from(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the embedded result of a session payload, or the payload itself."""

    result = payload.get("result")
    return result if isinstance(result, dict) else payload


def result_to_junit_xml(
    payload: Mapping[str, Any], *, name: str = "self-correct"
) -> str:
    """Render verification verdicts as a JUnit XML test suite.

    Every claim verdict in the verification log becomes a testcase; flagged
    claims carry a ``<failure>`` element whose message is the critique text.
    Log entries without a per-claim verdict (phase warnings, budget halts)
    are skipped. Accepts a full session written by :func:`save_session` or
    the bare result object it embeds.
    """

    result = _result_from(payload)
    suite = ET.Element("testsuite", {"name": name})

    cases = 0
    failures = 0
    for entry in result.get("verification_log") or []:
        if not isinstance(entry, dict):
            continue
        if "is_valid" not in entry or not entry.get("claim"):
            continue
        claim = str(entry["claim"])
        case = ET.SubElement(suite, "testcase", {"name": claim})
        cases += 1
        if entry["is_valid"] is False:
            critique = str(entry.get("critique", ""))
            failure = ET.SubElement(case, "failure", {"message": critique})
            failure.text = critique
            failures += 1

    suite.set("tests", str(cases))
    suite.set("failures", str(failures))
    elapsed = result.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)):
        suite.set("time", f"{float(elapsed):.3f}")

    ET.indent(suite)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        suite, encoding="unicode"
    )
