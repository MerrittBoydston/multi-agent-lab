from __future__ import annotations

import json
from typing import Any


class OutputValidationError(ValueError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise OutputValidationError("output must be a JSON object")
    return value


def _require_string(value: dict[str, Any], field: str) -> None:
    if not isinstance(value.get(field), str) or not value[field].strip():
        raise OutputValidationError(f"{field} must be a non-empty string")


def _require_string_list(value: dict[str, Any], field: str) -> None:
    items = value.get(field)
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise OutputValidationError(f"{field} must be an array of strings")


def _require_confidence(value: dict[str, Any]) -> None:
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise OutputValidationError("confidence must be a number from 0 to 1")


def validate_output(role: str, text: str) -> str:
    value = _extract_json(text)
    if role == "producer":
        _require_string(value, "proposal")
        _require_string_list(value, "assumptions")
        _require_string_list(value, "risks")
    elif role == "critic":
        _require_string_list(value, "findings")
        if value.get("recommendation") not in {"approve", "revise", "reject"}:
            raise OutputValidationError("recommendation must be approve, revise, or reject")
    elif role == "reviewer":
        _require_string(value, "decision")
        _require_string(value, "final")
        _require_string_list(value, "unresolved_questions")
    else:
        raise OutputValidationError(f"unknown agent role {role!r}")
    _require_confidence(value)
    return json.dumps(value, indent=2, ensure_ascii=False)
