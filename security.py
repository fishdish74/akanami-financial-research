"""Security helpers for useful diagnostics that never expose credentials."""

from __future__ import annotations

import os
import re
from typing import Any


_KEY_SHAPE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_AUTHORIZATION = re.compile(r"(?i)(\bauthorization\s*[=:]\s*)[^,;]+")
_NAMED_SECRET = re.compile(r"(?i)((?:api[_ -]?key)\s*[=:]\s*)[^\s,;]+")


def redact_secrets(value: object, *, limit: int = 500) -> str:
    """Return bounded, one-line diagnostic text with secrets removed."""
    text = str(value).replace("\r", " ").replace("\n", " ")
    configured_key = os.getenv("OPENAI_API_KEY", "")
    if configured_key:
        text = text.replace(configured_key, "[REDACTED]")
    text = _KEY_SHAPE.sub("[REDACTED]", text)
    text = _AUTHORIZATION.sub(r"\1[REDACTED]", text)
    text = _NAMED_SECRET.sub(r"\1[REDACTED]", text)
    return text[:limit]


def safe_error(exc: BaseException, *, category: str = "runtime") -> dict[str, Any]:
    """Extract stable, sanitized metadata from SDK, HTTP, and local exceptions."""
    body = getattr(exc, "body", None)
    nested = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(nested, dict):
        nested = {}
    status = getattr(exc, "status_code", None)
    code = nested.get("code") or getattr(exc, "code", None)
    error_type = nested.get("type") or type(exc).__name__
    message = nested.get("message") or str(exc) or type(exc).__name__
    result: dict[str, Any] = {
        "category": category,
        "type": redact_secrets(error_type, limit=120),
        "message": redact_secrets(message),
    }
    if isinstance(status, int):
        result["status"] = status
    if code is not None:
        result["code"] = redact_secrets(code, limit=120)
    return result
