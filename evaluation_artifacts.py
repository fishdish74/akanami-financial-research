"""Append-only, inert JSON artifacts for live semantic-validation failures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from config import PROJECT_DIR, ModelConfig
from renderer import escape_external_markdown
from security import redact_secrets


ARTIFACT_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION = "ModelAnalysisPayload-v1"
EVALUATION_ROOT = (PROJECT_DIR / "evaluations").resolve()
DEFAULT_REJECTED_CLAIM_ROOT = (
    EVALUATION_ROOT / "market-analysis-v5" / "rejected-claims"
)
_SAFE_IDENTIFIER = re.compile(r"[^a-z0-9_-]+")
_CODE_OWNED_RUN_ID = re.compile(
    r"^run-\d{8}T\d{12}Z-[a-f0-9]{12}$"
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_MAX_TEXT_CHARACTERS = 10_000


def sanitize_artifact_text(value: object, *, limit: int = _MAX_TEXT_CHARACTERS) -> str:
    """Return bounded, one-line, secret-redacted text with active markup neutralized."""

    text = _CONTROL_CHARACTERS.sub(" ", str(value))
    text = escape_external_markdown(text)
    text = redact_secrets(text, limit=limit)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _sanitize_string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [sanitize_artifact_text(item, limit=500) for item in value]


def _safe_identifier(value: object, *, fallback: str) -> str:
    normalized = _SAFE_IDENTIFIER.sub("-", str(value).casefold()).strip("-_")
    return normalized[:80] or fallback


def new_run_id(fixture_sha256: str, *, now: datetime | None = None) -> str:
    """Create a code-owned sortable run ID; caller/model text never becomes a path."""

    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("run timestamp must be timezone-aware")
    utc = instant.astimezone(timezone.utc)
    stamp = utc.strftime("%Y%m%dT%H%M%S%fZ")
    suffix = hashlib.sha256(f"{stamp}\0{fixture_sha256}".encode()).hexdigest()[:12]
    return f"run-{stamp}-{suffix}"


def build_rejected_claim_records(
    *,
    case_id: object,
    error: BaseException,
    config: ModelConfig,
    run_id: str,
    timestamp: datetime | None = None,
) -> list[dict[str, object]]:
    """Whitelist and sanitize only claim-level fields attached by local validation."""

    instant = timestamp or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("failure timestamp must be timezone-aware")
    safe_case_id = _safe_identifier(case_id, fallback="case")
    records: list[dict[str, object]] = []
    for raw in getattr(error, "rejected_claims", ()):
        if not isinstance(raw, Mapping):
            continue
        try:
            claim_index = int(raw.get("claim_index", 0))
        except (TypeError, ValueError):
            continue
        if claim_index <= 0:
            continue
        validator_errors = _sanitize_string_list(raw.get("validator_errors"))
        records.append({
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "record_type": "rejected_claim",
            "run_id": run_id,
            "case_id": safe_case_id,
            "claim_index": claim_index,
            "claim_text": sanitize_artifact_text(raw.get("claim_text", "")),
            "causal_role": sanitize_artifact_text(raw.get("causal_role", ""), limit=100),
            "category": sanitize_artifact_text(raw.get("category", ""), limit=100),
            "support": sanitize_artifact_text(raw.get("support", ""), limit=100),
            "evidence_ids": _sanitize_string_list(raw.get("evidence_ids")),
            "semantic_facets": _sanitize_string_list(raw.get("facets")),
            "validator_error": "; ".join(validator_errors),
            "prompt_version": sanitize_artifact_text(config.prompt_version, limit=100),
            "pipeline_version": sanitize_artifact_text(config.pipeline_version, limit=100),
            "model": sanitize_artifact_text(
                getattr(error, "response_model", None) or config.model, limit=200
            ),
            "request_id": sanitize_artifact_text(
                getattr(error, "request_id", None) or "", limit=200
            ) or None,
            "timestamp": instant.astimezone(timezone.utc).isoformat(),
        })
    return records


def _checked_artifact_root(root: Path) -> Path:
    resolved = root.resolve()
    try:
        resolved.relative_to(EVALUATION_ROOT)
    except ValueError as exc:
        raise ValueError("failure artifacts must stay under the evaluation directory") from exc
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("failure artifact directory cannot be a symlink")
    return resolved


def validate_evaluation_output_path(path: Path) -> Path:
    """Allow create-once JSON summaries only in evaluation or OS temporary storage."""

    if path.suffix.casefold() != ".json":
        raise ValueError("evaluation summary output must be JSON")
    resolved = path.resolve()
    allowed_roots = {
        EVALUATION_ROOT,
        Path(tempfile.gettempdir()).resolve(),
        Path("/private/tmp").resolve(),
        Path("/tmp").resolve(),
    }
    if not any(
        resolved == root or root in resolved.parents
        for root in allowed_roots
    ):
        raise ValueError("evaluation summary must stay in evaluation or temporary storage")
    if path.exists() or path.is_symlink():
        raise FileExistsError("evaluation summary output already exists")
    return resolved


def _write_json_exclusive(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _accepted_case_summary(case: Mapping[str, object]) -> dict[str, object]:
    summary = case.get("accepted_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("accepted case lacks a validated accepted_summary")
    result: dict[str, object] = {
        "case_id": sanitize_artifact_text(case.get("case_id", ""), limit=200),
        "status": "accepted",
        "title": sanitize_artifact_text(summary.get("title", "")),
        "executive_summary": sanitize_artifact_text(summary.get("executive_summary", "")),
        "claim_count": int(summary.get("claim_count", 0)),
        "request_id": sanitize_artifact_text(case.get("request_id", ""), limit=200) or None,
        "model": sanitize_artifact_text(case.get("response_model", ""), limit=200),
    }
    confidence_result = case.get("confidence_result")
    if isinstance(confidence_result, Mapping):
        encoded = json.dumps(confidence_result, allow_nan=False, sort_keys=True)
        if len(encoded) > 100_000:
            raise ValueError("confidence audit record is unexpectedly large")
        result["confidence_result"] = json.loads(encoded)
    return result


def write_canary_artifacts(
    output: Mapping[str, object], *, artifact_root: Path, run_id: str
) -> tuple[Path, list[Path]]:
    """Write one run manifest plus separate rejected-claim records, all append-only."""

    root = _checked_artifact_root(artifact_root)
    if not _CODE_OWNED_RUN_ID.fullmatch(run_id):
        raise ValueError("run ID is not code-owned safe syntax")
    safe_run_id = run_id
    accepted: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []
    for case in output.get("cases", ()):
        if not isinstance(case, Mapping):
            continue
        if case.get("status") == "PASS":
            accepted.append(_accepted_case_summary(case))
            continue
        for record in case.get("rejected_claim_records", ()):
            if isinstance(record, Mapping):
                rejected_records.append(dict(record))

    record_paths: list[Path] = []
    for record in rejected_records:
        case_slug = _safe_identifier(record.get("case_id"), fallback="case")
        claim_index = int(record.get("claim_index", 0))
        if claim_index <= 0:
            raise ValueError("rejected claim index must be positive")
        record_paths.append(root / f"{safe_run_id}--{case_slug}--claim-{claim_index:03d}.json")
    manifest_path = root / f"{safe_run_id}--manifest.json"
    planned = [*record_paths, manifest_path]
    if len({path.name for path in planned}) != len(planned):
        raise ValueError("duplicate rejected claim identity in one run")
    for path in planned:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"evaluation artifact already exists: {path.name}")

    for path, record in zip(record_paths, rejected_records, strict=True):
        _write_json_exclusive(path, record)
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_type": "live_canary_run_manifest",
        "run_id": safe_run_id,
        "evaluation_id": sanitize_artifact_text(output.get("evaluation_id", ""), limit=200),
        "fixture_sha256": sanitize_artifact_text(output.get("fixture_sha256", ""), limit=100),
        "model_requested": sanitize_artifact_text(output.get("model_requested", ""), limit=200),
        "prompt_version": sanitize_artifact_text(output.get("prompt_version", ""), limit=100),
        "pipeline_version": sanitize_artifact_text(output.get("pipeline_version", ""), limit=100),
        "accepted_cases": accepted,
        "rejected_claim_files": [path.name for path in record_paths],
        "validator_errors": [
            sanitize_artifact_text(case.get("error", {}).get("message", ""))
            for case in output.get("cases", ())
            if isinstance(case, Mapping)
            and case.get("status") != "PASS"
            and isinstance(case.get("error"), Mapping)
        ],
        "totals": output.get("totals", {}),
        "stopped_early": bool(output.get("stopped_early", False)),
        "stop_reason": (
            sanitize_artifact_text(output["stop_reason"])
            if output.get("stop_reason") is not None else None
        ),
        "started_at": output.get("started_at"),
        "finished_at": output.get("finished_at"),
    }
    _write_json_exclusive(manifest_path, manifest)
    return manifest_path, record_paths
