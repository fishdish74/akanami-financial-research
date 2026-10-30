"""Deterministic Obsidian views for current and historical rankings.

This module deliberately has a narrower write scope than the collector's research
note writer. Current ranking views can only be refreshed at four exact generated
paths, and historical snapshots remain create-once.
"""

from __future__ import annotations

import hashlib
import fcntl
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from config import writes_enabled
from obsidian import write_new_note_atomic

from .models import RankingResult
from .pipeline import AnalyticsRun
from .quality import freshness_report

HORIZON_FILE_NAMES = {
    "one_day": "1 Day Ranking.md",
    "one_week": "1 Week Ranking.md",
    "one_month": "1 Month Ranking.md",
    "one_year": "1 Year Ranking.md",
}

HISTORY_DIRECTORY_NAMES = {
    "one_day": "Daily",
    "one_week": "Weekly",
    "one_month": "Monthly",
    "one_year": "Yearly",
}

HORIZON_LABELS = {
    "one_day": "1 Day",
    "one_week": "1 Week",
    "one_month": "1 Month",
    "one_year": "1 Year",
}

RANKING_VIEW_WRITE_ENV = "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES"


def ranking_view_writes_enabled() -> bool:
    """Require both the collector gate and a ranking-view-specific gate."""

    return writes_enabled() and os.getenv(RANKING_VIEW_WRITE_ENV, "0") == "1"


def _validate_horizon(horizon: str) -> None:
    if horizon not in HORIZON_FILE_NAMES:
        allowed = ", ".join(HORIZON_FILE_NAMES)
        raise ValueError(f"Unknown ranking horizon {horizon!r}; expected one of {allowed}")


def current_ranking_path(vault_root: Path, horizon: str) -> Path:
    """Return the sole allowed Current path for a ranking horizon."""

    _validate_horizon(horizon)
    return vault_root / "Rankings" / "Current" / HORIZON_FILE_NAMES[horizon]


def historical_filename(run: AnalyticsRun, horizon: str) -> str:
    """Build a deterministic, same-day-safe snapshot filename."""

    _validate_horizon(horizon)
    run_date = run.manifest.timestamp.date().isoformat()
    return f"{run_date} - {run.manifest.run_id}.md"


def historical_ranking_path(vault_root: Path, run: AnalyticsRun, horizon: str) -> Path:
    """Return the immutable History path for a run and horizon."""

    _validate_horizon(horizon)
    return (
        vault_root
        / "Rankings"
        / "History"
        / HISTORY_DIRECTORY_NAMES[horizon]
        / historical_filename(run, horizon)
    )


def proposed_ranking_paths(vault_root: Path, run: AnalyticsRun) -> dict[str, dict[str, Path]]:
    """Describe all output paths without touching the filesystem."""

    return {
        horizon: {
            "current": current_ranking_path(vault_root, horizon),
            "history": historical_ranking_path(vault_root, run, horizon),
        }
        for horizon in HORIZON_FILE_NAMES
    }


def _escape(value: Any) -> str:
    if value is None:
        return "—"
    text = " ".join(str(value).splitlines())
    for character in ("\\", "`", "*", "_", "[", "]", "#", ">", "|"):
        text = text.replace(character, f"\\{character}")
    return text


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def _frontmatter(
    run: AnalyticsRun, horizon: str, *, snapshot: bool, document_hash: str
) -> str:
    ranked_count = sum(
        result.ranking_status == "ranked" for result in run.rankings[horizon]
    )
    metadata = {
        "type": "generated-ranking-snapshot" if snapshot else "generated-ranking-view",
        "generated": True,
        "immutable": snapshot,
        "data_scope": "fictional-demo" if run.fixture_notice else "production",
        "generated_at": run.manifest.timestamp.isoformat(),
        "input_cutoff": run.manifest.input_cutoff.isoformat(),
        "ranking_horizon": horizon,
        "ranking_version": run.manifest.ranking_version,
        "pipeline_version": run.manifest.pipeline_version,
        "calculation_version": run.manifest.calculation_version,
        "run_id": run.manifest.run_id,
        "input_hash": run.manifest.input_hash,
        "config_hash": run.manifest.config_hash,
        "generated_document_sha256": document_hash,
        "companies_ranked": ranked_count,
        "files_written": run.manifest.files_written,
    }
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False).rstrip() + "\n---"


def _opportunity_table(results: list[RankingResult], companies: dict[str, Any]) -> str:
    lines = [
        "| Rank | Ticker | Company | Opportunity | Research Priority | Valuation | Quality | Growth | Financial Strength | Catalyst | Risk Resilience | Confidence | Status | Last Updated |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    ranked = [result for result in results if result.ranking_status == "ranked"]
    not_ranked = [result for result in results if result.ranking_status != "ranked"]
    for rank, result in enumerate(ranked, 1):
        company = companies[result.ticker]
        components = result.components
        last_updated = max(
            (point.as_of for point in company.financials.values()),
            default=None,
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    _escape(result.ticker),
                    _escape(result.company),
                    _score(result.opportunity_score),
                    _score(result.research_priority_score),
                    _score(components.valuation),
                    _score(components.quality),
                    _score(components.growth),
                    _score(components.financial_strength),
                    _score(components.catalysts),
                    _score(components.risk),
                    _escape(result.confidence),
                    _escape(result.ranking_status),
                    _escape(last_updated),
                ]
            )
            + " |"
        )
    if not_ranked:
        lines.extend(
            [
                "",
                "### Not Ranked",
                "",
                "| Ticker | Company | Status | Reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for result in not_ranked:
            reason = "; ".join(result.warnings) or "; ".join(result.missing_inputs)
            lines.append(
                f"| {_escape(result.ticker)} | {_escape(result.company)} | "
                f"{_escape(result.ranking_status)} | {_escape(reason or 'Not rankable')} |"
            )
    return "\n".join(lines)


def _priority_table(results: list[RankingResult]) -> str:
    lines = [
        "| Priority Rank | Ticker | Company | Research Priority | Opportunity | Confidence |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for rank, result in enumerate(results, 1):
        lines.append(
            f"| {rank} | {_escape(result.ticker)} | "
            f"{_escape(result.company)} | {_score(result.research_priority_score)} | "
            f"{_score(result.opportunity_score)} | {_escape(result.confidence)} |"
        )
    return "\n".join(lines)


def _freshness_table(results: list[RankingResult], companies: dict[str, Any], as_of: str) -> str:
    lines = [
        "| Ticker | Price Age | Fundamentals Age | Estimate Age | News Age | Confidence | Insufficient Data |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        report = freshness_report(companies[result.ticker], as_of=date.fromisoformat(as_of))
        missing = ", ".join(result.missing_inputs) if result.missing_inputs else "None"
        lines.append(
            f"| {_escape(result.ticker)} | {_escape(report.price_age)} | "
            f"{_escape(report.fundamentals_age)} | {_escape(report.estimate_age)} | "
            f"{_escape(report.news_age)} | {_escape(result.confidence)} | {_escape(missing)} |"
        )
    return "\n".join(lines)


def _catalysts_section(results: list[RankingResult], companies: dict[str, Any]) -> str:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    lines: list[str] = []
    for result in results:
        items = sorted(
            companies[result.ticker].catalysts,
            key=lambda item: (order.get(item.importance, 99), item.expected_date or item.source_date, item.name),
        )
        if not items:
            continue
        lines.append(f"### {_escape(result.ticker)} — {_escape(result.company)}")
        for item in items:
            lines.append(
                f"- **{_escape(item.importance.title())}: {_escape(item.name)}** — "
                f"{_escape(item.catalyst_type)}; expected: {_escape(item.expected_date or 'unknown')}; "
                f"status: {_escape(item.status)}; confidence: {_escape(item.confidence)}; "
                f"source: {_escape(item.source)}."
            )
        lines.append("")
    return "\n".join(lines).rstrip() if lines else "No catalysts were supplied."


def _risks_section(results: list[RankingResult], companies: dict[str, Any]) -> str:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    lines: list[str] = []
    for result in results:
        items = sorted(
            companies[result.ticker].risks,
            key=lambda item: (order.get(item.severity, 99), item.category, item.evidence),
        )
        if not items:
            continue
        lines.append(f"### {_escape(result.ticker)} — {_escape(result.company)}")
        for item in items:
            lines.append(
                f"- **{_escape(item.severity.title())}: "
                f"{_escape(item.category.replace('_', ' ').title())}** — "
                f"{_escape(item.evidence)} Confidence: {_escape(item.confidence)}; "
                f"freshness: {_escape(item.freshness)}."
            )
        lines.append("")
    return "\n".join(lines).rstrip() if lines else "No risks were supplied."


def _warnings_section(results: list[RankingResult]) -> str:
    lines = [
        f"- **{_escape(result.ticker)}:** {_escape(warning)}"
        for result in results
        for warning in result.warnings
    ]
    return "\n".join(lines) if lines else "No ranking warnings were generated."


def _insufficient_data_section(results: list[RankingResult]) -> str:
    lines = [
        f"- **{_escape(result.ticker)}:** "
        + _escape("; ".join(f"{name} — {reason}" for name, reason in result.unavailable_inputs.items()))
        for result in results
        if result.unavailable_inputs
    ]
    return "\n".join(lines) if lines else "No insufficient-data flags were generated."


def render_ranking_markdown(run: AnalyticsRun, horizon: str, *, snapshot: bool = False) -> str:
    """Render a stable human-readable ranking view or immutable snapshot."""

    _validate_horizon(horizon)
    results = run.rankings[horizon]
    priorities = run.research_priorities[horizon]
    companies = run.companies
    label = HORIZON_LABELS[horizon]
    title = f"{label} Ranking"
    kind = "Historical Ranking Snapshot" if snapshot else "Current Ranking View"
    ranked_count = sum(result.ranking_status == "ranked" for result in results)
    notice = (
        f"> [!warning] FICTIONAL DEMO DATA\n> {run.fixture_notice}"
        if run.fixture_notice
        else "> [!info] Generated ranking output\n> Verify freshness and warnings before use."
    )

    body_sections = [
        f"# {title}",
        notice,
        "## Ranking Metadata",
        f"- **View:** {kind}\n"
        f"- **Generated:** {run.manifest.timestamp.isoformat()}\n"
        f"- **Horizon:** {label} (`{horizon}`)\n"
        f"- **Ranking version:** `{run.manifest.ranking_version}`\n"
        f"- **Pipeline version:** `{run.manifest.pipeline_version}`\n"
        f"- **Calculation version:** `{run.manifest.calculation_version}`\n"
        f"- **Companies ranked:** {ranked_count}\n"
        f"- **Run ID:** `{run.manifest.run_id}`\n"
        f"- **Input hash:** `{run.manifest.input_hash}`\n"
        f"- **Config hash:** `{run.manifest.config_hash}`",
        "## Data Freshness and Confidence",
        _freshness_table(results, companies, run.analysis_date),
        "## Opportunity Ranking",
        _opportunity_table(results, companies),
        "## Research-Priority Ranking",
        _priority_table(priorities),
        "## Ranking Movement",
        "Movement explanations are unavailable because no earlier comparable snapshot was supplied.",
        "## Major Catalysts",
        _catalysts_section(results, companies),
        "## Major Risks",
        _risks_section(results, companies),
        "## Important Warnings",
        _warnings_section(results),
        "## Insufficient-Data Flags",
        _insufficient_data_section(results),
    ]
    body = "\n\n".join(body_sections).rstrip() + "\n"
    marker = "__DOCUMENT_SHA256__"
    unsigned = _frontmatter(
        run, horizon, snapshot=snapshot, document_hash=marker
    ) + "\n\n" + body
    document_hash = hashlib.sha256(unsigned.encode("utf-8")).hexdigest()
    return unsigned.replace(marker, document_hash, 1)


def _parse_frontmatter(markdown: str) -> dict[str, Any]:
    if not markdown.startswith("---\n"):
        raise ValueError("Generated ranking content must start with YAML frontmatter")
    try:
        raw = markdown.split("---\n", 2)[1]
        parsed = yaml.safe_load(raw)
    except (IndexError, yaml.YAMLError) as exc:
        raise ValueError("Generated ranking content contains invalid YAML") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Generated ranking frontmatter must be a mapping")
    return parsed


def _validate_generated_content(markdown: str, horizon: str, *, snapshot: bool) -> None:
    if "\x00" in markdown:
        raise ValueError("Ranking Markdown contains a NUL byte")
    metadata = _parse_frontmatter(markdown)
    expected_type = "generated-ranking-snapshot" if snapshot else "generated-ranking-view"
    expected = {
        "type": expected_type,
        "generated": True,
        "immutable": snapshot,
        "ranking_horizon": horizon,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Ranking metadata {key!r} must be {value!r}")
    stored_hash = metadata.get("generated_document_sha256")
    if not isinstance(stored_hash, str) or len(stored_hash) != 64:
        raise ValueError("Generated ranking document fingerprint is missing")
    signed_line = f"generated_document_sha256: {stored_hash}"
    if markdown.count(signed_line) != 1:
        raise ValueError("Generated ranking document fingerprint is malformed")
    unsigned = markdown.replace(
        signed_line, "generated_document_sha256: __DOCUMENT_SHA256__", 1
    )
    actual_hash = hashlib.sha256(unsigned.encode("utf-8")).hexdigest()
    if stored_hash != actual_hash:
        raise ValueError("Generated ranking document was modified outside the generator")


def _require_ranking_write_gate() -> None:
    if not ranking_view_writes_enabled():
        raise PermissionError(
            "Ranking writes require MARKET_COLLECTOR_ENABLE_WRITES=1 and "
            f"{RANKING_VIEW_WRITE_ENV}=1"
        )


def _require_scoped_parent(vault_root: Path, destination: Path, *parts: str) -> None:
    """Reject intermediate symlinks or any path outside the exact generated tree."""

    root = vault_root.resolve()
    expected_parent = root.joinpath(*parts)
    actual_parent = destination.parent.resolve()
    if actual_parent != expected_parent:
        raise ValueError("Ranking destination was redirected outside its allowed generated folder")


def refresh_current_ranking_view(
    vault_root: Path, run: AnalyticsRun, horizon: str, markdown: str | None = None
) -> Path:
    """Atomically create/refresh one exact generated Current ranking file.

    An existing file must already identify itself as the matching generated view.
    Human-authored or research files can never be refreshed by this function.
    """

    _require_ranking_write_gate()
    _validate_horizon(horizon)
    content = markdown if markdown is not None else render_ranking_markdown(run, horizon)
    _validate_generated_content(content, horizon, snapshot=False)
    destination = current_ranking_path(vault_root, horizon)
    _require_scoped_parent(vault_root, destination, "Rankings", "Current")
    if destination.is_symlink():
        raise ValueError("Refusing to refresh a symlinked ranking view")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / ".ranking-refresh.lock"
    if lock_path.is_symlink():
        raise ValueError("Refusing a symlinked ranking refresh lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        original = destination.read_bytes() if destination.exists() else None
        if original is not None:
            _validate_generated_content(original.decode("utf-8"), horizon, snapshot=False)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent, delete=False, prefix=".ranking-"
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            current = destination.read_bytes() if destination.exists() else None
            if current != original:
                raise RuntimeError("ranking view changed concurrently; refresh aborted")
            os.replace(temp_path, destination)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return destination


def write_historical_ranking_snapshot(
    vault_root: Path, run: AnalyticsRun, horizon: str, markdown: str | None = None
) -> Path:
    """Create one immutable snapshot; a repeat run can never overwrite it."""

    _require_ranking_write_gate()
    _validate_horizon(horizon)
    content = markdown if markdown is not None else render_ranking_markdown(
        run, horizon, snapshot=True
    )
    _validate_generated_content(content, horizon, snapshot=True)
    destination = historical_ranking_path(vault_root, run, horizon)
    _require_scoped_parent(
        vault_root, destination, "Rankings", "History", HISTORY_DIRECTORY_NAMES[horizon]
    )
    write_new_note_atomic(destination, content, allowed_root=vault_root)
    return destination
