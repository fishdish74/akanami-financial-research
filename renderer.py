"""Deterministic Markdown rendering for validated analysis payloads."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, Optional

import yaml

from dedup import candidate_id, content_hash
from schemas import AnalysisPayload, Candidate, DuplicateStatus, ProcessingMetadata


REQUIRED_SECTIONS = (
    "Executive Summary",
    "Reported Facts",
    "Company / Management Statements",
    "Analyst or Source Opinion",
    "AI Inference",
    "Companies Mentioned",
    "Potential Catalysts",
    "Potential Risks",
    "Bullish Evidence",
    "Bearish Evidence",
    "What Is Actually New?",
    "Connections to Existing Research",
    "Contradictions",
    "Open Questions",
    "Information Needed Next",
    "Source Information",
    "Claim-Level Evidence",
    "My Notes",
)


def _clean_text(value: object) -> str:
    return str(value).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _wikilink_text(match: re.Match[str], *, embedded: bool) -> str:
    target = match.group(1).strip()
    readable = target.rsplit("|", 1)[-1].strip() or target
    prefix = "Embedded reference: " if embedded else ""
    return prefix + readable


def _neutralize_active_constructs(value: object) -> str:
    """Make untrusted Obsidian/Markdown constructs inert while retaining their text."""

    text = _clean_text(value)
    text = re.sub(
        r"!\[\[([^\]\n]+)\]\]",
        lambda match: _wikilink_text(match, embedded=True),
        text,
    )
    text = re.sub(
        r"(?<!!)\[\[([^\]\n]+)\]\]",
        lambda match: _wikilink_text(match, embedded=False),
        text,
    )
    text = re.sub(
        r"!\[([^\]\n]*)\]\(([^)\n]*)\)",
        lambda match: f"Embedded image: {match.group(1) or 'image'} ({match.group(2)})",
        text,
    )
    text = re.sub(
        r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)",
        lambda match: f"{match.group(1)} ({match.group(2)})",
        text,
    )
    text = re.sub(
        r"`\s*(?:\$?=)\s*([^`\n]*)`",
        lambda match: f"Dataview expression: {match.group(1).strip()}",
        text,
    )
    text = text.replace("<%", "&lt;%").replace("%>", "%&gt;")
    text = re.sub(
        r"<(?=[!/A-Za-z])([^>]*)>",
        lambda match: f"&lt;{match.group(1)}&gt;",
        text,
    )
    return text


def neutralize_external_text(value: object) -> str:
    """Public boundary for rendering untrusted source text as inert readable text."""

    return _neutralize_active_constructs(value)


def escape_external_markdown(value: object) -> str:
    """Public boundary for rendering untrusted source text without Markdown structure."""

    return _escape_markdown_structure(value)


def _safe_filename_part(value: str) -> str:
    cleaned = unicodedata.normalize("NFKC", value)
    cleaned = re.sub(r"[\x00-\x1f\\/:*?\"<>|]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if cleaned.upper() in reserved:
        cleaned = f"_{cleaned}"
    return cleaned or "Unknown"


def deterministic_filename(
    payload: AnalysisPayload, candidate: Candidate | None = None, *, max_bytes: int = 220
) -> str:
    """Return a stable, cross-platform, byte-bounded filename."""
    note_date = payload.published or payload.accessed.date()
    source = _safe_filename_part(payload.source)
    title = _safe_filename_part(payload.title)
    identity = candidate_id(candidate)[:10] if candidate is not None else None
    identity_suffix = f"--{identity}" if identity else ""
    filename = f"{note_date.isoformat()} - {source} - {title}{identity_suffix}.md"
    if len(filename.encode("utf-8")) <= max_bytes:
        return filename
    digest = identity or hashlib.sha256(f"{source}\0{title}".encode("utf-8")).hexdigest()[:10]
    prefix = f"{note_date.isoformat()} - "
    suffix = f"--{digest}.md"
    available = max_bytes - len((prefix + suffix).encode("utf-8"))
    if available < 1:
        raise ValueError("max_bytes is too small for a deterministic Markdown filename")
    body = f"{source} - {title}"
    while len(body.encode("utf-8")) > available:
        body = body[:-1]
    body = body.rstrip(" .-") or "_"
    filename = f"{prefix}{body}{suffix}"
    if len(filename.encode("utf-8")) > max_bytes:
        raise AssertionError("deterministic filename exceeded its byte bound")
    return filename


def _escape_markdown_structure(value: object) -> str:
    lines = _neutralize_active_constructs(value).splitlines()
    escaped: list[str] = []
    for line in lines:
        match = re.match(r"^(\s{0,3})(#{1,6}\s|```|~~~|(?:---|___|\*\*\*)\s*$)", line)
        if match:
            line = f"{match.group(1)}\\{line[len(match.group(1)):]}"
        escaped.append(line)
    return "\n".join(escaped)


def _bullet(value: object) -> str:
    lines = _escape_markdown_structure(value).splitlines() or [""]
    return lines[0] + "".join(f"\n  {line}" for line in lines[1:])


def _section(title: str, items: Iterable[str]) -> str:
    values = [_bullet(item) for item in items if _clean_text(item)]
    body = "\n".join(f"- {item}" for item in values) or "- None identified."
    return f"## {title}\n\n{body}"


def _claim_evidence_section(payload: AnalysisPayload) -> str:
    evidence = {item.evidence_id: item for item in payload.evidence_catalog}
    lines: list[str] = []
    for claim in payload.claims:
        references = ", ".join(
            f"`{item}` ({evidence[item].evidence_class})" for item in claim.evidence_ids
        ) or "none — explicitly unsupported"
        lines.append(
            f"- `{claim.claim_id}` [{claim.category}; {claim.support}] "
            f"{_bullet(claim.text)} — Evidence: {references}"
        )
    return "## Claim-Level Evidence\n\n" + ("\n".join(lines) or "- None identified.")


def render_article(
    payload: AnalysisPayload,
    candidate: Candidate,
    duplicate_status: DuplicateStatus,
    metadata: Optional[ProcessingMetadata] = None,
) -> str:
    """Render only validated objects; model/source text never controls YAML keys."""
    created = payload.accessed.date().isoformat()
    analysis_completed = metadata.analysis_completed if metadata is not None else True
    analysis_mode = metadata.analysis_mode if metadata is not None else "live_ai"
    frontmatter_text = lambda value: re.sub(
        r"\s+", " ", _neutralize_active_constructs(value)
    ).strip()
    frontmatter = {
        "type": "article",
        "status": "inbox",
        "created": created,
        "updated": created,
        "published": payload.published.isoformat() if payload.published else None,
        "accessed": payload.accessed.isoformat(),
        "source": frontmatter_text(payload.source),
        "source_url": str(payload.source_url),
        "author": frontmatter_text(payload.author) if payload.author else None,
        "companies": [frontmatter_text(item) for item in payload.companies],
        "tickers": payload.tickers,
        "themes": [frontmatter_text(item) for item in payload.themes],
        "sector": frontmatter_text(payload.sector) if payload.sector else None,
        "importance": payload.importance if analysis_completed else "not_analyzed",
        "confidence": payload.confidence if analysis_completed else "not_analyzed",
        "analysis_version": payload.analysis_version,
        "confidence_owner": payload.confidence_owner,
        "confidence_rule_version": payload.confidence_rule_version,
        "sentiment": payload.sentiment if analysis_completed else "not_analyzed",
        "reviewed": False,
        "archived": False,
        "needs_review": payload.needs_review if analysis_completed else True,
        "candidate_id": candidate_id(candidate),
        "content_hash": content_hash(candidate),
        "duplicate_status": duplicate_status,
        "analysis_mode": analysis_mode,
        "analysis_completed": analysis_completed,
    }
    if metadata is not None:
        frontmatter.update(
            {
                "model": metadata.model_id,
                "prompt_version": metadata.prompt_version,
                "pipeline_version": metadata.pipeline_version,
            }
        )
    yaml_text = yaml.safe_dump(
        frontmatter,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()
    published = payload.published.isoformat() if payload.published else ""
    source_lines = [
        f"- Publication: {_bullet(payload.source)}",
        f"- Author: {_bullet(payload.author or '')}",
        f"- Publication date: {published}",
        f"- Original URL: {payload.source_url}",
        f"- Access date: {payload.accessed.isoformat()}",
    ]
    heading = re.sub(
        r"\s+", " ", _neutralize_active_constructs(payload.title).replace("#", "\\#")
    )
    sections = [
        f"# {heading}",
        f"## Executive Summary\n\n{_escape_markdown_structure(payload.executive_summary)}",
        _section("Reported Facts", payload.reported_facts),
        _section("Company / Management Statements", payload.company_management_statements),
        _section("Analyst or Source Opinion", payload.analyst_source_opinion),
        _section("AI Inference", payload.ai_inference),
        _section("Companies Mentioned", [*payload.companies, *payload.tickers]),
        _section("Potential Catalysts", payload.potential_catalysts),
        _section("Potential Risks", payload.potential_risks),
        _section("Bullish Evidence", payload.bullish_evidence),
        _section("Bearish Evidence", payload.bearish_evidence),
        _section("What Is Actually New?", payload.what_is_new),
        _section("Connections to Existing Research", []),
        _section("Contradictions", payload.contradictions),
        _section("Open Questions", payload.open_questions),
        _section("Information Needed Next", payload.information_needed_next),
        "## Source Information\n\n" + "\n".join(source_lines),
        _claim_evidence_section(payload),
        "## My Notes",
    ]
    return f"---\n{yaml_text}\n---\n\n" + "\n\n".join(sections) + "\n"


def validate_rendered_markdown(markdown: str) -> dict[str, object]:
    if not markdown.startswith("---\n"):
        raise ValueError("rendered note does not start with YAML frontmatter")
    parts = markdown.split("---", 2)
    if len(parts) != 3:
        raise ValueError("rendered note has invalid frontmatter boundaries")
    data = yaml.safe_load(parts[1])
    if not isinstance(data, dict):
        raise ValueError("rendered frontmatter must be a mapping")
    required = {
        "type", "status", "source", "source_url", "importance", "confidence",
        "sentiment", "reviewed", "archived", "needs_review", "candidate_id",
        "analysis_mode", "analysis_completed",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"rendered YAML missing fields: {sorted(missing)}")
    if data["type"] != "article" or data["status"] != "inbox":
        raise ValueError("rendered YAML has unexpected type/status")
    headings = re.findall(r"(?m)^## ([^\n]+?)\s*$", parts[2])
    if headings != list(REQUIRED_SECTIONS):
        raise ValueError("rendered Markdown sections are missing, duplicated, or out of order")
    if len(re.findall(r"(?m)^# [^#\n].*$", parts[2])) != 1:
        raise ValueError("rendered Markdown must contain exactly one document title")
    return data
