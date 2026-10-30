"""Inspectable deterministic filtering performed before any model call."""

from __future__ import annotations

import re
from dataclasses import dataclass

from dedup import ExistingResearchIndex, candidate_id, normalize_url
from entities import Entity
from schemas import Candidate, RejectionReason


JUNK_TERMS = {
    "sponsored",
    "advertisement",
    "promo code",
    "horoscope",
    "click here now",
}
MATERIAL_EVENT_TERMS = {
    "earnings",
    "guidance",
    "outlook",
    "acquisition",
    "merger",
    "filing",
    "10-k",
    "10-q",
    "8-k",
    "investigation",
    "lawsuit",
    "financing",
    "bankruptcy",
    "restructuring",
    "contract",
    "launch",
    "regulatory",
}

# These patterns require a direct match and no material-event term. They are
# intentionally narrow so dramatic but substantive reporting is not rejected.
LOW_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "pure long-term price prediction",
        re.compile(r"\b(?:prediction|forecast):?.*\b(?:stock|share price)\b|\bwhere will .* stock be\b", re.I),
    ),
    (
        "investment-value clickbait",
        re.compile(
            r"\b(?:turn|turned|could turn)\s+\$?[\d,.]+(?:\s+(?:thousand|million|billion))?"
            r"\s+into\s+\$?[\d,.]+(?:\s+(?:thousand|million|billion))?"
            r"|\bworth\s+\$?[\d,.]+(?:\s+(?:thousand|million|billion))?\s+by\s+20\d{2}\b",
            re.I,
        ),
    ),
    (
        "generic buy prompt",
        re.compile(r"\bshould you buy\b|\bis .* a buy\b|\bbuy .* before\b", re.I),
    ),
    (
        "millionaire-maker framing",
        re.compile(r"\bmillionaire[- ]maker\b|\bmake you a millionaire\b", re.I),
    ),
    (
        "SEO listicle",
        re.compile(r"\b\d+ (?:best|top) (?:stocks|reasons|things)\b", re.I),
    ),
)

COMMENTARY_PATTERNS = (
    re.compile(r"\bjim cramer\b", re.I),
    re.compile(r"\bprice target\b", re.I),
    re.compile(r"\bstock (?:jumps|falls|rises|drops) today\b", re.I),
)


@dataclass(frozen=True)
class FilterResult:
    accepted: bool
    reason: str
    normalized_url: str
    candidate_id: str
    rejection_reason: RejectionReason | None = None
    priority_adjustment: int = 0


def _has_material_event(text: str) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text, re.I) for term in MATERIAL_EVENT_TERMS)


def cheap_filter(
    candidate: Candidate,
    target: Entity,
    index: ExistingResearchIndex,
    min_content_chars: int,
) -> FilterResult:
    url = normalize_url(str(candidate.source_url))
    identity = candidate_id(candidate)
    if candidate.source_tier not in {1, 2, 3, 4}:
        return FilterResult(False, "source tier is not allowed", url, identity, "low-quality source")
    if candidate.content_scope == "metadata_only":
        return FilterResult(
            False,
            "metadata-only record is not filing-body evidence",
            url,
            identity,
            "inaccessible / insufficient content",
        )

    searchable = f"{candidate.title} {candidate.snippet} {candidate.content}"
    explicit_entity = (
        target.ticker in {ticker.upper() for ticker in candidate.tickers}
        or target.canonical_name.casefold() in {name.casefold() for name in candidate.companies}
    )
    if not explicit_entity and not target.appears_in(searchable):
        return FilterResult(False, "target company/ticker not found as a complete name", url, identity, "irrelevant")

    folded = searchable.casefold()
    if any(term in folded for term in JUNK_TERMS):
        return FilterResult(False, "obvious junk or promotional content", url, identity, "promotional")
    if index.exact_duplicate(candidate):
        return FilterResult(False, "normalized URL or canonical ID already exists", url, identity, "duplicate")

    available = (candidate.content or candidate.snippet).strip()
    if len(available) < min_content_chars:
        return FilterResult(
            False,
            f"accessible content is shorter than {min_content_chars} characters",
            url,
            identity,
            "inaccessible / insufficient content",
        )

    title_has_material_event = _has_material_event(candidate.title)
    for label, pattern in LOW_VALUE_PATTERNS:
        if pattern.search(candidate.title) and not title_has_material_event:
            return FilterResult(
                False,
                label,
                url,
                identity,
                "low-value prediction or clickbait",
            )

    has_material_event = _has_material_event(searchable)
    if not has_material_event:
        if candidate.source_tier == 4 and re.search(r"\b(?:rumor|unconfirmed|could|might)\b", folded):
            return FilterResult(
                False,
                "weak source presents an unsupported or speculative claim",
                url,
                identity,
                "unsupported rumor",
            )

    if not has_material_event and any(pattern.search(searchable) for pattern in COMMENTARY_PATTERNS):
        return FilterResult(
            True,
            "qualified but deprioritized: commentary without an identifiable material event",
            url,
            identity,
            priority_adjustment=-1,
        )
    if not has_material_event:
        return FilterResult(
            True,
            "qualified but deprioritized: no identifiable material event",
            url,
            identity,
            priority_adjustment=-1,
        )
    return FilterResult(True, "qualified", url, identity)
