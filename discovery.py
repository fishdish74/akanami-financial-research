"""Deterministic, watchlist-independent broad discovery and daily-note rendering.

Discovery priority is research triage. It is not investment attractiveness,
expected return, personal suitability, or a trading instruction.
"""

from __future__ import annotations

import hashlib
import json
import errno
import os
import re
import tempfile
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from dedup import normalize_url
from renderer import escape_external_markdown, neutralize_external_text
from schemas import Candidate


DISCOVERY_RULES_VERSION = "discovery-v1"
DAILY_NOTE_DISCLAIMER = "This is a research-priority signal, not a buy recommendation."
NO_QUALIFIED_CANDIDATE = "NO QUALIFIED CANDIDATE TODAY"
DISCOVERY_WEIGHT_KEYS = (
    "novelty",
    "event_materiality",
    "evidence_quality",
    "catalyst_strength",
    "fundamental_interest",
    "growth_acceleration",
    "urgency",
    "data_completeness",
)
DEFAULT_DISCOVERY_WEIGHTS = {
    "novelty": 20,
    "event_materiality": 25,
    "evidence_quality": 20,
    "catalyst_strength": 10,
    "fundamental_interest": 10,
    "growth_acceleration": 5,
    "urgency": 5,
    "data_completeness": 5,
}


class SourceTier(StrEnum):
    PRIMARY = "primary"
    HIGH_QUALITY = "high_quality"
    SECONDARY = "secondary"
    LOW_QUALITY = "low_quality"


class EvidenceLevel(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    UNAVAILABLE = "unavailable"


class HistoryStatus(StrEnum):
    NORMAL = "normal"
    LIMITED_HISTORY = "limited_history"
    PROVISIONAL = "provisional"
    NOT_RANKABLE = "not_rankable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DiscoveryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str = Field(min_length=1)
    ticker: str | None = None
    cik: str | None = None
    identity_verified: bool = False
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_url: HttpUrl
    published_at: datetime | None = None
    discovered_at: datetime
    event_type: str = Field(min_length=1)
    event_date: date
    event_key: str | None = None
    source_snapshot_id: str = Field(min_length=1)
    source_tier: SourceTier
    source_independence_key: str | None = None
    primary_source: bool = False
    summary: str = ""
    material_facts: list[str] = Field(default_factory=list)
    material_numbers: dict[str, str] = Field(default_factory=dict)
    direction: str | None = None
    new_information: bool = True
    material_change: bool = False
    direction_change: bool = False
    absolute_scale_context: bool = False
    rumor: bool = False
    catalyst_strength: EvidenceLevel = EvidenceLevel.UNAVAILABLE
    fundamental_interest: EvidenceLevel = EvidenceLevel.UNAVAILABLE
    growth_acceleration: EvidenceLevel = EvidenceLevel.UNAVAILABLE
    valuation_interest: EvidenceLevel = EvidenceLevel.UNAVAILABLE
    urgency: EvidenceLevel = EvidenceLevel.UNAVAILABLE
    risk_severity: EvidenceLevel = EvidenceLevel.UNAVAILABLE
    risk_flags: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    limited_history_status: HistoryStatus = HistoryStatus.NORMAL
    syndication_key: str | None = None

    @model_validator(mode="after")
    def validate_boundaries(self) -> "DiscoveryItem":
        if self.published_at is not None and self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware when supplied")
        if self.discovered_at.tzinfo is None:
            raise ValueError("discovery timestamps must be timezone-aware")
        if self.ticker:
            self.ticker = self.ticker.strip().upper()
        if self.cik:
            self.cik = self.cik.strip().lstrip("0") or "0"
        return self

    @property
    def company_id(self) -> str:
        if self.cik:
            return f"cik:{self.cik}"
        if self.ticker:
            return f"ticker:{self.ticker}"
        return "name:" + re.sub(r"[^a-z0-9]+", "-", self.company.casefold()).strip("-")

    @property
    def normalized_url(self) -> str:
        return normalize_url(str(self.source_url))

    @property
    def independence_key(self) -> str:
        return (self.source_independence_key or urlsplit(self.normalized_url).hostname or self.source).casefold()


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rules_version: str = DISCOVERY_RULES_VERSION
    minimum_score: int = Field(default=55, ge=0, le=100, strict=True)
    minimum_materiality: int = Field(default=50, ge=0, le=100, strict=True)
    minimum_evidence_quality: int = Field(default=60, ge=0, le=100, strict=True)
    weights: dict[str, int] = Field(
        default_factory=lambda: dict(DEFAULT_DISCOVERY_WEIGHTS)
    )

    @field_validator("weights", mode="before")
    @classmethod
    def weights_are_finite_integers(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("discovery weights must be a mapping")
        if any(isinstance(weight, bool) or not isinstance(weight, int) for weight in value.values()):
            raise ValueError("discovery weights must be finite integers")
        return value

    @model_validator(mode="after")
    def validate_scoring_contract(self) -> "DiscoveryConfig":
        expected = set(DISCOVERY_WEIGHT_KEYS)
        supplied = set(self.weights)
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        if missing:
            raise ValueError("missing discovery weight keys: " + ", ".join(missing))
        if unexpected:
            raise ValueError("unexpected discovery weight keys: " + ", ".join(unexpected))
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("discovery weights must be nonnegative")
        if sum(self.weights.values()) != 100:
            raise ValueError("discovery weights must total 100")
        return self


class DiscoveryDimensions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    novelty: int = Field(ge=0, le=100)
    event_materiality: int = Field(ge=0, le=100)
    evidence_quality: int = Field(ge=0, le=100)
    catalyst_strength: int | None = Field(default=None, ge=0, le=100)
    fundamental_interest: int | None = Field(default=None, ge=0, le=100)
    growth_acceleration: int | None = Field(default=None, ge=0, le=100)
    valuation_interest: int | None = Field(default=None, ge=0, le=100)
    urgency: int | None = Field(default=None, ge=0, le=100)
    data_completeness: int = Field(ge=0, le=100)
    risk_severity: int | None = Field(default=None, ge=0, le=100)
    limited_history_status: HistoryStatus


class MaterialNumberObservation(BaseModel):
    """One source-qualified material value retained for audit and conflict review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_snapshot_id: str
    source: str
    source_url: HttpUrl
    independence_key: str
    raw_value: str
    normalized_value: str


class DiscoveryCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    company_id: str
    company: str
    ticker: str | None
    event_type: str
    event_date: date
    first_seen_at: datetime
    latest_meaningful_update_at: datetime
    source_count: int
    independent_source_count: int
    strongest_source: str
    strongest_source_url: HttpUrl
    strongest_source_published_at: datetime | None
    strongest_source_tier: SourceTier
    corroborating_source_urls: list[HttpUrl]
    source_snapshot_ids: list[str]
    new_information_added: bool
    material_numbers: dict[str, str]
    material_number_evidence: dict[str, list[MaterialNumberObservation]] = Field(
        default_factory=dict
    )
    material_number_conflicts: list[str] = Field(default_factory=list)
    direction_change: bool
    strongest_evidence: list[str]
    summary: str
    dimensions: DiscoveryDimensions
    research_priority_score: int
    qualifies: bool
    risk_flags: list[str]
    missing_information: list[str]
    recommended_questions: list[str]
    event_fingerprint: str


class DiscoveryHandoff(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["HIGHEST_RESEARCH_PRIORITY", "NO_QUALIFIED_CANDIDATE_TODAY"]
    company_id: str | None = None
    company: str | None = None
    ticker: str | None = None
    event_id: str | None = None
    event_type: str | None = None
    why_it_surfaced: str | None = None
    strongest_evidence: list[str] = Field(default_factory=list)
    source_urls: list[HttpUrl] = Field(default_factory=list)
    novelty: int | None = None
    materiality: int | None = None
    risk_flags: list[str] = Field(default_factory=list)
    limited_history_status: HistoryStatus | None = None
    missing_information: list[str] = Field(default_factory=list)
    recommended_questions: list[str] = Field(default_factory=list)


class DiscoveryRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    discovery_timestamp: datetime
    rules_version: str
    input_snapshot_id: str
    discovered_item_count: int
    filtered_count: int
    duplicate_count: int
    event_count: int
    shortlist_count: int
    deep_analysis_count: int
    candidates: list[DiscoveryCandidate]
    selected: DiscoveryCandidate | None
    handoff: DiscoveryHandoff
    rejection_reasons: dict[str, int]


def load_discovery_items_json(path: Path) -> list[DiscoveryItem]:
    """Load a bounded, reviewable public-search snapshot for deterministic replay."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("discovery snapshot must contain an items list")
    return [DiscoveryItem.model_validate(item) for item in payload["items"]]


_MATERIALITY = {
    "regulatory_approval": 85,
    "acquisition": 85,
    "material_contract": 80,
    "restructuring": 75,
    "financing": 75,
    "guidance": 70,
    "ipo_filing": 70,
    "earnings": 65,
    "filing_change": 60,
    "new_listing": 60,
    "other": 40,
}
_LEVEL_SCORE = {
    EvidenceLevel.STRONG: 90,
    EvidenceLevel.MODERATE: 65,
    EvidenceLevel.WEAK: 35,
    EvidenceLevel.UNAVAILABLE: None,
}
_TIER_SCORE = {
    SourceTier.PRIMARY: 90,
    SourceTier.HIGH_QUALITY: 75,
    SourceTier.SECONDARY: 55,
    SourceTier.LOW_QUALITY: 20,
}
_REJECT_PATTERNS = (
    re.compile(r"\btop \d+ (?:stocks?|picks?)\b", re.I),
    re.compile(r"\b(?:stock|share) price (?:prediction|forecast)\b", re.I),
    re.compile(r"\b(?:could|might) (?:soar|skyrocket|explode)\b", re.I),
)
_MATERIAL_NUMBER_PATTERN = re.compile(
    r"^(?:(?P<prefix>\$|€|£|usd|eur|gbp)\s*)?"
    r"(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*"
    r"(?P<scale>k|m|b|thousand|million|billion)?\s*"
    r"(?P<suffix>usd|eur|gbp)?\s*"
    r"(?P<unit>%|percent|percentage|years?|months?|shares?)?$",
    re.IGNORECASE,
)
_CURRENCIES = {"$": "USD", "€": "EUR", "£": "GBP"}
_SCALES = {
    None: Decimal(1),
    "k": Decimal(1_000),
    "thousand": Decimal(1_000),
    "m": Decimal(1_000_000),
    "million": Decimal(1_000_000),
    "b": Decimal(1_000_000_000),
    "billion": Decimal(1_000_000_000),
}


def _normalized_material_number(
    raw: str,
) -> tuple[Decimal, str | None, str | None] | None:
    """Normalize only exact numeric scales; never convert currencies or units."""

    text = " ".join(raw.strip().replace(",", "").split()).casefold()
    match = _MATERIAL_NUMBER_PATTERN.fullmatch(text)
    if match is None:
        return None
    try:
        value = Decimal(match.group("number")) * _SCALES[match.group("scale")]
    except (InvalidOperation, KeyError):
        return None
    if not value.is_finite():
        return None
    prefix = match.group("prefix")
    suffix = match.group("suffix")
    currency = (_CURRENCIES.get(prefix, prefix.upper()) if prefix else None)
    suffix_currency = suffix.upper() if suffix else None
    if currency and suffix_currency and currency != suffix_currency:
        return None
    currency = currency or suffix_currency
    unit = match.group("unit")
    if unit in {"percent", "percentage"}:
        unit = "%"
    elif unit:
        unit = unit.rstrip("s")
    return value, currency, unit


def _normalized_material_display(raw: str) -> str:
    parsed = _normalized_material_number(raw)
    if parsed is None:
        return " ".join(raw.split()).casefold()
    value, currency, unit = parsed
    parts = [format(value, "f")]
    if currency:
        parts.append(currency)
    if unit:
        parts.append(unit)
    return " ".join(parts)


def _material_values_agree(values: list[str]) -> bool:
    parsed = [_normalized_material_number(value) for value in values]
    if all(value is not None for value in parsed):
        numeric = [value for value in parsed if value is not None]
        amounts = {value[0] for value in numeric}
        currencies = {value[1] for value in numeric if value[1] is not None}
        units = {value[2] for value in numeric}
        return len(amounts) == 1 and len(currencies) <= 1 and len(units) == 1
    if any(value is not None for value in parsed):
        return False
    return len({" ".join(value.split()).casefold() for value in values}) == 1


def _aggregate_material_numbers(
    items: list[DiscoveryItem],
) -> tuple[
    dict[str, str],
    dict[str, list[MaterialNumberObservation]],
    list[str],
]:
    grouped: defaultdict[str, list[tuple[DiscoveryItem, str]]] = defaultdict(list)
    for item in items:
        for field, value in item.material_numbers.items():
            grouped[field].append((item, value))
    selected: dict[str, str] = {}
    evidence: dict[str, list[MaterialNumberObservation]] = {}
    conflicts: list[str] = []
    for field in sorted(grouped):
        observations = sorted(
            grouped[field],
            key=lambda pair: (
                pair[0].source_snapshot_id,
                pair[0].normalized_url,
                pair[1],
            ),
        )
        evidence[field] = [
            MaterialNumberObservation(
                source_snapshot_id=item.source_snapshot_id,
                source=item.source,
                source_url=item.source_url,
                independence_key=item.independence_key,
                raw_value=value,
                normalized_value=_normalized_material_display(value),
            )
            for item, value in observations
        ]
        raw_values = [value for _, value in observations]
        if _material_values_agree(raw_values):
            selected[field] = raw_values[0]
        else:
            conflicts.append(field)
    return selected, evidence, conflicts


def _fingerprint(items: list[DiscoveryItem]) -> str:
    content = {
        "numbers": sorted((key, value) for item in items for key, value in item.material_numbers.items()),
        "directions": sorted({item.direction for item in items if item.direction}),
        "facts": sorted({fact.casefold().strip() for item in items for fact in item.material_facts}),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _event_id(item: DiscoveryItem) -> str:
    key = item.event_key or f"{item.company_id}|{item.event_type.casefold()}|{item.event_date.isoformat()}"
    return "event:" + hashlib.sha256(key.encode()).hexdigest()[:20]


def _score_level(level: EvidenceLevel) -> int | None:
    return _LEVEL_SCORE[level]


def _strongest(items: list[DiscoveryItem]) -> DiscoveryItem:
    return sorted(
        items,
        key=lambda item: (-_TIER_SCORE[item.source_tier], not item.primary_source, item.normalized_url),
    )[0]


def run_discovery(
    items: list[DiscoveryItem],
    *,
    known_company_ids: set[str] | None = None,
    known_event_fingerprints: dict[str, str] | None = None,
    discovery_timestamp: datetime | None = None,
    input_snapshot_id: str = "manual-public-search",
    config: DiscoveryConfig | None = None,
) -> DiscoveryRun:
    """Filter cheaply, cluster by event, score explainably, and select at most one."""

    config = (
        DiscoveryConfig.model_validate(config.model_dump())
        if config is not None
        else DiscoveryConfig()
    )
    known_company_ids = known_company_ids or set()
    known_event_fingerprints = known_event_fingerprints or {}
    discovery_timestamp = discovery_timestamp or datetime.now(timezone.utc)
    if discovery_timestamp.tzinfo is None:
        raise ValueError("discovery_timestamp must be timezone-aware")

    rejection_reasons: defaultdict[str, int] = defaultdict(int)
    accepted: list[DiscoveryItem] = []
    seen_urls: set[str] = set()
    seen_syndication: set[str] = set()
    duplicate_count = 0
    for item in items:
        haystack = f"{item.title} {item.summary}"
        reason: str | None = None
        if not item.identity_verified:
            reason = "unresolved_identity"
        elif item.rumor:
            reason = "unsupported_rumor"
        elif any(pattern.search(haystack) for pattern in _REJECT_PATTERNS):
            reason = "promotional_or_price_prediction"
        elif _MATERIALITY.get(item.event_type, _MATERIALITY["other"]) < config.minimum_materiality:
            reason = "low_materiality"
        if reason:
            rejection_reasons[reason] += 1
            continue
        if item.normalized_url in seen_urls or (item.syndication_key and item.syndication_key in seen_syndication):
            duplicate_count += 1
            continue
        seen_urls.add(item.normalized_url)
        if item.syndication_key:
            seen_syndication.add(item.syndication_key)
        accepted.append(item)

    clusters: defaultdict[str, list[DiscoveryItem]] = defaultdict(list)
    for item in accepted:
        clusters[_event_id(item)].append(item)

    candidates: list[DiscoveryCandidate] = []
    for event_id, event_items in sorted(clusters.items()):
        event_items = sorted(event_items, key=lambda item: (item.discovered_at, item.normalized_url))
        strongest = _strongest(event_items)
        fingerprint = _fingerprint(event_items)
        known_fingerprint = known_event_fingerprints.get(event_id)
        unchanged = known_fingerprint == fingerprint
        changed = known_fingerprint is not None and known_fingerprint != fingerprint
        new_company = strongest.company_id not in known_company_ids
        novelty = 0 if unchanged else (80 if new_company else (45 if changed else 65))
        numbers, number_evidence, number_conflicts = _aggregate_material_numbers(event_items)
        materiality = _MATERIALITY.get(strongest.event_type, _MATERIALITY["other"])
        if strongest.material_change:
            materiality = min(100, materiality + 10)
        if strongest.material_numbers and strongest.absolute_scale_context and not number_conflicts:
            materiality = min(100, materiality + 5)
        independent_count = len({item.independence_key for item in event_items})
        evidence_quality = min(
            100,
            _TIER_SCORE[strongest.source_tier]
            + (10 if independent_count >= 2 and not number_conflicts else 0),
        )
        completeness_fields = (
            bool(strongest.company), bool(strongest.ticker or strongest.cik),
            bool(strongest.material_facts), bool(numbers),
            bool(strongest.summary), bool(strongest.missing_information == []),
        )
        completeness = round(100 * sum(completeness_fields) / len(completeness_fields))
        dimensions = DiscoveryDimensions(
            novelty=novelty,
            event_materiality=materiality,
            evidence_quality=evidence_quality,
            catalyst_strength=_score_level(strongest.catalyst_strength),
            fundamental_interest=_score_level(strongest.fundamental_interest),
            growth_acceleration=_score_level(strongest.growth_acceleration),
            valuation_interest=_score_level(strongest.valuation_interest),
            urgency=_score_level(strongest.urgency),
            data_completeness=completeness,
            risk_severity=_score_level(strongest.risk_severity),
            limited_history_status=strongest.limited_history_status,
        )
        score_values = dimensions.model_dump()
        weighted = sum((score_values[name] or 0) * weight for name, weight in config.weights.items())
        score = round(weighted / 100)
        qualifies = (
            not unchanged
            and score >= config.minimum_score
            and materiality >= config.minimum_materiality
            and evidence_quality >= config.minimum_evidence_quality
        )
        facts = list(dict.fromkeys(fact for item in event_items for fact in item.material_facts))
        risks = list(dict.fromkeys(flag for item in event_items for flag in item.risk_flags))
        missing = list(dict.fromkeys(value for item in event_items for value in item.missing_information))
        for field in number_conflicts:
            risks.append(f"Conflicting material number: {field}")
            missing.append(f"Resolve conflicting source values for material number: {field}.")
        if strongest.limited_history_status != HistoryStatus.NORMAL:
            risks.append(f"Limited-history status: {strongest.limited_history_status.value}")
        questions = [
            "What primary evidence confirms the event and its exact terms?",
            "How material is the event relative to the company's existing scale?",
            "What conditions, counterparties, dilution, or execution risks could negate the thesis?",
            "What price, valuation, estimate, and financial-history data are still missing?",
        ]
        latest_updates = [item.discovered_at for item in event_items if item.new_information or item.material_change or item.direction_change]
        candidates.append(DiscoveryCandidate(
            event_id=event_id,
            company_id=strongest.company_id,
            company=strongest.company,
            ticker=strongest.ticker,
            event_type=strongest.event_type,
            event_date=strongest.event_date,
            first_seen_at=min(item.discovered_at for item in event_items),
            latest_meaningful_update_at=max(latest_updates or [event_items[0].discovered_at]),
            source_count=len(event_items),
            independent_source_count=independent_count,
            strongest_source=strongest.source,
            strongest_source_url=strongest.source_url,
            strongest_source_published_at=strongest.published_at,
            strongest_source_tier=strongest.source_tier,
            corroborating_source_urls=[item.source_url for item in event_items if item is not strongest],
            source_snapshot_ids=sorted({item.source_snapshot_id for item in event_items}),
            new_information_added=not unchanged and any(item.new_information for item in event_items),
            material_numbers=numbers,
            material_number_evidence=number_evidence,
            material_number_conflicts=number_conflicts,
            direction_change=any(item.direction_change for item in event_items),
            strongest_evidence=facts,
            summary=strongest.summary,
            dimensions=dimensions,
            research_priority_score=score,
            qualifies=qualifies,
            risk_flags=risks,
            missing_information=missing,
            recommended_questions=questions,
            event_fingerprint=fingerprint,
        ))

    shortlist = [candidate for candidate in candidates if candidate.qualifies]
    shortlist.sort(key=lambda value: (-value.research_priority_score, -value.dimensions.evidence_quality,
                                      -value.dimensions.event_materiality, value.event_id))
    selected = shortlist[0] if shortlist else None
    if selected:
        handoff = DiscoveryHandoff(
            status="HIGHEST_RESEARCH_PRIORITY",
            company_id=selected.company_id, company=selected.company, ticker=selected.ticker,
            event_id=selected.event_id, event_type=selected.event_type,
            why_it_surfaced=(f"Score {selected.research_priority_score}/100 from a new, material "
                             f"{selected.event_type.replace('_', ' ')} event with "
                             f"{selected.dimensions.evidence_quality}/100 evidence quality."),
            strongest_evidence=selected.strongest_evidence,
            source_urls=[selected.strongest_source_url, *selected.corroborating_source_urls],
            novelty=selected.dimensions.novelty, materiality=selected.dimensions.event_materiality,
            risk_flags=selected.risk_flags,
            limited_history_status=selected.dimensions.limited_history_status,
            missing_information=selected.missing_information,
            recommended_questions=selected.recommended_questions,
        )
    else:
        handoff = DiscoveryHandoff(status="NO_QUALIFIED_CANDIDATE_TODAY")
    return DiscoveryRun(
        discovery_timestamp=discovery_timestamp,
        rules_version=config.rules_version,
        input_snapshot_id=input_snapshot_id,
        discovered_item_count=len(items),
        filtered_count=sum(rejection_reasons.values()),
        duplicate_count=duplicate_count,
        event_count=len(candidates),
        shortlist_count=len(shortlist),
        # Selection creates a handoff; it is not itself deep analysis.
        deep_analysis_count=0,
        candidates=candidates,
        selected=selected,
        handoff=handoff,
        rejection_reasons=dict(sorted(rejection_reasons.items())),
    )


def to_main_pipeline_candidate(selected: DiscoveryCandidate) -> Candidate:
    """Adapt the selected event to the existing validated main-pipeline boundary."""

    tier = {
        SourceTier.PRIMARY: 1,
        SourceTier.HIGH_QUALITY: 2,
        SourceTier.SECONDARY: 3,
        SourceTier.LOW_QUALITY: 4,
    }[selected.strongest_source_tier]
    content = "\n".join(
        [selected.summary, *selected.strongest_evidence]
        + [f"Risk flag: {value}" for value in selected.risk_flags]
        + [f"Missing information: {value}" for value in selected.missing_information]
    )
    return Candidate(
        title=f"{selected.company}: {selected.event_type.replace('_', ' ')}",
        source=selected.strongest_source,
        source_url=selected.strongest_source_url,
        published=selected.strongest_source_published_at,
        discovered=selected.latest_meaningful_update_at,
        content=content,
        snippet=selected.summary,
        source_tier=tier,
        companies=[selected.company],
        tickers=[selected.ticker] if selected.ticker else [],
        source_type="discovery-handoff",
        content_scope="excerpt",
    )


def render_daily_highest_potential(run: DiscoveryRun, research_date: date) -> str:
    """Render an immutable, auditable daily discovery note."""

    selected = run.selected
    metadata = {
        "type": "daily-highest-potential",
        "date": research_date.isoformat(),
        "discovery_timestamp": run.discovery_timestamp.isoformat(),
        "rules_version": run.rules_version,
        "input_snapshot_id": run.input_snapshot_id,
        "source_snapshot_ids": selected.source_snapshot_ids if selected else [],
        "candidate_set_size": run.event_count,
        "rejected_or_deprioritized_count": (
            run.filtered_count + run.duplicate_count + run.event_count - (1 if selected else 0)
        ),
        "selected_candidate": selected.company if selected else None,
        "event_id": selected.event_id if selected else None,
        "research_priority": selected.research_priority_score if selected else None,
        "discovery_status": run.handoff.status,
        "immutable_history": True,
    }
    yaml_text = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
    if not selected:
        candidate = NO_QUALIFIED_CANDIDATE
        reason = "No event cleared the configured minimum materiality, evidence-quality, and research-priority thresholds."
        evidence = risks = limitations = questions = "- None."
        trigger = matter = novelty = context = catalyst = priority = comparison = "Unavailable."
    else:
        clean = escape_external_markdown
        candidate = f"- Company: {clean(selected.company)}\n- Ticker: {clean(selected.ticker or 'Unknown')}\n- Date: {research_date}\n- Discovery status: HIGHEST_RESEARCH_PRIORITY"
        reason = clean(run.handoff.why_it_surfaced or "")
        trigger = f"{clean(selected.event_type.replace('_', ' ').title())} dated {selected.event_date}. {clean(selected.summary)}"
        matter = "\n".join(f"- {clean(item)}" for item in selected.strongest_evidence) or "- No sufficiently specific fact captured."
        urls = [selected.strongest_source_url, *selected.corroborating_source_urls]
        related_label = "Corroborating" if selected.independent_source_count >= 2 else "Related same-source evidence"
        evidence = "\n".join([f"- Primary/strongest: {clean(selected.strongest_source)} — {urls[0]}", *[f"- {related_label}: {url}" for url in urls[1:]]])
        novelty = f"Novelty score: {selected.dimensions.novelty}/100. New information added: {str(selected.new_information_added).lower()}."
        context = (f"Fundamental interest: {selected.dimensions.fundamental_interest if selected.dimensions.fundamental_interest is not None else 'unavailable'}/100. "
                   f"Growth acceleration: {selected.dimensions.growth_acceleration if selected.dimensions.growth_acceleration is not None else 'unavailable'}/100. "
                   f"Valuation interest: {selected.dimensions.valuation_interest if selected.dimensions.valuation_interest is not None else 'unavailable'}. "
                   f"Limited-history status: {selected.dimensions.limited_history_status.value}.")
        catalyst = f"Catalyst strength: {selected.dimensions.catalyst_strength if selected.dimensions.catalyst_strength is not None else 'unavailable'}."
        risks = "\n".join(f"- {clean(item)}" for item in selected.risk_flags) or "- No explicit risk flag in the discovery inputs; deeper research is still required."
        limitations = "\n".join(f"- {clean(item)}" for item in selected.missing_information) or "- No limitation supplied."
        priority = f"{selected.research_priority_score}/100 (deterministic discovery priority; not expected return)."
        others = sorted((item for item in run.candidates if item.event_id != selected.event_id), key=lambda item: -item.research_priority_score)
        comparison = (f"It had the highest qualifying deterministic score. Next event score: {others[0].research_priority_score}/100."
                      if others else "It was the only event that cleared every qualification threshold.")
        questions = "\n".join(f"- {clean(item)}" for item in selected.recommended_questions)
    sections = [
        f"# Daily Highest Potential\n\n> {DAILY_NOTE_DISCLAIMER}",
        f"## Candidate\n\n{candidate}",
        f"## Why It Surfaced\n\n{reason}",
        f"## Trigger / Event\n\n{trigger}",
        f"## Why It May Matter\n\n{matter}",
        f"## Evidence\n\n{evidence}",
        f"## Novelty\n\n{novelty}",
        f"## Fundamental Context\n\n{context}",
        f"## Catalyst\n\n{catalyst}",
        f"## Risks / Reasons It Could Be Wrong\n\n{risks}",
        f"## Data Limitations\n\n{limitations}",
        f"## Research Priority\n\n{priority}",
        f"## Why It Beat Other Candidates Today\n\n{comparison}",
        f"## Questions for Deep Research\n\n{questions}",
        "## Main Research Result\n\nDeterministic handoff created. No OpenAI or paid-API deep analysis was run in this pilot.",
        "## Decision Status\n\nRESEARCH ONLY — NO AUTOMATIC TRADE",
    ]
    return f"---\n{yaml_text}\n---\n\n" + "\n\n".join(sections) + "\n"


def daily_note_path(vault_root: Path, research_date: date) -> Path:
    return vault_root.resolve() / "Daily Highest Potential" / f"{research_date.isoformat()} - Daily Highest Potential.md"


def write_authorized_daily_note(vault_root: Path, research_date: date, content: str, *, authorized: bool) -> Path:
    """Create only the exact daily discovery path; never overwrite or broaden write authority."""

    if not authorized:
        raise PermissionError("explicit daily discovery write authorization is required")
    root = vault_root.resolve()
    folder = root / "Daily Highest Potential"
    destination = daily_note_path(root, research_date)
    folder.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink() or destination.is_symlink() or destination.parent.resolve() != folder.resolve():
        raise ValueError("daily discovery path failed containment or symlink checks")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite daily discovery note: {destination.name}")
    if "\x00" in content:
        raise ValueError("note content contains a NUL byte")
    descriptor, name = tempfile.mkstemp(prefix=".daily-discovery-", suffix=".tmp", dir=folder)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite daily discovery note: {destination.name}"
            ) from exc
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
            destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(destination_fd, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                destination.unlink(missing_ok=True)
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return destination
