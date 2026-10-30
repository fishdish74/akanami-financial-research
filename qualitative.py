"""Offline qualitative research models layered above deterministic evidence.

This module has no network or filesystem side effects. Forum/community material is
deliberately absent from analytics/ranking inputs and can only appear in its own
presentation section.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Literal, Optional

from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, field_validator,
    model_validator,
)

from identifiers import normalize_ticker
from schemas import EvidenceClaim, EvidenceItem


CLAIM_CATEGORIES = {
    "reported_fact": "REPORTED FACT",
    "management_statement": "COMPANY / MANAGEMENT STATEMENT",
    "source_opinion": "ANALYST / SOURCE OPINION",
    "ai_inference": "AI INFERENCE",
    "forum_community_opinion": "FORUM / COMMUNITY OPINION",
}


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _clean_id_list(values: object) -> list[str]:
    if values is None:
        return []
    raw = [values] if isinstance(values, str) else list(values)  # type: ignore[arg-type]
    result: list[str] = []
    for value in raw:
        cleaned = str(value).strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _validate_claim_bundle(
    evidence_catalog: list[EvidenceItem], claims: list[EvidenceClaim]
) -> tuple[dict[str, EvidenceItem], dict[str, EvidenceClaim]]:
    evidence = {item.evidence_id: item for item in evidence_catalog}
    claim_map = {item.claim_id: item for item in claims}
    if len(evidence) != len(evidence_catalog):
        raise ValueError("duplicate evidence ID in narrative catalog")
    if len(claim_map) != len(claims):
        raise ValueError("duplicate claim ID in narrative")
    for claim in claims:
        unknown = set(claim.evidence_ids) - set(evidence)
        if unknown:
            raise ValueError(f"claim references nonexistent evidence IDs: {sorted(unknown)}")
        classes = {evidence[value].evidence_class for value in claim.evidence_ids}
        if claim.category == "forum_community_opinion":
            if classes and classes != {"forum_community"}:
                raise ValueError("forum claims may cite only forum/community evidence")
        elif "forum_community" in classes:
            raise ValueError("forum evidence cannot support a core narrative claim")
    return evidence, claim_map


class CompanyImpact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    company: str
    business_overview: Optional[str] = None
    claim_ids: list[str] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def valid_ticker(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator("claim_ids", mode="before")
    @classmethod
    def clean_claim_ids(cls, value: object) -> list[str]:
        return _clean_id_list(value)


class CompanyNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str
    company: str
    evidence_catalog: list[EvidenceItem]
    claims: list[EvidenceClaim]
    business_overview: Optional[str] = None
    current_picture: list[str] = Field(default_factory=list)
    financial_picture: list[str] = Field(default_factory=list)
    important_developments: list[str] = Field(default_factory=list)
    interesting: list[str] = Field(default_factory=list)
    major_risks: list[str] = Field(default_factory=list)
    evidence_confidence: Literal["low", "medium", "high"]
    confidence_reasons: list[str]
    watch_next: list[str]

    @field_validator("ticker")
    @classmethod
    def valid_ticker(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator(
        "current_picture", "financial_picture", "important_developments",
        "interesting", "major_risks", mode="before",
    )
    @classmethod
    def clean_claim_ids(cls, value: object) -> list[str]:
        return _clean_id_list(value)

    @model_validator(mode="after")
    def known_claims_only(self) -> "CompanyNarrative":
        _, claims = _validate_claim_bundle(self.evidence_catalog, self.claims)
        referenced = {
            item for field in (
                self.current_picture, self.financial_picture,
                self.important_developments, self.interesting, self.major_risks,
            ) for item in field
        }
        unknown = referenced - set(claims)
        if unknown:
            raise ValueError(f"company narrative references unknown claims: {sorted(unknown)}")
        return self


class ForumSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signal_id: str = Field(pattern=r"^forum:[A-Za-z0-9._-]{1,100}$")
    forum: str = Field(min_length=1)
    author: str = Field(min_length=1)
    title: str = Field(min_length=1)
    post_date: AwareDatetime
    ticker: Optional[str] = None
    claim: EvidenceClaim
    evidence: EvidenceItem
    prediction_horizon: Optional[str] = None
    author_cited_evidence: list[str] = Field(default_factory=list)
    corroboration: Literal["none", "partial", "independent"] = "none"
    historical_reliability: Optional[str] = None
    source_url: HttpUrl
    needs_review: bool = True
    status: Literal["unverified", "opinion"] = "unverified"
    thesis_cluster_id: Optional[str] = None
    cluster_status: Literal["unclustered", "proposed", "human_confirmed"] = "unclustered"

    @field_validator("post_date", mode="after")
    @classmethod
    def normalize_post_date(cls, value: datetime) -> datetime:
        return _utc(value)

    @field_validator("ticker")
    @classmethod
    def normalize_optional_ticker(cls, value: str | None) -> str | None:
        return normalize_ticker(value) if value else None

    @model_validator(mode="after")
    def enforce_forum_boundary(self) -> "ForumSignal":
        if self.evidence.evidence_class != "forum_community":
            raise ValueError("forum signal evidence must use forum_community class")
        if self.claim.category != "forum_community_opinion":
            raise ValueError("forum signal claim must remain forum/community opinion")
        if self.claim.evidence_ids != [self.evidence.evidence_id]:
            raise ValueError("forum claim must cite its exact forum evidence ID")
        return self


def load_forum_signals(path: Path) -> tuple[list[ForumSignal], list[str]]:
    """Parse manual JSON per item; malformed signals never abort valid siblings."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("signals") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("forum fixture must be a list or contain a signals list")
    signals: list[ForumSignal] = []
    warnings: list[str] = []
    for position, item in enumerate(items, 1):
        try:
            signals.append(ForumSignal.model_validate(item))
        except (TypeError, ValueError):
            warnings.append(f"skipped malformed forum signal {position}")
    return signals, warnings


PredictionType = Literal[
    "stock_direction", "relative_performance", "revenue", "eps", "margins",
    "valuation", "business_development", "product_development",
    "management_expectation", "industry_development",
]


class FrozenForumPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    prediction_id: str = Field(pattern=r"^prediction:[A-Za-z0-9._-]{1,100}$")
    signal_id: str
    forum: str
    author: str
    source_url: HttpUrl
    ticker: Optional[str] = None
    prediction_created_at: AwareDatetime
    prediction_frozen_at: AwareDatetime
    prediction_type: PredictionType
    subject: str
    industry: Optional[str] = None
    topic: Optional[str] = None
    direction: Optional[Literal["up", "down", "increase", "decrease", "outperform", "underperform", "occur", "not_occur"]] = None
    magnitude: Optional[Decimal] = None
    horizon_days: Optional[int] = Field(default=None, gt=0)
    conditions: list[str] = Field(default_factory=list)
    specificity: Literal["low", "medium", "high"]
    falsifiable: bool
    evaluability: Literal["evaluable", "non_evaluable"]
    non_evaluable_reason: Optional[str] = None
    original_evidence_id: str
    frozen_interpretation: str
    thesis_cluster_id: Optional[str] = None
    derivative_of: Optional[str] = None
    cluster_status: Literal["unclustered", "proposed", "human_confirmed"] = "unclustered"

    @field_validator("prediction_created_at", "prediction_frozen_at", mode="after")
    @classmethod
    def normalize_times(cls, value: datetime) -> datetime:
        return _utc(value)

    @field_validator("ticker")
    @classmethod
    def normalize_optional_ticker(cls, value: str | None) -> str | None:
        return normalize_ticker(value) if value else None

    @model_validator(mode="after")
    def freeze_contract(self) -> "FrozenForumPrediction":
        if self.prediction_frozen_at < self.prediction_created_at:
            raise ValueError("prediction cannot be frozen before it was created")
        if self.evaluability == "evaluable":
            if not self.falsifiable or self.direction is None or self.horizon_days is None:
                raise ValueError("evaluable prediction requires direction, horizon, and falsifiability")
        elif not self.non_evaluable_reason:
            raise ValueError("non-evaluable prediction requires an explicit reason")
        return self


class PredictionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: str
    outcome_available_at: AwareDatetime
    evaluation_at: AwareDatetime
    observed_at: Optional[AwareDatetime] = None
    observed_direction: Optional[str] = None
    observed_magnitude: Optional[Decimal] = None
    evidence_id: str

    @field_validator("outcome_available_at", "evaluation_at", "observed_at", mode="after")
    @classmethod
    def normalize_times(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None

    @model_validator(mode="after")
    def observed_before_available(self) -> "PredictionOutcome":
        if self.observed_at and self.observed_at > self.outcome_available_at:
            raise ValueError("an outcome cannot be observed after its availability timestamp")
        return self


class PredictionEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    prediction_id: str
    author: str
    evaluation_result: Literal["correct", "incorrect", "partial", "non_evaluable"]
    outcome_available_at: Optional[AwareDatetime] = None
    evaluation_at: AwareDatetime
    prediction_created_at: AwareDatetime
    direction_correct: Optional[bool] = None
    timing_correct: Optional[bool] = None
    magnitude_error: Optional[Decimal] = None
    prediction_type: PredictionType
    horizon_bucket: str
    specificity: Literal["low", "medium", "high"]
    industry: Optional[str] = None
    topic: Optional[str] = None
    outcome_evidence_id: Optional[str] = None
    thesis_cluster_id: Optional[str] = None


def freeze_prediction(
    signal: ForumSignal,
    *,
    frozen_at: datetime,
    prediction_type: PredictionType,
    subject: str,
    direction: str | None,
    horizon_days: int | None,
    specificity: Literal["low", "medium", "high"],
    falsifiable: bool,
    frozen_interpretation: str,
    magnitude: Decimal | None = None,
    conditions: list[str] | None = None,
    industry: str | None = None,
    topic: str | None = None,
) -> FrozenForumPrediction:
    """Freeze only T0 signal data; outcome data is intentionally not accepted."""

    evaluable = bool(falsifiable and direction and horizon_days)
    digest = hashlib.sha256(
        f"{signal.signal_id}:{signal.post_date.isoformat()}:{frozen_interpretation}".encode()
    ).hexdigest()[:20]
    return FrozenForumPrediction(
        prediction_id=f"prediction:{digest}", signal_id=signal.signal_id,
        forum=signal.forum, author=signal.author, source_url=signal.source_url,
        ticker=signal.ticker, prediction_created_at=signal.post_date,
        prediction_frozen_at=frozen_at, prediction_type=prediction_type,
        subject=subject, industry=industry, topic=topic,
        direction=direction, magnitude=magnitude,
        horizon_days=horizon_days, conditions=conditions or [], specificity=specificity,
        falsifiable=falsifiable,
        evaluability="evaluable" if evaluable else "non_evaluable",
        non_evaluable_reason=None if evaluable else "statement is too vague to evaluate without invented precision",
        original_evidence_id=signal.evidence.evidence_id,
        frozen_interpretation=frozen_interpretation,
    )


def evaluate_prediction(
    prediction: FrozenForumPrediction,
    outcome: PredictionOutcome,
    evidence_catalog: list[EvidenceItem] | None = None,
) -> PredictionEvaluation:
    if outcome.prediction_id != prediction.prediction_id:
        raise ValueError("outcome does not belong to prediction")
    if outcome.evaluation_at < outcome.outcome_available_at:
        raise ValueError("prediction cannot be evaluated before outcome availability")
    if outcome.outcome_available_at < prediction.prediction_frozen_at:
        raise ValueError("future outcome cannot precede frozen interpretation")
    observed_at = outcome.observed_at or outcome.outcome_available_at
    if observed_at < prediction.prediction_created_at:
        raise ValueError("outcome observation cannot predate prediction creation")
    if outcome.outcome_available_at < prediction.prediction_created_at:
        raise ValueError("outcome availability cannot predate prediction creation")
    if evidence_catalog is None:
        raise ValueError("outcome evaluation requires a known evidence catalog")
    evidence = {item.evidence_id: item for item in evidence_catalog}.get(outcome.evidence_id)
    if evidence is None:
        raise ValueError("outcome evidence ID is not present in the evidence catalog")
    if evidence.available_at is None:
        raise ValueError("outcome evidence requires an availability timestamp")
    if evidence.available_at < prediction.prediction_created_at:
        raise ValueError("outcome evidence availability cannot predate prediction creation")
    if evidence.available_at > outcome.evaluation_at:
        raise ValueError("outcome evidence was unavailable at evaluation time")
    bucket = horizon_bucket(prediction.horizon_days)
    if prediction.evaluability == "non_evaluable":
        return PredictionEvaluation(
            prediction_id=prediction.prediction_id, author=prediction.author,
            evaluation_result="non_evaluable",
            evaluation_at=outcome.evaluation_at,
            prediction_created_at=prediction.prediction_created_at,
            prediction_type=prediction.prediction_type, horizon_bucket=bucket,
            specificity=prediction.specificity, industry=prediction.industry,
            topic=prediction.topic, outcome_evidence_id=outcome.evidence_id,
            thesis_cluster_id=prediction.thesis_cluster_id,
        )
    eligible_at = prediction.prediction_created_at + timedelta(days=prediction.horizon_days or 0)
    if outcome.outcome_available_at < eligible_at:
        raise ValueError("outcome is not yet available for the frozen prediction horizon")
    direction_correct = outcome.observed_direction == prediction.direction
    timing_correct = observed_at <= eligible_at
    magnitude_error = None
    if prediction.magnitude is not None and outcome.observed_magnitude is not None:
        magnitude_error = abs(outcome.observed_magnitude - prediction.magnitude)
    result = "correct" if direction_correct and timing_correct else "incorrect"
    if direction_correct and (not timing_correct or (
        magnitude_error is not None
        and magnitude_error > abs(prediction.magnitude or 0) * Decimal("0.5")
    )):
        result = "partial"
    return PredictionEvaluation(
        prediction_id=prediction.prediction_id, author=prediction.author,
        evaluation_result=result,
        outcome_available_at=outcome.outcome_available_at,
        evaluation_at=outcome.evaluation_at,
        prediction_created_at=prediction.prediction_created_at,
        direction_correct=direction_correct, timing_correct=timing_correct,
        magnitude_error=magnitude_error,
        prediction_type=prediction.prediction_type, horizon_bucket=bucket,
        specificity=prediction.specificity, industry=prediction.industry,
        topic=prediction.topic, outcome_evidence_id=outcome.evidence_id,
        thesis_cluster_id=prediction.thesis_cluster_id,
    )


def horizon_bucket(days: int | None) -> str:
    if days is None:
        return "unspecified"
    if days <= 7:
        return "short_term"
    if days <= 90:
        return "medium_term"
    return "long_term"


class ReliabilitySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    author: str
    prediction_type: PredictionType
    horizon_bucket: str
    resolved_predictions: int
    directional_accuracy: Optional[Decimal]
    timing_accuracy: Optional[Decimal]
    mean_magnitude_error: Optional[Decimal]
    shrunken_accuracy: Optional[Decimal]
    recency_weighted_accuracy: Optional[Decimal]
    uncertainty_low: Optional[Decimal]
    uncertainty_high: Optional[Decimal]
    historical_reliability: Literal["insufficient_history", "weak", "moderate", "strong"]
    independent_thesis_count: int
    specificity_counts: dict[str, int]


def historical_reliability(
    author: str,
    prediction_type: PredictionType,
    bucket: str,
    evaluations: list[PredictionEvaluation],
    *,
    as_of: datetime,
    minimum_history: int = 5,
    industry: str | None = None,
    topic: str | None = None,
) -> ReliabilitySummary:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("historical reliability as_of must be timezone-aware")
    cutoff = as_of.astimezone(timezone.utc)
    relevant = [
        item for item in evaluations
        if item.author == author
        and item.prediction_type == prediction_type
        and item.horizon_bucket == bucket
        and (industry is None or item.industry == industry)
        and (topic is None or item.topic == topic)
        and item.direction_correct is not None
        and item.evaluation_at <= cutoff
    ]
    count = len(relevant)
    correct = sum(item.direction_correct is True for item in relevant)
    independent = len({item.thesis_cluster_id or item.prediction_id for item in relevant})
    specificity_counts = {
        level: sum(item.specificity == level for item in relevant)
        for level in ("low", "medium", "high")
    }
    if count < minimum_history:
        return ReliabilitySummary(
            author=author, prediction_type=prediction_type, horizon_bucket=bucket,
            resolved_predictions=count, directional_accuracy=None, timing_accuracy=None,
            mean_magnitude_error=None, shrunken_accuracy=None,
            recency_weighted_accuracy=None, uncertainty_low=None, uncertainty_high=None,
            historical_reliability="insufficient_history",
            independent_thesis_count=independent, specificity_counts=specificity_counts,
        )
    raw = Decimal(correct) / Decimal(count)
    # Beta(2,2) shrinkage prevents tiny samples from receiving extreme estimates.
    shrunk = Decimal(correct + 2) / Decimal(count + 4)
    timing = [item.timing_correct for item in relevant if item.timing_correct is not None]
    timing_accuracy = Decimal(sum(value is True for value in timing)) / Decimal(len(timing)) if timing else None
    magnitude = [item.magnitude_error for item in relevant if item.magnitude_error is not None]
    mean_magnitude = sum(magnitude, Decimal(0)) / Decimal(len(magnitude)) if magnitude else None
    latest = max(item.evaluation_at for item in relevant)
    weights = [Decimal(str(0.5 ** ((latest - item.evaluation_at).days / 365))) for item in relevant]
    weighted = sum(
        weight * Decimal(1 if item.direction_correct else 0)
        for item, weight in zip(relevant, weights)
    ) / sum(weights)
    z = 1.96
    p = correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * count)) / count) / denominator
    low = Decimal(str(max(0.0, center - margin)))
    high = Decimal(str(min(1.0, center + margin)))
    label = "strong" if low >= Decimal("0.60") else "weak" if high <= Decimal("0.50") else "moderate"
    quant = Decimal("0.01")
    return ReliabilitySummary(
        author=author, prediction_type=prediction_type, horizon_bucket=bucket,
        resolved_predictions=count, directional_accuracy=raw.quantize(quant, rounding=ROUND_HALF_UP),
        timing_accuracy=timing_accuracy.quantize(quant, rounding=ROUND_HALF_UP) if timing_accuracy is not None else None,
        mean_magnitude_error=mean_magnitude.quantize(quant, rounding=ROUND_HALF_UP) if mean_magnitude is not None else None,
        shrunken_accuracy=shrunk.quantize(quant, rounding=ROUND_HALF_UP),
        recency_weighted_accuracy=weighted.quantize(quant, rounding=ROUND_HALF_UP),
        uncertainty_low=low.quantize(quant, rounding=ROUND_HALF_UP),
        uncertainty_high=high.quantize(quant, rounding=ROUND_HALF_UP),
        historical_reliability=label, independent_thesis_count=independent,
        specificity_counts=specificity_counts,
    )


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) > 2}


def propose_prediction_clusters(
    predictions: list[FrozenForumPrediction], *, threshold: Decimal = Decimal("0.70")
) -> list[FrozenForumPrediction]:
    """Propose derivative clusters deterministically; never marks them confirmed."""

    output: list[FrozenForumPrediction] = []
    for prediction in sorted(predictions, key=lambda item: (item.prediction_created_at, item.prediction_id)):
        matched: FrozenForumPrediction | None = None
        current_tokens = _tokens(prediction.frozen_interpretation)
        for prior in output:
            if (
                prior.ticker != prediction.ticker
                or prior.prediction_type != prediction.prediction_type
                or prior.horizon_days != prediction.horizon_days
                or prediction.prediction_created_at - prior.prediction_created_at > timedelta(days=30)
            ):
                continue
            prior_tokens = _tokens(prior.frozen_interpretation)
            union = current_tokens | prior_tokens
            similarity = Decimal(len(current_tokens & prior_tokens)) / Decimal(len(union) or 1)
            if similarity >= threshold:
                matched = prior
                break
        if matched is None:
            cluster = "cluster:" + hashlib.sha256(prediction.prediction_id.encode()).hexdigest()[:16]
            output.append(prediction.model_copy(update={
                "thesis_cluster_id": cluster, "cluster_status": "proposed",
            }))
        else:
            output.append(prediction.model_copy(update={
                "thesis_cluster_id": matched.thesis_cluster_id,
                "derivative_of": matched.prediction_id,
                "cluster_status": "proposed",
            }))
    return output


class NewListingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    limited_history_days: int = Field(default=30, ge=0)
    provisional_days: int = Field(default=90, gt=0)
    normal_coverage_threshold: Decimal = Field(default=Decimal("0.70"), ge=0, le=1)

    @model_validator(mode="after")
    def ordered_windows(self) -> "NewListingConfig":
        if self.provisional_days <= self.limited_history_days:
            raise ValueError("provisional window must exceed limited-history window")
        return self


class NewListingCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prospectus: bool = False
    audited_financial_history: bool = False
    peer_coverage: bool = False
    trading_history: bool = False
    earnings_history: bool = False
    analyst_coverage: bool = False
    verified_news: bool = False
    unresolved_contradictions: bool = False

    def ratio(self) -> Decimal:
        fields = (
            self.prospectus, self.audited_financial_history, self.peer_coverage,
            self.trading_history, self.earnings_history, self.analyst_coverage,
            self.verified_news,
        )
        return Decimal(sum(fields)) / Decimal(len(fields))


class NewListingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    ticker: str
    company: str
    detected_at: AwareDatetime
    listing_date: date
    listing_type: Literal[
        "traditional_ipo", "direct_listing", "spac_de_spac", "spin_off",
        "relisting", "unknown_other",
    ] = "unknown_other"
    coverage: NewListingCoverage
    evidence_catalog: list[EvidenceItem] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def valid_ticker(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator("detected_at", mode="after")
    @classmethod
    def normalize_detected(cls, value: datetime) -> datetime:
        return _utc(value)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def clean_evidence_ids(cls, value: object) -> list[str]:
        return _clean_id_list(value)

    @model_validator(mode="after")
    def known_non_forum_evidence_only(self) -> "NewListingCandidate":
        evidence = {item.evidence_id: item for item in self.evidence_catalog}
        if len(evidence) != len(self.evidence_catalog):
            raise ValueError("new-listing evidence catalog contains duplicate IDs")
        unknown = set(self.evidence_ids) - set(evidence)
        if unknown:
            raise ValueError(f"new listing references unknown evidence IDs: {sorted(unknown)}")
        if any(evidence[item].evidence_class == "forum_community" for item in self.evidence_ids):
            raise ValueError("forum/community evidence cannot qualify a new listing")
        return self


class NewListingAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate: NewListingCandidate
    assessment_date: date
    listing_age_days: int
    mode: Literal["limited_history", "provisional", "normal"]
    ranking_status: Literal["limited_history", "provisional", "not_rankable", "rankable"]
    evidence_confidence: Literal["low", "medium", "high"]
    coverage_ratio: Decimal
    confidence_limitations: list[str]
    labels: list[str]


def assess_new_listing(
    candidate: NewListingCandidate,
    *,
    assessment_date: date,
    config: NewListingConfig | None = None,
) -> NewListingAssessment:
    policy = config or NewListingConfig()
    age = (assessment_date - candidate.listing_date).days
    if age < 0:
        raise ValueError("assessment cannot precede listing date")
    coverage = candidate.coverage.ratio()
    if age <= policy.limited_history_days:
        mode, status = "limited_history", "limited_history"
    elif age <= policy.provisional_days:
        mode, status = "provisional", "provisional"
    else:
        mode = "normal"
        status = "rankable" if coverage >= policy.normal_coverage_threshold else "not_rankable"
    confidence = "high" if coverage >= Decimal("0.80") else "medium" if coverage >= Decimal("0.50") else "low"
    limitations: list[str] = []
    names = {
        "prospectus": "missing prospectus/registration evidence",
        "audited_financial_history": "limited audited financial history",
        "peer_coverage": "limited peer evidence",
        "trading_history": "short or missing public trading history",
        "earnings_history": "limited public earnings history",
        "analyst_coverage": "limited analyst coverage",
        "verified_news": "limited verified news coverage",
    }
    for field, message in names.items():
        if not getattr(candidate.coverage, field):
            limitations.append(message)
    if candidate.coverage.unresolved_contradictions:
        limitations.append("unresolved contradictions")
    return NewListingAssessment(
        candidate=candidate, assessment_date=assessment_date, listing_age_days=age,
        mode=mode, ranking_status=status, evidence_confidence=confidence,
        coverage_ratio=coverage.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        confidence_limitations=limitations,
        labels=["NEW LISTING", "LIMITED PUBLIC-MARKET HISTORY", "PROVISIONAL ASSESSMENT"],
    )


class DailyResearchBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")
    research_date: date
    evidence_catalog: list[EvidenceItem]
    claims: list[EvidenceClaim]
    executive_claim_ids: list[str]
    core_verified_development_ids: list[str] = Field(default_factory=list)
    company_impacts: list[CompanyImpact] = Field(default_factory=list)
    catalyst_claim_ids: list[str] = Field(default_factory=list)
    risk_claim_ids: list[str] = Field(default_factory=list)
    novelty_claim_ids: list[str] = Field(default_factory=list)
    contradiction_claim_ids: list[str] = Field(default_factory=list)
    data_quality_notes: list[str] = Field(default_factory=list)
    new_listings: list[NewListingAssessment] = Field(default_factory=list)
    forum_signals: list[ForumSignal] = Field(default_factory=list)
    watch_next: list[str] = Field(default_factory=list)

    @field_validator(
        "executive_claim_ids", "core_verified_development_ids", "catalyst_claim_ids",
        "risk_claim_ids", "novelty_claim_ids", "contradiction_claim_ids", mode="before",
    )
    @classmethod
    def clean_claim_ids(cls, value: object) -> list[str]:
        return _clean_id_list(value)

    @model_validator(mode="after")
    def evidence_contract(self) -> "DailyResearchBrief":
        _, claims = _validate_claim_bundle(self.evidence_catalog, self.claims)
        referenced = set(self.executive_claim_ids)
        referenced.update(self.core_verified_development_ids)
        referenced.update(self.catalyst_claim_ids)
        referenced.update(self.risk_claim_ids)
        referenced.update(self.novelty_claim_ids)
        referenced.update(self.contradiction_claim_ids)
        for impact in self.company_impacts:
            referenced.update(impact.claim_ids)
        unknown = referenced - set(claims)
        if unknown:
            raise ValueError(f"daily brief references unknown claims: {sorted(unknown)}")
        if not self.executive_claim_ids:
            raise ValueError("daily brief requires an evidence-linked executive brief")
        if any(claim.category == "forum_community_opinion" for claim in self.claims):
            raise ValueError("forum claims belong in forum_signals, not core brief claims")
        return self


def _md(value: object) -> str:
    text = " ".join(str(value).replace("\x00", "").splitlines())
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_!\[\]#|])", r"\\\1", text)


def _render_claims(ids: list[str], claims: dict[str, EvidenceClaim]) -> str:
    if not ids:
        return "- None supported by supplied evidence."
    lines = []
    for claim_id in ids:
        claim = claims[claim_id]
        refs = ", ".join(f"`{item}`" for item in claim.evidence_ids) or "unsupported"
        lines.append(
            f"- **{CLAIM_CATEGORIES[claim.category]}** — {_md(claim.text)} "
            f"_Evidence: {refs}; support: {claim.support}._"
        )
    return "\n".join(lines)


def render_daily_research_brief(brief: DailyResearchBrief) -> str:
    claims = {item.claim_id: item for item in brief.claims}
    company_lines: list[str] = []
    for impact in brief.company_impacts:
        company_lines.append(f"### {_md(impact.ticker)} — {_md(impact.company)}")
        if impact.business_overview:
            company_lines.append(_md(impact.business_overview))
        company_lines.append(_render_claims(impact.claim_ids, claims))
    listing_lines = []
    for item in brief.new_listings:
        listing_lines.extend([
            f"### {_md(item.candidate.ticker)} — {_md(item.candidate.company)}",
            f"**{' · '.join(item.labels)}**",
            f"- Listing type: `{item.candidate.listing_type}`",
            f"- Mode: `{item.mode}`; ranking status: `{item.ranking_status}`",
            f"- Evidence confidence: **{item.evidence_confidence}**",
            f"- Confidence limited by: {_md('; '.join(item.confidence_limitations) or 'no identified coverage limitation')}",
        ])
    forum_lines = []
    for item in brief.forum_signals:
        forum_lines.extend([
            f"### {_md(item.title)}",
            f"- **UNVERIFIED / OPINION — research signal only**",
            f"- Forum/source: {_md(item.forum)}; author: {_md(item.author)}; date: {item.post_date.date()}",
            f"- Company/ticker: {_md(item.ticker or 'not assigned')}",
            f"- Claim: {_md(item.claim.text)}",
            f"- Prediction horizon: {_md(item.prediction_horizon or 'not specified')}",
            f"- Author-cited evidence: {_md('; '.join(item.author_cited_evidence) or 'none supplied')}",
            f"- Independent corroboration: `{item.corroboration}`",
            f"- Historical reliability: {_md(item.historical_reliability or 'insufficient_history')}",
            f"- Source URL: {item.source_url}",
            f"- Needs review: `{str(item.needs_review).lower()}`",
        ])
    sections = [
        f"# Daily Market Research — {brief.research_date.isoformat()}",
        "## Executive Research Brief\n\n" + _render_claims(brief.executive_claim_ids, claims),
        "## Core Verified Developments\n\n" + _render_claims(brief.core_verified_development_ids, claims),
        "## Company Impact\n\n" + ("\n\n".join(company_lines) or "No evidence-linked company impact was supplied."),
        "## Catalysts\n\n" + _render_claims(brief.catalyst_claim_ids, claims),
        "## Risks\n\n" + _render_claims(brief.risk_claim_ids, claims),
        "## New Information vs Existing Research\n\n" + _render_claims(brief.novelty_claim_ids, claims),
        "## Contradictions\n\n" + _render_claims(brief.contradiction_claim_ids, claims),
        "## Evidence Confidence / Data Quality\n\n" + "\n".join(f"- {_md(item)}" for item in brief.data_quality_notes),
        "## New Listings\n\n" + ("\n".join(listing_lines) or "No new-listing candidates supplied."),
        "## Forum / Community Signals\n\n" + ("\n".join(forum_lines) or "No forum/community signals supplied."),
        "## What to Watch Next\n\n" + "\n".join(f"- {_md(item)}" for item in brief.watch_next),
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def render_company_narrative(narrative: CompanyNarrative) -> str:
    claims = {item.claim_id: item for item in narrative.claims}
    sections = [
        f"# {_md(narrative.ticker)} — {_md(narrative.company)} Research Summary",
        "## Business Overview\n\n" + _md(narrative.business_overview or "Business context was not supplied."),
        "## Current Picture\n\n" + _render_claims(narrative.current_picture, claims),
        "## Financial / Fundamental Picture\n\n" + _render_claims(narrative.financial_picture, claims),
        "## Important Developments\n\n" + _render_claims(narrative.important_developments, claims),
        "## What Looks Interesting\n\n" + _render_claims(narrative.interesting, claims),
        "## Major Risks\n\n" + _render_claims(narrative.major_risks, claims),
        f"## Evidence Confidence\n\n**{narrative.evidence_confidence.title()}**\n\n"
        + "\n".join(f"- {_md(item)}" for item in narrative.confidence_reasons),
        "## What to Watch Next\n\n" + "\n".join(f"- {_md(item)}" for item in narrative.watch_next),
        "## Supporting Evidence\n\n" + "\n".join(
            f"- `{item.evidence_id}` — {_md(item.label)}" for item in narrative.evidence_catalog
        ),
    ]
    return "\n\n".join(sections).rstrip() + "\n"
