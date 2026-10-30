"""Deterministic, no-write forward paper-tracking records."""

from __future__ import annotations

import calendar
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from .capabilities import CapabilityReason, SecurityCapability
from .exchange_calendar import (
    US_EQUITY_CALENDAR_POLICY_VERSION,
    ExchangeCalendar,
    us_equity_calendar,
)
from .models import PriceObservation


SelectionStatus = Literal["selected_for_tracking", "not_selected", "blocked"]
EvaluationStatus = Literal[
    "evaluated", "pending", "missing_price", "provider_gap",
    "inactive_unresolved", "terminal_value_verified",
    "unsupported_identity_transition", "unsupported_adjustment_basis",
]
OutcomeReason = Literal[
    "regular_session_close", "target_session_not_closed", "provider_bar_missing",
    "split_adjusted_close_missing", "verified_cash_acquisition",
    "terminal_event_evidence_missing", "identity_continuity_unverified",
    "verified_ticker_continuity", "temporary_halt",
    "split_coverage_unavailable", "intervening_split_detected",
]

CALENDAR_POLICY_VERSION = US_EQUITY_CALENDAR_POLICY_VERSION
LEGACY_CALENDAR_POLICY_VERSION = "us-equity-xnas-xnys-2025-2030-v1"
LEGACY_HORIZON_POLICY_VERSION = "calendar-offset-next-session-v1"
HORIZON_POLICY_VERSION = "exchange-local-calendar-offset-next-session-v2"
SUPPORTED_HORIZON_POLICY_VERSIONS = frozenset({
    LEGACY_HORIZON_POLICY_VERSION,
    HORIZON_POLICY_VERSION,
})
DECISION_PRICE_POLICY_VERSION = "latest-eligible-completed-session-v1"
DECISION_PRICE_AVAILABILITY_POLICY_VERSION = (
    "decision-price-regular-close-plus-15m-v1"
)
DECISION_PRICE_AVAILABILITY_BUFFER = timedelta(minutes=15)
DELISTING_POLICY_VERSION = "fail-closed-terminal-event-v1"
RETURN_DEFINITION_VERSION = "split-adjusted-price-return-v1"
ADJUSTMENT_BASIS_POLICY_VERSION = "intervening-split-fail-closed-v2"
DECISION_PRICE_BASIS = "frozen-provider-split-adjusted-at-decision-vintage"
FUTURE_PRICE_BASIS = "provider-split-adjusted-at-evaluation-vintage"
HISTORICAL_LEGACY_DECISION_IDS = frozenset({"000000000000000000000001"})


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _add_months(value: date, months: int) -> date:
    target_month = value.month - 1 + months
    year = value.year + target_month // 12
    month = target_month % 12 + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


class DecisionPriceAudit(BaseModel):
    """Compact evidence for prospective decision-price selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = DECISION_PRICE_POLICY_VERSION
    availability_policy_version: str = DECISION_PRICE_AVAILABILITY_POLICY_VERSION
    expected_session_date: date
    selected_session_date: date
    provider_response_statuses: tuple[str, ...]
    returned_bar_dates: tuple[date, ...]
    returned_bar_timestamps_ms: tuple[int, ...]
    page_hashes: tuple[str, ...]
    retrieved_at: AwareDatetime

    @field_validator("retrieved_at", mode="after")
    @classmethod
    def normalize_retrieved_at(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def selected_expected_session(self) -> "DecisionPriceAudit":
        if self.policy_version != DECISION_PRICE_POLICY_VERSION:
            raise ValueError("unsupported decision-price policy version")
        if self.availability_policy_version != DECISION_PRICE_AVAILABILITY_POLICY_VERSION:
            raise ValueError("unsupported decision-price availability policy version")
        if self.selected_session_date != self.expected_session_date:
            raise ValueError("selected decision-price session does not match expected session")
        if self.selected_session_date not in self.returned_bar_dates:
            raise ValueError("selected decision-price session is absent from returned dates")
        if (
            not self.provider_response_statuses
            or not self.returned_bar_timestamps_ms
            or not self.page_hashes
        ):
            raise ValueError(
                "decision-price audit requires provider status, timestamp, and page identity"
            )
        for values, label in (
            (self.provider_response_statuses, "provider statuses"),
            (self.returned_bar_dates, "returned dates"),
            (self.returned_bar_timestamps_ms, "returned timestamps"),
            (self.page_hashes, "page hashes"),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"decision-price audit {label} must be sorted and unique")
        return self


class DecisionPriceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation: PriceObservation
    audit: DecisionPriceAudit


class DecisionPriceTimingError(ValueError):
    """Deterministic fail-closed decision-price freshness failure."""

    def __init__(
        self, reason_code: str, *, expected_session_date: date,
        returned_bar_dates: tuple[date, ...],
    ) -> None:
        self.reason_code = reason_code
        self.expected_session_date = expected_session_date
        self.returned_bar_dates = returned_bar_dates
        super().__init__(
            f"{reason_code}: expected {expected_session_date.isoformat()}; "
            f"returned {','.join(value.isoformat() for value in returned_bar_dates) or 'none'}"
        )


class FrozenMarketInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_close: Decimal = Field(gt=0)
    split_adjusted_close: Decimal = Field(gt=0)
    price_date: date
    available_at: AwareDatetime
    retrieved_at: AwareDatetime
    volume: Decimal | None = Field(default=None, ge=0)
    currency: str
    provider_snapshot_ids: tuple[str, ...]
    volume_semantics_warning: str = "adjusted_volume_unverified"
    decision_price_audit: DecisionPriceAudit | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("available_at", "retrieved_at", mode="after")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_times(self) -> "FrozenMarketInput":
        if self.retrieved_at < self.available_at:
            raise ValueError("market retrieval cannot precede availability")
        if not self.provider_snapshot_ids:
            raise ValueError("market input requires provider snapshot identity")
        return self


class FrozenFundamentalInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    value: Decimal
    unit: str
    currency: str | None = None
    period_type: str
    start_date: date | None = None
    end_date: date
    available_at: AwareDatetime
    source: str
    source_artifact_id: int | None = Field(default=None, gt=0)
    quality_status: str

    @field_validator("available_at", mode="after")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)


class FrozenMetricContribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    value: Decimal | None = None


class FrozenPeerIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    company: str
    exchange: str
    cik: str
    sic: str
    industry: str
    classification_rule: str
    source_ids: tuple[str, ...]


class PaperSecurityInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    company_id: int = Field(gt=0)
    ticker: str
    company: str
    exchange: str
    capability: SecurityCapability
    market: FrozenMarketInput
    fundamentals: tuple[FrozenFundamentalInput, ...]
    ranking_status: Literal["ranked", "insufficient_evidence", "blocked"]
    opportunity_score: Decimal | None = Field(default=None, ge=0, le=100)
    available_weight: Decimal | None = Field(default=None, ge=0, le=1)
    metric_contributions: tuple[FrozenMetricContribution, ...]
    peer_identities: tuple[FrozenPeerIdentity, ...] = ()
    peer_classification_rule: str | None = None
    confidence: Literal["low", "medium", "high"]
    missing_or_blocked_reasons: tuple[str, ...] = ()
    evidence_summary: tuple[str, ...] = ()
    source_artifact_ids: tuple[int, ...] = ()
    mapping_versions: tuple[str, ...] = ()
    analytics_run_id: str
    ranking_version: str
    pipeline_version: str
    calculation_version: str
    config_hash: str

    @model_validator(mode="after")
    def ranking_contract(self) -> "PaperSecurityInput":
        if self.ranking_status == "ranked" and self.opportunity_score is None:
            raise ValueError("ranked paper input requires an opportunity score")
        if self.ranking_status != "ranked" and self.opportunity_score is not None:
            raise ValueError("non-ranked paper input cannot carry an opportunity score")
        return self


class EvaluationHorizon(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["one_trading_week", "one_month", "three_months"]
    target_not_before: date
    resolved_session_date: date | None = None
    evaluation_timestamp_rule: str
    price_field: Literal["split_adjusted_close"] = "split_adjusted_close"
    return_type: Literal["price_return"] = "price_return"
    corporate_action_treatment: str
    missing_price_policy: str
    delisting_policy: str
    benchmark_policy: str
    calendar_policy_version: str
    horizon_policy_version: str
    delisting_policy_version: str
    return_definition_version: str


class PaperDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rehearsal_id: str
    decision_id: str
    company_id: int = Field(gt=0)
    ticker: str
    exchange: str
    decision_timestamp: AwareDatetime
    eligibility_status: Literal["eligible", "blocked"]
    selection_status: SelectionStatus
    capability_reason_codes: tuple[CapabilityReason, ...]
    ranking_status: Literal["ranked", "insufficient_evidence", "blocked"]
    opportunity_score: Decimal | None = Field(default=None, ge=0, le=100)
    confidence: Literal["low", "medium", "high"]
    available_weight: Decimal | None = Field(default=None, ge=0, le=1)
    decision_raw_close: Decimal = Field(gt=0)
    decision_split_adjusted_close: Decimal = Field(gt=0)
    decision_price_date: date
    evidence_summary: tuple[str, ...]
    key_metrics: tuple[FrozenFundamentalInput, ...]
    metric_contributions: tuple[FrozenMetricContribution, ...]
    peer_identities: tuple[FrozenPeerIdentity, ...]
    peer_classification_rule: str | None
    unresolved_risks: tuple[str, ...]
    source_artifact_ids: tuple[int, ...]
    mapping_versions: tuple[str, ...]
    provider_snapshot_ids: tuple[str, ...]
    snapshot_hash: str
    analytics_run_id: str
    ranking_version: str
    pipeline_version: str
    calculation_version: str
    config_hash: str
    model_metadata: None = None
    evaluation_horizons: tuple[EvaluationHorizon, ...]
    security_input: PaperSecurityInput
    status: Literal["paper_only"] = "paper_only"

    @field_validator("decision_timestamp", mode="after")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)


class PaperRehearsal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rehearsal_id: str
    decision_timestamp: AwareDatetime
    decisions: tuple[PaperDecision, ...]
    frozen_payload_hash: str
    evaluation_horizons: tuple[EvaluationHorizon, ...]
    status: Literal["paper_only"] = "paper_only"
    files_written: Literal[0] = 0
    database_rows_written: Literal[0] = 0


class PriceReturnEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str
    ticker: str
    horizon: str
    return_type: Literal["price_return"] = "price_return"
    decision_split_adjusted_close: Decimal
    future_split_adjusted_close: Decimal
    price_return: Decimal
    total_return_included: Literal[False] = False
    adjustment_basis_policy_version: str = ADJUSTMENT_BASIS_POLICY_VERSION
    rebasing_applied: Literal[False] = False


class EvaluationPriceBar(BaseModel):
    """One provider-observed regular-session candidate for outcome evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    exchange: str
    trading_date: date
    split_adjusted_close: Decimal | None = Field(default=None, gt=0)
    regular_session: bool
    available_at: AwareDatetime
    provider_snapshot_id: str
    source_page_ids: tuple[str, ...] = ()

    @field_validator("available_at", mode="after")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)


class TerminalEventEvidence(BaseModel):
    """Explicit evidence only; missing bars can never manufacture an event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_type: Literal[
        "cash_acquisition", "delisting", "ticker_change", "bankruptcy",
        "liquidation", "temporary_halt", "inactive_unknown",
    ]
    effective_date: date
    available_at: AwareDatetime
    source_ids: tuple[str, ...]
    independently_verified: bool = False
    identity_continuity_verified: bool = False
    old_ticker: str | None = None
    new_ticker: str | None = None
    terminal_cash_value: Decimal | None = Field(default=None, gt=0)

    @field_validator("available_at", mode="after")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def evidence_contract(self) -> "TerminalEventEvidence":
        if not self.source_ids:
            raise ValueError("terminal-event evidence requires source identity")
        if self.event_type == "cash_acquisition" and self.terminal_cash_value is not None:
            if not self.identity_continuity_verified:
                raise ValueError("terminal cash requires verified identity continuity")
        if self.event_type == "ticker_change" and self.identity_continuity_verified:
            if not self.old_ticker or not self.new_ticker:
                raise ValueError("verified ticker continuity requires old and new tickers")
        return self


class SplitAdjustmentEvidence(BaseModel):
    """One immutable split-like action relevant to price-basis compatibility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str
    event_type: Literal["stock_split", "reverse_split", "stock_dividend"]
    effective_date: date
    ratio: Decimal = Field(gt=0)
    available_at: AwareDatetime
    source_id: str

    @field_validator("available_at", mode="after")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)

    @field_validator("source_id")
    @classmethod
    def nonempty_source_id(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("split evidence identity fields must be non-empty")
        return clean

    @field_validator("ticker")
    @classmethod
    def normalized_ticker(cls, value: str) -> str:
        clean = value.strip().upper()
        if not clean:
            raise ValueError("split evidence ticker must be non-empty")
        return clean


class SplitBasisCoverage(BaseModel):
    """Complete provider coverage for the decision-to-evaluation split interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_start: date
    requested_end: date
    complete: bool
    available_at: AwareDatetime
    source_ids: tuple[str, ...]
    evidence: tuple[SplitAdjustmentEvidence, ...] = ()
    # Transport provenance is diagnostic only; it never enters the outcome or
    # canonical evaluation-event identity.
    raw_page_source_ids: tuple[str, ...] = ()

    @field_validator("available_at", mode="after")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def coverage_contract(self) -> "SplitBasisCoverage":
        if self.requested_end < self.requested_start:
            raise ValueError("split coverage end precedes start")
        if self.complete and not self.source_ids:
            raise ValueError("complete split coverage requires source identity")
        if any(not item.strip() for item in self.source_ids):
            raise ValueError("split coverage source IDs must be non-empty")
        if tuple(sorted(set(self.source_ids))) != self.source_ids:
            raise ValueError("split coverage source IDs must be sorted and unique")
        return self


class AdmittedEvaluationEvidence(BaseModel):
    """Validation inputs, not opaque provenance assertions, for a price return."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    retrieval_started_at: AwareDatetime
    retrieval_completed_at: AwareDatetime
    authorized_boundary: AwareDatetime
    operator_cutoff: AwareDatetime
    retrieval_policy_version: str | None = None
    bar: EvaluationPriceBar
    coverage: SplitBasisCoverage


class PaperEvaluationOutcome(BaseModel):
    """Fail-closed outcome that preserves every tracked decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str
    ticker: str
    horizon: str
    target_not_before: date
    resolved_session_date: date
    status: EvaluationStatus
    reason_codes: tuple[OutcomeReason, ...]
    evaluation_value: Decimal | None = None
    evaluation_value_type: Literal[
        "split_adjusted_close", "terminal_cash_consideration"
    ] | None = None
    price_return: Decimal | None = None
    return_type: Literal["price_return"] = "price_return"
    total_return_included: Literal[False] = False
    provider_snapshot_id: str | None = None
    terminal_event_source_ids: tuple[str, ...] = ()
    successor_ticker: str | None = None
    calendar_policy_version: str
    horizon_policy_version: str
    delisting_policy_version: str
    return_definition_version: str
    adjustment_basis_policy_version: str = ADJUSTMENT_BASIS_POLICY_VERSION
    decision_price_basis: str = DECISION_PRICE_BASIS
    future_price_basis: str | None = None
    split_coverage_source_ids: tuple[str, ...] = ()
    intervening_split_source_ids: tuple[str, ...] = ()
    cumulative_split_factor: Decimal | None = None
    rebasing_applied: Literal[False] = False

    admitted_evidence: AdmittedEvaluationEvidence | None = None


class HorizonEvaluationResult(BaseModel):
    """Outcome plus the exact evidence timestamps admitted by its evaluator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: PaperEvaluationOutcome
    admitted_evidence_available_at: tuple[AwareDatetime, ...] = ()

    @field_validator("admitted_evidence_available_at", mode="after")
    @classmethod
    def normalize_evidence_times(
        cls, values: tuple[datetime, ...]
    ) -> tuple[datetime, ...]:
        return tuple(value.astimezone(timezone.utc) for value in values)


def expected_decision_price_session(
    decision_timestamp: datetime, exchange: str
) -> date:
    """Return the exact completed session required for a prospective decision."""

    if decision_timestamp.tzinfo is None or decision_timestamp.utcoffset() is None:
        raise ValueError("decision_timestamp must be timezone-aware")
    cutoff = decision_timestamp.astimezone(timezone.utc)
    calendar_owner = us_equity_calendar(exchange)
    local_date = calendar_owner.trading_date_for(cutoff)
    local_session = calendar_owner.session(local_date)
    if (
        local_session.kind != "CLOSED"
        and local_session.closes_at is not None
        and cutoff >= local_session.closes_at + DECISION_PRICE_AVAILABILITY_BUFFER
    ):
        return local_session.trading_date
    return calendar_owner.previous_session_on_or_before(
        local_date - timedelta(days=1), max_calendar_days=7
    ).trading_date


def select_decision_price(
    observations: list[PriceObservation], *, decision_timestamp: datetime, exchange: str
) -> DecisionPriceSelection:
    """Select only the expected completed session and retain compact provider evidence."""

    expected = expected_decision_price_session(decision_timestamp, exchange)
    returned_dates = tuple(sorted({item.trading_date for item in observations}))
    if len(returned_dates) != len(observations):
        raise DecisionPriceTimingError(
            "decision_price_duplicate_session",
            expected_session_date=expected,
            returned_bar_dates=returned_dates,
        )
    cutoff = decision_timestamp.astimezone(timezone.utc)
    if any(item.available_at > cutoff or item.retrieved_at > cutoff for item in observations):
        raise DecisionPriceTimingError(
            "decision_price_future_input",
            expected_session_date=expected,
            returned_bar_dates=returned_dates,
        )
    if not observations:
        raise DecisionPriceTimingError(
            "decision_price_provider_gap",
            expected_session_date=expected,
            returned_bar_dates=returned_dates,
        )
    latest = max(observations, key=lambda item: item.trading_date)
    if latest.trading_date != expected:
        reason = (
            "decision_price_stale"
            if latest.trading_date < expected
            else "decision_price_ineligible_future_session"
        )
        raise DecisionPriceTimingError(
            reason,
            expected_session_date=expected,
            returned_bar_dates=returned_dates,
        )
    if latest.split_adjusted_close is None:
        raise DecisionPriceTimingError(
            "decision_price_split_adjusted_close_missing",
            expected_session_date=expected,
            returned_bar_dates=returned_dates,
        )

    statuses: set[str] = set()
    hashes: set[str] = set()
    timestamps: set[int] = set()
    retrieved = max(item.retrieved_at for item in observations)
    for item in observations:
        raw_timestamp = item.provenance.get("raw_timestamp_ms")
        if isinstance(raw_timestamp, int) and not isinstance(raw_timestamp, bool):
            timestamps.add(raw_timestamp)
        for key in ("raw_pages", "adjusted_pages"):
            pages = item.provenance.get(key, ())
            if not isinstance(pages, (list, tuple)):
                continue
            for page in pages:
                if not isinstance(page, dict):
                    continue
                content_hash = page.get("content_hash")
                if isinstance(content_hash, str) and content_hash:
                    hashes.add(content_hash)
                source = page.get("source")
                status = source.get("response_status") if isinstance(source, dict) else None
                statuses.add(str(status) if status is not None else "UNSPECIFIED")
    audit = DecisionPriceAudit(
        expected_session_date=expected,
        selected_session_date=latest.trading_date,
        provider_response_statuses=tuple(sorted(statuses)),
        returned_bar_dates=returned_dates,
        returned_bar_timestamps_ms=tuple(sorted(timestamps)),
        page_hashes=tuple(sorted(hashes)),
        retrieved_at=retrieved,
    )
    return DecisionPriceSelection(observation=latest, audit=audit)


def _evaluation_horizons_for_anchor(
    anchor: date, horizon_policy_version: str
) -> tuple[EvaluationHorizon, ...]:
    rule = "first validated regular-session close on or after target_not_before"
    common = {
        "evaluation_timestamp_rule": rule,
        "corporate_action_treatment": (
            "use provider split-adjusted close; dividends are excluded"
        ),
        "missing_price_policy": (
            "provider gap on the frozen open session; do not advance, forward-fill, or substitute"
        ),
        "delisting_policy": (
            "fail closed unless verified terminal value or identity continuity evidence exists"
        ),
        "benchmark_policy": "none in paper-tracking v1",
        "calendar_policy_version": CALENDAR_POLICY_VERSION,
        "horizon_policy_version": horizon_policy_version,
        "delisting_policy_version": DELISTING_POLICY_VERSION,
        "return_definition_version": RETURN_DEFINITION_VERSION,
    }
    return (
        EvaluationHorizon(
            name="one_trading_week",
            target_not_before=anchor + timedelta(days=7),
            **common,
        ),
        EvaluationHorizon(
            name="one_month", target_not_before=_add_months(anchor, 1), **common
        ),
        EvaluationHorizon(
            name="three_months", target_not_before=_add_months(anchor, 3), **common
        ),
    )


def evaluation_horizons(
    decision_timestamp: datetime, exchange: str
) -> tuple[EvaluationHorizon, ...]:
    """Prospective V2 horizons anchored to the exchange-local decision date."""

    if decision_timestamp.tzinfo is None or decision_timestamp.utcoffset() is None:
        raise ValueError("decision_timestamp must be timezone-aware")
    anchor = us_equity_calendar(exchange).trading_date_for(decision_timestamp)
    return _evaluation_horizons_for_anchor(anchor, HORIZON_POLICY_VERSION)


def legacy_evaluation_horizons(
    decision_timestamp: datetime,
) -> tuple[EvaluationHorizon, ...]:
    """Historical V1 UTC-date schedule; retained only for immutable record replay."""

    if decision_timestamp.tzinfo is None or decision_timestamp.utcoffset() is None:
        raise ValueError("decision_timestamp must be timezone-aware")
    anchor = decision_timestamp.astimezone(timezone.utc).date()
    return _evaluation_horizons_for_anchor(anchor, LEGACY_HORIZON_POLICY_VERSION)


def resolved_evaluation_horizons(
    decision_timestamp: datetime, exchange: str, *,
    horizon_policy_version: str = HORIZON_POLICY_VERSION,
) -> tuple[EvaluationHorizon, ...]:
    calendar_owner = us_equity_calendar(exchange)
    if horizon_policy_version == HORIZON_POLICY_VERSION:
        horizons = evaluation_horizons(decision_timestamp, exchange)
    elif horizon_policy_version == LEGACY_HORIZON_POLICY_VERSION:
        horizons = legacy_evaluation_horizons(decision_timestamp)
    else:
        raise ValueError("unsupported horizon policy version")
    return tuple(
        item.model_copy(update={
            "resolved_session_date": calendar_owner.next_session_on_or_after(
                item.target_not_before, max_calendar_days=7
            ).trading_date
        })
        for item in horizons
    )


def build_paper_rehearsal(
    securities: list[PaperSecurityInput],
    *,
    decision_timestamp: datetime,
    max_selected: int = 2,
    horizon_policy_version: str = HORIZON_POLICY_VERSION,
) -> PaperRehearsal:
    """Freeze one deterministic decision set without filesystem/database persistence."""

    if decision_timestamp.tzinfo is None or decision_timestamp.utcoffset() is None:
        raise ValueError("decision_timestamp must be timezone-aware")
    cutoff = decision_timestamp.astimezone(timezone.utc)
    if not 1 <= max_selected <= 2:
        raise ValueError("paper rehearsal may select only one or two securities")
    if len({item.ticker for item in securities}) != len(securities):
        raise ValueError("paper rehearsal tickers must be unique")
    if horizon_policy_version != HORIZON_POLICY_VERSION:
        if horizon_policy_version == LEGACY_HORIZON_POLICY_VERSION:
            raise ValueError(
                "legacy V1 paper construction is replay-only; use the validated "
                "historical artifact loader"
            )
        raise ValueError("unsupported horizon policy version")
    for item in securities:
        timestamps = [item.market.available_at, item.market.retrieved_at]
        timestamps.extend(value.available_at for value in item.fundamentals)
        if any(value > cutoff for value in timestamps):
            raise ValueError(f"future input exceeds decision timestamp: {item.ticker}")
        if horizon_policy_version == HORIZON_POLICY_VERSION:
            audit = item.market.decision_price_audit
            if audit is None:
                raise ValueError(
                    f"prospective decision requires decision-price audit: {item.ticker}"
                )
            expected = expected_decision_price_session(cutoff, item.exchange)
            if (
                audit.expected_session_date != expected
                or audit.selected_session_date != item.market.price_date
                or audit.retrieved_at != item.market.retrieved_at
            ):
                raise ValueError(
                    f"decision-price audit does not match frozen market input: {item.ticker}"
                )
            snapshot_hashes = {
                value.rsplit(":", 1)[-1] for value in item.market.provider_snapshot_ids
            }
            if not set(audit.page_hashes).issubset(snapshot_hashes):
                raise ValueError(
                    f"decision-price audit page identity is not frozen: {item.ticker}"
                )

    horizons_by_ticker = {
        item.ticker: resolved_evaluation_horizons(
            cutoff, item.exchange, horizon_policy_version=horizon_policy_version
        )
        for item in securities
    }
    unique_schedules = {
        tuple(value.resolved_session_date for value in horizons)
        for horizons in horizons_by_ticker.values()
    }
    horizons = (
        next(iter(horizons_by_ticker.values()))
        if horizons_by_ticker and len(unique_schedules) == 1
        else ()
    )
    frozen_inputs = [item.model_dump(mode="json") for item in sorted(securities, key=lambda x: x.ticker)]
    frozen_payload_hash = _canonical_hash({
        "decision_timestamp": cutoff.isoformat(),
        "securities": frozen_inputs,
        "evaluation_horizons_by_ticker": {
            ticker: [item.model_dump(mode="json") for item in schedule]
            for ticker, schedule in sorted(horizons_by_ticker.items())
        },
    })
    rehearsal_id = _canonical_hash(
        {
            "kind": (
                "forward-paper-rehearsal-v2"
                if horizon_policy_version == HORIZON_POLICY_VERSION
                else "forward-paper-rehearsal-v1"
            ),
            "payload": frozen_payload_hash,
        }
    )[:24]
    eligible_ranked = sorted(
        (
            item for item in securities
            if item.capability.ranking_eligible
            and item.ranking_status == "ranked"
            and item.opportunity_score is not None
        ),
        key=lambda item: (-item.opportunity_score, item.ticker),
    )
    selected = {item.ticker for item in eligible_ranked[:max_selected]}
    decisions: list[PaperDecision] = []
    for item in sorted(securities, key=lambda value: value.ticker):
        item_horizons = horizons_by_ticker[item.ticker]
        blocked = not item.capability.ranking_eligible or item.ranking_status == "blocked"
        selection: SelectionStatus = (
            "blocked" if blocked else
            "selected_for_tracking" if item.ticker in selected else
            "not_selected"
        )
        snapshot_payload = {
            "decision_timestamp": cutoff.isoformat(),
            "security": item.model_dump(mode="json"),
            "evaluation_horizons": [
                value.model_dump(mode="json") for value in item_horizons
            ],
        }
        snapshot_hash = _canonical_hash(snapshot_payload)
        decision_id = _canonical_hash(
            {"rehearsal_id": rehearsal_id, "ticker": item.ticker, "snapshot_hash": snapshot_hash}
        )[:24]
        decisions.append(PaperDecision(
            rehearsal_id=rehearsal_id,
            decision_id=decision_id,
            company_id=item.company_id,
            ticker=item.ticker,
            exchange=item.exchange,
            decision_timestamp=cutoff,
            eligibility_status="blocked" if blocked else "eligible",
            selection_status=selection,
            capability_reason_codes=tuple(item.capability.reason_codes),
            ranking_status=item.ranking_status,
            opportunity_score=item.opportunity_score,
            confidence=item.confidence,
            available_weight=item.available_weight,
            decision_raw_close=item.market.raw_close,
            decision_split_adjusted_close=item.market.split_adjusted_close,
            decision_price_date=item.market.price_date,
            evidence_summary=item.evidence_summary,
            key_metrics=item.fundamentals,
            metric_contributions=item.metric_contributions,
            peer_identities=item.peer_identities,
            peer_classification_rule=item.peer_classification_rule,
            unresolved_risks=item.missing_or_blocked_reasons,
            source_artifact_ids=tuple(sorted(set(item.source_artifact_ids))),
            mapping_versions=tuple(sorted(set(item.mapping_versions))),
            provider_snapshot_ids=tuple(sorted(set(item.market.provider_snapshot_ids))),
            snapshot_hash=snapshot_hash,
            analytics_run_id=item.analytics_run_id,
            ranking_version=item.ranking_version,
            pipeline_version=item.pipeline_version,
            calculation_version=item.calculation_version,
            config_hash=item.config_hash,
            evaluation_horizons=item_horizons,
            security_input=item,
        ))
    return PaperRehearsal(
        rehearsal_id=rehearsal_id,
        decision_timestamp=cutoff,
        decisions=tuple(decisions),
        frozen_payload_hash=frozen_payload_hash,
        evaluation_horizons=horizons,
    )


def build_legacy_paper_rehearsal(
    securities: list[PaperSecurityInput], *, decision_timestamp: datetime,
    max_selected: int = 2,
) -> PaperRehearsal:
    """Reject prospective V1 construction; legacy V1 is artifact-replay only."""

    raise ValueError(
        "legacy V1 paper construction is replay-only; use the validated historical "
        "artifact loader or the current RC2 builder"
    )


def evaluate_price_return(
    decision: PaperDecision,
    *,
    horizon: str,
    future_split_adjusted_close: Decimal,
    same_split_basis_proven: bool = False,
) -> PriceReturnEvaluation:
    """Evaluate only the frozen decision price; never reconstruct current state."""

    allowed = {item.name for item in decision.evaluation_horizons}
    if horizon not in allowed:
        raise ValueError("evaluation horizon was not frozen with the decision")
    if future_split_adjusted_close <= 0:
        raise ValueError("future split-adjusted close must be positive")
    if not same_split_basis_proven:
        raise ValueError("a common split-adjustment basis must be proven")
    result = (
        future_split_adjusted_close / decision.decision_split_adjusted_close
    ) - Decimal(1)
    return PriceReturnEvaluation(
        decision_id=decision.decision_id,
        ticker=decision.ticker,
        horizon=horizon,
        decision_split_adjusted_close=decision.decision_split_adjusted_close,
        future_split_adjusted_close=future_split_adjusted_close,
        price_return=result,
    )


def resolve_evaluation_schedule(
    decision_timestamp: datetime, exchange: str, *,
    horizon_policy_version: str = HORIZON_POLICY_VERSION,
) -> dict[str, date]:
    """Resolve calendar offsets to the first validated session on/after each target."""

    return {
        horizon.name: horizon.resolved_session_date
        for horizon in resolved_evaluation_horizons(
            decision_timestamp, exchange,
            horizon_policy_version=horizon_policy_version,
        )
    }


def evaluate_horizon_outcome_with_evidence(
    decision: PaperDecision,
    *,
    horizon: str,
    calendar_owner: ExchangeCalendar,
    evaluation_timestamp: datetime,
    bars: tuple[EvaluationPriceBar, ...] = (),
    terminal_event: TerminalEventEvidence | None = None,
    split_basis_coverage: SplitBasisCoverage | None = None,
) -> HorizonEvaluationResult:
    """Evaluate a horizon and retain only evidence admitted by the outcome."""

    if evaluation_timestamp.tzinfo is None or evaluation_timestamp.utcoffset() is None:
        raise ValueError("evaluation_timestamp must be timezone-aware")
    evaluated_at = evaluation_timestamp.astimezone(timezone.utc)
    exchange_alias = {"NASDAQ": "XNAS", "NYSE": "XNYS"}
    decision_exchange = exchange_alias.get(
        decision.exchange.strip().upper(), decision.exchange.strip().upper()
    )
    if calendar_owner.exchange != decision_exchange:
        raise ValueError("calendar exchange does not match the frozen decision exchange")
    try:
        frozen = next(item for item in decision.evaluation_horizons if item.name == horizon)
    except StopIteration as exc:
        raise ValueError("evaluation horizon was not frozen with the decision") from exc
    exact_legacy_replay = (
        decision.decision_id in HISTORICAL_LEGACY_DECISION_IDS
        and frozen.calendar_policy_version == LEGACY_CALENDAR_POLICY_VERSION
        and frozen.horizon_policy_version == LEGACY_HORIZON_POLICY_VERSION
    )
    if (
        (
            not exact_legacy_replay
            and frozen.calendar_policy_version != calendar_owner.policy_version
        )
        or frozen.horizon_policy_version not in SUPPORTED_HORIZON_POLICY_VERSIONS
        or frozen.delisting_policy_version != DELISTING_POLICY_VERSION
        or frozen.return_definition_version != RETURN_DEFINITION_VERSION
    ):
        raise ValueError("frozen evaluation policy versions do not match the evaluator")
    expected_session = calendar_owner.next_session_on_or_after(
        frozen.target_not_before, max_calendar_days=7
    )
    if (
        frozen.resolved_session_date is not None
        and frozen.resolved_session_date != expected_session.trading_date
    ):
        raise ValueError("frozen resolved session does not match its calendar policy")
    session = expected_session
    common = {
        "decision_id": decision.decision_id,
        "ticker": decision.ticker,
        "horizon": horizon,
        "target_not_before": frozen.target_not_before,
        "resolved_session_date": session.trading_date,
        "calendar_policy_version": frozen.calendar_policy_version,
        "horizon_policy_version": frozen.horizon_policy_version,
        "delisting_policy_version": frozen.delisting_policy_version,
        "return_definition_version": frozen.return_definition_version,
    }
    admitted: tuple[datetime, ...] = ()

    expected_split_start = decision.decision_price_date + timedelta(days=1)
    # Massive's split execution dates are exchange-local dates. The historical
    # target bar reflects the provider's adjustment vintage at retrieval, even
    # when a split occurred after the target session. A shorter interval cannot
    # prove that the frozen decision price and retrieved bar share a basis.
    expected_split_end = evaluated_at.astimezone(
        ZoneInfo(calendar_owner.timezone_name)
    ).date()

    def assess_split_basis() -> tuple[bool, OutcomeReason, tuple[str, ...], tuple[datetime, ...]]:
        coverage = split_basis_coverage
        if (
            coverage is None
            or not coverage.complete
            or coverage.requested_start != expected_split_start
            or coverage.requested_end != expected_split_end
            or coverage.available_at > evaluated_at
        ):
            return False, "split_coverage_unavailable", (), ()
        relevant = tuple(
            item for item in coverage.evidence
            if expected_split_start <= item.effective_date <= expected_split_end
        )
        if any(
            item.available_at > evaluated_at
            or item.ticker not in {decision.ticker, terminal_event.new_ticker if terminal_event else None}
            for item in relevant
        ):
            return False, "split_coverage_unavailable", (), ()
        evidence_times = tuple(sorted({coverage.available_at, *(item.available_at for item in relevant)}))
        split_ids = tuple(sorted(item.source_id for item in relevant))
        if split_ids:
            return False, "intervening_split_detected", split_ids, evidence_times
        return True, "split_coverage_unavailable", (), evidence_times

    def unsupported_basis(
        reason: OutcomeReason, split_ids: tuple[str, ...], basis_admitted: tuple[datetime, ...],
        *, snapshot: str | None = None, successor: str | None = None,
        terminal_sources: tuple[str, ...] = (), continuity: tuple[OutcomeReason, ...] = (),
    ) -> HorizonEvaluationResult:
        coverage_sources = (
            split_basis_coverage.source_ids
            if split_basis_coverage is not None and basis_admitted else ()
        )
        return HorizonEvaluationResult(
            outcome=PaperEvaluationOutcome(
                **common,
                status="unsupported_adjustment_basis",
                reason_codes=(reason,) + continuity,
                provider_snapshot_id=snapshot,
                terminal_event_source_ids=terminal_sources,
                successor_ticker=successor,
                future_price_basis=FUTURE_PRICE_BASIS if snapshot is not None else None,
                split_coverage_source_ids=coverage_sources,
                intervening_split_source_ids=split_ids,
            ),
            admitted_evidence_available_at=tuple(sorted({*admitted, *basis_admitted})),
        )
    event_is_known = (
        terminal_event is not None
        and terminal_event.available_at <= evaluated_at
        and terminal_event.effective_date <= session.trading_date
    )
    if event_is_known and terminal_event is not None:
        admitted = (terminal_event.available_at,)
        if terminal_event.event_type == "temporary_halt":
            return HorizonEvaluationResult(
                outcome=PaperEvaluationOutcome(
                    **common, status="pending", reason_codes=("temporary_halt",),
                    terminal_event_source_ids=terminal_event.source_ids,
                ),
                admitted_evidence_available_at=admitted,
            )
        if terminal_event.event_type == "ticker_change":
            if not terminal_event.identity_continuity_verified:
                return HorizonEvaluationResult(
                    outcome=PaperEvaluationOutcome(
                        **common, status="unsupported_identity_transition",
                        reason_codes=("identity_continuity_unverified",),
                        terminal_event_source_ids=terminal_event.source_ids,
                    ),
                    admitted_evidence_available_at=admitted,
                )
        elif (
            terminal_event.event_type == "cash_acquisition"
            and terminal_event.independently_verified
            and terminal_event.identity_continuity_verified
            and terminal_event.terminal_cash_value is not None
        ):
            basis_ok, basis_reason, split_ids, basis_admitted = assess_split_basis()
            if not basis_ok:
                return unsupported_basis(
                    basis_reason, split_ids, basis_admitted,
                    terminal_sources=terminal_event.source_ids,
                )
            value = terminal_event.terminal_cash_value
            return HorizonEvaluationResult(
                outcome=PaperEvaluationOutcome(
                    **common, status="terminal_value_verified",
                    reason_codes=("verified_cash_acquisition",),
                    evaluation_value=value,
                    evaluation_value_type="terminal_cash_consideration",
                    price_return=value / decision.decision_split_adjusted_close - Decimal(1),
                    terminal_event_source_ids=terminal_event.source_ids,
                    future_price_basis="verified-terminal-cash-consideration-current-share-basis",
                    split_coverage_source_ids=split_basis_coverage.source_ids,
                ),
                admitted_evidence_available_at=tuple(sorted({*admitted, *basis_admitted})),
            )
        else:
            return HorizonEvaluationResult(
                outcome=PaperEvaluationOutcome(
                    **common, status="inactive_unresolved",
                    reason_codes=("terminal_event_evidence_missing",),
                    terminal_event_source_ids=terminal_event.source_ids,
                ),
                admitted_evidence_available_at=admitted,
            )

    if session.closes_at is None or evaluated_at < session.closes_at:
        return HorizonEvaluationResult(
            outcome=PaperEvaluationOutcome(
                **common, status="pending", reason_codes=("target_session_not_closed",)
            )
        )

    allowed_tickers = {decision.ticker}
    successor: str | None = None
    continuity_reason: tuple[OutcomeReason, ...] = ()
    if (
        event_is_known
        and terminal_event is not None
        and terminal_event.event_type == "ticker_change"
        and terminal_event.identity_continuity_verified
        and terminal_event.new_ticker is not None
    ):
        allowed_tickers.add(terminal_event.new_ticker)
        successor = terminal_event.new_ticker
        continuity_reason = ("verified_ticker_continuity",)
    candidates = [
        bar for bar in bars
        if bar.trading_date == session.trading_date
        and bar.ticker in allowed_tickers
        and bar.exchange == calendar_owner.exchange
        and bar.regular_session
        and bar.available_at <= evaluated_at
    ]
    if not candidates:
        return HorizonEvaluationResult(
            outcome=PaperEvaluationOutcome(
                **common, status="provider_gap",
                reason_codes=("provider_bar_missing",) + continuity_reason,
                terminal_event_source_ids=terminal_event.source_ids if event_is_known and terminal_event else (),
                successor_ticker=successor,
            ),
            admitted_evidence_available_at=admitted,
        )
    if len(candidates) != 1:
        raise ValueError("multiple provider bars match one frozen evaluation session")
    bar = candidates[0]
    admitted = (*admitted, bar.available_at)
    if bar.split_adjusted_close is None:
        return HorizonEvaluationResult(
            outcome=PaperEvaluationOutcome(
                **common, status="missing_price",
                reason_codes=("split_adjusted_close_missing",) + continuity_reason,
                provider_snapshot_id=bar.provider_snapshot_id,
                terminal_event_source_ids=terminal_event.source_ids if event_is_known and terminal_event else (),
                successor_ticker=successor,
            ),
            admitted_evidence_available_at=admitted,
        )
    value = bar.split_adjusted_close
    basis_ok, basis_reason, split_ids, basis_admitted = assess_split_basis()
    if not basis_ok:
        return unsupported_basis(
            basis_reason, split_ids, basis_admitted,
            snapshot=bar.provider_snapshot_id,
            successor=successor,
            terminal_sources=(
                terminal_event.source_ids if event_is_known and terminal_event else ()
            ),
            continuity=continuity_reason,
        )
    admitted = tuple(sorted({*admitted, *basis_admitted}))
    return HorizonEvaluationResult(
        outcome=PaperEvaluationOutcome(
            **common, status="evaluated",
            reason_codes=("regular_session_close",) + continuity_reason,
            evaluation_value=value, evaluation_value_type="split_adjusted_close",
            price_return=value / decision.decision_split_adjusted_close - Decimal(1),
            provider_snapshot_id=bar.provider_snapshot_id,
            terminal_event_source_ids=terminal_event.source_ids if event_is_known and terminal_event else (),
            successor_ticker=successor,
            future_price_basis=FUTURE_PRICE_BASIS,
            split_coverage_source_ids=split_basis_coverage.source_ids,
        ),
        admitted_evidence_available_at=admitted,
    )


def evaluate_horizon_outcome(
    decision: PaperDecision,
    *,
    horizon: str,
    calendar_owner: ExchangeCalendar,
    evaluation_timestamp: datetime,
    bars: tuple[EvaluationPriceBar, ...] = (),
    terminal_event: TerminalEventEvidence | None = None,
    split_basis_coverage: SplitBasisCoverage | None = None,
) -> PaperEvaluationOutcome:
    """Compatibility wrapper returning only the validated horizon outcome."""

    return evaluate_horizon_outcome_with_evidence(
        decision,
        horizon=horizon,
        calendar_owner=calendar_owner,
        evaluation_timestamp=evaluation_timestamp,
        bars=bars,
        terminal_event=terminal_event,
        split_basis_coverage=split_basis_coverage,
    ).outcome


def evaluate_rehearsal_outcomes(
    decisions: tuple[PaperDecision, ...],
    *,
    horizon: str,
    calendars: dict[str, ExchangeCalendar],
    evaluation_timestamp: datetime,
    bars_by_decision: dict[str, tuple[EvaluationPriceBar, ...]],
    events_by_decision: dict[str, TerminalEventEvidence] | None = None,
    split_coverage_by_decision: dict[str, SplitBasisCoverage] | None = None,
) -> tuple[PaperEvaluationOutcome, ...]:
    """Return one outcome per frozen decision; unresolved securities are never dropped."""

    events = events_by_decision or {}
    split_coverages = split_coverage_by_decision or {}
    aliases = {"NASDAQ": "XNAS", "NYSE": "XNYS"}
    return tuple(
        evaluate_horizon_outcome(
            decision,
            horizon=horizon,
            calendar_owner=calendars[
                aliases.get(decision.exchange.strip().upper(), decision.exchange.strip().upper())
            ],
            evaluation_timestamp=evaluation_timestamp,
            bars=bars_by_decision.get(decision.decision_id, ()),
            terminal_event=events.get(decision.decision_id),
            split_basis_coverage=split_coverages.get(decision.decision_id),
        )
        for decision in decisions
    )
