"""Strict shared models for deterministic analytics and reporting."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from identifiers import normalize_ticker
from .capabilities import CapabilityReason, SecurityCapability, active_us_stock_capability


Confidence = Literal["low", "medium", "high"]
MetricStatus = Literal["valid", "missing_data", "not_applicable", "invalid_input", "warning"]
Severity = Literal["INFO", "WARNING", "ERROR", "BLOCKING"]
PeriodKind = Literal["FY", "Q", "TTM", "FTM", "POINT_IN_TIME"]
UnitKind = Literal[
    "raw", "thousands", "millions", "billions", "percent", "decimal_ratio", "basis_points"
]
Horizon = Literal["one_day", "one_week", "one_month", "one_year"]
Importance = Literal["low", "medium", "high", "critical"]
RiskSeverity = Literal["low", "medium", "high", "critical"]
ListingType = Literal[
    "common_stock", "preferred_stock", "adr", "reit", "etf", "fund",
    "partnership", "warrant", "unit", "other",
]
CorporateActionType = Literal[
    "stock_split", "reverse_split", "cash_dividend", "stock_dividend",
    "ticker_change", "merger", "acquisition", "spin_off", "delisting", "relisting",
]
PriceSeriesKind = Literal[
    "raw_close", "split_adjusted_close", "dividend_adjusted_close",
    "provider_adjusted_close", "total_return_index",
]
ProviderAdjustmentBasis = Literal["split", "split_and_dividend", "total_return", "unknown"]
AdjustmentOrigin = Literal["provider_supplied", "internally_derived"]
DividendTreatment = Literal["cash_reinvested", "cash_not_reinvested", "provider_defined"]


class CompanyMetadataRecord(BaseModel):
    """One immutable, point-in-time provider observation of company metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    company_id: Optional[int] = Field(default=None, gt=0)
    ticker: str
    company_name: str
    cik: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    sic: Optional[str] = None
    listing_date: Optional[date] = None
    delisting_date: Optional[date] = None
    active: Optional[bool] = None
    listing_type: Optional[ListingType] = None
    prior_ticker: Optional[str] = None
    source: str
    source_record_id: str
    effective_from: AwareDatetime
    effective_to: Optional[AwareDatetime] = None
    observed_at: AwareDatetime
    available_at: AwareDatetime
    retrieved_at: AwareDatetime
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def normalize_metadata_ticker(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator("prior_ticker")
    @classmethod
    def normalize_prior_ticker(cls, value: str | None) -> str | None:
        return normalize_ticker(value) if value else None

    @field_validator("cik", mode="before")
    @classmethod
    def normalize_metadata_cik(cls, value: object) -> str | None:
        cik = str(value or "").strip()
        if not cik:
            return None
        if not cik.isdigit() or len(cik) > 10:
            raise ValueError("metadata CIK must contain at most 10 digits")
        return cik.zfill(10)

    @field_validator("exchange", mode="before")
    @classmethod
    def normalize_exchange(cls, value: object) -> str | None:
        exchange = str(value or "").strip().upper()
        return exchange or None

    @field_validator("effective_from", "effective_to", "observed_at", "available_at", "retrieved_at", mode="after")
    @classmethod
    def normalize_metadata_time(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(timezone.utc) if value is not None else None

    @model_validator(mode="after")
    def metadata_time_order(self) -> "CompanyMetadataRecord":
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("metadata effective_to must follow effective_from")
        if self.available_at < self.observed_at:
            raise ValueError("metadata cannot be available before it was observed")
        if self.retrieved_at < self.available_at:
            raise ValueError("metadata cannot be retrieved before it was available")
        if self.listing_date and self.delisting_date and self.delisting_date < self.listing_date:
            raise ValueError("delisting date cannot precede listing date")
        return self


class PriceObservation(BaseModel):
    """One daily price observation with non-interchangeable adjustment semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    company_id: Optional[int] = Field(default=None, gt=0)
    ticker: str
    trading_date: date
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Decimal = Field(gt=0)
    split_adjusted_close: Optional[Decimal] = Field(default=None, gt=0)
    dividend_adjusted_close: Optional[Decimal] = Field(default=None, gt=0)
    provider_adjusted_close: Optional[Decimal] = Field(default=None, gt=0)
    provider_adjustment_basis: Optional[ProviderAdjustmentBasis] = None
    total_return_index: Optional[Decimal] = Field(default=None, gt=0)
    adjustment_origin: Optional[AdjustmentOrigin] = None
    split_event_ids: list[str] = Field(default_factory=list)
    dividend_treatment: Optional[DividendTreatment] = None
    volume: Optional[Decimal] = Field(default=None, ge=0)
    currency: str
    exchange: Optional[str] = None
    source: str
    source_record_id: str
    observed_at: AwareDatetime
    available_at: AwareDatetime
    retrieved_at: AwareDatetime
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def normalize_price_ticker(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_price_currency(cls, value: object) -> str:
        code = str(value).strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", code):
            raise ValueError("price currency must be a three-letter code")
        return code

    @field_validator("exchange", mode="before")
    @classmethod
    def normalize_price_exchange(cls, value: object) -> str | None:
        exchange = str(value or "").strip().upper()
        return exchange or None

    @field_validator("observed_at", "available_at", "retrieved_at", mode="after")
    @classmethod
    def normalize_price_time(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def price_contract(self) -> "PriceObservation":
        if self.available_at < self.observed_at or self.retrieved_at < self.available_at:
            raise ValueError("price observation/availability/retrieval times are out of order")
        if self.observed_at.date() < self.trading_date:
            raise ValueError("price cannot be observed before its trading date")
        if self.high is not None and self.high < max(value for value in (self.open, self.close, self.low) if value is not None):
            raise ValueError("price high is below another OHLC value")
        if self.low is not None and self.low > min(value for value in (self.open, self.close, self.high) if value is not None):
            raise ValueError("price low is above another OHLC value")
        if (self.provider_adjusted_close is None) != (self.provider_adjustment_basis is None):
            raise ValueError("provider adjusted close requires an explicit adjustment basis")
        adjusted_values = (
            self.split_adjusted_close, self.dividend_adjusted_close,
            self.provider_adjusted_close, self.total_return_index,
        )
        if any(value is not None for value in adjusted_values) and self.adjustment_origin is None:
            raise ValueError("adjusted price data requires an explicit adjustment origin")
        if (
            self.adjustment_origin == "internally_derived"
            and self.split_adjusted_close is not None
            and not self.split_event_ids
        ):
            raise ValueError("internally derived split-adjusted prices require split event IDs")
        if self.total_return_index is not None and self.dividend_treatment is None:
            raise ValueError("total-return series requires explicit dividend treatment")
        if self.dividend_treatment is not None and self.total_return_index is None:
            raise ValueError("dividend treatment applies only to a total-return series")
        return self

    def value_for(self, series: PriceSeriesKind) -> Decimal:
        values = {
            "raw_close": self.close,
            "split_adjusted_close": self.split_adjusted_close,
            "dividend_adjusted_close": self.dividend_adjusted_close,
            "provider_adjusted_close": self.provider_adjusted_close,
            "total_return_index": self.total_return_index,
        }
        value = values[series]
        if value is None:
            raise ValueError(f"requested {series} is unavailable; adjustment methods cannot be mixed")
        return value


class CorporateActionRecord(BaseModel):
    """Immutable provider observation of a corporate action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    company_id: Optional[int] = Field(default=None, gt=0)
    ticker: str
    event_type: CorporateActionType
    announcement_date: Optional[date] = None
    declaration_date: Optional[date] = None
    record_date: Optional[date] = None
    ex_date: Optional[date] = None
    payment_date: Optional[date] = None
    effective_date: Optional[date] = None
    ratio: Optional[Decimal] = Field(default=None, gt=0)
    amount: Optional[Decimal] = Field(default=None, ge=0)
    currency: Optional[str] = None
    old_ticker: Optional[str] = None
    new_ticker: Optional[str] = None
    source: str
    source_record_id: str
    observed_at: AwareDatetime
    available_at: AwareDatetime
    retrieved_at: AwareDatetime
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker", "old_ticker", "new_ticker")
    @classmethod
    def normalize_action_ticker(cls, value: str | None) -> str | None:
        return normalize_ticker(value) if value else None

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_action_currency(cls, value: object) -> str | None:
        code = str(value or "").strip().upper()
        if code and not re.fullmatch(r"[A-Z]{3}", code):
            raise ValueError("action currency must be a three-letter code")
        return code or None

    @field_validator("observed_at", "available_at", "retrieved_at", mode="after")
    @classmethod
    def normalize_action_time(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def action_contract(self) -> "CorporateActionRecord":
        if self.available_at < self.observed_at or self.retrieved_at < self.available_at:
            raise ValueError("action observation/availability/retrieval times are out of order")
        if self.event_type in {"stock_split", "reverse_split", "stock_dividend"} and self.ratio is None:
            raise ValueError("split and stock-dividend actions require a ratio")
        if self.event_type == "cash_dividend" and (self.amount is None or self.currency is None):
            raise ValueError("cash dividends require amount and currency")
        if self.event_type == "ticker_change" and (not self.old_ticker or not self.new_ticker):
            raise ValueError("ticker changes require old_ticker and new_ticker")
        return self


class MetricResult(BaseModel):
    """Safe calculation result used instead of an unqualified float."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: Optional[Decimal] = None
    unit: str
    status: MetricStatus
    confidence: Confidence = "high"
    input_fields: list[str] = Field(default_factory=list)
    warning: Optional[str] = None
    applicable: bool = True
    calculation_version: str = "financial-math-v1"

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("metric values must be finite")
        return value


class FiscalPeriod(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PeriodKind
    label: str
    end_date: date
    start_date: Optional[date] = None
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = Field(default=None, ge=1, le=4)

    @model_validator(mode="after")
    def validate_kind_fields(self) -> "FiscalPeriod":
        if self.kind == "Q" and self.fiscal_quarter is None:
            raise ValueError("quarterly periods require fiscal_quarter")
        if self.kind in {"FY", "Q"} and self.fiscal_year is None:
            raise ValueError("fiscal periods require fiscal_year")
        if self.kind != "Q" and self.fiscal_quarter is not None:
            raise ValueError("fiscal_quarter is only valid for Q periods")
        return self


class DataPoint(BaseModel):
    """One sourced value before normalization into base units or decimal ratios."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[Decimal]
    unit: UnitKind = "raw"
    currency: Optional[str] = None
    period: FiscalPeriod
    as_of: date
    source: str = "fixture"

    @field_validator("value")
    @classmethod
    def finite_input(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("input values must be finite")
        return value

    @field_validator("currency", mode="before")
    @classmethod
    def currency_code(cls, value: object) -> Optional[str]:
        code = str(value or "").strip().upper()
        if code and not re.fullmatch(r"[A-Z]{3}", code):
            raise ValueError("currency must be a three-letter ISO-style code")
        return code or None


class Catalyst(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    company: str
    name: str
    catalyst_type: Literal[
        "earnings", "product_launch", "regulatory_decision", "court_ruling",
        "contract_decision", "merger_vote", "investor_day", "guidance_event",
        "financing_event", "other",
    ]
    expected_date: Optional[date] = None
    horizon: Horizon
    importance: Importance
    confidence: Confidence
    status: Literal["upcoming", "occurred", "delayed", "cancelled", "uncertain"]
    source: str
    source_date: date

    @field_validator("ticker")
    @classmethod
    def valid_ticker(cls, value: str) -> str:
        return normalize_ticker(value)


class Risk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "valuation", "execution", "financial", "regulatory", "litigation",
        "customer_concentration", "supplier_concentration", "competition",
        "dilution", "debt", "macro_sensitivity", "technological_disruption", "other",
    ]
    severity: RiskSeverity
    confidence: Confidence
    evidence: str
    freshness: date


class QualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Severity
    message: str
    fields: list[str] = Field(default_factory=list)


class CompanyData(BaseModel):
    """Normalized-input boundary for one fictional or externally sourced company."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    company: str
    sector: str
    industry: str
    as_of: date
    cik: Optional[str] = None
    exchange: Optional[str] = None
    sic: Optional[str] = None
    metadata_available_at: Optional[AwareDatetime] = None
    metadata_provenance: list[str] = Field(default_factory=list)
    financials: dict[str, DataPoint]
    catalysts: list[Catalyst] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    news_score: Optional[Decimal] = Field(default=None, ge=0, le=100)
    source_quality: Confidence = "medium"
    capability: SecurityCapability = Field(default_factory=active_us_stock_capability)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator("cik", mode="before")
    @classmethod
    def validate_cik(cls, value: object) -> Optional[str]:
        cik = str(value or "").strip()
        if not cik:
            return None
        if not cik.isdigit() or len(cik) > 10:
            raise ValueError("malformed CIK")
        return cik.zfill(10)

    @field_validator("metadata_available_at", mode="after")
    @classmethod
    def normalize_company_metadata_time(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(timezone.utc) if value is not None else None


class PeerComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    metric: str
    value: Optional[Decimal]
    peer_median: Optional[Decimal]
    percentile: Optional[Decimal]
    z_score: Optional[Decimal]
    relative_spread: Optional[Decimal]
    direction: Literal["higher", "lower", "context"]
    status: MetricStatus
    warning: Optional[str] = None
    peer_scope: Optional[Literal["watchlist", "sector", "industry"]] = None
    peer_group: Optional[str] = None
    peer_provenance: list[str] = Field(default_factory=list)


class ComponentScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valuation: Optional[Decimal] = Field(default=None, ge=0, le=100)
    quality: Optional[Decimal] = Field(default=None, ge=0, le=100)
    growth: Optional[Decimal] = Field(default=None, ge=0, le=100)
    financial_strength: Optional[Decimal] = Field(default=None, ge=0, le=100)
    catalysts: Optional[Decimal] = Field(default=None, ge=0, le=100)
    news: Optional[Decimal] = Field(default=None, ge=0, le=100)
    risk: Optional[Decimal] = Field(default=None, ge=0, le=100)
    data_quality: Decimal = Field(ge=0, le=100)


class RankingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    company: str
    horizon: Horizon
    components: ComponentScores
    opportunity_score: Optional[Decimal] = Field(default=None, ge=0, le=100)
    ranking_status: Literal["ranked", "insufficient_evidence", "blocked"] = "ranked"
    research_priority_score: Decimal = Field(ge=0, le=100)
    research_priority: Literal["low", "medium", "high", "critical"]
    confidence: Confidence
    positive_contributors: list[str]
    negative_contributors: list[str]
    missing_inputs: list[str]
    unavailable_inputs: dict[str, str] = Field(default_factory=dict)
    capability_reason_codes: list[CapabilityReason] = Field(default_factory=list)
    warnings: list[str]
    ranking_version: str = "ranking-v1"


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp: AwareDatetime
    input_cutoff: AwareDatetime
    pipeline_version: str
    ranking_version: str
    calculation_version: str
    input_hash: str
    config_hash: str
    tickers: list[str]
    warnings: list[str]
    errors: list[str]
    candidates: int
    files_proposed: list[str]
    files_written: int = 0

    @field_validator("timestamp", "input_cutoff", mode="after")
    @classmethod
    def timezone_required(cls, value: object) -> object:
        return value.astimezone(timezone.utc)


class BacktestRecord(BaseModel):
    """Future point-in-time record; return fields remain empty until real data exists."""

    model_config = ConfigDict(extra="forbid")

    ranking_cutoff_at: AwareDatetime
    ticker: str
    horizon: Literal["1d", "1w", "1m", "3m", "6m", "1y"]
    opportunity_score: Decimal = Field(ge=0, le=100)
    research_priority: Importance
    data_snapshot_id: str
    inputs_available_at_time: AwareDatetime
    future_return: Optional[Decimal] = None
    benchmark_return: Optional[Decimal] = None
    sector_return: Optional[Decimal] = None

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_date_only_cutoff(cls, value: object) -> object:
        if isinstance(value, dict) and "ranking_date" in value:
            raise ValueError(
                "legacy date-only ranking_date cannot support an exact point-in-time claim; "
                "supply an explicit timezone-aware ranking_cutoff_at"
            )
        return value

    @field_validator("ranking_cutoff_at", "inputs_available_at_time", mode="after")
    @classmethod
    def normalize_cutoff_times(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def prevent_lookahead_metadata(self) -> "BacktestRecord":
        if self.inputs_available_at_time > self.ranking_cutoff_at:
            raise ValueError("inputs_available_at_time cannot be after ranking_cutoff_at")
        return self


class TimeSeriesPoint(BaseModel):
    """Database-ready point-in-time value with availability metadata."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    metric: str
    period: FiscalPeriod
    value: Decimal
    unit: UnitKind
    currency: Optional[str] = None
    source: str
    available_at: AwareDatetime

    @field_validator("available_at", mode="after")
    @classmethod
    def normalize_available_at(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def availability_cannot_precede_period(self) -> "TimeSeriesPoint":
        if self.available_at.date() < self.period.end_date:
            raise ValueError("available_at cannot precede the measurement period end")
        return self


class FreshnessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    price_age: Optional[int] = None
    filing_age: Optional[int] = None
    fundamentals_age: Optional[int] = None
    estimate_age: Optional[int] = None
    news_age: Optional[int] = None
    thesis_age: Optional[int] = None
