"""Vendor-neutral market-data protocol and deterministic fixture implementation."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from .dcf import DCFScenario
from .models import (
    Catalyst, CompanyData, CompanyMetadataRecord, CorporateActionRecord,
    DataPoint, FiscalPeriod, PriceObservation, Risk,
)


MONETARY_FIELDS = {
    "market_cap", "enterprise_value", "revenue", "prior_revenue", "revenue_start",
    "gross_profit", "operating_income", "net_income", "free_cash_flow",
    "prior_free_cash_flow", "fcf_start", "ebitda", "equity", "average_equity",
    "nopat", "invested_capital", "total_debt", "cash", "ebit", "interest_expense",
    "current_assets", "current_liabilities", "receivables", "working_capital",
    "total_assets", "retained_earnings", "total_liabilities", "operating_cash_flow",
    "net_working_capital", "net_fixed_assets",
}
POINT_IN_TIME_FIELDS = {
    "price_per_share", "market_cap", "enterprise_value", "equity",
    "invested_capital", "total_debt", "cash", "current_assets", "current_liabilities",
    "receivables", "shares_outstanding", "prior_shares_outstanding", "total_assets",
    "total_liabilities", "retained_earnings", "book_value_per_share",
    "working_capital", "net_working_capital", "net_fixed_assets",
}
RATIO_FIELDS = {
    "forward_eps_growth", "roa", "prior_roa", "debt_to_assets", "prior_debt_to_assets",
    "prior_current_ratio", "prior_gross_margin", "asset_turnover", "prior_asset_turnover",
    "dsri", "gmi", "aqi", "sgi", "depi", "sgai", "tata", "lvgi",
}


class MarketDataProvider(Protocol):
    def get_price(self, ticker: str, as_of: date) -> DataPoint | None: ...
    def get_historical_prices(self, ticker: str, start: date, end: date) -> list[DataPoint]: ...
    def get_market_cap(self, ticker: str, as_of: date) -> DataPoint | None: ...
    def get_shares_outstanding(self, ticker: str, as_of: date) -> DataPoint | None: ...
    def get_fundamentals(self, ticker: str, as_of: date) -> dict[str, DataPoint]: ...
    def get_estimates(self, ticker: str, as_of: date) -> dict[str, DataPoint]: ...
    def get_company_metadata(self, ticker: str) -> dict[str, str]: ...


class CompanyMetadataProvider(Protocol):
    def metadata(self, ticker: str, input_cutoff: datetime) -> list[CompanyMetadataRecord]: ...


class HistoricalPriceProvider(Protocol):
    def prices(
        self, ticker: str, start: date, end: date, input_cutoff: datetime
    ) -> list[PriceObservation]: ...


class CorporateActionProvider(Protocol):
    def corporate_actions(
        self, ticker: str, start: date, end: date, input_cutoff: datetime
    ) -> list[CorporateActionRecord]: ...


class SyntheticProviderFixture:
    """Offline-only vendor-neutral provider fixture; all records are fictional."""

    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        notice = str(raw.get("fixture_notice", ""))
        if "synthetic" not in notice.casefold() and "fictional" not in notice.casefold():
            raise ValueError("provider fixture must be visibly labeled synthetic or fictional")
        self.fixture_notice = notice
        self._metadata = [CompanyMetadataRecord.model_validate(item) for item in raw.get("metadata", [])]
        self._prices = [PriceObservation.model_validate(item) for item in raw.get("prices", [])]
        self._actions = [CorporateActionRecord.model_validate(item) for item in raw.get("corporate_actions", [])]

    def metadata(self, ticker: str, input_cutoff: datetime) -> list[CompanyMetadataRecord]:
        if input_cutoff.tzinfo is None or input_cutoff.utcoffset() is None:
            raise ValueError("provider input cutoff must be timezone-aware")
        cutoff = input_cutoff.astimezone(timezone.utc)
        symbol = ticker.upper()
        return [
            item for item in self._metadata
            if (item.ticker == symbol or item.prior_ticker == symbol)
            and item.available_at <= cutoff
            and item.effective_from <= cutoff
            and (item.effective_to is None or cutoff < item.effective_to)
        ]

    def prices(self, ticker: str, start: date, end: date, input_cutoff: datetime) -> list[PriceObservation]:
        if input_cutoff.tzinfo is None or input_cutoff.utcoffset() is None:
            raise ValueError("provider input cutoff must be timezone-aware")
        cutoff = input_cutoff.astimezone(timezone.utc)
        symbol = ticker.upper()
        return sorted(
            [item for item in self._prices if item.ticker == symbol and start <= item.trading_date <= end and item.available_at <= cutoff],
            key=lambda item: (item.trading_date, item.available_at, item.source_record_id),
        )

    def corporate_actions(self, ticker: str, start: date, end: date, input_cutoff: datetime) -> list[CorporateActionRecord]:
        if input_cutoff.tzinfo is None or input_cutoff.utcoffset() is None:
            raise ValueError("provider input cutoff must be timezone-aware")
        cutoff = input_cutoff.astimezone(timezone.utc)
        symbol = ticker.upper()
        return sorted(
            [
                item for item in self._actions
                if (item.ticker == symbol or item.old_ticker == symbol or item.new_ticker == symbol)
                and item.available_at <= cutoff
                and start <= (
                    item.effective_date or item.ex_date or item.record_date
                    or item.declaration_date or item.announcement_date
                    or item.payment_date or start
                ) <= end
            ],
            key=lambda item: (
                item.effective_date or item.ex_date or item.record_date
                or item.declaration_date or item.announcement_date
                or item.payment_date or start,
                item.source_record_id,
            ),
        )


class FixtureProvider:
    """Read clearly fictional compact JSON and expose the future provider contract."""

    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("companies"), list):
            raise ValueError("analytics fixture must contain a companies list")
        self.analysis_date = date.fromisoformat(raw["analysis_date"])
        self._raw = raw
        self.fixture_notice = str(raw.get("fixture_notice", ""))
        self.defaults = dict(raw.get("defaults", {}))
        self._companies: dict[str, CompanyData] = {}
        self._dcf: dict[str, list[DCFScenario]] = {}
        for item in raw["companies"]:
            company = self._parse_company(item, self.defaults)
            if company.ticker in self._companies:
                raise ValueError(f"duplicate fixture ticker: {company.ticker}")
            self._companies[company.ticker] = company
            self._dcf[company.ticker] = [DCFScenario.model_validate(value) for value in item.get("dcf_scenarios", [])]

    @staticmethod
    def _parse_company(item: dict[str, object], defaults: dict[str, object]) -> CompanyData:
        as_of = date.fromisoformat(str(item["as_of"]))
        financials_as_of = date.fromisoformat(str(item.get("financials_as_of", as_of.isoformat())))
        financials: dict[str, DataPoint] = {}
        for name, raw_point in dict(item.get("financials", {})).items():
            point = dict(raw_point) if isinstance(raw_point, dict) else {"value": raw_point}
            if "unit" not in point:
                if name in MONETARY_FIELDS:
                    point["unit"] = defaults.get("monetary_unit", "millions")
                    point["currency"] = defaults.get("currency", "USD")
                elif name in {"price_per_share", "eps", "prior_eps", "eps_start", "forward_eps", "book_value_per_share"}:
                    point["unit"] = "raw"
                    point["currency"] = defaults.get("currency", "USD")
                elif name in {"shares_outstanding", "prior_shares_outstanding"}:
                    point["unit"] = defaults.get("shares_unit", "millions")
                elif name in RATIO_FIELDS:
                    point["unit"] = "decimal_ratio"
                else:
                    point["unit"] = "raw"
            default_kind = "POINT_IN_TIME" if name in POINT_IN_TIME_FIELDS else "FTM" if name.startswith("forward_") else "TTM"
            period_kind = str(point.pop("period_kind", default_kind))
            if name.startswith("prior_"):
                inferred_period_end = as_of.replace(year=as_of.year - 1)
            elif name.endswith("_start"):
                inferred_period_end = as_of.replace(year=as_of.year - 3)
            else:
                inferred_period_end = as_of
            period_end = date.fromisoformat(str(point.pop("period_end", inferred_period_end.isoformat())))
            period_start = point.pop("period_start", None)
            # Only the explicitly declared synthetic-fixture duration supplies
            # omitted starts. Real/provider observations get no inferred dates.
            if period_start is None and period_kind == "TTM" and defaults.get("duration_years") == 1:
                period_start = (period_end.replace(year=period_end.year - 1) + timedelta(days=1)).isoformat()
            if period_start is None and period_kind == "FTM" and defaults.get("duration_years") == 1:
                period_start = (period_end.replace(year=period_end.year - 1) + timedelta(days=1)).isoformat()
            fiscal_year = point.pop("fiscal_year", period_end.year if period_kind in {"FY", "Q"} else None)
            fiscal_quarter = point.pop("fiscal_quarter", None)
            label = str(point.pop("period_label", f"{period_kind}-{period_end.isoformat()}"))
            point_as_of = date.fromisoformat(str(point.pop("as_of", financials_as_of.isoformat())))
            financials[name] = DataPoint(
                **point,
                period=FiscalPeriod(
                    kind=period_kind, label=label, end_date=period_end,
                    start_date=period_start,
                    fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
                ),
                as_of=point_as_of,
                source="fictional-fixture",
            )
        return CompanyData(
            ticker=item["ticker"], company=item["company"], sector=item["sector"],
            industry=item["industry"], as_of=as_of, cik=item.get("cik"),
            financials=financials,
            catalysts=[Catalyst.model_validate(value) for value in item.get("catalysts", [])],
            risks=[Risk.model_validate(value) for value in item.get("risks", [])],
            news_score=(
                Decimal(str(item["news_score"]))
                if item.get("news_score") is not None else None
            ),
            source_quality=item.get("source_quality", "high"),
        )

    def list_companies(self) -> list[CompanyData]:
        return [self._companies[ticker] for ticker in sorted(self._companies)]

    def company(self, ticker: str) -> CompanyData:
        try:
            return self._companies[ticker.upper()]
        except KeyError as exc:
            raise ValueError(f"ticker not present in fixture: {ticker.upper()}") from exc

    def dcf_scenarios(self, ticker: str) -> list[DCFScenario]:
        self.company(ticker)
        return self._dcf[ticker.upper()]

    def provenance_payload(self) -> dict[str, object]:
        return self._raw

    def get_price(self, ticker: str, as_of: date) -> DataPoint | None:
        return self.company(ticker).financials.get("price_per_share")

    def get_historical_prices(self, ticker: str, start: date, end: date) -> list[DataPoint]:
        self.company(ticker)
        return []

    def get_market_cap(self, ticker: str, as_of: date) -> DataPoint | None:
        return self.company(ticker).financials.get("market_cap")

    def get_shares_outstanding(self, ticker: str, as_of: date) -> DataPoint | None:
        return self.company(ticker).financials.get("shares_outstanding")

    def get_fundamentals(self, ticker: str, as_of: date) -> dict[str, DataPoint]:
        return dict(self.company(ticker).financials)

    def get_estimates(self, ticker: str, as_of: date) -> dict[str, DataPoint]:
        return {name: value for name, value in self.company(ticker).financials.items() if name.startswith("forward_")}

    def get_company_metadata(self, ticker: str) -> dict[str, str]:
        company = self.company(ticker)
        return {"ticker": company.ticker, "company": company.company, "sector": company.sector, "industry": company.industry}
