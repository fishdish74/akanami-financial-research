"""Bridge point-in-time SQLite records into the existing canonical analytics models."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timezone, timedelta
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

from .database import MarketDatabase
from .dcf import DCFScenario
from .models import (
    CompanyData, CompanyMetadataRecord, CorporateActionRecord, DataPoint,
    FiscalPeriod, PriceObservation,
)


FLOW_METRICS = {
    "revenue", "gross_profit", "operating_income", "net_income",
    "operating_cash_flow", "free_cash_flow", "interest_expense", "eps",
}
POINT_METRICS = {
    "total_assets", "total_liabilities", "current_assets", "current_liabilities",
    "cash", "equity", "retained_earnings", "receivables", "shares_outstanding",
    "total_debt",
}


class DatabaseProvider:
    """Read only facts available by a requested date; SQL stays in the repository."""

    def __init__(self, database: MarketDatabase, as_of: date | datetime) -> None:
        if database.schema_version() == 0:
            raise ValueError("database schema has not been initialized")
        self.database = database
        if isinstance(as_of, datetime):
            if as_of.tzinfo is None or as_of.utcoffset() is None:
                raise ValueError("database input cutoff must be timezone-aware")
            self.input_cutoff = as_of.astimezone(timezone.utc)
        else:
            self.input_cutoff = datetime.combine(as_of, time.max, tzinfo=timezone.utc)
        self.analysis_date = self.input_cutoff.date()
        self.fixture_notice = ""

    @property
    def _boundary(self) -> str:
        return self.input_cutoff.isoformat()

    @contextmanager
    def read_snapshot(self):
        with self.database.read_snapshot():
            yield

    @staticmethod
    def _period(row: Any) -> FiscalPeriod:
        kind = row["period_type"]
        if kind not in {"FY", "Q", "POINT_IN_TIME"}:
            kind = "POINT_IN_TIME"
        return FiscalPeriod(
            kind=kind,
            label=f"{kind}-{row['end_date']}",
            end_date=date.fromisoformat(row["end_date"]),
            start_date=date.fromisoformat(row["start_date"]) if row["start_date"] else None,
            fiscal_year=row["fiscal_year"] if kind in {"FY", "Q"} else None,
            fiscal_quarter=row["fiscal_quarter"] if kind == "Q" else None,
        )

    @classmethod
    def _point(
        cls,
        row: Any,
        *,
        value: Decimal | None = None,
        source: str | None = None,
        fiscal_year: int | None = None,
    ) -> DataPoint:
        period = cls._period(row)
        if fiscal_year is not None and period.kind == "FY":
            period = period.model_copy(update={"fiscal_year": fiscal_year})
        return DataPoint(
            value=value if value is not None else Decimal(row["normalized_value"]),
            unit="raw",
            currency=row["currency"],
            period=period,
            as_of=date.fromisoformat(row["available_at"][:10]),
            source=source or f"SEC:{row['accession_number']}:{row['source_concept']}",
        )

    @staticmethod
    def _preferred(rows: list[Any], metric: str) -> list[Any]:
        eligible = [
            row for row in rows
            if row["canonical_metric"] == metric and row["quality_status"] == "valid"
        ]
        desired = "FY" if metric in FLOW_METRICS else "POINT_IN_TIME"
        matching = [row for row in eligible if row["period_type"] == desired]
        return sorted(
            matching,
            key=lambda row: (row["end_date"], row["available_at"], -row["mapping_priority"]),
            reverse=True,
        )

    @staticmethod
    def _annual_sequence(rows: list[Any]) -> list[Any]:
        """Return contiguous annual economic periods, ignoring SEC filing-year labels.

        SEC Company Facts repeats comparative periods in later filings and assigns the
        filing's ``fy`` to those rows. Actual duration and period-end spacing are the
        stable economic identity.
        """

        annual_rows = []
        for row in rows:
            if not row["start_date"]:
                continue
            start = date.fromisoformat(row["start_date"])
            end = date.fromisoformat(row["end_date"])
            if 300 <= (end - start).days <= 400:
                annual_rows.append(row)
        if not annual_rows:
            return []
        sequence = [annual_rows[0]]
        for row in annual_rows[1:]:
            previous_end = date.fromisoformat(sequence[-1]["end_date"])
            candidate_end = date.fromisoformat(row["end_date"])
            gap = (previous_end - candidate_end).days
            if gap == 0:
                continue
            if 300 <= gap <= 430:
                sequence.append(row)
                continue
            if gap > 430:
                break
        return sequence

    def _financials(self, company_id: int) -> tuple[dict[str, DataPoint], bool]:
        rows = self.database.get_metrics_as_of_company_id(company_id, self._boundary)
        has_conflict = any(row["quality_status"] == "conflict" for row in rows)
        financials: dict[str, DataPoint] = {}
        for metric in sorted(FLOW_METRICS | POINT_METRICS):
            values = self._preferred(rows, metric)
            if not values:
                continue
            annual_values = self._annual_sequence(values) if metric in FLOW_METRICS else []
            current_year = values[0]["fiscal_year"]
            financials[metric] = self._point(values[0], fiscal_year=current_year)
            if metric in {"revenue", "free_cash_flow", "eps"}:
                economic_year = (
                    int(current_year)
                    if current_year is not None
                    else date.fromisoformat(values[0]["end_date"]).year
                )
                if len(annual_values) >= 2:
                    financials[f"prior_{metric}"] = self._point(
                        annual_values[1], fiscal_year=economic_year - 1
                    )
                if len(annual_values) >= 4:
                    start = annual_values[3]
                    prefix = {"revenue": "revenue", "free_cash_flow": "fcf", "eps": "eps"}[metric]
                    financials[f"{prefix}_start"] = self._point(
                        start, fiscal_year=economic_year - 3
                    )
                    financials[f"{prefix}_cagr_years"] = DataPoint(
                        value=Decimal(3), unit="raw", currency=None,
                        period=self._point(values[0], fiscal_year=economic_year).period,
                        as_of=date.fromisoformat(values[0]["available_at"][:10]),
                        source="database-derived metric-specific period span",
                    )
        equity_values = self._preferred(rows, "equity")
        income_values = self._preferred(rows, "net_income")
        if income_values:
            income = income_values[0]
            income_end = date.fromisoformat(income["end_date"])
            closing = next(
                (row for row in equity_values if row["end_date"] == income["end_date"]),
                None,
            )
            opening_candidates = [
                row for row in equity_values
                if income["start_date"] and
                date.fromisoformat(row["end_date"]) + timedelta(days=1)
                == date.fromisoformat(income["start_date"])
            ]
            opening = min(
                opening_candidates,
                key=lambda row: abs(
                    (income_end - date.fromisoformat(row["end_date"])).days - 365
                ),
                default=None,
            )
        else:
            income = closing = opening = None
        if closing is not None and opening is not None:
            average = (
                Decimal(closing["normalized_value"])
                + Decimal(opening["normalized_value"])
            ) / Decimal(2)
            income_period = self._period(income)
            required_inputs = (
                ("opening_equity", opening),
                ("closing_equity", closing),
                ("net_income", income),
            )
            financials["average_equity"] = DataPoint(
                value=average,
                unit="raw",
                currency=closing["currency"] if closing["currency"] == opening["currency"] else None,
                period=income_period,
                as_of=max(
                    date.fromisoformat(row["available_at"][:10])
                    for _, row in required_inputs
                ),
                source=(
                    "database-derived annual opening/closing average equity:"
                    f"{opening['end_date']}:{closing['end_date']};"
                    + ";".join(
                        f"{role}=SEC:{row['accession_number']}:{row['source_concept']}"
                        f"@{row['available_at']}"
                        for role, row in required_inputs
                    )
                ),
            )
        return financials, has_conflict

    def _base_company(self, ticker: str):
        metadata = self.database.metadata_as_of(ticker, self._boundary)
        company_ids = {int(item["company_id"]) for item in metadata}
        if len(company_ids) > 1:
            raise ValueError(
                f"ticker resolves to multiple companies at cutoff: {ticker.upper()}"
            )
        if company_ids:
            return self.database.get_company_by_id(next(iter(company_ids)))
        return None

    def _metadata(self, company_id: int) -> dict[str, Any]:
        rows = self.database.metadata_as_of("", self._boundary, company_id=company_id)
        if not rows:
            return {}
        first = rows[0]
        cohort = [
            row for row in rows
            if (row["available_at"], row["observed_at"], row["effective_from"])
            == (first["available_at"], first["observed_at"], first["effective_from"])
        ]

        def resolved(field: str) -> Any:
            values = {row[field] for row in cohort if row[field] not in (None, "")}
            return next(iter(values)) if len(values) == 1 else None

        provenance = [
            f"{row['source']}:{row['source_record_id']}:{row['metadata_identity']}"
            for row in cohort
        ]
        conflicts = [
            field for field in ("ticker", "company_name", "exchange", "sector", "industry", "sic")
            if len({row[field] for row in cohort if row[field] not in (None, "")}) > 1
        ]
        if conflicts:
            provenance.append("conflicting_fields:" + ",".join(sorted(conflicts)))
        return {
            field: resolved(field)
            for field in (
                "ticker", "company_name", "cik", "exchange", "sector", "industry",
                "sic", "listing_date", "delisting_date", "is_active", "listing_type",
                "prior_ticker",
            )
        } | {"available_at": first["available_at"], "provenance": provenance}

    def list_companies(self) -> list[CompanyData]:
        return [
            self._company_by_id(company_id)
            for company_id in self.database.metadata_company_ids_as_of(self._boundary)
        ]

    def list_active_companies(self) -> list[CompanyData]:
        output: list[CompanyData] = []
        for company_id in self.database.metadata_company_ids_as_of(self._boundary):
            metadata = self._metadata(company_id)
            if metadata.get("is_active") == 0:
                continue
            if metadata.get("delisting_date"):
                delisted = date.fromisoformat(metadata["delisting_date"])
                if delisted <= self.analysis_date:
                    continue
            output.append(self._company_by_id(company_id))
        return output

    def company(self, ticker: str) -> CompanyData:
        row = self._base_company(ticker)
        if row is None:
            raise ValueError(
                f"ticker identity is unavailable at cutoff: {ticker.upper()}"
            )
        return self._company_by_id(int(row["id"]))

    def _company_by_id(self, company_id: int) -> CompanyData:
        row = self.database.get_company_by_id(company_id)
        if row is None:
            raise ValueError("metadata references an unknown company")
        metadata = self._metadata(int(row["id"]))
        if not metadata or not metadata.get("ticker") or not metadata.get("company_name"):
            raise ValueError("company identity is unavailable at cutoff")
        effective_ticker = metadata["ticker"]
        financials, has_conflict = self._financials(int(row["id"]))
        price_rows = self.database.prices_as_of(
            int(row["id"]), date.min, self.analysis_date, self._boundary
        )
        if price_rows:
            latest = price_rows[-1]
            financials["price_per_share"] = DataPoint(
                value=Decimal(latest["close"]), unit="raw", currency=latest["currency"],
                period=FiscalPeriod(
                    kind="POINT_IN_TIME", label=f"PRICE-{latest['price_date']}",
                    end_date=date.fromisoformat(latest["price_date"]),
                ),
                as_of=date.fromisoformat(latest["available_at"][:10]),
                source=f"{latest['source']}:{latest['source_identifier']}:raw_close",
            )
        return CompanyData(
            ticker=effective_ticker,
            company=metadata["company_name"],
            sector=metadata.get("sector") or "Unknown",
            industry=metadata.get("industry") or "Unknown",
            as_of=self.analysis_date,
            cik=metadata.get("cik"),
            exchange=metadata.get("exchange"),
            sic=metadata.get("sic"),
            metadata_available_at=(
                datetime.fromisoformat(metadata["available_at"])
                if metadata.get("available_at") else None
            ),
            metadata_provenance=metadata.get("provenance", []),
            financials=financials,
            catalysts=[],
            risks=[],
            news_score=None,
            # SEC fundamentals are authoritative, but this provider intentionally
            # lacks price, estimates, and structured news/catalyst coverage.
            source_quality="low" if has_conflict else "medium",
        )

    def get_fundamentals(self, ticker: str, as_of: date) -> dict[str, DataPoint]:
        if as_of != self.analysis_date:
            return DatabaseProvider(self.database, as_of).company(ticker).financials
        return self.company(ticker).financials

    def get_price(self, ticker: str, as_of: date) -> DataPoint | None:
        observations = self.get_historical_prices(ticker, date.min, as_of)
        return observations[-1] if observations else None

    def get_historical_prices(self, ticker: str, start: date, end: date) -> list[DataPoint]:
        return [
            DataPoint(
                value=item.close, unit="raw", currency=item.currency,
                period=FiscalPeriod(
                    kind="POINT_IN_TIME", label=f"PRICE-{item.trading_date}",
                    end_date=item.trading_date,
                ),
                as_of=item.available_at.date(),
                source=f"{item.source}:{item.source_record_id}:raw_close",
            )
            for item in self.get_price_observations(ticker, start, end)
        ]

    def get_price_observations(
        self, ticker: str, start: date, end: date
    ) -> list[PriceObservation]:
        row = self._base_company(ticker)
        if row is None:
            raise ValueError(f"ticker not present in database: {ticker.upper()}")
        return [
            PriceObservation(
                company_id=int(item["company_id"]), ticker=item["ticker_at_observation"],
                trading_date=item["price_date"], open=item["open"], high=item["high"],
                low=item["low"], close=item["close"],
                provider_adjusted_close=item["adjusted_close"],
                split_adjusted_close=item["split_adjusted_close"],
                dividend_adjusted_close=item["dividend_adjusted_close"],
                provider_adjustment_basis=item["provider_adjustment_basis"],
                total_return_index=item["total_return_index"], volume=item["volume"],
                adjustment_origin=item["adjustment_origin"],
                split_event_ids=json.loads(item["split_event_ids_json"] or "[]"),
                dividend_treatment=item["dividend_treatment"],
                currency=item["currency"], exchange=item["exchange"], source=item["source"],
                source_record_id=item["source_identifier"], observed_at=item["observed_at"],
                available_at=item["available_at"], retrieved_at=item["retrieved_at"],
                provenance=json.loads(item["provenance_json"] or "{}"),
            )
            for item in self.database.prices_as_of(
                int(row["id"]), start, end, self._boundary
            )
        ]

    def get_market_cap(self, ticker: str, as_of: date) -> DataPoint | None:
        return None

    def get_shares_outstanding(self, ticker: str, as_of: date) -> DataPoint | None:
        return self.get_fundamentals(ticker, as_of).get("shares_outstanding")

    def get_estimates(self, ticker: str, as_of: date) -> dict[str, DataPoint]:
        return {}

    def get_company_metadata(self, ticker: str) -> dict[str, str]:
        row = self._base_company(ticker)
        if row is None:
            raise ValueError(f"ticker not present in database: {ticker.upper()}")
        metadata = self._metadata(int(row["id"]))
        return {
            "ticker": metadata["ticker"],
            "company": metadata["company_name"],
            "sector": metadata.get("sector") or "Unknown",
            "industry": metadata.get("industry") or "Unknown",
            "currency": row["currency"], "cik": metadata.get("cik") or "",
        }

    def get_corporate_actions(
        self, ticker: str, start: date, end: date
    ) -> list[CorporateActionRecord]:
        row = self._base_company(ticker)
        if row is None:
            raise ValueError(f"ticker not present in database: {ticker.upper()}")
        return [
            CorporateActionRecord(
                company_id=int(item["company_id"]), ticker=item["ticker"],
                event_type=item["event_type"], announcement_date=item["announcement_date"],
                declaration_date=item["declaration_date"], record_date=item["record_date"],
                ex_date=item["ex_date"], payment_date=item["payment_date"],
                effective_date=item["effective_date"],
                ratio=item["ratio"], amount=item["amount"], currency=item["currency"],
                old_ticker=item["old_ticker"], new_ticker=item["new_ticker"],
                source=item["source"], source_record_id=item["source_record_id"],
                observed_at=item["observed_at"], available_at=item["available_at"],
                retrieved_at=item["retrieved_at"],
                provenance=json.loads(item["provenance_json"] or "{}"),
            )
            for item in self.database.corporate_actions_as_of(
                int(row["id"]), start, end, self._boundary
            )
        ]

    def dcf_scenarios(self, ticker: str) -> list[DCFScenario]:
        self.company(ticker)
        return []

    def provenance_payload(self) -> dict[str, Any]:
        identities = [
            (company_id, self._metadata(company_id))
            for company_id in self.database.metadata_company_ids_as_of(self._boundary)
        ]
        return {
            "analysis_date": self.analysis_date.isoformat(),
            "input_cutoff": self.input_cutoff.isoformat(),
            "companies": {
                metadata["ticker"]: [
                    {key: value for key, value in dict(metric).items() if key != "revision_rank"}
                    for metric in self.database.get_metrics_as_of_company_id(company_id, self._boundary)
                ]
                for company_id, metadata in identities
            },
            "metadata": {
                metadata["ticker"]: [dict(item) for item in self.database.metadata_as_of("", self._boundary, company_id=company_id)]
                for company_id, metadata in identities
            },
            "prices": {
                metadata["ticker"]: [dict(item) for item in self.database.prices_as_of(company_id, date.min, self.analysis_date, self._boundary)]
                for company_id, metadata in identities
            },
            "corporate_actions": {
                metadata["ticker"]: [dict(item) for item in self.database.corporate_actions_as_of(company_id, date.min, self.analysis_date, self._boundary)]
                for company_id, metadata in identities
            },
        }
