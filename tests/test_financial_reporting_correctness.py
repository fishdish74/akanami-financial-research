from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import yaml

from market_analytics.database import MarketDatabase
from market_analytics.database_provider import DatabaseProvider
from market_analytics.models import CompanyMetadataRecord
from market_analytics.pipeline import run_fixture_pipeline
from market_analytics.ranking_views import render_ranking_markdown
from market_analytics.sec_xbrl import NormalizedFinancialFact
from tests.helpers import ANALYTICS_FIXTURE


def _fact(
    metric: str,
    *,
    value: str,
    end_date: str,
    available_at: str,
    identity: str,
    concept: str,
    start_date: str | None = None,
) -> NormalizedFinancialFact:
    period_type = "FY" if start_date else "POINT_IN_TIME"
    return NormalizedFinancialFact(
        canonical_metric=metric,
        normalized_value=Decimal(value),
        unit="USD",
        currency="USD",
        taxonomy="us-gaap",
        source_concept=concept,
        source_unit="USD",
        source_value=Decimal(value),
        source_fact_id=identity,
        mapping_version="sec-xbrl-v2",
        mapping_priority=1,
        period_type=period_type,
        start_date=start_date,
        end_date=end_date,
        fiscal_year=2025 if end_date.startswith("2025") else 2024,
        fiscal_quarter=None,
        filed_at=available_at,
        available_at=available_at,
        accession_number=f"accession-{identity}",
        form="10-K",
        is_amendment=False,
        quality_status="valid",
        warning=None,
        retrieved_at="2026-03-01T00:00:00+00:00",
    )


def _roe_financials(
    opening_available: str,
    closing_available: str,
    income_available: str,
    *,
    include_opening: bool = True,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as folder, MarketDatabase(
        Path(folder) / "roe.sqlite3"
    ) as database:
        database.migrate()
        company_id = database.upsert_company(
            ticker="ROEX", canonical_name="ROE Availability Fixture",
            exchange="TEST", cik="99",
        )
        database.insert_company_metadata(company_id, CompanyMetadataRecord(
            ticker="ROEX", company_name="ROE Availability Fixture", exchange="TEST",
            source="fixture-identity", source_record_id="roex-identity-v1",
            effective_from="2020-01-01T00:00:00Z",
            observed_at="2020-01-01T00:00:00Z",
            available_at="2020-01-01T00:00:00Z",
            retrieved_at="2020-01-01T00:00:00Z",
        ))
        facts = [
            _fact(
                "equity", value="80", end_date="2025-12-31",
                available_at=closing_available, identity="closing-equity",
                concept="StockholdersEquity",
            ),
            _fact(
                "net_income", value="18", start_date="2025-01-01",
                end_date="2025-12-31", available_at=income_available,
                identity="annual-income", concept="NetIncomeLoss",
            ),
        ]
        if include_opening:
            facts.append(
                _fact(
                    "equity", value="60", end_date="2024-12-31",
                    available_at=opening_available, identity="opening-equity",
                    concept="StockholdersEquity",
                )
            )
        database.insert_financial_facts(company_id, facts)
        provider = DatabaseProvider(
            database, datetime(2026, 12, 31, tzinfo=timezone.utc)
        )
        return dict(provider.company("ROEX").financials)


class RoeAvailabilityRegressionTests(unittest.TestCase):
    def test_average_equity_uses_latest_availability_of_all_roe_inputs(self) -> None:
        scenarios = {
            "net income latest": (
                "2026-01-10T12:00:00+00:00", "2026-02-10T12:00:00+00:00",
                "2026-03-10T12:00:00+00:00", date(2026, 3, 10),
            ),
            "opening equity latest": (
                "2026-03-11T12:00:00+00:00", "2026-02-10T12:00:00+00:00",
                "2026-03-10T12:00:00+00:00", date(2026, 3, 11),
            ),
            "closing equity latest": (
                "2026-01-10T12:00:00+00:00", "2026-03-12T12:00:00+00:00",
                "2026-03-10T12:00:00+00:00", date(2026, 3, 12),
            ),
            "equal timestamps": (
                "2026-03-10T12:00:00+00:00", "2026-03-10T12:00:00+00:00",
                "2026-03-10T12:00:00+00:00", date(2026, 3, 10),
            ),
        }
        for name, (opening, closing, income, expected) in scenarios.items():
            with self.subTest(name=name):
                point = _roe_financials(opening, closing, income)["average_equity"]
                self.assertEqual(point.as_of, expected)
                self.assertIn("accession-opening-equity:StockholdersEquity", point.source)
                self.assertIn("accession-closing-equity:StockholdersEquity", point.source)
                self.assertIn("accession-annual-income:NetIncomeLoss", point.source)

    def test_average_equity_is_absent_when_a_required_input_is_missing(self) -> None:
        financials = _roe_financials(
            "2026-01-10T12:00:00+00:00",
            "2026-02-10T12:00:00+00:00",
            "2026-03-10T12:00:00+00:00",
            include_opening=False,
        )
        self.assertNotIn("average_equity", financials)


class DataStatusRetirementRegressionTests(unittest.TestCase):
    IMPORTANT = {
        "revenue", "net_income", "cash", "total_debt", "equity",
        "operating_cash_flow", "shares_outstanding",
    }

    def _status(self, states: tuple[str, ...]) -> tuple[dict[str, object], int]:
        with tempfile.TemporaryDirectory() as folder, MarketDatabase(
            Path(folder) / "status.sqlite3"
        ) as database:
            database.migrate()
            company_id = database.upsert_company(
                ticker="STAT", canonical_name="Status Fixture", exchange="TEST", cik="98"
            )
            facts = []
            metrics = ("revenue", "net_income")
            concepts = ("Revenues", "NetIncomeLoss")
            for index, state in enumerate(states):
                fact = _fact(
                    metrics[index], value=str(100 - index * 10),
                    start_date="2025-01-01", end_date="2025-12-31",
                    available_at=f"2026-02-{10 + index:02d}T12:00:00+00:00",
                    identity=f"status-{state}-{index}", concept=concepts[index],
                )
                facts.append(replace(fact, quality_status=state))
            if facts:
                database.insert_financial_facts(company_id, facts)
            status = database.data_status()[0]
            retained = int(
                database.connection.execute(
                    "SELECT COUNT(*) FROM financial_metrics WHERE quality_status='retired'"
                ).fetchone()[0]
            )
            return status, retained

    def test_all_active_facts_define_current_status(self) -> None:
        status, retained = self._status(("valid", "valid"))
        self.assertEqual(status["annual_periods"], 1)
        self.assertEqual(status["data_quality_issues"], 0)
        self.assertNotIn("revenue", status["missing_important_metrics"])
        self.assertNotIn("net_income", status["missing_important_metrics"])
        self.assertEqual(retained, 0)

    def test_retired_facts_do_not_contribute_to_mixed_current_status(self) -> None:
        status, retained = self._status(("valid", "retired"))
        self.assertEqual(status["annual_periods"], 1)
        self.assertEqual(status["data_quality_issues"], 0)
        self.assertNotIn("revenue", status["missing_important_metrics"])
        self.assertIn("net_income", status["missing_important_metrics"])
        self.assertEqual(retained, 1)

    def test_all_retired_facts_produce_zero_current_coverage(self) -> None:
        status, retained = self._status(("retired", "retired"))
        self.assertEqual(status["annual_periods"], 0)
        self.assertEqual(status["quarterly_periods"], 0)
        self.assertIsNone(status["last_filing"])
        self.assertIsNone(status["latest_financial_period"])
        self.assertIsNone(status["last_sync"])
        self.assertEqual(status["data_quality_issues"], 0)
        self.assertEqual(set(status["missing_important_metrics"]), self.IMPORTANT)
        self.assertEqual(retained, 2)

    def test_zero_data_produces_zero_current_coverage(self) -> None:
        status, retained = self._status(())
        self.assertEqual(status["annual_periods"], 0)
        self.assertEqual(status["quarterly_periods"], 0)
        self.assertIsNone(status["last_filing"])
        self.assertIsNone(status["latest_financial_period"])
        self.assertIsNone(status["last_sync"])
        self.assertEqual(status["data_quality_issues"], 0)
        self.assertEqual(set(status["missing_important_metrics"]), self.IMPORTANT)
        self.assertEqual(retained, 0)


class NonRankableRenderingRegressionTests(unittest.TestCase):
    def test_only_ranked_companies_receive_opportunity_positions_and_count(self) -> None:
        run = run_fixture_pipeline(ANALYTICS_FIXTURE)
        original = run.rankings["one_day"]
        mixed = [
            original[0],
            original[1].model_copy(
                update={"ranking_status": "insufficient_evidence", "opportunity_score": None}
            ),
            original[2],
            original[3].model_copy(
                update={"ranking_status": "blocked", "opportunity_score": None}
            ),
        ]
        run = run.model_copy(
            update={"rankings": {**run.rankings, "one_day": mixed}}
        )

        markdown = render_ranking_markdown(run, "one_day")
        metadata = yaml.safe_load(markdown.split("---", 2)[1])
        opportunity = markdown.split("## Opportunity Ranking", 1)[1].split(
            "## Research-Priority Ranking", 1
        )[0]

        self.assertEqual(metadata["companies_ranked"], 2)
        self.assertIn("- **Companies ranked:** 2", markdown)
        self.assertIn(f"| 1 | {original[0].ticker} |", opportunity)
        self.assertIn(f"| 2 | {original[2].ticker} |", opportunity)
        self.assertNotIn(f"| 2 | {original[1].ticker} |", opportunity)
        self.assertNotIn(f"| 4 | {original[3].ticker} |", opportunity)
        self.assertIn("### Not Ranked", opportunity)
        self.assertIn(original[1].ticker, opportunity)
        self.assertIn("insufficient\\_evidence", opportunity)
        self.assertIn(original[3].ticker, opportunity)
        self.assertIn("blocked", opportunity)


if __name__ == "__main__":
    unittest.main()
