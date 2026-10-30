from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from entities import Entity
from market_analytics.capabilities import (
    active_us_stock_capability,
    classify_corporate_action,
    compare_reported_volume,
    evaluate_security_capability,
    require_fundamental_normalization,
)
from market_analytics.config import load_config
from market_analytics.database import MarketDatabase
from market_analytics.pipeline import run_fixture_pipeline
from market_analytics.providers import FixtureProvider
from market_analytics.ranking import rank_company
from market_analytics.sec_ingestion import ingest_company_facts_payload
from market_analytics.sec_xbrl import (
    assess_company_facts_capability,
    parse_company_facts,
)
from tests.helpers import ANALYTICS_FIXTURE


CUSTOM_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "sec-companyfacts-vtsi-style-synthetic.json"
)
RETRIEVED = datetime(2026, 4, 1, tzinfo=timezone.utc)


def _rank_with_capability(capability):
    run = run_fixture_pipeline(ANALYTICS_FIXTURE)
    company = run.companies["QLTY"].model_copy(
        update={"ticker": "CASE", "company": "Capability Case", "capability": capability}
    )
    return rank_company(
        company,
        run.comparisons["QLTY"],
        run.quality_issues["QLTY"],
        load_config(),
        "one_year",
        date.fromisoformat(run.analysis_date),
    )


class InactiveIdentityCapabilityTests(unittest.TestCase):
    def test_missing_current_identity_stays_incomplete_with_historical_data(self) -> None:
        capability = evaluate_security_capability(
            current_metadata_available=False,
            active=None,
            listing_type=None,
            foreign_issuer=False,
            fundamental_basis="unknown",
            historical_prices_available=True,
            historical_actions_available=True,
            ticker_events_available=False,
        )
        self.assertEqual(capability.security_type, "unknown")
        self.assertEqual(capability.identity_continuity, "incomplete")
        self.assertEqual(capability.price_history, "partial")
        self.assertEqual(capability.corporate_actions, "partial")
        self.assertIn("incomplete_identity_continuity", capability.reason_codes)
        self.assertIn("provider_identity_unavailable", capability.reason_codes)
        self.assertNotEqual(capability.security_type, "active_us_common_stock")

    def test_missing_identity_cannot_rank_or_merge_into_a_replacement(self) -> None:
        capability = evaluate_security_capability(
            current_metadata_available=False,
            active=None,
            listing_type=None,
            foreign_issuer=False,
            fundamental_basis="us_gaap",
            historical_prices_available=True,
            historical_actions_available=True,
            ticker_events_available=False,
        )
        result = _rank_with_capability(capability)
        self.assertEqual(result.ranking_status, "blocked")
        self.assertIsNone(result.opportunity_score)
        self.assertIn("incomplete_identity_continuity", result.capability_reason_codes)
        self.assertNotIn("ticker_change", capability.explanations)

    def test_explicit_inactive_security_is_not_rankable(self) -> None:
        capability = evaluate_security_capability(
            current_metadata_available=True,
            active=False,
            listing_type="common_stock",
            foreign_issuer=False,
            fundamental_basis="us_gaap",
            historical_prices_available=True,
            historical_actions_available=True,
            ticker_events_available=False,
        )
        self.assertEqual(capability.security_type, "inactive_delisted")
        self.assertIn("inactive_security", capability.reason_codes)
        self.assertFalse(capability.ranking_eligible)


class ForeignAccountingCapabilityTests(unittest.TestCase):
    @staticmethod
    def ifrs_payload() -> bytes:
        return json.dumps({
            "cik": 1046179,
            "entityName": "Synthetic Foreign ADR",
            "facts": {
                "ifrs-full": {
                    "Revenue": {
                        "units": {
                            "USD": [{
                                "start": "2025-01-01", "end": "2025-12-31",
                                "val": 10, "accn": "foreign-1", "fy": 2025,
                                "fp": "FY", "form": "20-F", "filed": "2026-04-01"
                            }]
                        }
                    }
                }
            },
        }).encode()

    def test_ifrs_adr_keeps_provider_support_but_blocks_fundamentals(self) -> None:
        capability = assess_company_facts_capability(
            self.ifrs_payload(), listing_type="adr", foreign_issuer=True
        )
        self.assertEqual(capability.security_type, "foreign_issuer_adr")
        self.assertEqual(capability.price_history, "supported")
        self.assertEqual(capability.corporate_actions, "supported")
        self.assertEqual(capability.fundamental_normalization, "unsupported")
        self.assertIn("unsupported_ifrs", capability.reason_codes)
        with self.assertRaisesRegex(ValueError, "unsupported_ifrs"):
            require_fundamental_normalization(capability)

    def test_ifrs_cannot_enter_us_gaap_parser_or_receive_a_numeric_rank(self) -> None:
        capability = assess_company_facts_capability(
            self.ifrs_payload(), listing_type="adr", foreign_issuer=True
        )
        with self.assertRaisesRegex(ValueError, "unsupported_ifrs"):
            parse_company_facts(
                self.ifrs_payload(), retrieved_at=RETRIEVED, capability=capability
            )
        result = _rank_with_capability(capability)
        self.assertEqual(result.ranking_status, "blocked")
        self.assertIsNone(result.opportunity_score)

    def test_ifrs_ingestion_gate_fails_before_any_database_write(self) -> None:
        with tempfile.TemporaryDirectory() as folder, MarketDatabase(
            Path(folder) / "ifrs.sqlite3"
        ) as database:
            database.migrate()
            with self.assertRaisesRegex(ValueError, "unsupported_ifrs"):
                ingest_company_facts_payload(
                    database,
                    Entity(
                        ticker="TSM", canonical_name="Synthetic Foreign ADR",
                        exchange="NYSE", cik="1046179",
                    ),
                    self.ifrs_payload(),
                    retrieved_at=RETRIEVED,
                    source_url="https://example.invalid/ifrs",
                    request_identifier="synthetic-ifrs",
                )
            self.assertEqual(database.connection.execute("SELECT COUNT(*) FROM companies").fetchone()[0], 0)
            self.assertEqual(database.connection.execute("SELECT COUNT(*) FROM source_artifacts").fetchone()[0], 0)

    def test_missing_sic_or_industry_does_not_infer_a_sector(self) -> None:
        capability = evaluate_security_capability(
            current_metadata_available=True,
            active=True,
            listing_type="adr",
            foreign_issuer=True,
            fundamental_basis="ifrs",
        )
        company = FixtureProvider(ANALYTICS_FIXTURE).company("QLTY").model_copy(
            update={"sector": "Unknown", "industry": "Unknown", "sic": None, "capability": capability}
        )
        self.assertEqual((company.sector, company.industry, company.sic), ("Unknown", "Unknown", None))


class CustomTaxonomyCapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = CUSTOM_FIXTURE.read_bytes()

    def test_custom_names_do_not_inherit_us_gaap_meaning(self) -> None:
        _, _, facts, warnings = parse_company_facts(
            self.payload, retrieved_at=RETRIEVED
        )
        metrics = {fact.canonical_metric for fact in facts}
        self.assertIn("total_assets", metrics)
        self.assertIn("shares_outstanding", metrics)
        self.assertNotIn("revenue", metrics)
        self.assertNotIn("net_income", metrics)
        self.assertTrue(any("unsupported taxonomy coverage" in item for item in warnings))

    def test_custom_taxonomy_requires_replay_before_normalization(self) -> None:
        capability = assess_company_facts_capability(self.payload)
        self.assertEqual(capability.custom_taxonomy, "custom")
        self.assertEqual(capability.fundamental_normalization, "requires_replay")
        self.assertIn("custom_taxonomy_requires_replay", capability.reason_codes)
        with self.assertRaisesRegex(ValueError, "custom_taxonomy_requires_replay"):
            parse_company_facts(
                self.payload, retrieved_at=RETRIEVED, capability=capability
            )

    def test_sparse_custom_taxonomy_security_is_not_ranked(self) -> None:
        result = _rank_with_capability(assess_company_facts_capability(self.payload))
        self.assertEqual(result.ranking_status, "blocked")
        self.assertIsNone(result.opportunity_score)
        self.assertIn("custom_taxonomy_requires_replay", result.capability_reason_codes)

    def test_custom_taxonomy_ingestion_gate_fails_before_any_database_write(self) -> None:
        with tempfile.TemporaryDirectory() as folder, MarketDatabase(
            Path(folder) / "custom.sqlite3"
        ) as database:
            database.migrate()
            with self.assertRaisesRegex(ValueError, "custom_taxonomy_requires_replay"):
                ingest_company_facts_payload(
                    database,
                    Entity(
                        ticker="VTSI", canonical_name="Synthetic Micro-Cap",
                        exchange="NASDAQ", cik="1085243",
                    ),
                    self.payload,
                    retrieved_at=RETRIEVED,
                    source_url="https://example.invalid/custom",
                    request_identifier="synthetic-custom",
                )
            self.assertEqual(database.connection.execute("SELECT COUNT(*) FROM companies").fetchone()[0], 0)
            self.assertEqual(database.connection.execute("SELECT COUNT(*) FROM financial_metrics").fetchone()[0], 0)


class CorporateActionAndVolumePolicyTests(unittest.TestCase):
    def test_stock_dividend_label_stays_ambiguous_without_external_evidence(self) -> None:
        value = classify_corporate_action("stock_dividend")
        self.assertEqual(value.raw_provider_type, "stock_dividend")
        self.assertIsNone(value.normalized_action_type)
        self.assertEqual(value.normalization_status, "ambiguous")
        self.assertEqual(value.reason_code, "ambiguous_corporate_action_type")

    def test_verified_economic_split_is_derived_without_losing_raw_type(self) -> None:
        value = classify_corporate_action(
            "stock_dividend",
            verified_economic_type="stock_split",
            evidence_reference="issuer:verified-event",
        )
        self.assertEqual(value.raw_provider_type, "stock_dividend")
        self.assertEqual(value.normalized_action_type, "stock_split")
        self.assertEqual(value.normalization_status, "derived")
        self.assertEqual(value.evidence_reference, "issuer:verified-event")

    def test_derived_action_requires_explicit_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "evidence_reference"):
            classify_corporate_action(
                "stock_dividend", verified_economic_type="stock_split"
            )

    def test_fractional_volume_rounding_is_not_a_false_mismatch(self) -> None:
        rounded = compare_reported_volume(Decimal("100.4"), Decimal("100"))
        different = compare_reported_volume(Decimal("100.6"), Decimal("100"))
        self.assertEqual(rounded.status, "equivalent_after_rounding")
        self.assertEqual(different.status, "different")
        self.assertEqual(rounded.adjusted_volume_semantics, "unverified")
        self.assertEqual(rounded.reason_code, "adjusted_volume_unverified")


class SupportedUniverseRegressionTests(unittest.TestCase):
    def test_supported_active_us_stocks_remain_rankable(self) -> None:
        baseline = run_fixture_pipeline(ANALYTICS_FIXTURE)
        original = next(
            item for item in baseline.rankings["one_year"] if item.ticker == "QLTY"
        )
        for ticker in ("AAPL", "JPM", "COST", "NVDA"):
            company = baseline.companies["QLTY"].model_copy(
                update={"ticker": ticker, "company": ticker, "capability": active_us_stock_capability()}
            )
            result = rank_company(
                company,
                baseline.comparisons["QLTY"],
                baseline.quality_issues["QLTY"],
                load_config(),
                "one_year",
                date.fromisoformat(baseline.analysis_date),
            )
            with self.subTest(ticker=ticker):
                self.assertEqual(result.ranking_status, original.ranking_status)
                self.assertEqual(result.opportunity_score, original.opportunity_score)
                self.assertEqual(
                    result.capability_reason_codes, ["adjusted_volume_unverified"]
                )


if __name__ == "__main__":
    unittest.main()
