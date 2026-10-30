from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from market_analytics.config import load_config, metric_relevance
from market_analytics.models import FiscalPeriod
from market_analytics.pipeline import run_fixture_pipeline, run_provider_pipeline
from market_analytics.providers import FixtureProvider
from market_analytics.quality import confidence_from_quality, freshness_report, validate_company, validate_records
from market_analytics.ranking import rank_company, rank_universe
from tests.helpers import ANALYTICS_FIXTURE


D = Decimal


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analytics_run = run_fixture_pipeline(ANALYTICS_FIXTURE)
        cls.config = load_config()

    def test_pipeline_is_deterministic(self) -> None:
        second = run_fixture_pipeline(ANALYTICS_FIXTURE)
        self.assertEqual(self.analytics_run.manifest.run_id, second.manifest.run_id)
        self.assertEqual(self.analytics_run.rankings, second.rankings)

    def test_all_six_fictional_profiles_are_loaded(self) -> None:
        self.assertEqual(sorted(self.analytics_run.companies), ["CATX", "GRWX", "LEVR", "QLTY", "STDY", "VALX"])

    def test_horizon_changes_top_opportunity(self) -> None:
        self.assertEqual(self.analytics_run.rankings["one_day"][0].ticker, "CATX")
        self.assertEqual(self.analytics_run.rankings["one_year"][0].ticker, "QLTY")

    def test_research_priority_is_separate_from_opportunity(self) -> None:
        opportunity_order = [item.ticker for item in self.analytics_run.rankings["one_year"]]
        priority_order = [item.ticker for item in self.analytics_run.research_priorities["one_year"]]
        self.assertNotEqual(opportunity_order, priority_order)
        self.assertEqual(priority_order[0], "LEVR")

    def test_scores_stay_within_bounds(self) -> None:
        for horizon, results in self.analytics_run.rankings.items():
            for result in results:
                self.assertGreaterEqual(result.opportunity_score, 0)
                self.assertLessEqual(result.opportunity_score, 100)
                self.assertGreaterEqual(result.research_priority_score, 0)
                self.assertLessEqual(result.research_priority_score, 100)

    def test_stale_data_reduces_confidence(self) -> None:
        result = next(item for item in self.analytics_run.rankings["one_year"] if item.ticker == "LEVR")
        self.assertEqual(result.confidence, "low")
        self.assertTrue(any("days old" in warning for warning in result.warnings))

    def test_freshness_report_tracks_input_categories(self) -> None:
        raw_company = FixtureProvider(ANALYTICS_FIXTURE).company("LEVR")
        report = freshness_report(raw_company, date(2026, 9, 29))
        self.assertEqual(report.price_age, 272)
        self.assertEqual(report.fundamentals_age, 272)

    def test_severely_stale_current_input_is_excluded_from_calculations(self) -> None:
        self.assertNotIn("price_per_share", self.analytics_run.companies["LEVR"].financials)
        self.assertTrue(
            any(
                issue.code == "stale_input"
                and issue.severity == "ERROR"
                and issue.fields == ["price_per_share"]
                for issue in self.analytics_run.quality_issues["LEVR"]
            )
        )

    def test_missing_data_reduces_confidence(self) -> None:
        result = next(item for item in self.analytics_run.rankings["one_year"] if item.ticker == "CATX")
        self.assertEqual(result.confidence, "low")
        self.assertIn("Forward P/E", result.missing_inputs)

    def test_catalyst_change_affects_one_day_more_than_one_year(self) -> None:
        company = self.analytics_run.companies["CATX"]
        no_catalyst = company.model_copy(update={"catalysts": []})
        comparisons = self.analytics_run.comparisons["CATX"]
        issues = self.analytics_run.quality_issues["CATX"]
        as_of = date.fromisoformat(self.analytics_run.analysis_date)
        day_with = rank_company(company, comparisons, issues, self.config, "one_day", as_of)
        day_without = rank_company(no_catalyst, comparisons, issues, self.config, "one_day", as_of)
        year_with = rank_company(company, comparisons, issues, self.config, "one_year", as_of)
        year_without = rank_company(no_catalyst, comparisons, issues, self.config, "one_year", as_of)
        self.assertGreater(day_with.opportunity_score - day_without.opportunity_score, year_with.opportunity_score - year_without.opportunity_score)

    def test_valuation_change_affects_one_year_more_than_one_day(self) -> None:
        company = self.analytics_run.companies["QLTY"]
        issues = self.analytics_run.quality_issues["QLTY"]
        base = self.analytics_run.comparisons["QLTY"]
        low = {name: value.model_copy(update={"percentile": D("10")}) if name in {"P/E", "Forward P/E", "EV / EBITDA", "Price / Sales", "Earnings Yield", "FCF Yield"} else value for name, value in base.items()}
        high = {name: value.model_copy(update={"percentile": D("90")}) if name in {"P/E", "Forward P/E", "EV / EBITDA", "Price / Sales", "Earnings Yield", "FCF Yield"} else value for name, value in base.items()}
        as_of = date.fromisoformat(self.analytics_run.analysis_date)
        day_delta = rank_company(company, high, issues, self.config, "one_day", as_of).opportunity_score - rank_company(company, low, issues, self.config, "one_day", as_of).opportunity_score
        year_delta = rank_company(company, high, issues, self.config, "one_year", as_of).opportunity_score - rank_company(company, low, issues, self.config, "one_year", as_of).opportunity_score
        self.assertGreater(year_delta, day_delta)

    def test_ties_use_ticker_for_stable_order(self) -> None:
        first = self.analytics_run.rankings["one_year"][0].model_copy(update={"ticker": "ZZZ", "opportunity_score": D("50")})
        second = first.model_copy(update={"ticker": "AAA"})
        self.assertEqual([item.ticker for item in rank_universe([first, second])], ["AAA", "ZZZ"])

    def test_weight_config_is_complete_and_sums_to_one(self) -> None:
        for weights in self.config.ranking_weights.values():
            self.assertEqual(sum(weights.values()), D("1.00"))
            self.assertEqual(len(weights), 8)

    def test_sector_relevance_is_explicit(self) -> None:
        self.assertEqual(metric_relevance(self.config, "Software", "Rule of 40"), "relevant")
        self.assertEqual(metric_relevance(self.config, "Banks", "EV / EBITDA"), "not_applicable")

    def test_two_negative_graham_inputs_stay_unavailable_in_pipeline(self) -> None:
        provider = FixtureProvider(ANALYTICS_FIXTURE)
        company = provider.company("QLTY")
        financials = dict(company.financials)
        for field in ("eps", "book_value_per_share"):
            financials[field] = financials[field].model_copy(
                update={"value": D("-2")}
            )
        provider._companies["QLTY"] = company.model_copy(
            update={"financials": financials}
        )
        run = run_provider_pipeline(provider)
        graham = run.metrics["QLTY"]["Graham Number"]
        self.assertEqual(graham.status, "not_applicable")
        self.assertIsNone(graham.value)
        self.assertEqual(run.comparisons["QLTY"]["Graham Number"].status, "not_applicable")
        self.assertEqual(run.metrics["QLTY"]["ROE"].status, "valid")


class QualityEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analytics_run = run_fixture_pipeline(ANALYTICS_FIXTURE)
        cls.config = load_config()

    def test_negative_market_cap_is_blocking(self) -> None:
        company = self.analytics_run.companies["QLTY"]
        financials = dict(company.financials)
        financials["market_cap"] = financials["market_cap"].model_copy(update={"value": D("-1")})
        broken = company.model_copy(update={"financials": financials})
        issues = validate_company(broken, date(2026, 9, 29), self.config.freshness_days)
        self.assertTrue(any(issue.severity == "BLOCKING" for issue in issues))

    def test_mixed_currency_is_blocking(self) -> None:
        company = self.analytics_run.companies["QLTY"]
        financials = dict(company.financials)
        financials["revenue"] = financials["revenue"].model_copy(update={"currency": "EUR"})
        broken = company.model_copy(update={"financials": financials})
        issues = validate_company(broken, date(2026, 9, 29), self.config.freshness_days)
        self.assertTrue(any(issue.code == "mixed_currency" and issue.severity == "BLOCKING" for issue in issues))

    def test_duplicate_and_conflicting_rows_are_detected(self) -> None:
        rows = [
            ("AAA", "revenue", "FY2026", D("10"), "millions", "USD"),
            ("AAA", "revenue", "FY2026", D("10"), "millions", "USD"),
            ("AAA", "revenue", "FY2026", D("11"), "millions", "EUR"),
        ]
        issues = validate_records(rows)
        self.assertEqual(issues[0].code, "duplicate_row")
        self.assertEqual(issues[1].code, "conflicting_values")
        self.assertEqual(issues[1].severity, "BLOCKING")

    def test_blocking_issue_forces_low_confidence(self) -> None:
        issues = validate_records([
            ("AAA", "revenue", "FY2026", D("10"), "millions", "USD"),
            ("AAA", "revenue", "FY2026", D("11"), "millions", "EUR"),
        ])
        self.assertEqual(confidence_from_quality(issues, 0, 10, "high"), "low")

    def test_recent_annual_fundamental_uses_annual_cadence(self) -> None:
        company = FixtureProvider(ANALYTICS_FIXTURE).company("QLTY")
        annual_period = FiscalPeriod(
            kind="FY", label="FY2025", end_date=date(2025, 9, 30), fiscal_year=2025
        )
        annual_revenue = company.financials["revenue"].model_copy(
            update={"period": annual_period, "as_of": date(2025, 10, 31)}
        )
        annual_company = company.model_copy(update={"financials": {"revenue": annual_revenue}})
        issues = validate_company(
            annual_company, date(2026, 9, 29), self.config.freshness_days
        )
        self.assertFalse(any(issue.code == "stale_input" for issue in issues))

    def test_old_annual_fundamental_warns_before_it_errors(self) -> None:
        company = FixtureProvider(ANALYTICS_FIXTURE).company("QLTY")
        annual_period = FiscalPeriod(
            kind="FY", label="FY2024", end_date=date(2024, 12, 31), fiscal_year=2024
        )
        annual_revenue = company.financials["revenue"].model_copy(
            update={"period": annual_period, "as_of": date(2025, 1, 1)}
        )
        annual_company = company.model_copy(update={"financials": {"revenue": annual_revenue}})
        issues = validate_company(
            annual_company, date(2026, 9, 29), self.config.freshness_days
        )
        stale = [issue for issue in issues if issue.code == "stale_input"]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0].severity, "WARNING")
        self.assertIn("annual fundamentals threshold is 455", stale[0].message)

    def test_historical_comparison_field_is_not_a_current_freshness_claim(self) -> None:
        company = FixtureProvider(ANALYTICS_FIXTURE).company("QLTY")
        historical = company.financials["prior_revenue"].model_copy(
            update={"as_of": date(2020, 1, 1)}
        )
        historical_company = company.model_copy(
            update={"financials": {"prior_revenue": historical}}
        )
        issues = validate_company(
            historical_company, date(2026, 9, 29), self.config.freshness_days
        )
        self.assertFalse(any(issue.code == "stale_input" for issue in issues))

    def test_old_quarterly_fundamental_uses_quarterly_cadence(self) -> None:
        company = FixtureProvider(ANALYTICS_FIXTURE).company("QLTY")
        quarterly_period = FiscalPeriod(
            kind="Q", label="2025-Q1", end_date=date(2025, 3, 31),
            fiscal_year=2025, fiscal_quarter=1,
        )
        quarterly_revenue = company.financials["revenue"].model_copy(
            update={"period": quarterly_period, "as_of": date(2025, 5, 1)}
        )
        quarterly_company = company.model_copy(
            update={"financials": {"revenue": quarterly_revenue}}
        )
        issues = validate_company(
            quarterly_company, date(2026, 9, 29), self.config.freshness_days
        )
        stale = [issue for issue in issues if issue.code == "stale_input"]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0].severity, "ERROR")
        self.assertIn("quarterly/instant fundamentals threshold is 150", stale[0].message)


if __name__ == "__main__":
    unittest.main()
