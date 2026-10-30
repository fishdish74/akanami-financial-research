"""R3 regressions for undefined ROA in the existing analytics pipeline."""

from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import patch

from market_analytics import frameworks, metrics
from market_analytics.pipeline import run_provider_pipeline
from market_analytics.providers import FixtureProvider
from market_analytics.ranking import COMPONENT_METRICS
from tests.helpers import ANALYTICS_FIXTURE


class ZeroAssetsPipelineTests(unittest.TestCase):
    @staticmethod
    def run_with_assets(value: Decimal | None, *, remove: bool = False):
        provider = FixtureProvider(ANALYTICS_FIXTURE)
        company = provider.company("QLTY")
        financials = dict(company.financials)
        raw_company = next(
            item for item in provider._raw["companies"] if item["ticker"] == "QLTY"
        )
        if remove:
            financials.pop("total_assets")
            raw_company["financials"].pop("total_assets")
        else:
            financials["total_assets"] = financials["total_assets"].model_copy(
                update={"value": value}
            )
            raw_company["financials"]["total_assets"] = str(value)
        provider._companies["QLTY"] = company.model_copy(
            update={"financials": financials}
        )
        return run_provider_pipeline(provider)

    def test_a_zero_assets_makes_roa_and_positive_roa_unavailable(self) -> None:
        run = self.run_with_assets(Decimal("0"))
        roa = run.metrics["QLTY"]["ROA"]
        f_score = run.metrics["QLTY"]["Piotroski F-Score"]
        self.assertEqual(roa.status, "invalid_input")
        self.assertIsNone(roa.value)
        self.assertIn("total_assets_zero", roa.warning)
        self.assertEqual(f_score.status, "invalid_input")
        self.assertIsNone(f_score.value)
        self.assertIn("positive_roa: total_assets_zero", f_score.warning)

    def test_b_positive_assets_keep_valid_roa_behavior(self) -> None:
        run = self.run_with_assets(Decimal("15000"))
        roa = run.metrics["QLTY"]["ROA"]
        self.assertEqual(roa.status, "valid")
        self.assertEqual(roa.value, Decimal("0.08"))
        self.assertEqual(metrics.roa(Decimal("10"), Decimal("100")).value, Decimal("0.1"))
        self.assertEqual(
            frameworks.piotroski_f_score({
                "positive_roa": True, "positive_cfo": True,
                "roa_improved": True, "cfo_exceeds_income": True,
                "leverage_declined": True, "current_ratio_improved": True,
                "no_new_shares": True, "gross_margin_improved": True,
                "asset_turnover_improved": True,
            }).value,
            Decimal("9"),
        )

    def test_valid_zero_percent_roa_is_distinct_from_undefined_roa(self) -> None:
        valid_zero = metrics.roa(Decimal("0"), Decimal("100"))
        undefined = metrics.roa(Decimal("0"), Decimal("0"))
        self.assertEqual((valid_zero.status, valid_zero.value), ("valid", Decimal("0")))
        self.assertEqual(undefined.status, "invalid_input")
        self.assertIsNone(undefined.value)

    def test_c_negative_assets_preserve_current_signed_ratio_policy(self) -> None:
        result = metrics.roa(Decimal("10"), Decimal("-100"))
        self.assertEqual((result.status, result.value), ("valid", Decimal("-0.1")))
        run = self.run_with_assets(Decimal("-15000"))
        self.assertEqual(run.metrics["QLTY"]["ROA"].value, Decimal("-0.08"))
        self.assertNotIn(
            "total_assets_zero",
            run.metrics["QLTY"]["Piotroski F-Score"].warning or "",
        )

    def test_d_missing_assets_preserve_missing_data(self) -> None:
        run = self.run_with_assets(None, remove=True)
        self.assertEqual(run.metrics["QLTY"]["ROA"].status, "missing_data")
        self.assertIsNone(run.metrics["QLTY"]["ROA"].value)
        self.assertEqual(
            run.metrics["QLTY"]["Piotroski F-Score"].status, "missing_data"
        )

    def test_e_other_metrics_continue_after_zero_assets(self) -> None:
        run = self.run_with_assets(Decimal("0"))
        self.assertEqual(run.metrics["QLTY"]["P/E"].status, "valid")
        self.assertEqual(run.metrics["QLTY"]["ROE"].status, "valid")
        self.assertEqual(run.metrics["QLTY"]["Altman Z-Score"].status, "not_applicable")
        self.assertEqual(len(run.rankings), 4)

    def test_f_ranking_never_receives_a_fabricated_roa_contribution(self) -> None:
        baseline = self.run_with_assets(Decimal("15000"))
        zero = self.run_with_assets(Decimal("0"))
        comparison = zero.comparisons["QLTY"]["ROA"]
        self.assertEqual(comparison.status, "invalid_input")
        self.assertIsNone(comparison.value)
        self.assertIsNone(comparison.percentile)
        self.assertTrue(all(
            "ROA" not in names and "Piotroski F-Score" not in names
            for names in COMPONENT_METRICS.values()
        ))
        for horizon in zero.rankings:
            base = next(item for item in baseline.rankings[horizon] if item.ticker == "QLTY")
            current = next(item for item in zero.rankings[horizon] if item.ticker == "QLTY")
            self.assertEqual(current.components, base.components)
            self.assertEqual(current.opportunity_score, base.opportunity_score)
            self.assertEqual(current.ranking_status, base.ranking_status)

    def test_g_identical_zero_assets_runs_are_deterministic(self) -> None:
        first = self.run_with_assets(Decimal("0"))
        second = self.run_with_assets(Decimal("0"))
        self.assertEqual(first.manifest.run_id, second.manifest.run_id)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.rankings, second.rankings)

    def test_zero_assets_with_reported_roa_still_withholds_roa_improvement_point(self) -> None:
        provider = FixtureProvider(ANALYTICS_FIXTURE)
        company = provider.company("QLTY")
        financials = dict(company.financials)
        financials["total_assets"] = financials["total_assets"].model_copy(
            update={"value": Decimal("0")}
        )
        for name, value in (("roa", "0.5"), ("prior_roa", "0.1")):
            financials[name] = financials["total_assets"].model_copy(
                update={"value": Decimal(value), "unit": "decimal_ratio"}
            )
        provider._companies["QLTY"] = company.model_copy(
            update={"financials": financials}
        )
        captured_signals = []
        original = frameworks.piotroski_f_score

        def capture_signals(signals, **kwargs):
            if kwargs.get("unavailable_reasons"):
                captured_signals.append(dict(signals))
            return original(signals, **kwargs)

        with patch(
            "market_analytics.pipeline.frameworks.piotroski_f_score",
            side_effect=capture_signals,
        ):
            run = run_provider_pipeline(provider)
        self.assertEqual(len(captured_signals), 1)
        self.assertIsNone(captured_signals[0]["positive_roa"])
        self.assertIsNone(captured_signals[0]["roa_improved"])
        self.assertIsNone(run.metrics["QLTY"]["Piotroski F-Score"].value)
        self.assertIn(
            "positive_roa: total_assets_zero",
            run.metrics["QLTY"]["Piotroski F-Score"].warning,
        )


if __name__ == "__main__":
    unittest.main()
