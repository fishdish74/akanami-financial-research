from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from market_analytics import frameworks, metrics
from market_analytics.models import DataPoint, FiscalPeriod
from market_analytics.normalization import normalize_data_point, periods_compatible


D = Decimal


class UnitNormalizationTests(unittest.TestCase):
    def point(self, value, unit="raw") -> DataPoint:
        return DataPoint(
            value=value, unit=unit,
            period=FiscalPeriod(kind="TTM", label="TTM-2026", end_date=date(2026, 9, 29)),
            as_of=date(2026, 9, 29),
        )

    def test_percent_and_decimal_ratio_are_not_confused(self) -> None:
        percent = normalize_data_point(self.point(D("20"), "percent"), "margin")
        ratio = normalize_data_point(self.point(D("0.20"), "decimal_ratio"), "margin")
        self.assertEqual(percent.value, D("0.2"))
        self.assertEqual(ratio.value, D("0.20"))

    def test_basis_points_normalize_to_decimal_ratio(self) -> None:
        self.assertEqual(normalize_data_point(self.point(D("125"), "basis_points"), "spread").value, D("0.0125"))

    def test_millions_normalize_to_base_units(self) -> None:
        self.assertEqual(normalize_data_point(self.point(D("2.5"), "millions"), "revenue").value, D("2500000.0"))

    def test_missing_value_returns_structured_missing_result(self) -> None:
        result = normalize_data_point(self.point(None, "millions"), "revenue")
        self.assertEqual(result.status, "missing_data")
        self.assertIsNone(result.value)

    def test_impossible_percentage_is_invalid(self) -> None:
        result = normalize_data_point(self.point(D("10001"), "percent"), "margin")
        self.assertEqual(result.status, "invalid_input")

    def test_unlike_periods_warn(self) -> None:
        fy = FiscalPeriod(kind="FY", label="FY2026", end_date=date(2026, 12, 31), fiscal_year=2026)
        ttm = FiscalPeriod(kind="TTM", label="TTM", end_date=date(2026, 9, 29))
        compatible, warning = periods_compatible(fy, ttm)
        self.assertFalse(compatible)
        self.assertIn("mismatch", warning)


class ValuationMetricTests(unittest.TestCase):
    def test_pe_and_forward_pe(self) -> None:
        self.assertEqual(metrics.pe(D("20"), D("4")).value, D("5"))
        self.assertEqual(metrics.forward_pe(D("20"), D("5")).value, D("4"))

    def test_negative_earnings_make_pe_not_applicable(self) -> None:
        result = metrics.pe(D("20"), D("-1"))
        self.assertEqual(result.status, "not_applicable")
        self.assertFalse(result.applicable)

    def test_peg_uses_growth_percent_not_raw_decimal(self) -> None:
        self.assertEqual(metrics.peg(D("20"), D("0.20")).value, D("1"))

    def test_ev_ebitda_rejects_zero_or_negative_ebitda(self) -> None:
        self.assertEqual(metrics.ev_to_ebitda(D("100"), D("0")).status, "not_applicable")
        self.assertEqual(metrics.ev_to_ebitda(D("100"), D("-5")).status, "not_applicable")

    def test_negative_fcf_yield_is_retained_with_warning(self) -> None:
        result = metrics.fcf_yield(D("-10"), D("100"))
        self.assertEqual(result.value, D("-0.1"))
        self.assertEqual(result.status, "warning")

    def test_missing_denominator_is_missing_data(self) -> None:
        self.assertEqual(metrics.price_to_sales(D("100"), None).status, "missing_data")


class QualityGrowthStrengthTests(unittest.TestCase):
    def test_quality_and_margin_formulas(self) -> None:
        self.assertEqual(metrics.roe(D("10"), D("50")).value, D("0.2"))
        self.assertEqual(metrics.roic(D("12"), D("60")).value, D("0.2"))
        self.assertEqual(metrics.gross_margin(D("40"), D("100")).value, D("0.4"))
        self.assertEqual(metrics.operating_margin(D("20"), D("100")).value, D("0.2"))
        self.assertEqual(metrics.net_margin(D("10"), D("100")).value, D("0.1"))
        self.assertEqual(metrics.fcf_margin(D("15"), D("100")).value, D("0.15"))

    def test_growth_from_negative_base_is_not_applicable(self) -> None:
        self.assertEqual(metrics.eps_growth(D("1"), D("-1")).status, "not_applicable")

    def test_growth_and_cagr(self) -> None:
        self.assertEqual(metrics.revenue_growth(D("120"), D("100")).value, D("0.2"))
        result = metrics.revenue_cagr(D("133.1"), D("100"), D("3"))
        self.assertAlmostEqual(float(result.value), 0.1, places=10)

    def test_financial_strength_and_net_cash(self) -> None:
        self.assertEqual(metrics.debt_to_equity(D("20"), D("40")).value, D("0.5"))
        self.assertEqual(metrics.net_debt(D("20"), D("30")).value, D("-10"))
        self.assertEqual(metrics.interest_coverage(D("50"), D("10")).value, D("5"))
        self.assertEqual(metrics.current_ratio(D("60"), D("30")).value, D("2"))
        self.assertEqual(metrics.quick_ratio(D("20"), D("10"), D("30")).value, D("1"))


class EstablishedFrameworkTests(unittest.TestCase):
    def test_piotroski_f_score(self) -> None:
        names = [
            "positive_roa", "positive_cfo", "roa_improved", "cfo_exceeds_income",
            "leverage_declined", "current_ratio_improved", "no_new_shares",
            "gross_margin_improved", "asset_turnover_improved",
        ]
        result = frameworks.piotroski_f_score({name: name != "no_new_shares" for name in names})
        self.assertEqual(result.value, D("8"))

    def test_piotroski_missing_signal_fails_safely(self) -> None:
        self.assertEqual(frameworks.piotroski_f_score({}).status, "missing_data")

    def test_altman_and_beneish_return_documented_warnings(self) -> None:
        altman = frameworks.altman_z_score(D("10"), D("100"), D("20"), D("15"), D("80"), D("40"), D("120"))
        beneish = frameworks.beneish_m_score(*([D("1")] * 8))
        self.assertEqual(altman.status, "warning")
        self.assertEqual(beneish.status, "warning")
        self.assertIsNotNone(altman.value)
        self.assertIsNotNone(beneish.value)

    def test_rule_of_40_and_graham_number(self) -> None:
        self.assertEqual(frameworks.rule_of_40(D("0.30"), D("0.15")).value, D("45.00"))
        self.assertEqual(frameworks.graham_number(D("4"), D("10")).value.quantize(D("0.01")), D("30.00"))

    def test_magic_formula_components(self) -> None:
        self.assertEqual(frameworks.magic_formula_earnings_yield(D("10"), D("100")).value, D("0.1"))
        self.assertEqual(frameworks.magic_formula_return_on_capital(D("10"), D("20"), D("30")).value, D("0.2"))


class GrahamInputValidityTests(unittest.TestCase):
    def assert_unavailable(self, eps: Decimal, book_value: Decimal) -> None:
        result = frameworks.graham_number(eps, book_value)
        self.assertEqual(result.status, "not_applicable")
        self.assertFalse(result.applicable)
        self.assertIsNone(result.value)
        self.assertIn("positive EPS and book value per share", result.warning)

    def test_a_positive_inputs_preserve_existing_graham_number(self) -> None:
        result = frameworks.graham_number(D("4"), D("10"))
        self.assertEqual(result.status, "warning")
        self.assertEqual(result.value, D("30"))

    def test_b_two_negative_inputs_are_not_applicable(self) -> None:
        self.assert_unavailable(D("-2"), D("-10"))

    def test_c_negative_eps_is_not_applicable(self) -> None:
        self.assert_unavailable(D("-2"), D("10"))

    def test_d_negative_book_value_is_not_applicable(self) -> None:
        self.assert_unavailable(D("2"), D("-10"))

    def test_e_zero_eps_is_not_applicable(self) -> None:
        self.assert_unavailable(D("0"), D("10"))

    def test_f_zero_book_value_is_not_applicable(self) -> None:
        self.assert_unavailable(D("2"), D("0"))

    def test_g_missing_eps_preserves_missing_data(self) -> None:
        result = frameworks.graham_number(None, D("10"))
        self.assertEqual(result.status, "missing_data")
        self.assertIsNone(result.value)

    def test_h_missing_book_value_preserves_missing_data(self) -> None:
        result = frameworks.graham_number(D("2"), None)
        self.assertEqual(result.status, "missing_data")
        self.assertIsNone(result.value)

    def test_i_identical_inputs_produce_identical_results(self) -> None:
        self.assertEqual(
            frameworks.graham_number(D("-2"), D("-10")),
            frameworks.graham_number(D("-2"), D("-10")),
        )

    def test_nonfinite_inputs_are_unavailable_without_numeric_output(self) -> None:
        for eps, book_value in ((D("NaN"), D("10")), (D("2"), D("Infinity"))):
            with self.subTest(eps=eps, book_value=book_value):
                result = frameworks.graham_number(eps, book_value)
                self.assertEqual(result.status, "not_applicable")
                self.assertIsNone(result.value)


if __name__ == "__main__":
    unittest.main()
