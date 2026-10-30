from __future__ import annotations

import unittest
from decimal import Decimal

from market_analytics.dcf import DCFScenario, calculate_dcf, run_scenarios, sensitivity_matrix
from market_analytics.peers import compare_metric, compare_scoped, percentile_rank


D = Decimal


def scenario(**updates) -> DCFScenario:
    values = {
        "name": "Base", "starting_fcf": D("100"),
        "growth_rates": [D("0.1"), D("0.08"), D("0.06")], "projection_period": 3,
        "discount_rate": D("0.1"), "terminal_growth_rate": D("0.03"),
        "net_debt": D("20"), "shares_outstanding": D("10"), "currency": "USD",
    }
    values.update(updates)
    return DCFScenario(**values)


class DCFTests(unittest.TestCase):
    def test_complete_dcf_has_all_outputs(self) -> None:
        result = calculate_dcf(scenario())
        self.assertTrue(result.complete)
        self.assertEqual(len(result.projections), 3)
        self.assertIsNotNone(result.terminal_value)
        self.assertIsNotNone(result.enterprise_value)
        self.assertIsNotNone(result.value_per_share)

    def test_bear_base_bull_are_supported(self) -> None:
        values = run_scenarios([
            scenario(name="Bear", growth_rates=[D("0"), D("0"), D("0")]),
            scenario(name="Base"),
            scenario(name="Bull", growth_rates=[D("0.2"), D("0.15"), D("0.1")]),
        ])
        self.assertEqual([item.scenario for item in values], ["Bear", "Base", "Bull"])
        self.assertLess(values[0].value_per_share, values[1].value_per_share)
        self.assertLess(values[1].value_per_share, values[2].value_per_share)

    def test_zero_terminal_growth_is_valid(self) -> None:
        self.assertTrue(calculate_dcf(scenario(terminal_growth_rate=D("0"))).complete)

    def test_discount_rate_must_exceed_terminal_growth(self) -> None:
        result = calculate_dcf(scenario(discount_rate=D("0.03"), terminal_growth_rate=D("0.03")))
        self.assertFalse(result.complete)

    def test_missing_shares_returns_incomplete_result(self) -> None:
        result = calculate_dcf(scenario(shares_outstanding=None))
        self.assertFalse(result.complete)
        self.assertIn("shares_outstanding", result.warnings[0])

    def test_net_cash_increases_equity_value(self) -> None:
        debt = calculate_dcf(scenario(net_debt=D("20")))
        cash = calculate_dcf(scenario(net_debt=D("-20")))
        self.assertGreater(cash.equity_value, debt.equity_value)

    def test_growth_length_must_match_projection_period(self) -> None:
        result = calculate_dcf(scenario(growth_rates=[D("0.1")]))
        self.assertFalse(result.complete)

    def test_sensitivity_matrix_dimensions_and_invalid_cells(self) -> None:
        matrix = sensitivity_matrix(
            scenario(), [D("0.08"), D("0.10")], [D("0.02"), D("0.10")]
        )
        self.assertEqual(len(matrix.values), 2)
        self.assertEqual(len(matrix.values[0]), 2)
        self.assertIsNone(matrix.values[1][1])


class PeerNormalizationTests(unittest.TestCase):
    def test_higher_is_better_percentile(self) -> None:
        self.assertEqual(percentile_rank(D("30"), [D("10"), D("20"), D("30")], "higher"), D("83.33333333333333333333333333"))

    def test_lower_is_better_inverts_percentile(self) -> None:
        low = percentile_rank(D("10"), [D("10"), D("20"), D("30")], "lower")
        high = percentile_rank(D("30"), [D("10"), D("20"), D("30")], "lower")
        self.assertGreater(low, high)

    def test_ties_share_same_percentile(self) -> None:
        first = percentile_rank(D("10"), [D("10"), D("10"), D("20")], "higher")
        second = percentile_rank(D("10"), [D("10"), D("10"), D("20")], "higher")
        self.assertEqual(first, second)

    def test_all_missing_returns_missing_comparison(self) -> None:
        result = compare_metric("AAA", "ROIC", None, {"AAA": None, "BBB": None}, "higher")
        self.assertEqual(result.status, "missing_data")

    def test_zero_variance_has_no_z_score(self) -> None:
        result = compare_metric("AAA", "ROIC", D("1"), {"AAA": D("1"), "BBB": D("1")}, "higher")
        self.assertIsNone(result.z_score)

    def test_outlier_has_positive_favorable_z_score(self) -> None:
        result = compare_metric("AAA", "ROIC", D("100"), {"AAA": D("100"), "BBB": D("1"), "CCC": D("2")}, "higher")
        self.assertGreater(result.z_score, 0)

    def test_sector_and_industry_scopes_use_matching_groups(self) -> None:
        values = {"AAA": {"ROIC": D("0.1")}, "BBB": {"ROIC": D("0.2")}, "CCC": {"ROIC": D("0.9")}}
        metadata = {
            "AAA": {"sector": "Tech", "industry": "Software"},
            "BBB": {"sector": "Tech", "industry": "Hardware"},
            "CCC": {"sector": "Health", "industry": "Biotech"},
        }
        sector = compare_scoped(values, {"ROIC": "higher"}, metadata, "sector")
        industry = compare_scoped(values, {"ROIC": "higher"}, metadata, "industry")
        self.assertLess(sector["AAA"]["ROIC"].percentile, sector["BBB"]["ROIC"].percentile)
        self.assertIsNone(industry["AAA"]["ROIC"].percentile)
        self.assertIn("insufficient_peer_set", industry["AAA"]["ROIC"].warning)

    def test_unknown_scopes_fail_closed_instead_of_cross_industry_comparison(self) -> None:
        values = {"AAA": {"ROIC": D("0.1")}, "BBB": {"ROIC": D("0.9")}}
        metadata = {
            "AAA": {"sector": "Unknown", "industry": "Unknown"},
            "BBB": {"sector": "Unknown", "industry": "Unknown"},
        }
        for scope in ("sector", "industry"):
            comparisons = compare_scoped(values, {"ROIC": "higher"}, metadata, scope)
            self.assertIsNone(comparisons["AAA"]["ROIC"].percentile)
            self.assertIsNone(comparisons["BBB"]["ROIC"].percentile)
            self.assertIn("insufficient_peer_set", comparisons["AAA"]["ROIC"].warning)


if __name__ == "__main__":
    unittest.main()
