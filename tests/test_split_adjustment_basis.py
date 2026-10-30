from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from pydantic import ValidationError

from market_analytics.exchange_calendar import us_equity_calendar
from market_analytics.forward_paper import (
    ADJUSTMENT_BASIS_POLICY_VERSION,
    EvaluationPriceBar,
    SplitAdjustmentEvidence,
    SplitBasisCoverage,
    TerminalEventEvidence,
    evaluate_horizon_outcome,
    evaluate_price_return,
)
from market_analytics.models import CorporateActionRecord
from market_analytics.paper_persistence import proposed_evaluation_identity
from tests.test_forward_policy import decision


EVALUATED_AT = datetime(2026, 10, 12, 22, 0, tzinfo=timezone.utc)


class SplitAdjustmentBasisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decision = decision()
        self.horizon = self.decision.evaluation_horizons[0]
        self.session_date = self.horizon.resolved_session_date
        self.calendar = us_equity_calendar(self.decision.exchange)

    def bar(self, *, ticker: str = "AAPL", value: str = "55") -> EvaluationPriceBar:
        return EvaluationPriceBar(
            ticker=ticker,
            exchange="XNAS",
            trading_date=self.session_date,
            split_adjusted_close=Decimal(value),
            regular_session=True,
            available_at=EVALUATED_AT,
            provider_snapshot_id=f"fixture:{ticker}:{self.session_date}",
        )

    def split(
        self, identity: str, *, kind: str = "stock_split", ratio: str = "2",
        effective_date: date | None = None, available_at: datetime = EVALUATED_AT,
        ticker: str = "AAPL",
    ) -> SplitAdjustmentEvidence:
        return SplitAdjustmentEvidence(
            ticker=ticker,
            event_type=kind,
            effective_date=effective_date or date(2026, 10, 8),
            ratio=Decimal(ratio),
            available_at=available_at,
            source_id=identity,
        )

    def coverage(
        self, *evidence: SplitAdjustmentEvidence, complete: bool = True,
        available_at: datetime = EVALUATED_AT,
    ) -> SplitBasisCoverage:
        return SplitBasisCoverage(
            requested_start=self.decision.decision_price_date + timedelta(days=1),
            requested_end=self.session_date,
            complete=complete,
            available_at=available_at,
            source_ids=("fixture:complete-split-page",) if complete else (),
            evidence=evidence,
        )

    def evaluate(self, *, coverage: SplitBasisCoverage | None, value: str = "55", **kwargs):
        return evaluate_horizon_outcome(
            self.decision,
            horizon=self.horizon.name,
            calendar_owner=self.calendar,
            evaluation_timestamp=EVALUATED_AT,
            bars=(self.bar(value=value, ticker=kwargs.pop("ticker", "AAPL")),),
            split_basis_coverage=coverage,
            **kwargs,
        )

    def test_a_no_intervening_split_computes_ten_percent(self) -> None:
        result = self.evaluate(coverage=self.coverage())
        self.assertEqual((result.status, result.price_return), ("evaluated", Decimal("0.1")))
        self.assertEqual(result.adjustment_basis_policy_version, ADJUSTMENT_BASIS_POLICY_VERSION)
        self.assertFalse(result.rebasing_applied)

    def test_b_two_for_one_unchanged_fails_closed_not_minus_fifty_percent(self) -> None:
        result = self.evaluate(
            coverage=self.coverage(self.split("split:2-for-1")), value="25"
        )
        self.assertEqual(result.status, "unsupported_adjustment_basis")
        self.assertIsNone(result.price_return)
        self.assertNotEqual(result.price_return, Decimal("-0.5"))

    def test_c_two_for_one_plus_real_gain_still_fails_closed(self) -> None:
        result = self.evaluate(
            coverage=self.coverage(self.split("split:gain")), value="27.5"
        )
        self.assertEqual(result.reason_codes[0], "intervening_split_detected")
        self.assertIsNone(result.evaluation_value)

    def test_d_reverse_split_fails_closed(self) -> None:
        result = self.evaluate(
            coverage=self.coverage(self.split(
                "split:reverse", kind="reverse_split", ratio="0.1"
            ))
        )
        self.assertEqual(result.status, "unsupported_adjustment_basis")

    def test_e_multiple_splits_preserve_all_evidence_and_fail_closed(self) -> None:
        result = self.evaluate(coverage=self.coverage(
            self.split("split:first", effective_date=date(2026, 10, 6)),
            self.split("split:second", ratio="3", effective_date=date(2026, 10, 9)),
        ))
        self.assertEqual(
            result.intervening_split_source_ids, ("split:first", "split:second")
        )
        self.assertIsNone(result.cumulative_split_factor)

    def test_f_missing_split_coverage_computes_no_return(self) -> None:
        result = self.evaluate(coverage=None)
        self.assertEqual(result.reason_codes, ("split_coverage_unavailable",))
        self.assertIsNone(result.price_return)

    def test_g_future_split_evidence_invalidates_coverage(self) -> None:
        result = self.evaluate(coverage=self.coverage(self.split(
            "split:future-known",
            available_at=EVALUATED_AT + timedelta(seconds=1),
        )))
        self.assertEqual(result.reason_codes[0], "split_coverage_unavailable")
        self.assertIsNone(result.price_return)

    def test_h_split_outside_interval_does_not_affect_return(self) -> None:
        result = self.evaluate(coverage=self.coverage(self.split(
            "split:outside", effective_date=self.decision.decision_price_date
        )))
        self.assertEqual((result.status, result.price_return), ("evaluated", Decimal("0.1")))

    def test_i_cash_dividend_is_not_split_evidence_and_return_stays_price_only(self) -> None:
        dividend = CorporateActionRecord(
            ticker="AAPL", event_type="cash_dividend",
            ex_date=date(2026, 10, 8), amount=Decimal("0.25"), currency="USD",
            source="fixture", source_record_id="dividend:1",
            observed_at=EVALUATED_AT, available_at=EVALUATED_AT,
            retrieved_at=EVALUATED_AT,
        )
        with self.assertRaises(ValidationError):
            SplitAdjustmentEvidence.model_validate({
                "ticker": dividend.ticker,
                "event_type": dividend.event_type,
                "effective_date": dividend.ex_date,
                "ratio": Decimal("1"),
                "available_at": dividend.available_at,
                "source_id": dividend.source_record_id,
            })
        result = self.evaluate(coverage=self.coverage())
        self.assertEqual(result.price_return, Decimal("0.1"))
        self.assertFalse(result.total_return_included)

    def test_j_verified_ticker_continuity_does_not_override_split_failure(self) -> None:
        transition = TerminalEventEvidence(
            event_type="ticker_change", effective_date=date(2026, 10, 7),
            available_at=EVALUATED_AT, source_ids=("ticker:change",),
            identity_continuity_verified=True, old_ticker="AAPL", new_ticker="APPL",
        )
        result = self.evaluate(
            coverage=self.coverage(self.split("split:with-ticker", ticker="APPL")),
            ticker="APPL", terminal_event=transition,
        )
        self.assertEqual(result.status, "unsupported_adjustment_basis")
        self.assertIn("verified_ticker_continuity", result.reason_codes)
        self.assertEqual(result.successor_ticker, "APPL")

    def test_k_adjustment_basis_changes_canonical_event_identity(self) -> None:
        supported = self.evaluate(coverage=self.coverage())
        unsupported = self.evaluate(coverage=self.coverage(self.split("split:identity")))
        args = {
            "evaluation_timestamp": EVALUATED_AT,
            "correction_of_event_id": None,
            "provenance": {"provider_request": ("fixture",)},
        }
        supported_id, supported_hash, _ = proposed_evaluation_identity(supported, **args)
        unsupported_id, unsupported_hash, _ = proposed_evaluation_identity(unsupported, **args)
        self.assertNotEqual((supported_id, supported_hash), (unsupported_id, unsupported_hash))

    def test_l_low_level_return_helper_requires_explicit_basis_proof(self) -> None:
        with self.assertRaisesRegex(ValueError, "common split-adjustment basis"):
            evaluate_price_return(
                self.decision, horizon=self.horizon.name,
                future_split_adjusted_close=Decimal("55"),
            )

    def test_m_incomplete_coverage_computes_no_return(self) -> None:
        result = self.evaluate(coverage=self.coverage(complete=False))
        self.assertEqual(result.status, "unsupported_adjustment_basis")
        self.assertIsNone(result.price_return)


if __name__ == "__main__":
    unittest.main()
