from __future__ import annotations

import unittest
from datetime import date, datetime, timezone
from decimal import Decimal

from pydantic import ValidationError

from market_analytics.exchange_calendar import (
    US_EQUITY_CALENDAR_POLICY_VERSION,
    us_equity_calendar,
)
from market_analytics.forward_paper import (
    DELISTING_POLICY_VERSION,
    HORIZON_POLICY_VERSION,
    RETURN_DEFINITION_VERSION,
    EvaluationPriceBar,
    SplitBasisCoverage,
    TerminalEventEvidence,
    build_paper_rehearsal,
    evaluate_horizon_outcome,
    evaluate_rehearsal_outcomes,
    resolve_evaluation_schedule,
)
from tests.test_forward_paper import CUTOFF, security


AFTER_ONE_WEEK_CLOSE = datetime(2026, 10, 12, 22, tzinfo=timezone.utc)


def decision(ticker: str = "AAPL", exchange: str = "XNAS"):
    frozen = security(ticker).model_copy(update={"exchange": exchange})
    return build_paper_rehearsal(
        [frozen], decision_timestamp=CUTOFF
    ).decisions[0]


def bar(
    ticker: str = "AAPL", *, trading_date: date = date(2026, 10, 12),
    close: Decimal | None = Decimal("55"), exchange: str = "XNAS",
) -> EvaluationPriceBar:
    return EvaluationPriceBar(
        ticker=ticker, exchange=exchange, trading_date=trading_date,
        split_adjusted_close=close, regular_session=True,
        available_at=AFTER_ONE_WEEK_CLOSE,
        provider_snapshot_id=f"provider:{ticker}:{trading_date.isoformat()}",
    )


def event(event_type: str, **updates: object) -> TerminalEventEvidence:
    values: dict[str, object] = {
        "event_type": event_type,
        "effective_date": date(2026, 10, 9),
        "available_at": datetime(2026, 10, 9, 20, tzinfo=timezone.utc),
        "source_ids": (f"official:{event_type}:1",),
    }
    values.update(updates)
    return TerminalEventEvidence.model_validate(values)


def split_coverage(owner, end: date, *, available_at=AFTER_ONE_WEEK_CLOSE):
    return SplitBasisCoverage(
        requested_start=owner.decision_price_date + date.resolution,
        requested_end=end,
        complete=True,
        available_at=available_at,
        source_ids=(f"fixture:split-coverage:{end.isoformat()}",),
    )


class UsEquityCalendarPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.xnas = us_equity_calendar("NASDAQ")
        self.xnys = us_equity_calendar("NYSE")

    def test_supported_exchange_identity_and_shared_policy(self) -> None:
        self.assertEqual(self.xnas.exchange, "XNAS")
        self.assertEqual(self.xnys.exchange, "XNYS")
        self.assertEqual(self.xnas.policy_version, US_EQUITY_CALENDAR_POLICY_VERSION)
        self.assertEqual(self.xnas.holidays, self.xnys.holidays)
        with self.assertRaisesRegex(ValueError, "supports only"):
            us_equity_calendar("XLON")

    def test_a_b_weekend_targets_resolve_to_next_session(self) -> None:
        self.assertEqual(
            self.xnas.next_session_on_or_after(date(2026, 10, 10)).trading_date,
            date(2026, 10, 12),
        )
        self.assertEqual(
            self.xnas.next_session_on_or_after(date(2026, 10, 11)).trading_date,
            date(2026, 10, 12),
        )

    def test_c_d_holiday_and_normal_session_resolution(self) -> None:
        thanksgiving = self.xnas.next_session_on_or_after(date(2026, 11, 26))
        self.assertEqual(thanksgiving.trading_date, date(2026, 11, 27))
        self.assertEqual(thanksgiving.kind, "EARLY_CLOSE")
        normal = self.xnas.next_session_on_or_after(date(2026, 11, 25))
        self.assertEqual(normal.trading_date, date(2026, 11, 25))
        self.assertEqual(normal.kind, "FULL")

    def test_e_f_holiday_weekend_and_year_boundary(self) -> None:
        self.assertEqual(
            self.xnas.next_session_on_or_after(date(2026, 12, 25)).trading_date,
            date(2026, 12, 28),
        )
        self.assertEqual(
            self.xnas.next_session_on_or_after(date(2027, 1, 1)).trading_date,
            date(2027, 1, 4),
        )

    def test_g_leap_day_is_an_ordinary_session_when_applicable(self) -> None:
        session = self.xnas.next_session_on_or_after(date(2028, 2, 29))
        self.assertEqual(session.trading_date, date(2028, 2, 29))
        self.assertEqual(session.kind, "FULL")

    def test_h_early_close_is_a_valid_regular_session_close(self) -> None:
        session = self.xnas.session(date(2026, 11, 27))
        self.assertEqual(session.kind, "EARLY_CLOSE")
        self.assertEqual(session.closes_at.hour, 18)

    def test_calendar_search_is_explicitly_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside the resolution bound"):
            self.xnas.next_session_on_or_after(
                date(2026, 11, 26), max_calendar_days=0
            )

    def test_aapl_offline_schedule_uses_calendar_offsets_then_sessions(self) -> None:
        schedule = resolve_evaluation_schedule(
            datetime(2026, 10, 4, 3, 40, 7, 277009, tzinfo=timezone.utc), "XNAS"
        )
        self.assertEqual(schedule, {
            "one_trading_week": date(2026, 10, 12),
            "one_month": date(2026, 11, 3),
            "three_months": date(2027, 1, 4),
        })


class DelistingAndEvaluationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calendar = us_equity_calendar("XNAS")
        self.aapl = decision()

    def evaluate(self, **updates: object):
        values = {
            "decision": self.aapl,
            "horizon": "one_trading_week",
            "calendar_owner": self.calendar,
            "evaluation_timestamp": AFTER_ONE_WEEK_CLOSE,
            "bars": (bar(),),
            "split_basis_coverage": split_coverage(
                self.aapl, self.aapl.evaluation_horizons[0].resolved_session_date
            ),
        }
        values.update(updates)
        return evaluate_horizon_outcome(**values)

    def test_e_missing_bar_is_provider_gap_not_market_closed(self) -> None:
        result = self.evaluate(bars=())
        self.assertEqual(result.status, "provider_gap")
        self.assertIn("provider_bar_missing", result.reason_codes)

    def test_f_next_day_bar_does_not_hide_expected_session_gap(self) -> None:
        result = self.evaluate(
            bars=(bar(trading_date=date(2026, 10, 13)),)
        )
        self.assertEqual(result.status, "provider_gap")
        self.assertEqual(result.resolved_session_date, date(2026, 10, 12))

    def test_missing_split_adjusted_value_is_explicit_missing_price(self) -> None:
        result = self.evaluate(bars=(bar(close=None),))
        self.assertEqual(result.status, "missing_price")
        self.assertIsNone(result.price_return)

    def test_early_close_bar_is_accepted_by_the_outcome_evaluator(self) -> None:
        horizons = tuple(
            item.model_copy(update={
                "target_not_before": date(2026, 11, 27),
                "resolved_session_date": date(2026, 11, 27),
            })
            if item.name == "one_trading_week" else item
            for item in self.aapl.evaluation_horizons
        )
        early_decision = self.aapl.model_copy(update={"evaluation_horizons": horizons})
        early_bar = EvaluationPriceBar(
            ticker="AAPL", exchange="XNAS", trading_date=date(2026, 11, 27),
            split_adjusted_close=Decimal("55"), regular_session=True,
            available_at=datetime(2026, 11, 27, 18, 30, tzinfo=timezone.utc),
            provider_snapshot_id="provider:AAPL:2026-11-27",
        )
        result = evaluate_horizon_outcome(
            early_decision, horizon="one_trading_week",
            calendar_owner=self.calendar,
            evaluation_timestamp=datetime(2026, 11, 27, 19, tzinfo=timezone.utc),
            bars=(early_bar,),
            split_basis_coverage=split_coverage(
                early_decision, date(2026, 11, 27),
                available_at=datetime(2026, 11, 27, 18, 30, tzinfo=timezone.utc),
            ),
        )
        self.assertEqual(result.status, "evaluated")
        self.assertEqual(result.resolved_session_date, date(2026, 11, 27))

    def test_g_verified_cash_acquisition_uses_terminal_value(self) -> None:
        acquisition = event(
            "cash_acquisition", independently_verified=True,
            identity_continuity_verified=True,
            terminal_cash_value=Decimal("60"),
        )
        result = self.evaluate(bars=(), terminal_event=acquisition)
        self.assertEqual(result.status, "terminal_value_verified")
        self.assertEqual(result.evaluation_value_type, "terminal_cash_consideration")
        self.assertEqual(result.price_return, Decimal("0.2"))

    def test_unverified_cash_consideration_remains_unresolved(self) -> None:
        acquisition = event(
            "cash_acquisition", identity_continuity_verified=True,
            terminal_cash_value=Decimal("60"),
        )
        result = self.evaluate(bars=(), terminal_event=acquisition)
        self.assertEqual(result.status, "inactive_unresolved")
        self.assertIsNone(result.evaluation_value)

    def test_later_terminal_evidence_does_not_leak_backward(self) -> None:
        acquisition = event(
            "cash_acquisition", independently_verified=True,
            identity_continuity_verified=True, terminal_cash_value=Decimal("60"),
            available_at=AFTER_ONE_WEEK_CLOSE.replace(day=13),
        )
        result = self.evaluate(bars=(), terminal_event=acquisition)
        self.assertEqual(result.status, "provider_gap")
        self.assertNotIn("verified_cash_acquisition", result.reason_codes)

    def test_h_delisted_or_inactive_without_terminal_value_is_unresolved(self) -> None:
        for kind in ("delisting", "bankruptcy", "liquidation", "inactive_unknown"):
            with self.subTest(kind=kind):
                result = self.evaluate(bars=(), terminal_event=event(kind))
                self.assertEqual(result.status, "inactive_unresolved")
                self.assertIsNone(result.evaluation_value)

    def test_i_verified_ticker_continuity_allows_successor_bar(self) -> None:
        fb = decision("FB")
        transition = event(
            "ticker_change", identity_continuity_verified=True,
            old_ticker="FB", new_ticker="META",
        )
        result = self.evaluate(
            decision=fb, bars=(bar("META"),), terminal_event=transition
        )
        self.assertEqual(result.status, "evaluated")
        self.assertEqual(result.successor_ticker, "META")
        self.assertIn("verified_ticker_continuity", result.reason_codes)

    def test_j_unverified_ticker_transition_is_rejected(self) -> None:
        result = self.evaluate(
            bars=(bar("META"),),
            terminal_event=event(
                "ticker_change", old_ticker="FB", new_ticker="META"
            ),
        )
        self.assertEqual(result.status, "unsupported_identity_transition")
        self.assertIsNone(result.evaluation_value)

    def test_temporary_halt_requires_evidence_and_remains_pending(self) -> None:
        result = self.evaluate(bars=(), terminal_event=event("temporary_halt"))
        self.assertEqual(result.status, "pending")
        self.assertEqual(result.reason_codes, ("temporary_halt",))

    def test_k_no_survivor_bias_dropping(self) -> None:
        atvi = decision("ATVI")
        results = evaluate_rehearsal_outcomes(
            (self.aapl, atvi),
            horizon="one_trading_week",
            calendars={"XNAS": self.calendar},
            evaluation_timestamp=AFTER_ONE_WEEK_CLOSE,
            bars_by_decision={self.aapl.decision_id: (bar(),)},
            events_by_decision={
                atvi.decision_id: event("inactive_unknown")
            },
            split_coverage_by_decision={
                self.aapl.decision_id: split_coverage(
                    self.aapl, self.aapl.evaluation_horizons[0].resolved_session_date
                )
            },
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(
            {item.ticker: item.status for item in results},
            {"AAPL": "evaluated", "ATVI": "inactive_unresolved"},
        )

    def test_l_m_decision_and_policy_versions_are_frozen(self) -> None:
        horizon = self.aapl.evaluation_horizons[0]
        self.assertEqual(horizon.calendar_policy_version, US_EQUITY_CALENDAR_POLICY_VERSION)
        self.assertEqual(horizon.horizon_policy_version, HORIZON_POLICY_VERSION)
        self.assertEqual(horizon.delisting_policy_version, DELISTING_POLICY_VERSION)
        self.assertEqual(horizon.return_definition_version, RETURN_DEFINITION_VERSION)
        with self.assertRaises(ValidationError):
            horizon.delisting_policy_version = "changed"
        with self.assertRaises(ValidationError):
            self.aapl.decision_split_adjusted_close = Decimal("999")

    def test_n_o_price_return_label_is_preserved_without_total_return_claim(self) -> None:
        result = self.evaluate()
        self.assertEqual(result.status, "evaluated")
        self.assertEqual(result.return_type, "price_return")
        self.assertFalse(result.total_return_included)
        self.assertEqual(result.price_return, Decimal("0.1"))


if __name__ == "__main__":
    unittest.main()
