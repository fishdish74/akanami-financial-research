from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from pydantic import ValidationError

from market_analytics.capabilities import (
    active_us_stock_capability,
    evaluate_security_capability,
)
from market_analytics.forward_paper import (
    DecisionPriceAudit,
    FrozenFundamentalInput,
    FrozenMarketInput,
    FrozenMetricContribution,
    PaperSecurityInput,
    build_paper_rehearsal,
    evaluate_price_return,
)


CUTOFF = datetime(2026, 10, 3, 20, 0, tzinfo=timezone.utc)


def security(
    ticker: str = "AAPL",
    *,
    score: Decimal | None = Decimal("70"),
    ranking_status: str = "ranked",
    capability=None,
    market_available: datetime = CUTOFF - timedelta(minutes=1),
    fact_available: datetime = CUTOFF - timedelta(days=30),
) -> PaperSecurityInput:
    return PaperSecurityInput(
        company_id={"AAPL": 1, "JPM": 2, "COST": 3, "NVDA": 4}.get(ticker, 9),
        ticker=ticker,
        company=f"{ticker} Company",
        exchange="XNAS",
        capability=capability or active_us_stock_capability(),
        market=FrozenMarketInput(
            raw_close=Decimal("100"),
            split_adjusted_close=Decimal("50"),
            price_date=date(2026, 10, 2),
            available_at=market_available,
            retrieved_at=market_available,
            volume=Decimal("100.4"),
            currency="USD",
            provider_snapshot_ids=(f"massive:{ticker}:raw_pages:{'f' * 64}",),
            decision_price_audit=DecisionPriceAudit(
                expected_session_date=date(2026, 10, 2),
                selected_session_date=date(2026, 10, 2),
                provider_response_statuses=("OK",),
                returned_bar_dates=(date(2026, 10, 2),),
                returned_bar_timestamps_ms=(1790900000000,),
                page_hashes=("f" * 64,),
                retrieved_at=market_available,
            ),
        ),
        fundamentals=(FrozenFundamentalInput(
            metric="revenue",
            value=Decimal("1000000"),
            unit="raw",
            currency="USD",
            period_type="FY",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            available_at=fact_available,
            source="SEC:test",
            source_artifact_id=1,
            quality_status="valid",
        ),),
        ranking_status=ranking_status,
        opportunity_score=score,
        metric_contributions=(
            FrozenMetricContribution(name="quality", value=Decimal("75")),
        ),
        confidence="medium",
        missing_or_blocked_reasons=(),
        evidence_summary=("Revenue input is frozen from an eligible SEC fact.",),
        source_artifact_ids=(1,),
        analytics_run_id="analytics-run",
        ranking_version="ranking-v1",
        pipeline_version="analytics-v1",
        calculation_version="financial-math-v1",
        config_hash="c" * 64,
    )


class ForwardPaperFreezeTests(unittest.TestCase):
    def test_future_sec_fact_is_excluded(self) -> None:
        with self.assertRaisesRegex(ValueError, "future input"):
            build_paper_rehearsal(
                [security(fact_available=CUTOFF + timedelta(microseconds=1))],
                decision_timestamp=CUTOFF,
            )

    def test_future_price_or_retrieval_is_excluded(self) -> None:
        with self.assertRaisesRegex(ValueError, "future input"):
            build_paper_rehearsal(
                [security(market_available=CUTOFF + timedelta(microseconds=1))],
                decision_timestamp=CUTOFF,
            )

    def test_exact_aware_cutoff_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            build_paper_rehearsal(
                [security()], decision_timestamp=CUTOFF.replace(tzinfo=None)
            )

    def test_later_correction_cannot_mutate_frozen_decision(self) -> None:
        rehearsal = build_paper_rehearsal([security()], decision_timestamp=CUTOFF)
        decision = rehearsal.decisions[0]
        with self.assertRaises(ValidationError):
            decision.decision_split_adjusted_close = Decimal("999")
        with self.assertRaisesRegex(ValueError, "future input"):
            build_paper_rehearsal(
                [security(fact_available=CUTOFF + timedelta(days=1))],
                decision_timestamp=CUTOFF,
            )
        self.assertEqual(decision.decision_split_adjusted_close, Decimal("50"))

    def test_identical_run_has_identical_ids_hashes_and_schedule(self) -> None:
        first = build_paper_rehearsal([security()], decision_timestamp=CUTOFF)
        second = build_paper_rehearsal([security()], decision_timestamp=CUTOFF)
        self.assertEqual(first.rehearsal_id, second.rehearsal_id)
        self.assertEqual(first.frozen_payload_hash, second.frozen_payload_hash)
        self.assertEqual(first.decisions[0].decision_id, second.decisions[0].decision_id)
        self.assertEqual(first.evaluation_horizons, second.evaluation_horizons)

    def test_input_order_does_not_change_ids_or_selection(self) -> None:
        values = [security("AAPL", score=Decimal("70")), security("NVDA", score=Decimal("80"))]
        first = build_paper_rehearsal(values, decision_timestamp=CUTOFF)
        second = build_paper_rehearsal(list(reversed(values)), decision_timestamp=CUTOFF)
        self.assertEqual(first, second)


class ForwardPaperSelectionTests(unittest.TestCase):
    def test_blocked_capability_cannot_enter_selection_or_carry_numeric_rank(self) -> None:
        blocked_capability = evaluate_security_capability(
            current_metadata_available=True,
            active=True,
            listing_type="adr",
            foreign_issuer=True,
            fundamental_basis="ifrs",
        )
        blocked = security(
            "NVDA", score=None, ranking_status="blocked", capability=blocked_capability
        )
        result = build_paper_rehearsal(
            [security("AAPL"), blocked], decision_timestamp=CUTOFF
        )
        decision = next(item for item in result.decisions if item.ticker == "NVDA")
        self.assertEqual(decision.selection_status, "blocked")
        self.assertEqual(decision.eligibility_status, "blocked")
        self.assertIsNone(decision.opportunity_score)
        self.assertIn("unsupported_ifrs", decision.capability_reason_codes)

    def test_only_top_two_ranked_eligible_names_are_selected(self) -> None:
        result = build_paper_rehearsal(
            [
                security("AAPL", score=Decimal("60")),
                security("JPM", score=Decimal("80")),
                security("COST", score=Decimal("70")),
            ],
            decision_timestamp=CUTOFF,
        )
        selected = {
            item.ticker for item in result.decisions
            if item.selection_status == "selected_for_tracking"
        }
        self.assertEqual(selected, {"JPM", "COST"})

    def test_insufficient_evidence_is_eligible_but_not_selected(self) -> None:
        result = build_paper_rehearsal(
            [security(score=None, ranking_status="insufficient_evidence")],
            decision_timestamp=CUTOFF,
        )
        decision = result.decisions[0]
        self.assertEqual(decision.eligibility_status, "eligible")
        self.assertEqual(decision.selection_status, "not_selected")


class ForwardPaperEvaluationTests(unittest.TestCase):
    def test_return_is_explicit_price_return_without_total_return_claim(self) -> None:
        decision = build_paper_rehearsal(
            [security()], decision_timestamp=CUTOFF
        ).decisions[0]
        result = evaluate_price_return(
            decision,
            horizon="one_month",
            future_split_adjusted_close=Decimal("55"),
            same_split_basis_proven=True,
        )
        self.assertEqual(result.return_type, "price_return")
        self.assertEqual(result.price_return, Decimal("0.1"))
        self.assertFalse(result.total_return_included)

    def test_evaluation_horizon_is_immutable_and_unplanned_horizon_fails(self) -> None:
        decision = build_paper_rehearsal(
            [security()], decision_timestamp=CUTOFF
        ).decisions[0]
        with self.assertRaises(ValidationError):
            decision.evaluation_horizons[0].target_not_before = date(2030, 1, 1)
        with self.assertRaisesRegex(ValueError, "not frozen"):
            evaluate_price_return(
                decision,
                horizon="one_year",
                future_split_adjusted_close=Decimal("55"),
            )

    def test_rehearsal_contract_guarantees_zero_persistence(self) -> None:
        result = build_paper_rehearsal([security()], decision_timestamp=CUTOFF)
        self.assertEqual(result.files_written, 0)
        self.assertEqual(result.database_rows_written, 0)
        self.assertEqual(result.status, "paper_only")


if __name__ == "__main__":
    unittest.main()
