from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from pydantic import ValidationError

from market_analytics.models import BacktestRecord
from qualitative import PredictionEvaluation, historical_reliability


EASTERN = timezone(timedelta(hours=-4))
CUTOFF = datetime(2026, 6, 1, 10, 0, tzinfo=EASTERN)


class ExactBacktestCutoffRegressionTests(unittest.TestCase):
    def record(self, available_at: datetime, *, cutoff: datetime = CUTOFF) -> BacktestRecord:
        return BacktestRecord.model_validate(
            {
                "ranking_cutoff_at": cutoff,
                "ticker": "PITX",
                "horizon": "1m",
                "opportunity_score": Decimal("60"),
                "research_priority": "medium",
                "data_snapshot_id": "snapshot:pit-regression",
                "inputs_available_at_time": available_at,
                "future_return": None,
                "benchmark_return": None,
                "sector_return": None,
            }
        )

    def test_same_day_input_after_decision_cutoff_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "after ranking_cutoff_at"):
            self.record(datetime(2026, 6, 1, 15, 0, tzinfo=EASTERN))

    def test_input_before_cutoff_is_eligible(self) -> None:
        record = self.record(datetime(2026, 6, 1, 9, 30, tzinfo=EASTERN))
        self.assertLess(record.inputs_available_at_time, record.ranking_cutoff_at)

    def test_input_exactly_at_cutoff_is_eligible(self) -> None:
        record = self.record(CUTOFF)
        self.assertEqual(record.inputs_available_at_time, record.ranking_cutoff_at)

    def test_input_one_microsecond_after_cutoff_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self.record(CUTOFF + timedelta(microseconds=1))

    def test_prior_day_is_eligible_and_next_day_is_rejected(self) -> None:
        self.assertIsInstance(self.record(CUTOFF - timedelta(days=1)), BacktestRecord)
        with self.assertRaises(ValidationError):
            self.record(CUTOFF + timedelta(days=1))

    def test_timezone_equivalent_instants_compare_equally(self) -> None:
        record = self.record(datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc))
        self.assertEqual(record.ranking_cutoff_at, record.inputs_available_at_time)
        self.assertEqual(record.ranking_cutoff_at.tzinfo, timezone.utc)

    def test_naive_cutoff_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self.record(datetime(2026, 6, 1, 9, 30, tzinfo=EASTERN), cutoff=datetime(2026, 6, 1, 10, 0))

    def test_serialization_round_trip_preserves_exact_instant(self) -> None:
        record = self.record(datetime(2026, 6, 1, 9, 30, tzinfo=EASTERN))
        restored = BacktestRecord.model_validate_json(record.model_dump_json())
        self.assertEqual(restored.ranking_cutoff_at, record.ranking_cutoff_at)
        self.assertEqual(restored.inputs_available_at_time, record.inputs_available_at_time)

    def test_legacy_date_only_record_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValidationError, "date-only"):
            BacktestRecord.model_validate(
                {
                    "ranking_date": "2026-06-01",
                    "ticker": "PITX",
                    "horizon": "1m",
                    "opportunity_score": 60,
                    "research_priority": "medium",
                    "data_snapshot_id": "legacy",
                    "inputs_available_at_time": "2026-06-01T15:00:00-04:00",
                }
            )


def _evaluation(
    identifier: str,
    evaluation_at: datetime,
    *,
    correct: bool,
    outcome_available_at: datetime | None = None,
) -> PredictionEvaluation:
    return PredictionEvaluation(
        prediction_id=f"prediction:{identifier}",
        author="PointInTimeAuthor",
        evaluation_result="correct" if correct else "incorrect",
        outcome_available_at=outcome_available_at,
        evaluation_at=evaluation_at,
        prediction_created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        direction_correct=correct,
        timing_correct=correct,
        prediction_type="revenue",
        horizon_bucket="medium_term",
        specificity="medium",
        thesis_cluster_id=f"cluster:{identifier}",
    )


class HistoricalReliabilityCutoffRegressionTests(unittest.TestCase):
    def summarize(self, evaluations, *, as_of: datetime = CUTOFF):
        return historical_reliability(
            "PointInTimeAuthor",
            "revenue",
            "medium_term",
            evaluations,
            as_of=as_of,
            minimum_history=1,
        )

    def test_evaluation_after_cutoff_does_not_influence_history(self) -> None:
        before = _evaluation("before", CUTOFF - timedelta(minutes=30), correct=True)
        after = _evaluation("after", CUTOFF + timedelta(hours=5), correct=False)
        summary = self.summarize([before, after])
        self.assertEqual(summary.resolved_predictions, 1)
        self.assertEqual(summary.directional_accuracy, Decimal("1.00"))

    def test_all_evaluations_before_cutoff_are_included(self) -> None:
        summary = self.summarize(
            [
                _evaluation("one", CUTOFF - timedelta(days=2), correct=True),
                _evaluation("two", CUTOFF - timedelta(days=1), correct=False),
            ]
        )
        self.assertEqual(summary.resolved_predictions, 2)
        self.assertEqual(summary.directional_accuracy, Decimal("0.50"))

    def test_all_evaluations_after_cutoff_produce_no_history(self) -> None:
        summary = self.summarize(
            [_evaluation("future", CUTOFF + timedelta(seconds=1), correct=True)]
        )
        self.assertEqual(summary.resolved_predictions, 0)
        self.assertEqual(summary.historical_reliability, "insufficient_history")

    def test_evaluation_exactly_at_cutoff_is_included(self) -> None:
        self.assertEqual(
            self.summarize([_evaluation("exact", CUTOFF, correct=True)]).resolved_predictions,
            1,
        )

    def test_timezone_equivalent_evaluation_is_included(self) -> None:
        equivalent = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
        self.assertEqual(
            self.summarize([_evaluation("equivalent", equivalent, correct=True)]).resolved_predictions,
            1,
        )

    def test_naive_as_of_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.summarize([], as_of=datetime(2026, 6, 1, 10, 0))

    def test_as_of_is_required_instead_of_defaulting_to_all_known_history(self) -> None:
        with self.assertRaises(TypeError):
            historical_reliability(
                "PointInTimeAuthor", "revenue", "medium_term", []
            )

    def test_later_evaluation_of_older_outcome_does_not_leak_backward(self) -> None:
        late_backfill = _evaluation(
            "backfill",
            CUTOFF + timedelta(days=2),
            correct=True,
            outcome_available_at=CUTOFF - timedelta(days=10),
        )
        self.assertEqual(self.summarize([late_backfill]).resolved_predictions, 0)


if __name__ == "__main__":
    unittest.main()
