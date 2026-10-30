from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from market_analytics.pipeline import run_fixture_pipeline
from qualitative import (
    CompanyImpact,
    CompanyNarrative,
    DailyResearchBrief,
    ForumSignal,
    NewListingCandidate,
    NewListingCoverage,
    PredictionEvaluation,
    PredictionOutcome,
    assess_new_listing,
    evaluate_prediction,
    freeze_prediction,
    historical_reliability,
    load_forum_signals,
    propose_prediction_clusters,
    render_company_narrative,
    render_daily_research_brief,
)
from schemas import EvidenceClaim, EvidenceItem
from tests.helpers import ANALYTICS_FIXTURE, analysis


UTC = timezone.utc
NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def evidence(
    evidence_id: str = "filing:fictional-1", evidence_class: str = "filing"
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        evidence_class=evidence_class,
        label="Fictional supplied evidence",
        source_url="https://example.com/fictional",
        available_at=NOW,
    )


def claim(
    claim_id: str = "claim:fictional-1",
    text: str = "FictionalCo reported higher quarterly revenue.",
    category: str = "reported_fact",
    support: str = "supported",
    evidence_ids: list[str] | None = None,
    *,
    causal_role: str = "none",
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        text=text,
        category=category,
        support=support,
        evidence_ids=["filing:fictional-1"] if evidence_ids is None else evidence_ids,
        causal_role=causal_role,
    )


def forum_signal(*, signal_id: str = "forum:fictional-1", author: str = "ExampleUser") -> ForumSignal:
    item = evidence("forum_community:fictional-1", "forum_community")
    return ForumSignal(
        signal_id=signal_id,
        forum="Example Forum",
        author=author,
        title="Fictional thesis",
        post_date=NOW,
        ticker="FIC",
        claim=claim(
            "claim:forum-1", "Community member expects demand to rise.",
            "forum_community_opinion", "supported", [item.evidence_id],
        ),
        evidence=item,
        prediction_horizon="30 days",
        source_url="https://example.com/forum/post",
    )


class ClaimEvidenceTests(unittest.TestCase):
    def test_analysis_payload_accepts_complete_claim_references(self) -> None:
        self.assertTrue(analysis().executive_summary_claim_ids)

    def test_executive_summary_must_equal_referenced_claim_text(self) -> None:
        payload = analysis().model_dump(mode="json")
        payload["executive_summary"] = "Unrelated arbitrary summary text."
        with self.assertRaisesRegex(ValidationError, "exactly match"):
            analysis(**payload)

    def test_supported_claim_requires_evidence(self) -> None:
        with self.assertRaises(ValidationError):
            claim(evidence_ids=[])

    def test_analysis_rejects_unknown_evidence_reference(self) -> None:
        payload = analysis().model_dump(mode="json")
        payload["claims"][0]["evidence_ids"] = ["filing:not-present"]
        with self.assertRaises(ValidationError):
            analysis(evidence_catalog=payload["evidence_catalog"], claims=payload["claims"],
                     executive_summary_claim_ids=payload["executive_summary_claim_ids"])

    def test_inference_must_be_explicit_and_evidence_linked(self) -> None:
        item = claim("claim:inference", "Demand may rise.", "ai_inference", "inference")
        self.assertEqual(item.support, "inference")
        with self.assertRaises(ValidationError):
            claim("claim:bad", "Demand rose.", "ai_inference", "supported")

    def test_unsupported_causality_remains_uncertain_inference(self) -> None:
        item = claim("claim:unsupported", "A possible scenario.", "ai_inference", "unsupported", [])
        self.assertEqual(item.evidence_ids, [])
        causal = claim(
            "claim:causal", "Demand may have increased because of a promotion.",
            "ai_inference", "unsupported", [], causal_role="inference",
        )
        self.assertEqual(causal.causal_role, "inference")
        with self.assertRaises(ValidationError):
            claim(
                "claim:definite", "A promotion caused demand to rise.",
                "ai_inference", "inference", ["filing:fictional-1"],
                causal_role="inference",
            )

    def test_forum_evidence_cannot_support_core_analysis(self) -> None:
        payload = analysis().model_dump(mode="json")
        payload["evidence_catalog"][0]["evidence_class"] = "forum_community"
        with self.assertRaises(ValidationError):
            analysis(evidence_catalog=payload["evidence_catalog"], claims=payload["claims"],
                     executive_summary_claim_ids=payload["executive_summary_claim_ids"])


class NarrativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = evidence()
        self.fact = claim()
        self.opinion = claim(
            "claim:opinion", "An analyst called the outlook encouraging.",
            "source_opinion", "supported",
        )

    def test_daily_brief_has_required_sections_and_labels(self) -> None:
        brief = DailyResearchBrief(
            research_date=date(2026, 9, 30), evidence_catalog=[self.evidence],
            claims=[self.fact, self.opinion], executive_claim_ids=[self.fact.claim_id],
            core_verified_development_ids=[self.fact.claim_id],
            company_impacts=[CompanyImpact(ticker="FIC", company="FictionalCo", claim_ids=[self.fact.claim_id])],
            contradiction_claim_ids=[self.opinion.claim_id],
            data_quality_notes=["Only supplied fictional evidence was used."],
            forum_signals=[forum_signal()], watch_next=["Next fictional filing"],
        )
        rendered = render_daily_research_brief(brief)
        for heading in (
            "## Executive Research Brief", "## Core Verified Developments",
            "## Company Impact", "## Catalysts", "## Risks",
            "## New Information vs Existing Research", "## Contradictions",
            "## Evidence Confidence / Data Quality", "## New Listings",
            "## Forum / Community Signals", "## What to Watch Next",
        ):
            self.assertIn(heading, rendered)
        self.assertIn("ANALYST / SOURCE OPINION", rendered)
        self.assertIn("UNVERIFIED / OPINION", rendered)

    def test_daily_brief_rejects_unknown_claim_and_forum_claim_in_core(self) -> None:
        with self.assertRaises(ValidationError):
            DailyResearchBrief(
                research_date=date(2026, 9, 30), evidence_catalog=[self.evidence],
                claims=[self.fact], executive_claim_ids=["claim:missing"],
            )
        signal = forum_signal()
        with self.assertRaises(ValidationError):
            DailyResearchBrief(
                research_date=date(2026, 9, 30), evidence_catalog=[signal.evidence],
                claims=[signal.claim], executive_claim_ids=[signal.claim.claim_id],
            )

    def test_company_narrative_renders_missing_values_without_invention(self) -> None:
        narrative = CompanyNarrative(
            ticker="FIC", company="FictionalCo", evidence_catalog=[self.evidence],
            claims=[self.fact], current_picture=[self.fact.claim_id],
            evidence_confidence="low", confidence_reasons=["One fictional source"],
            watch_next=["Next filing"],
        )
        rendered = render_company_narrative(narrative)
        self.assertIn("Business context was not supplied.", rendered)
        self.assertIn("None supported by supplied evidence.", rendered)

    def test_markdown_source_text_is_escaped(self) -> None:
        hostile = claim(text="# Injected\n> quote")
        narrative = CompanyNarrative(
            ticker="FIC", company="FictionalCo", evidence_catalog=[self.evidence],
            claims=[hostile], current_picture=[hostile.claim_id], evidence_confidence="low",
            confidence_reasons=["fixture"], watch_next=[],
        )
        rendered = render_company_narrative(narrative)
        self.assertIn(r"\# Injected &gt; quote", rendered)
        self.assertNotIn("\n# Injected", rendered)


class ForumPredictionTests(unittest.TestCase):
    def test_manual_forum_loader_isolates_bad_items(self) -> None:
        valid = forum_signal().model_dump(mode="json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.json"
            path.write_text(json.dumps({"signals": [valid, {"bad": True}]}), encoding="utf-8")
            signals, warnings = load_forum_signals(path)
        self.assertEqual(len(signals), 1)
        self.assertEqual(warnings, ["skipped malformed forum signal 2"])

    def test_forum_layer_does_not_change_ranking_outputs(self) -> None:
        before = run_fixture_pipeline(ANALYTICS_FIXTURE).rankings
        _ = DailyResearchBrief(
            research_date=date(2026, 9, 30), evidence_catalog=[evidence()], claims=[claim()],
            executive_claim_ids=["claim:fictional-1"], forum_signals=[forum_signal()],
        )
        after = run_fixture_pipeline(ANALYTICS_FIXTURE).rankings
        self.assertEqual(before, after)

    def _prediction(self, *, text: str = "Revenue will increase over the next month", horizon: int | None = 30):
        signal = forum_signal()
        return freeze_prediction(
            signal, frozen_at=NOW, prediction_type="revenue", subject="Fictional revenue",
            direction="increase" if horizon else None, horizon_days=horizon,
            specificity="medium", falsifiable=horizon is not None,
            frozen_interpretation=text,
        )

    def test_prediction_freeze_and_point_in_time_evaluation(self) -> None:
        prediction = self._prediction()
        outcome = PredictionOutcome(
            prediction_id=prediction.prediction_id,
            outcome_available_at=NOW + timedelta(days=31),
            evaluation_at=NOW + timedelta(days=31), observed_direction="increase",
            observed_at=NOW + timedelta(days=30),
            evidence_id="metric:fictional-outcome",
        )
        catalog = [evidence("metric:fictional-outcome", "metric").model_copy(
            update={"available_at": NOW + timedelta(days=31)}
        )]
        result = evaluate_prediction(prediction, outcome, catalog)
        self.assertEqual(result.evaluation_result, "correct")
        with self.assertRaises(ValueError):
            evaluate_prediction(
                prediction,
                outcome.model_copy(update={"outcome_available_at": NOW + timedelta(days=5)}),
                catalog,
            )

    def test_outcome_must_be_delayed_known_and_point_in_time_safe(self) -> None:
        prediction = self._prediction()
        valid_evidence = evidence("metric:fictional-outcome", "metric").model_copy(
            update={"available_at": NOW + timedelta(days=31)}
        )
        outcome = PredictionOutcome(
            prediction_id=prediction.prediction_id,
            outcome_available_at=NOW + timedelta(days=31),
            evaluation_at=NOW + timedelta(days=31),
            observed_at=NOW + timedelta(days=30),
            observed_direction="increase",
            evidence_id=valid_evidence.evidence_id,
        )
        self.assertEqual(evaluate_prediction(prediction, outcome, [valid_evidence]).evaluation_result, "correct")
        with self.assertRaisesRegex(ValueError, "evidence ID"):
            evaluate_prediction(prediction, outcome.model_copy(update={"evidence_id": "metric:fabricated"}), [valid_evidence])
        with self.assertRaisesRegex(ValueError, "predate prediction"):
            evaluate_prediction(
                prediction,
                outcome.model_copy(update={"observed_at": NOW - timedelta(seconds=1)}),
                [valid_evidence],
            )

    def test_vague_prediction_is_non_evaluable(self) -> None:
        prediction = self._prediction(text="Things may be interesting", horizon=None)
        self.assertEqual(prediction.evaluability, "non_evaluable")
        self.assertIn("vague", prediction.non_evaluable_reason or "")

    def test_reliability_requires_history_and_is_scoped(self) -> None:
        evaluations = [
            PredictionEvaluation(
                prediction_id=f"prediction:{index}", author="ExampleUser",
                evaluation_result="correct" if index < 8 else "incorrect",
                evaluation_at=NOW, prediction_created_at=NOW - timedelta(days=30),
                direction_correct=index < 8, timing_correct=index < 7,
                prediction_type="revenue", horizon_bucket="medium_term",
                specificity="medium", industry="Software", topic="demand",
                thesis_cluster_id=f"cluster:{index}",
            )
            for index in range(10)
        ]
        summary = historical_reliability(
            "ExampleUser", "revenue", "medium_term", evaluations, as_of=NOW
        )
        self.assertEqual(summary.resolved_predictions, 10)
        self.assertEqual(summary.directional_accuracy, Decimal("0.80"))
        self.assertEqual(summary.timing_accuracy, Decimal("0.70"))
        self.assertNotEqual(summary.historical_reliability, "insufficient_history")
        empty = historical_reliability(
            "DifferentUser", "revenue", "medium_term", evaluations, as_of=NOW
        )
        self.assertEqual(empty.historical_reliability, "insufficient_history")

    def test_two_of_two_does_not_outrank_large_history(self) -> None:
        def evaluation(index: int, correct: bool, *, author: str) -> PredictionEvaluation:
            return PredictionEvaluation(
                prediction_id=f"prediction:{author}-{index}", author=author,
                evaluation_result="correct" if correct else "incorrect",
                evaluation_at=NOW - timedelta(days=index),
                prediction_created_at=NOW - timedelta(days=index + 30),
                direction_correct=correct, timing_correct=correct,
                prediction_type="stock_direction", horizon_bucket="medium_term",
                specificity="high", thesis_cluster_id=f"cluster:{author}-{index}",
            )

        tiny = [evaluation(index, True, author="Tiny") for index in range(2)]
        large = [evaluation(index, index < 70, author="Large") for index in range(100)]
        tiny_summary = historical_reliability(
            "Tiny", "stock_direction", "medium_term", tiny, as_of=NOW
        )
        large_summary = historical_reliability(
            "Large", "stock_direction", "medium_term", large, as_of=NOW
        )
        self.assertEqual(tiny_summary.historical_reliability, "insufficient_history")
        self.assertEqual(large_summary.directional_accuracy, Decimal("0.70"))
        self.assertIn(large_summary.historical_reliability, {"moderate", "strong"})

    def test_derivative_predictions_share_only_a_proposed_cluster(self) -> None:
        first = self._prediction()
        second = first.model_copy(update={
            "prediction_id": "prediction:second", "signal_id": "forum:second",
            "prediction_created_at": NOW + timedelta(days=1),
        })
        clustered = propose_prediction_clusters([first, second])
        self.assertEqual(clustered[0].thesis_cluster_id, clustered[1].thesis_cluster_id)
        self.assertEqual(clustered[1].derivative_of, clustered[0].prediction_id)
        self.assertTrue(all(item.cluster_status == "proposed" for item in clustered))


class NewListingTests(unittest.TestCase):
    def _candidate(self, age: int, coverage: NewListingCoverage) -> NewListingCandidate:
        item = evidence("filing:fictional-prospectus", "filing")
        return NewListingCandidate(
            candidate_id="listing:fictional", ticker="NEW", company="New Fictional Co",
            detected_at=NOW, listing_date=date(2026, 9, 30) - timedelta(days=age),
            listing_type="traditional_ipo", coverage=coverage,
            evidence_catalog=[item], evidence_ids=[item.evidence_id],
        )

    def test_new_listing_modes_and_coverage_gate(self) -> None:
        sparse = NewListingCoverage(prospectus=True)
        self.assertEqual(assess_new_listing(self._candidate(10, sparse), assessment_date=date(2026, 9, 30)).mode, "limited_history")
        self.assertEqual(assess_new_listing(self._candidate(45, sparse), assessment_date=date(2026, 9, 30)).ranking_status, "provisional")
        self.assertEqual(assess_new_listing(self._candidate(120, sparse), assessment_date=date(2026, 9, 30)).ranking_status, "not_rankable")
        full = NewListingCoverage(
            prospectus=True, audited_financial_history=True, peer_coverage=True,
            trading_history=True, earnings_history=True, analyst_coverage=True,
            verified_news=True,
        )
        assessment = assess_new_listing(self._candidate(120, full), assessment_date=date(2026, 9, 30))
        self.assertEqual(assessment.ranking_status, "rankable")
        self.assertIn("PROVISIONAL ASSESSMENT", assessment.labels)

    def test_new_listing_rejects_unknown_or_forum_evidence(self) -> None:
        with self.assertRaises(ValidationError):
            NewListingCandidate(
                candidate_id="listing:bad", ticker="NEW", company="New Fictional Co",
                detected_at=NOW, listing_date=date(2026, 9, 30),
                coverage=NewListingCoverage(), evidence_ids=["filing:missing"],
            )
        item = evidence("forum_community:listing-rumor", "forum_community")
        with self.assertRaises(ValidationError):
            NewListingCandidate(
                candidate_id="listing:bad-forum", ticker="NEW", company="New Fictional Co",
                detected_at=NOW, listing_date=date(2026, 9, 30),
                coverage=NewListingCoverage(), evidence_catalog=[item], evidence_ids=[item.evidence_id],
            )


if __name__ == "__main__":
    unittest.main()
