from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from discovery import (
    DAILY_NOTE_DISCLAIMER,
    DiscoveryItem,
    EvidenceLevel,
    HistoryStatus,
    SourceTier,
    daily_note_path,
    render_daily_highest_potential,
    run_discovery,
    to_main_pipeline_candidate,
    write_authorized_daily_note,
)


UTC = timezone.utc
NOW = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)


def item(**updates: object) -> DiscoveryItem:
    values: dict[str, object] = {
        "company": "NewCo, Inc.", "ticker": "NEW", "cik": "0001234567",
        "identity_verified": True, "title": "NewCo announces material contract",
        "source": "NewCo Investor Relations", "source_url": "https://newco.example/contract",
        "published_at": NOW, "discovered_at": NOW, "event_type": "material_contract",
        "event_date": date(2026, 10, 1), "event_key": "newco-contract-2026",
        "source_snapshot_id": "snapshot:1", "source_tier": SourceTier.PRIMARY,
        "primary_source": True, "summary": "NewCo signed a multi-year customer contract.",
        "material_facts": ["NewCo disclosed a signed multi-year customer contract."],
        "material_numbers": {"term": "five years"}, "absolute_scale_context": True,
        "new_information": True, "material_change": True,
        "catalyst_strength": EvidenceLevel.STRONG,
        "fundamental_interest": EvidenceLevel.MODERATE,
        "urgency": EvidenceLevel.STRONG,
        "risk_severity": EvidenceLevel.MODERATE,
        "risk_flags": ["Contract value and margin were not disclosed."],
        "missing_information": ["Price, valuation, estimates, and audited contribution are unavailable."],
    }
    values.update(updates)
    return DiscoveryItem(**values)


class DiscoveryTests(unittest.TestCase):
    def test_discovers_company_outside_watchlist(self) -> None:
        run = run_discovery([item()], known_company_ids={"ticker:AAPL"}, discovery_timestamp=NOW)
        self.assertEqual(run.selected.company, "NewCo, Inc.")
        self.assertEqual(run.selected.dimensions.novelty, 80)

    def test_novelty_alone_cannot_overcome_weak_evidence(self) -> None:
        run = run_discovery([item(source_tier=SourceTier.LOW_QUALITY)], discovery_timestamp=NOW)
        self.assertEqual(run.candidates[0].dimensions.novelty, 80)
        self.assertFalse(run.candidates[0].qualifies)
        self.assertIsNone(run.selected)

    def test_duplicate_articles_form_one_event(self) -> None:
        second = item(source_url="https://wire.example/rewrite", source="Wire",
                      source_snapshot_id="snapshot:2", syndication_key="wire-copy", primary_source=False)
        third = item(source_url="https://other.example/rewrite", source="Other",
                     source_snapshot_id="snapshot:3", syndication_key="wire-copy", primary_source=False)
        run = run_discovery([item(), second, third], discovery_timestamp=NOW)
        self.assertEqual(run.event_count, 1)
        self.assertEqual(run.duplicate_count, 1)
        self.assertEqual(run.candidates[0].source_count, 2)

    def test_unchanged_repeated_event_does_not_qualify(self) -> None:
        first = run_discovery([item()], discovery_timestamp=NOW)
        repeated = run_discovery([item()], known_event_fingerprints={first.candidates[0].event_id: first.candidates[0].event_fingerprint}, discovery_timestamp=NOW)
        self.assertIsNone(repeated.selected)
        self.assertEqual(repeated.candidates[0].dimensions.novelty, 0)

    def test_meaningful_update_qualifies(self) -> None:
        first = run_discovery([item(material_numbers={"term": "three years"})], discovery_timestamp=NOW)
        updated = run_discovery([item(material_numbers={"term": "five years"}, direction_change=True)],
                                known_event_fingerprints={first.candidates[0].event_id: first.candidates[0].event_fingerprint},
                                known_company_ids={"cik:1234567"}, discovery_timestamp=NOW)
        self.assertTrue(updated.selected.new_information_added)
        self.assertTrue(updated.selected.direction_change)

    def test_weak_rumor_is_rejected(self) -> None:
        run = run_discovery([item(rumor=True, source_tier=SourceTier.LOW_QUALITY)], discovery_timestamp=NOW)
        self.assertIsNone(run.selected)
        self.assertEqual(run.rejection_reasons["unsupported_rumor"], 1)

    def test_limited_history_warning_survives_selection(self) -> None:
        run = run_discovery([item(limited_history_status=HistoryStatus.PROVISIONAL)], discovery_timestamp=NOW)
        self.assertEqual(run.selected.dimensions.limited_history_status, HistoryStatus.PROVISIONAL)
        self.assertIn("Limited-history status: provisional", run.selected.risk_flags)

    def test_no_forced_winner(self) -> None:
        run = run_discovery([item(event_type="other")], discovery_timestamp=NOW)
        self.assertIsNone(run.selected)
        self.assertEqual(run.handoff.status, "NO_QUALIFIED_CANDIDATE_TODAY")

    def test_deterministic_selection(self) -> None:
        lower = item(company="Lower Co", ticker="LOW", cik="2", event_key="lower", event_type="filing_change",
                     source_url="https://lower.example/filing", material_change=False,
                     catalyst_strength=EvidenceLevel.WEAK, urgency=EvidenceLevel.WEAK)
        expected = run_discovery([lower, item()], discovery_timestamp=NOW).selected.event_id
        actual = run_discovery([item(), lower], discovery_timestamp=NOW).selected.event_id
        self.assertEqual(actual, expected)

    def test_handoff_has_required_provenance(self) -> None:
        run = run_discovery([item()], discovery_timestamp=NOW)
        handoff = run.handoff
        self.assertEqual(handoff.status, "HIGHEST_RESEARCH_PRIORITY")
        self.assertTrue(handoff.event_id and handoff.company_id and handoff.source_urls)
        self.assertTrue(handoff.strongest_evidence and handoff.recommended_questions)
        main_candidate = to_main_pipeline_candidate(run.selected)
        self.assertEqual(main_candidate.source_type, "discovery-handoff")
        self.assertEqual(main_candidate.tickers, ["NEW"])
        self.assertIn("Risk flag:", main_candidate.content)

    def test_note_rendering_and_yaml_are_safe(self) -> None:
        malicious = item(summary="[[Secret]] <% tp.file.title %> <script>x</script>")
        rendered = render_daily_highest_potential(run_discovery([malicious], discovery_timestamp=NOW), date(2026, 10, 1))
        self.assertIn(DAILY_NOTE_DISCLAIMER, rendered)
        self.assertNotIn("[[Secret]]", rendered)
        self.assertNotIn("<%", rendered)
        frontmatter = yaml.safe_load(rendered.split("---", 2)[1])
        self.assertEqual(frontmatter["rules_version"], "discovery-v1")
        self.assertTrue(frontmatter["immutable_history"])
        for heading in ("## Candidate", "## Evidence", "## Main Research Result", "## Decision Status"):
            self.assertIn(heading, rendered)

    def test_daily_note_is_create_once_and_narrowly_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            content = render_daily_highest_potential(run_discovery([item()], discovery_timestamp=NOW), date(2026, 10, 1))
            with self.assertRaises(PermissionError):
                write_authorized_daily_note(vault, date(2026, 10, 1), content, authorized=False)
            result = write_authorized_daily_note(vault, date(2026, 10, 1), content, authorized=True)
            self.assertEqual(result, daily_note_path(vault, date(2026, 10, 1)))
            with self.assertRaises(FileExistsError):
                write_authorized_daily_note(vault, date(2026, 10, 1), "replacement", authorized=True)
            self.assertEqual(result.read_text(), content)
            next_day = write_authorized_daily_note(vault, date(2026, 10, 2), "next day", authorized=True)
            self.assertEqual(next_day.read_text(), "next day")
            self.assertEqual(result.read_text(), content)


if __name__ == "__main__":
    unittest.main()
