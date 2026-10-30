from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from dedup import (
    ExistingResearchIndex,
    ResearchRecord,
    candidate_id,
    classify_duplicate,
    content_hash,
    normalize_url,
    normalized_title,
    token_similarity,
)
from entities import Entity
from filters import cheap_filter
from tests.helpers import candidate


class UrlAndIdentityTests(unittest.TestCase):
    def test_tracking_fragments_ports_and_query_order_are_normalized(self) -> None:
        first = normalize_url("HTTPS://Example.COM:443/a//b/?utm_source=x&b=2&a=1#top")
        self.assertEqual(first, "https://example.com/a/b?a=1&b=2")

    def test_non_http_url_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_url("file:///tmp/secret")

    def test_candidate_id_ignores_tracking(self) -> None:
        first = candidate(source_url="https://example.com/story?utm_campaign=a")
        second = candidate(source_url="https://EXAMPLE.com/story/")
        self.assertEqual(candidate_id(first), candidate_id(second))

    def test_content_hash_normalizes_case_and_whitespace(self) -> None:
        first = candidate(content="NVIDIA  RESULTS\nTODAY")
        second = candidate(content="nvidia results today")
        self.assertEqual(content_hash(first), content_hash(second))

    def test_title_normalization_supports_unicode(self) -> None:
        self.assertEqual(normalized_title("Café—RESULTS!"), "café results")

    def test_token_similarity_is_bounded(self) -> None:
        score = token_similarity("NVIDIA reports earnings", "NVIDIA earnings reported")
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


class DuplicateClassificationTests(unittest.TestCase):
    def test_exact_duplicate_from_vault_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            vault = Path(folder)
            (vault / "03 Articles").mkdir()
            (vault / "03 Articles" / "existing.md").write_text(
                "---\nsource_url: https://example.com/nvidia-results\n---\n# Existing\n",
                encoding="utf-8",
            )
            self.assertEqual(
                classify_duplicate(candidate(), ExistingResearchIndex.scan(vault)),
                "EXACT_DUPLICATE",
            )

    def test_duplicate_canonical_id_from_vault_metadata(self) -> None:
        item = candidate(source_url="https://example.com/id-only")
        with tempfile.TemporaryDirectory() as folder:
            vault = Path(folder)
            vault.mkdir(exist_ok=True)
            (vault / "existing.md").write_text(
                f"---\ncandidate_id: {candidate_id(item)}\n---\n# Existing\n",
                encoding="utf-8",
            )
            self.assertTrue(ExistingResearchIndex.scan(vault).exact_duplicate(item))

    def test_malformed_frontmatter_does_not_abort_scan(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            vault = Path(folder)
            (vault / "bad.md").write_text("---\n[not: valid\n---\n# Still indexed\n", encoding="utf-8")
            index = ExistingResearchIndex.scan(vault)
            self.assertIn("still indexed", index.titles)

    def test_same_run_url_is_exact_duplicate(self) -> None:
        first = candidate()
        second = candidate(title="Different headline", source_url=str(first.source_url))
        self.assertEqual(classify_duplicate(second, ExistingResearchIndex(), [first]), "EXACT_DUPLICATE")

    def test_same_content_different_url_is_syndicated(self) -> None:
        first = candidate()
        second = candidate(source_url="https://other.example.org/copy")
        self.assertEqual(
            classify_duplicate(second, ExistingResearchIndex(), [first]),
            "SYNDICATED_DUPLICATE",
        )

    def test_same_content_different_company_is_not_hash_deduplicated(self) -> None:
        boilerplate = "Standard market disclosure text shared by unrelated issuers."
        known = candidate(
            title="AMD reports quarterly earnings",
            source_url="https://example.com/amd",
            content=boilerplate,
            companies=["Advanced Micro Devices, Inc."],
            tickers=["AMD"],
        )
        current = candidate(
            title="NVIDIA reports quarterly earnings",
            source_url="https://example.com/nvidia",
            content=boilerplate,
        )
        self.assertEqual(
            classify_duplicate(current, ExistingResearchIndex(), [known]),
            "NEW_EVENT",
        )

    def test_same_content_far_apart_in_time_is_not_hash_deduplicated(self) -> None:
        boilerplate = "Repeated issuer boilerplate that is not itself an event identity."
        known = candidate(content=boilerplate)
        current = candidate(
            source_url="https://example.com/later-event",
            content=boilerplate,
            published=known.published + timedelta(days=120),
        )
        self.assertEqual(
            classify_duplicate(current, ExistingResearchIndex(), [known]),
            "NEW_EVENT",
        )

    def test_same_event_revised_number_is_meaningful_update(self) -> None:
        known = candidate(title="NVIDIA reports earnings guidance of $10 billion")
        current = candidate(
            title="NVIDIA revises earnings guidance to $12 billion",
            source_url="https://example.com/revision",
            content="Distinct content for the revision with material details.",
        )
        self.assertEqual(
            classify_duplicate(current, ExistingResearchIndex(), [known]),
            "SAME_EVENT_MEANINGFUL_UPDATE",
        )

    def test_same_event_without_new_information(self) -> None:
        known = candidate(title="NVIDIA reports quarterly earnings guidance")
        current = candidate(
            title="NVIDIA reports quarterly earnings and guidance",
            source_url="https://example.com/same-event",
            content="Different wording but no update signal in this fixture content.",
        )
        self.assertEqual(
            classify_duplicate(current, ExistingResearchIndex(), [known]),
            "SAME_EVENT_NO_NEW_INFO",
        )

    def test_second_source_can_be_confirmation(self) -> None:
        known = candidate(title="NVIDIA reports quarterly earnings guidance", source="Wire A")
        current = candidate(
            title="NVIDIA reports quarterly earnings and guidance",
            source="Wire B",
            source_url="https://example.com/confirmation",
            content="Independent fixture wording for a second publication.",
        )
        self.assertEqual(
            classify_duplicate(current, ExistingResearchIndex(), [known]),
            "SAME_EVENT_CONFIRMATION",
        )

    def test_different_entity_is_new_event(self) -> None:
        index = ExistingResearchIndex(
            records=[
                ResearchRecord(
                    title="AMD reports quarterly earnings and guidance",
                    companies=("advanced micro devices",),
                    tickers=("AMD",),
                )
            ]
        )
        self.assertEqual(classify_duplicate(candidate(), index), "NEW_EVENT")


class FilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entity = Entity(
            ticker="NVDA",
            canonical_name="NVIDIA Corporation",
            aliases=["NVIDIA"],
        )
        self.index = ExistingResearchIndex()

    def test_irrelevant_candidate_is_rejected(self) -> None:
        item = candidate(companies=[], tickers=[], title="Other company earnings", content="Other company reported earnings with enough unrelated test content.", snippet="")
        result = cheap_filter(item, self.entity, self.index, 20)
        self.assertFalse(result.accepted)
        self.assertEqual(result.rejection_reason, "irrelevant")

    def test_short_content_is_rejected(self) -> None:
        result = cheap_filter(candidate(content="short", snippet=""), self.entity, self.index, 80)
        self.assertEqual(result.rejection_reason, "inaccessible / insufficient content")

    def test_clickbait_prediction_is_rejected(self) -> None:
        item = candidate(
            title="Could NVIDIA turn $1,000 into $1 million by 2035?",
            content="NVIDIA is discussed in a generic long range prediction with enough filler text but no concrete corporate event or filing details.",
            snippet="",
            source_tier=4,
        )
        result = cheap_filter(item, self.entity, self.index, 20)
        self.assertEqual(result.rejection_reason, "low-value prediction or clickbait")

    def test_material_event_is_not_rejected_for_buy_prompt(self) -> None:
        item = candidate(title="Should you buy NVIDIA after earnings guidance?")
        self.assertTrue(cheap_filter(item, self.entity, self.index, 20).accepted)

    def test_weak_unsupported_rumor_is_rejected(self) -> None:
        item = candidate(
            title="NVIDIA might pursue an unspecified idea",
            content="An unconfirmed rumor claims NVIDIA might do something, but it supplies no material event or direct evidence despite this filler text.",
            snippet="",
            source_tier=4,
        )
        self.assertEqual(
            cheap_filter(item, self.entity, self.index, 20).rejection_reason,
            "unsupported rumor",
        )

    def test_commentary_is_accepted_but_deprioritized(self) -> None:
        item = candidate(
            title="Analyst raises NVIDIA price target",
            content="An analyst raises an NVIDIA price target based on commentary, with adequate fixture-only text and no newly reported corporate event.",
            snippet="",
        )
        result = cheap_filter(item, self.entity, self.index, 20)
        self.assertTrue(result.accepted)
        self.assertEqual(result.priority_adjustment, -1)

    def test_no_identifiable_event_is_deprioritized(self) -> None:
        item = candidate(
            title="NVIDIA profile and general background",
            content="A general NVIDIA Corporation profile with adequate fixture text and no identifiable event claim for the deterministic prefilter.",
            snippet="",
        )
        result = cheap_filter(item, self.entity, self.index, 20)
        self.assertTrue(result.accepted)
        self.assertEqual(result.priority_adjustment, -1)


if __name__ == "__main__":
    unittest.main()
