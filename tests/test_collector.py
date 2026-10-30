from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import ValidationError

from dedup import ExistingResearchIndex, candidate_id, normalize_url
from entities import Entity, EntityRegistry
from renderer import deterministic_filename, render_article, validate_rendered_markdown
from schemas import AnalysisPayload, Candidate
from tests.helpers import analysis as make_fixture_analysis


def make_candidate(url: str = "https://example.com/story?utm_source=test#top") -> Candidate:
    return Candidate(
        title="Test Company publishes a material update",
        source="Example Primary Source",
        source_url=url,
        published=datetime(2026, 9, 24, tzinfo=timezone.utc),
        discovered=datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc),
        author="Example Author",
        content="Test Company published an update with enough text for deterministic testing.",
        snippet="Test Company published an update with enough text for deterministic testing.",
        source_tier=1,
        companies=["Test Company"],
        tickers=["TEST"],
        source_type="manual",
    )


def make_analysis() -> AnalysisPayload:
    return make_fixture_analysis(
        title="Test Company publishes a material update",
        source="Example Primary Source",
        source_url="https://example.com/story",
        author="Example Author",
        published="2026-09-24",
        accessed="2026-09-24T12:00:00Z",
        companies=["Test Company"],
        tickers=["TEST"],
        themes=[],
        sector=None,
        importance="medium",
        confidence="high",
        sentiment="neutral",
        needs_review=False,
        executive_summary="A schema-only test payload.",
        reported_facts=["A test fact."],
        company_management_statements=[],
        analyst_source_opinion=[],
        ai_inference=[],
        potential_catalysts=[],
        potential_risks=[],
        bullish_evidence=[],
        bearish_evidence=[],
        what_is_new=["A test update."],
        contradictions=[],
        open_questions=[],
        information_needed_next=[],
    )


class UrlAndIdentityTests(unittest.TestCase):
    def test_url_normalization(self) -> None:
        value = normalize_url(
            "HTTPS://Example.COM/story/?utm_source=x&b=2&a=1#section"
        )
        self.assertEqual(value, "https://example.com/story?a=1&b=2")

    def test_candidate_id_is_deterministic(self) -> None:
        first = make_candidate("https://example.com/story?utm_campaign=x")
        second = make_candidate("https://EXAMPLE.com/story/")
        self.assertEqual(candidate_id(first), candidate_id(second))


class SchemaTests(unittest.TestCase):
    def test_schema_rejects_unknown_importance(self) -> None:
        raw = make_analysis().model_dump(mode="json")
        raw["importance"] = "urgent"
        with self.assertRaises(ValidationError):
            AnalysisPayload.model_validate(raw)


class RendererTests(unittest.TestCase):
    def test_renderer_produces_valid_yaml(self) -> None:
        candidate = make_candidate()
        analysis = make_analysis()
        markdown = render_article(analysis, candidate, "NEW_EVENT")
        validate_rendered_markdown(markdown)
        frontmatter = yaml.safe_load(markdown.split("---", 2)[1])
        self.assertEqual(frontmatter["status"], "inbox")
        self.assertFalse(frontmatter["reviewed"])
        self.assertIn("## AI Inference", markdown)
        self.assertTrue(deterministic_filename(analysis).endswith(".md"))


class EntityTests(unittest.TestCase):
    def test_alias_normalization(self) -> None:
        entity = Entity(
            ticker="TEST",
            canonical_name="Test Company",
            aliases=["Test Co"],
            alternate_listings=[],
        )
        registry = EntityRegistry([entity])
        self.assertEqual(registry.resolve("test co"), entity)
        self.assertEqual(registry.resolve("TEST"), entity)


class DuplicateTests(unittest.TestCase):
    def test_exact_duplicate_detection(self) -> None:
        candidate = make_candidate()
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            article_dir = vault / "03 Articles"
            article_dir.mkdir()
            (article_dir / "existing.md").write_text(
                "---\nsource_url: https://example.com/story\n---\n# Existing\n",
                encoding="utf-8",
            )
            index = ExistingResearchIndex.scan(vault)
            self.assertTrue(index.exact_duplicate(candidate))


if __name__ == "__main__":
    unittest.main()
