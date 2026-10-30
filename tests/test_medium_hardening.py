from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from collector import resolve_target
from dedup import ExistingResearchIndex, classify_duplicate
from entities import COMMON_WORD_TICKERS, Entity, EntityRegistry
from market_analytics.cache import CacheEntry, CacheMetadata, FileCache
from market_analytics.config import load_config
from market_analytics.models import PeerComparison
from market_analytics.pipeline import run_fixture_pipeline
from market_analytics.ranking import rank_company
from schemas import EvidenceClaim
from sources.manual import ManualJsonSource
from sources.sec_fundamentals import RawSecCache
from tests.helpers import ANALYTICS_FIXTURE, candidate


UTC = timezone.utc


class EntityAndTargetHardeningTests(unittest.TestCase):
    def test_common_word_tickers_require_explicit_notation_or_company_context(self) -> None:
        self.assertEqual(COMMON_WORD_TICKERS, {"COST", "NOW", "ON", "ALL", "IT", "AI", "A", "T", "F"})
        for ticker in sorted(COMMON_WORD_TICKERS):
            entity = Entity(ticker=ticker, canonical_name=f"Example {ticker} Corporation")
            with self.subTest(ticker=ticker):
                self.assertFalse(entity.appears_in(f"The ordinary prose uses {ticker.lower()} in context."))
                for text in (f"${ticker}", f"NASDAQ:{ticker}", f"NASDAQ: {ticker}", f"({ticker})"):
                    self.assertTrue(entity.appears_in(text), text)
                self.assertTrue(entity.appears_in(entity.canonical_name))

    def test_target_pair_must_resolve_to_same_entity(self) -> None:
        apple = Entity(ticker="AAPL", canonical_name="Apple Inc.", aliases=["Apple"])
        tesla = Entity(ticker="TSLA", canonical_name="Tesla, Inc.", aliases=["Tesla"])
        registry = EntityRegistry([apple, tesla])
        self.assertIs(resolve_target(argparse.Namespace(ticker="AAPL", company="Apple"), registry), apple)
        self.assertIs(resolve_target(argparse.Namespace(ticker="AAPL", company="Apple Inc."), registry), apple)
        with self.assertRaisesRegex(ValueError, "contradictory"):
            resolve_target(argparse.Namespace(ticker="TSLA", company="Apple"), registry)
        unknown = resolve_target(argparse.Namespace(ticker="ZZZZ", company="Unknown Holdings"), registry)
        self.assertEqual((unknown.ticker, unknown.canonical_name), ("ZZZZ", "Unknown Holdings"))


class ManualAndDedupHardeningTests(unittest.TestCase):
    def test_unrelated_manual_record_is_not_stamped_with_target(self) -> None:
        item = candidate(
            title="Unrelated issuer reports results", content="Another issuer reported results.",
            snippet="Another issuer update.", companies=[], tickers=[],
        ).model_dump(mode="json")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manual.json"
            path.write_text(json.dumps([item]), encoding="utf-8")
            result = ManualJsonSource(path).discover(
                Entity(ticker="NVDA", canonical_name="NVIDIA Corporation", aliases=["NVIDIA"]), 1
            )[0]
        self.assertEqual(result.companies, [])
        self.assertEqual(result.tickers, [])

    def test_all_plausible_matches_are_scored_before_selection(self) -> None:
        current = candidate(
            title="NVIDIA earnings guidance $12 billion", source_url="https://example.com/current",
            content="Current distinct content.",
        )
        no_info = candidate(
            title="NVIDIA quarterly earnings guidance", source_url="https://example.com/no-info",
            content="Prior distinct content.",
        )
        changed_value = candidate(
            title="NVIDIA earnings guidance $10 billion", source_url="https://example.com/value",
            content="Earlier distinct content.",
        )
        self.assertEqual(
            classify_duplicate(current, ExistingResearchIndex(), [no_info, changed_value]),
            "SAME_EVENT_MEANINGFUL_UPDATE",
        )

    def test_tie_is_deterministic_and_opposite_direction_is_reviewable(self) -> None:
        current = candidate(title="NVIDIA reports earnings guidance", source="Wire B", source_url="https://x.example/current")
        first = candidate(title="NVIDIA reports quarterly earnings guidance", source="Wire B", source_url="https://x.example/a")
        second = candidate(title="NVIDIA reports annual earnings guidance", source="Wire A", source_url="https://x.example/b")
        left = classify_duplicate(current, ExistingResearchIndex(), [first, second])
        right = classify_duplicate(current, ExistingResearchIndex(), [second, first])
        self.assertEqual(left, right)
        rising = candidate(title="NVIDIA stock rises after earnings", source_url="https://x.example/rise")
        falling = candidate(title="NVIDIA stock falls after earnings", source_url="https://x.example/fall")
        self.assertEqual(
            classify_duplicate(falling, ExistingResearchIndex(), [rising]),
            "SAME_EVENT_MEANINGFUL_UPDATE",
        )


class CacheAndProvenanceHardeningTests(unittest.TestCase):
    @staticmethod
    def entry(retrieved: datetime, expiry: datetime, value: int) -> CacheEntry:
        payload = {"value": value}
        packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return CacheEntry(
            metadata=CacheMetadata(
                source="fixture", retrieval_timestamp=retrieved, expiry_timestamp=expiry,
                request_identifier="fixture:key",
                content_hash=hashlib.sha256(packed.encode()).hexdigest(),
            ), payload=payload,
        )

    def test_cache_rejects_tamper_and_replaces_only_expired_entry(self) -> None:
        now = datetime(2026, 9, 30, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as folder:
            cache = FileCache(Path(folder))
            first = self.entry(now, now + timedelta(hours=1), 1)
            path = cache.put(first)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["payload"]["value"] = 99
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content-hash"):
                cache.get("fixture:key", now)
            path.unlink()
            expired = self.entry(now, now + timedelta(hours=1), 1)
            cache.put(expired)
            replacement = self.entry(now + timedelta(hours=2), now + timedelta(hours=3), 2)
            cache.put(replacement)
            self.assertEqual(cache.get("fixture:key", now + timedelta(hours=2)).payload, {"value": 2})

    def test_identical_sec_content_retains_distinct_retrieval_events(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            cache = RawSecCache(Path(folder))
            for hour in (1, 2):
                cache.put(
                    payload=b'{"facts": {}}', source_url="https://data.sec.gov/example",
                    request_identifier="companyfacts:CIK0000000001",
                    retrieved_at=datetime(2026, 9, 30, hour, tzinfo=UTC),
                )
            self.assertEqual(len(list((Path(folder) / "raw").glob("*.json"))), 1)
            self.assertEqual(len(list((Path(folder) / "retrieval_events").glob("*.json"))), 2)
            latest = cache.latest(
                "companyfacts:CIK0000000001",
                now=datetime(2026, 9, 30, 2, 30, tzinfo=UTC),
                max_age=timedelta(hours=1),
            )
            self.assertIsNotNone(latest)
            self.assertEqual(latest.retrieved_at.hour, 2)


class RankingSemanticsTests(unittest.TestCase):
    def test_unavailable_reasons_and_conservative_missing_weight_policy(self) -> None:
        run = run_fixture_pipeline(ANALYTICS_FIXTURE)
        company = run.companies["QLTY"].model_copy(update={"catalysts": [], "risks": [], "news_score": None})
        comparisons = {
            name: PeerComparison(
                ticker="QLTY", metric=name, value=Decimal("1"), peer_median=Decimal("1"),
                percentile=Decimal("50"), z_score=Decimal("0"), relative_spread=Decimal("0"),
                direction="higher", status="valid",
            )
            for names in (
                ["P/E", "Forward P/E", "EV / EBITDA", "Price / Sales", "Earnings Yield", "FCF Yield"],
                ["ROE", "ROIC", "Gross Margin", "Operating Margin", "Net Margin", "FCF Margin"],
                ["Revenue Growth", "EPS Growth", "FCF Growth"],
                ["Debt / Equity", "Net Debt / EBITDA", "Interest Coverage", "Current Ratio"],
            ) for name in names
        }
        comparisons["ROE"] = comparisons["ROE"].model_copy(update={"status": "not_applicable", "percentile": None,
            "warning": "ROE is configured as not applicable for Synthetic"})
        comparisons["P/E"] = comparisons["P/E"].model_copy(update={"status": "invalid_input", "percentile": None})
        comparisons["EV / EBITDA"] = comparisons["EV / EBITDA"].model_copy(
            update={"status": "not_applicable", "percentile": None, "warning": "negative EBITDA makes this metric misleading"}
        )
        comparisons["Revenue Growth"] = comparisons["Revenue Growth"].model_copy(update={"status": "missing_data", "percentile": None, "warning": "insufficient_peer_set: 1 usable peer"})
        config = load_config()
        weights = {name: Decimal("0") for name in config.ranking_weights["one_day"]}
        weights.update({"valuation": Decimal("0.50"), "catalysts": Decimal("0.50")})
        config = config.model_copy(update={"ranking_weights": {**config.ranking_weights, "one_day": weights}})
        result = rank_company(company, comparisons, [], config, "one_day", company.as_of)
        self.assertEqual(result.unavailable_inputs["ROE"], "structural not-applicable")
        self.assertEqual(result.unavailable_inputs["P/E"], "adverse or economically invalid input")
        self.assertEqual(result.unavailable_inputs["EV / EBITDA"], "adverse or economically invalid input")
        self.assertEqual(result.unavailable_inputs["Revenue Growth"], "insufficient peer set")
        # F5 authorization supersedes available-only constituent renormalization:
        # four valid valuation metrics at 50 / six applicable metrics = 33,
        # then the unchanged 0.50 component weight rounds 16.5 to 17.
        self.assertEqual(result.opportunity_score, Decimal("17"))


class EvidenceHardeningTests(unittest.TestCase):
    def test_causal_language_requires_role_and_locations_must_bind_to_evidence(self) -> None:
        with self.assertRaisesRegex(ValidationError, "causal language"):
            EvidenceClaim(
                claim_id="claim:causal", text="Higher demand caused revenue growth.",
                category="reported_fact", support="supported", evidence_ids=["filing:1"],
            )
        with self.assertRaisesRegex(ValidationError, "locations"):
            EvidenceClaim(
                claim_id="claim:span", text="Revenue increased.", category="reported_fact",
                support="supported", evidence_ids=["filing:1"],
                evidence_locations=[{
                    "evidence_id": "filing:other", "location_type": "fact_id",
                    "identifier": "us-gaap:Revenues:2025",
                }],
            )


if __name__ == "__main__":
    unittest.main()
