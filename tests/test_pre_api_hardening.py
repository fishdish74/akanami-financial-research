from __future__ import annotations

import errno
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from config import SourceRequestConfig
from entities import Entity
from market_analytics.cache import CacheEntry, CacheMetadata, FileCache
from market_analytics.config import load_config
from market_analytics.models import CompanyData, FiscalPeriod, DataPoint, QualityIssue
from market_analytics.peers import compare_metric
from market_analytics.ranking import rank_company
from obsidian import write_new_note_atomic
from sources.base import HttpClient, SourceRequestError, _safe_endpoint
from sources.google_news import parse_google_news_feed
from sources.sec import parse_sec_submissions


NOW = datetime(2026, 9, 29, tzinfo=timezone.utc)


class EntityAndSourceBoundaryTests(unittest.TestCase):
    def test_scalar_alias_is_one_alias_not_characters(self) -> None:
        entity = Entity(ticker="NVDA", canonical_name="NVIDIA", aliases="Nvidia Corp")
        self.assertEqual(entity.aliases, ["Nvidia Corp"])

    def test_common_word_ticker_requires_explicit_notation(self) -> None:
        entity = Entity(ticker="IT", canonical_name="Gartner, Inc.")
        self.assertFalse(entity.appears_in("It reported another routine result."))
        self.assertTrue(entity.appears_in("NYSE: IT reported another result."))

    def test_google_query_target_is_not_automatically_an_actual_mention(self) -> None:
        entity = Entity(ticker="NVDA", canonical_name="NVIDIA Corporation")
        xml = b"<rss><channel><item><title>Unrelated semiconductor story - Reuters</title><link>https://example.com/a</link><description>No named company appears here.</description></item></channel></rss>"
        values, _ = parse_google_news_feed(xml, entity, 1, discovered_at=NOW)
        self.assertEqual(values[0].companies, [])
        self.assertEqual(values[0].tickers, [])

    def test_sec_submission_is_explicitly_metadata_only(self) -> None:
        entity = Entity(ticker="NVDA", canonical_name="NVIDIA Corporation", cik="1045810")
        payload = json.dumps({"filings": {"recent": {
            "accessionNumber": ["0001045810-26-000001"],
            "filingDate": ["2026-09-01"], "reportDate": ["2026-08-01"],
            "form": ["8-K"], "primaryDocument": ["filing.htm"],
        }}}).encode()
        values, _ = parse_sec_submissions(payload, entity, 1, discovered_at=NOW)
        self.assertEqual(values[0].content_scope, "metadata_only")
        self.assertEqual(values[0].content, "")


class HttpAndCacheBoundaryTests(unittest.TestCase):
    @staticmethod
    def config(retries: int = 2, delay: float = 0) -> SourceRequestConfig:
        return SourceRequestConfig(delay, 1, retries, {"User-Agent": "offline-test"})

    def test_permanent_404_is_not_retried_and_url_credentials_are_hidden(self) -> None:
        calls = 0

        def opener(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise HTTPError("https://user:pass@example.com/a", 404, "missing", {}, None)

        with self.assertRaises(SourceRequestError):
            HttpClient(self.config(), opener=opener, sleeper=lambda _: None).get(
                "https://user:pass@example.com/a?token=secret"
            )
        self.assertEqual(calls, 1)
        self.assertEqual(_safe_endpoint("https://user:pass@example.com/a?x=1"), "https://example.com/a")

    def test_retry_after_is_bounded(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def opener(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise HTTPError("https://example.com", 429, "slow", {"Retry-After": "99"}, None)

            class Response:
                headers = {}
                def __enter__(self): return self
                def __exit__(self, *_args): return False
                def read(self, size=-1): return b"ok"
            return Response()

        value = HttpClient(self.config(retries=1), opener=opener, sleeper=sleeps.append).get("https://example.com")
        self.assertEqual(value, b"ok")
        self.assertEqual(sleeps, [30.0])

    def test_future_cache_timestamp_is_rejected(self) -> None:
        payload = {"value": 1}
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        import hashlib
        entry = CacheEntry(metadata=CacheMetadata(
            source="fixture", retrieval_timestamp=NOW + timedelta(hours=1),
            expiry_timestamp=NOW + timedelta(hours=2), request_identifier="future",
            content_hash=hashlib.sha256(serialized.encode()).hexdigest(),
        ), payload=payload)
        with tempfile.TemporaryDirectory() as folder:
            cache = FileCache(Path(folder))
            cache.put(entry)
            with self.assertRaisesRegex(ValueError, "future"):
                cache.get("future", NOW)


class RankingAndWriteBoundaryTests(unittest.TestCase):
    def company(self) -> CompanyData:
        point = DataPoint(
            value=Decimal("1"), unit="raw",
            period=FiscalPeriod(kind="FY", label="FY2025", end_date="2025-12-31", fiscal_year=2025),
            as_of="2026-01-31", source="fixture",
        )
        return CompanyData(
            ticker="TEST", company="Test", sector="Tech", industry="Software",
            as_of="2026-09-29", financials={"revenue": point}, source_quality="low",
        )

    def test_blocking_issue_prevents_opportunity_ranking(self) -> None:
        issue = QualityIssue(code="bad", severity="BLOCKING", message="blocked")
        result = rank_company(
            self.company(), {}, [issue], load_config(), "one_year", NOW.date()
        )
        self.assertEqual(result.ranking_status, "blocked")
        self.assertIsNone(result.opportunity_score)

    def test_missing_news_catalyst_and_risk_are_not_favorable_scores(self) -> None:
        result = rank_company(
            self.company(), {}, [], load_config(), "one_day", NOW.date()
        )
        self.assertIsNone(result.components.news)
        self.assertIsNone(result.components.catalysts)
        self.assertIsNone(result.components.risk)
        self.assertEqual(result.ranking_status, "insufficient_evidence")

    def test_single_company_is_an_insufficient_peer_set(self) -> None:
        comparison = compare_metric(
            "TEST", "ROE", Decimal("1"), {"TEST": Decimal("1")}, "higher"
        )
        self.assertIsNone(comparison.percentile)
        self.assertIn("insufficient_peer_set", comparison.warning)

    def test_atomic_writer_portable_fallback_keeps_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"MARKET_COLLECTOR_ENABLE_WRITES": "1"}
        ), patch("obsidian.os.link", side_effect=OSError(errno.EXDEV, "cross-device")):
            root = Path(folder)
            destination = root / "03 Articles" / "portable.md"
            write_new_note_atomic(destination, "safe", allowed_root=root)
            self.assertEqual(destination.read_text(), "safe")
            with self.assertRaises(FileExistsError):
                write_new_note_atomic(destination, "replace", allowed_root=root)


if __name__ == "__main__":
    unittest.main()
