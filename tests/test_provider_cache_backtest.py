from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from market_analytics.cache import CacheEntry, CacheMetadata, FileCache
from market_analytics.models import BacktestRecord, FiscalPeriod, TimeSeriesPoint
from market_analytics.providers import FixtureProvider
from tests.helpers import ANALYTICS_FIXTURE


class FixtureProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = FixtureProvider(ANALYTICS_FIXTURE)

    def test_provider_supports_future_interface(self) -> None:
        self.assertIsNotNone(self.provider.get_price("QLTY", self.provider.analysis_date))
        self.assertIsNotNone(self.provider.get_market_cap("QLTY", self.provider.analysis_date))
        self.assertIsNotNone(self.provider.get_shares_outstanding("QLTY", self.provider.analysis_date))
        self.assertIn("revenue", self.provider.get_fundamentals("QLTY", self.provider.analysis_date))
        self.assertIn("forward_eps", self.provider.get_estimates("QLTY", self.provider.analysis_date))
        self.assertEqual(self.provider.get_company_metadata("QLTY")["industry"], "Software")

    def test_historical_prices_are_empty_not_fabricated(self) -> None:
        self.assertEqual(
            self.provider.get_historical_prices("QLTY", date(2025, 1, 1), date(2026, 1, 1)),
            [],
        )

    def test_unknown_ticker_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present"):
            self.provider.company("NOPE")


class CacheTests(unittest.TestCase):
    def entry(self, now: datetime, expiry: datetime) -> CacheEntry:
        payload = {"fictional": True, "value": 10}
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return CacheEntry(
            metadata=CacheMetadata(
                source="fixture", retrieval_timestamp=now, expiry_timestamp=expiry,
                request_identifier="fixture:QLTY:fundamentals",
                content_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            ),
            payload=payload,
        )

    def test_cache_round_trip_and_expiry(self) -> None:
        now = datetime(2026, 9, 29, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as folder:
            cache = FileCache(Path(folder))
            entry = self.entry(now, now + timedelta(days=1))
            cache.put(entry)
            self.assertIsNotNone(cache.get(entry.metadata.request_identifier, now))
            self.assertIsNone(cache.get(entry.metadata.request_identifier, now + timedelta(days=2)))

    def test_cache_refuses_hash_mismatch(self) -> None:
        now = datetime(2026, 9, 29, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as folder:
            cache = FileCache(Path(folder))
            entry = self.entry(now, now + timedelta(days=1)).model_copy(deep=True)
            entry.metadata.content_hash = "0" * 64
            with self.assertRaises(ValueError):
                cache.put(entry)

    def test_cache_refuses_overwrite(self) -> None:
        now = datetime(2026, 9, 29, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as folder:
            cache = FileCache(Path(folder))
            entry = self.entry(now, now + timedelta(days=1))
            cache.put(entry)
            with self.assertRaises(FileExistsError):
                cache.put(entry)


class BacktestModelTests(unittest.TestCase):
    def valid(self, **updates) -> BacktestRecord:
        values = {
            "ranking_cutoff_at": "2026-09-29T16:00:00Z",
            "ticker": "QLTY", "horizon": "1y",
            "opportunity_score": Decimal("75"), "research_priority": "medium",
            "data_snapshot_id": "fixture-snapshot",
            "inputs_available_at_time": "2026-09-29T16:00:00Z",
            "future_return": None, "benchmark_return": None, "sector_return": None,
        }
        values.update(updates)
        return BacktestRecord.model_validate(values)

    def test_future_outcomes_are_optional(self) -> None:
        record = self.valid()
        self.assertIsNone(record.future_return)

    def test_lookahead_metadata_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self.valid(inputs_available_at_time="2026-09-29T16:00:00.000001Z")

    def test_invalid_score_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self.valid(opportunity_score=Decimal("101"))

    def test_time_series_point_requires_point_in_time_availability(self) -> None:
        period = FiscalPeriod(kind="FY", label="FY2026", end_date=date(2026, 12, 31), fiscal_year=2026)
        with self.assertRaises(ValidationError):
            TimeSeriesPoint(
                ticker="QLTY", metric="revenue", period=period, value=Decimal("10"),
                unit="millions", currency="USD", source="fixture",
                available_at="2026-12-01T00:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
