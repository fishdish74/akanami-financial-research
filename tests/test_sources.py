from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError

from config import SourceRequestConfig, require_sec_user_agent
from entities import Entity
from sources.base import HttpClient, SourceRequestError
from sources.google_news import SourceParseError, parse_google_news_feed
from sources.manual import ManualJsonSource
from sources.sec import SecSubmissionsSource, parse_sec_submissions


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class _Response:
    def __init__(self, payload: bytes, encoding: str = "") -> None:
        self.payload = payload
        self.headers = {"Content-Encoding": encoding}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.payload if size < 0 else self.payload[:size]


class HttpClientTests(unittest.TestCase):
    MAX_RESPONSE_BYTES = 16 * 1024 * 1024
    MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024

    def config(self, retries: int = 0) -> SourceRequestConfig:
        return SourceRequestConfig(0, 1, retries, {"User-Agent": "offline-test"})

    def test_retries_then_succeeds(self) -> None:
        calls: list[int] = []

        def opener(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise URLError("offline failure")
            return _Response(b"ok")

        sleeps: list[float] = []
        client = HttpClient(self.config(retries=1), opener=opener, sleeper=sleeps.append)
        self.assertEqual(client.get("https://example.com/path?secret=no"), b"ok")
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [0.5])

    def test_failure_message_omits_query(self) -> None:
        def opener(*args, **kwargs):
            raise URLError("failed")

        with self.assertRaises(SourceRequestError) as caught:
            HttpClient(self.config(), opener=opener).get("https://example.com/path?token=secret")
        self.assertNotIn("token", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_gzip_is_decompressed(self) -> None:
        payload = gzip.compress(b"fixture")
        client = HttpClient(
            self.config(), opener=lambda *args, **kwargs: _Response(payload, "gzip")
        )
        self.assertEqual(client.get("https://example.com/feed"), b"fixture")

    def test_small_uncompressed_response_succeeds(self) -> None:
        client = HttpClient(
            self.config(), opener=lambda *args, **kwargs: _Response(b"fixture")
        )
        self.assertEqual(client.get("https://example.com/feed"), b"fixture")

    def test_oversized_raw_response_is_rejected_without_partial_result(self) -> None:
        payload = b"x" * (self.MAX_RESPONSE_BYTES + 1)
        client = HttpClient(
            self.config(), opener=lambda *args, **kwargs: _Response(payload)
        )
        with self.assertRaisesRegex(SourceRequestError, "response size"):
            client.get("https://example.com/feed")

    def test_gzip_expansion_past_limit_is_rejected(self) -> None:
        payload = gzip.compress(b"x" * (self.MAX_DECOMPRESSED_BYTES + 1))
        client = HttpClient(
            self.config(), opener=lambda *args, **kwargs: _Response(payload, "gzip")
        )
        with self.assertRaisesRegex(SourceRequestError, "decompressed response size"):
            client.get("https://example.com/feed")

    def test_malformed_gzip_is_rejected_safely(self) -> None:
        client = HttpClient(
            self.config(), opener=lambda *args, **kwargs: _Response(b"not-gzip", "gzip")
        )
        with self.assertRaises(SourceRequestError):
            client.get("https://example.com/feed")


class SecUserAgentValidationTests(unittest.TestCase):
    def test_rejects_missing_application_or_unusable_contact(self) -> None:
        for value in ("", "   ", "@", "foo@", "http", "Market Research Collector"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                require_sec_user_agent(value)

    def test_accepts_application_with_email_or_absolute_https_contact(self) -> None:
        self.assertEqual(
            require_sec_user_agent("Market Research Collector research@example.com"),
            "Market Research Collector research@example.com",
        )
        self.assertEqual(
            require_sec_user_agent("Market Research Collector https://example.com/contact"),
            "Market Research Collector https://example.com/contact",
        )


class SourceParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entity = Entity(
            ticker="NVDA",
            canonical_name="NVIDIA Corporation",
            aliases=["NVIDIA"],
            cik="1045810",
        )
        self.now = datetime(2026, 9, 29, tzinfo=timezone.utc)

    def test_google_feed_skips_bad_item_and_parses_good_item(self) -> None:
        values, warnings = parse_google_news_feed(
            (FIXTURES / "google-news-sample.xml").read_bytes(),
            self.entity,
            10,
            discovered_at=self.now,
        )
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0].source, "Reuters")
        self.assertEqual(warnings, ["skipped malformed Google News item 2"])

    def test_google_malformed_xml_is_explicit(self) -> None:
        with self.assertRaises(SourceParseError):
            parse_google_news_feed(b"<rss>", self.entity, 10)

    def test_google_bad_date_becomes_unknown(self) -> None:
        xml = b"<rss><channel><item><title>NVIDIA update - Reuters</title><link>https://example.com/a</link><pubDate>bad</pubDate><description>NVIDIA fixture text</description></item></channel></rss>"
        values, warnings = parse_google_news_feed(xml, self.entity, 1, discovered_at=self.now)
        self.assertIsNone(values[0].published)
        self.assertEqual(warnings, [])

    def test_sec_fixture_parses_valid_and_skips_bad_row(self) -> None:
        values, warnings = parse_sec_submissions(
            (FIXTURES / "sec-submissions-sample.json").read_bytes(),
            self.entity,
            10,
            discovered_at=self.now,
        )
        self.assertEqual(len(values), 1)
        self.assertIn("10-Q", values[0].title)
        self.assertEqual(warnings, ["skipped malformed SEC filing row 2"])

    def test_sec_malformed_json_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_sec_submissions(b"{", self.entity, 10)

    def test_sec_source_requires_user_agent(self) -> None:
        config = SourceRequestConfig(0, 1, 0, {"User-Agent": ""})
        with self.assertRaises(ValueError):
            SecSubmissionsSource(config)

    def test_manual_source_enriches_missing_entities(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidates.json"
            raw = json.loads((FIXTURES / "offline-demo.json").read_text(encoding="utf-8"))
            raw["candidates"][0]["companies"] = []
            raw["candidates"][0]["tickers"] = []
            path.write_text(json.dumps(raw), encoding="utf-8")
            value = ManualJsonSource(path).discover(self.entity, 1)[0]
            self.assertEqual(value.companies, ["NVIDIA Corporation"])
            self.assertEqual(value.tickers, ["NVDA"])

    def test_manual_source_isolates_schema_error_per_item(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            valid = json.loads((FIXTURES / "offline-demo.json").read_text(encoding="utf-8"))["candidates"][0]
            path.write_text(json.dumps([{"title": "missing fields"}, valid]), encoding="utf-8")
            source = ManualJsonSource(path)
            values = source.discover(self.entity, 1)
            self.assertEqual(len(values), 1)
            self.assertEqual(source.last_warnings, ["skipped manual candidate 1: schema validation failed"])
            self.assertEqual(values[0].source_type, "manual")
            self.assertEqual(values[0].source_tier, 4)


if __name__ == "__main__":
    unittest.main()
