from __future__ import annotations

import json
import os
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

from market_analytics.massive_provider import (
    MASSIVE_API_KEY_ENV,
    MASSIVE_ERROR_DETAIL_CHARS,
    MassiveValidationProvider,
    massive_api_key_is_configured,
    massive_capabilities,
)
from market_analytics.provider_runtime import (
    HttpResult,
    PaginationError,
    ProviderBoundaryError,
    ProviderRequestError,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "massive-validation-synthetic.json"
)
NOW = datetime(2026, 10, 3, 18, 0, tzinfo=timezone.utc)
CUTOFF = datetime.max.replace(tzinfo=timezone.utc)


class FixtureTransport:
    def __init__(self, fixture: dict[str, object]) -> None:
        self.fixture = fixture
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(
        self, url: str, headers: object, timeout: float, maximum: int
    ) -> HttpResult:
        header_copy = dict(headers)  # type: ignore[arg-type]
        self.calls.append((url, header_copy))
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        if parsed.path == "/v3/reference/tickers/AAPL":
            key = "metadata"
        elif parsed.path.startswith("/v2/aggs/ticker/AAPL/range/1/day/"):
            key = "adjusted_prices" if query.get("adjusted") == ["true"] else "raw_prices"
        elif parsed.path == "/stocks/v1/splits":
            key = "splits"
        elif parsed.path == "/stocks/v1/dividends":
            key = "dividends"
        elif parsed.path == "/vX/reference/tickers/META/events":
            key = "ticker_events"
        else:
            raise AssertionError(f"unexpected synthetic endpoint: {parsed.path}")
        return HttpResult(status=200, body=json.dumps(self.fixture[key]), headers={})


class MassiveProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def provider(self, transport: object, **kwargs: object) -> MassiveValidationProvider:
        return MassiveValidationProvider(
            "test-secret-never-output",
            transport=transport,  # type: ignore[arg-type]
            now=lambda: NOW,
            clock=lambda: 0.0,
            sleeper=lambda _: None,
            **kwargs,
        )

    def test_credential_uses_one_exact_environment_variable(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(massive_api_key_is_configured())
            with self.assertRaisesRegex(ValueError, MASSIVE_API_KEY_ENV):
                MassiveValidationProvider.from_env()
        with patch.dict(os.environ, {MASSIVE_API_KEY_ENV: "configured"}, clear=True):
            self.assertTrue(massive_api_key_is_configured())

    def test_capabilities_do_not_overclaim_unverified_provider_behavior(self) -> None:
        capabilities = massive_capabilities()
        self.assertEqual(capabilities["daily_prices"].historical_depth_days, 730)
        self.assertEqual(capabilities["ticker_events"].status, "UNVERIFIED")
        self.assertEqual(capabilities["daily_prices"].correction_behavior, "UNVERIFIED")
        for capability in capabilities.values():
            self.assertIn("retrieval-only", capability.point_in_time_quality)
            self.assertEqual(capability.storage_policy.local_storage, "UNVERIFIED")

    def test_metadata_maps_identity_and_retrieval_only_availability(self) -> None:
        transport = FixtureTransport(self.fixture)
        provider = self.provider(transport)
        row = provider.metadata("aapl", CUTOFF)[0]
        self.assertEqual((row.ticker, row.company_name), ("AAPL", "Apple Inc."))
        self.assertEqual((row.cik, row.exchange, row.sic), ("0000320193", "XNAS", "3571"))
        self.assertEqual(row.listing_date, date(1980, 12, 12))
        self.assertEqual(row.available_at, NOW)
        self.assertIn("retrieval_only", row.provenance["pit_availability"])
        url, headers = transport.calls[0]
        self.assertNotIn("test-secret", url)
        self.assertNotIn("apiKey", url)
        self.assertEqual(headers["Authorization"], "Bearer test-secret-never-output")
        self.assertEqual(provider.metadata("AAPL", NOW.replace(microsecond=0))[0].ticker, "AAPL")

    def test_prices_combine_raw_and_split_adjusted_without_calling_it_total_return(self) -> None:
        transport = FixtureTransport(self.fixture)
        provider = self.provider(transport)
        rows = provider.prices("AAPL", date(2025, 9, 29), date(2025, 9, 30), CUTOFF)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].close, Decimal("200.0"))
        self.assertEqual(rows[0].split_adjusted_close, Decimal("100.0"))
        self.assertIsNone(rows[0].total_return_index)
        self.assertIsNone(rows[0].dividend_adjusted_close)
        self.assertEqual(rows[0].volume, Decimal("50000000"))
        self.assertEqual(rows[0].available_at, NOW)
        self.assertEqual(rows[0].provenance["volume_adjustment_semantics"], "provider_not_explicitly_verified")
        self.assertEqual(
            rows[0].provenance["volume_capability_reason_code"],
            "adjusted_volume_unverified",
        )
        self.assertEqual([row.trading_date for row in rows], sorted(row.trading_date for row in rows))
        self.assertEqual(len({row.trading_date for row in rows}), 2)

    def test_delayed_plan_aggregate_status_is_an_endpoint_scoped_success(self) -> None:
        fixture = dict(self.fixture)
        fixture["raw_prices"] = {**dict(self.fixture["raw_prices"]), "status": "DELAYED"}
        fixture["adjusted_prices"] = {
            **dict(self.fixture["adjusted_prices"]), "status": "DELAYED"
        }
        rows = self.provider(FixtureTransport(fixture)).prices(
            "AAPL", date(2025, 9, 29), date(2025, 9, 30), CUTOFF
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0].provenance["raw_pages"][0]["source"]["response_status"],
            "DELAYED",
        )

    def test_paged_provider_boundary_failure_is_not_relabeled_as_pagination(self) -> None:
        fixture = dict(self.fixture)
        fixture["splits"] = {**dict(self.fixture["splits"]), "status": "ERROR"}
        with self.assertRaises(ProviderBoundaryError) as caught:
            self.provider(FixtureTransport(fixture)).corporate_actions(
                "AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF
            )
        self.assertNotIsInstance(caught.exception, PaginationError)
        self.assertIn("allowed success state", str(caught.exception))

    def test_splits_and_dividends_preserve_distinct_fields(self) -> None:
        transport = FixtureTransport(self.fixture)
        rows = self.provider(transport).corporate_actions(
            "AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF
        )
        split = next(row for row in rows if row.event_type == "stock_split")
        dividend = next(row for row in rows if row.event_type == "cash_dividend")
        self.assertEqual((split.effective_date, split.ratio), (date(2025, 9, 30), Decimal("2")))
        self.assertIsNone(split.announcement_date)
        self.assertEqual(
            split.provenance["action_classification"],
            {
                "raw_provider_type": "forward_split",
                "normalized_action_type": "stock_split",
                "normalization_status": "provider_as_reported",
                "reason_code": None,
                "evidence_reference": None,
            },
        )
        self.assertEqual(dividend.amount, Decimal("0.26"))
        self.assertEqual(dividend.declaration_date, date(2025, 7, 31))
        self.assertEqual(dividend.ex_date, date(2025, 8, 11))
        self.assertEqual(dividend.record_date, date(2025, 8, 11))
        self.assertEqual(dividend.payment_date, date(2025, 8, 14))

        split_query = parse_qs(urlsplit(transport.calls[0][0]).query)
        dividend_query = parse_qs(urlsplit(transport.calls[1][0]).query)
        self.assertEqual(split_query["ticker"], ["AAPL"])
        self.assertEqual(split_query["execution_date.gte"], ["2025-01-01"])
        self.assertEqual(split_query["execution_date.lte"], ["2025-12-31"])
        self.assertEqual(split_query["sort"], ["execution_date.asc"])
        self.assertEqual(dividend_query["ticker"], ["AAPL"])
        self.assertEqual(dividend_query["ex_dividend_date.gte"], ["2025-01-01"])
        self.assertEqual(dividend_query["ex_dividend_date.lte"], ["2025-12-31"])
        self.assertEqual(dividend_query["sort"], ["ex_dividend_date.asc"])

    def test_stock_dividend_provider_label_is_preserved_as_ambiguous(self) -> None:
        fixture = dict(self.fixture)
        splits = dict(self.fixture["splits"])
        splits["results"] = [
            {
                **dict(splits["results"][0]),
                "id": "provider-stock-dividend",
                "adjustment_type": "stock_dividend",
                "split_from": 1,
                "split_to": 4,
            }
        ]
        fixture["splits"] = splits
        rows = self.provider(FixtureTransport(fixture)).corporate_actions(
            "AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF
        )
        action = next(item for item in rows if item.source_record_id.endswith("provider-stock-dividend"))
        self.assertEqual(action.event_type, "stock_dividend")
        self.assertEqual(action.ratio, Decimal("4"))
        self.assertEqual(
            action.provenance["action_classification"],
            {
                "raw_provider_type": "stock_dividend",
                "normalized_action_type": None,
                "normalization_status": "ambiguous",
                "reason_code": "ambiguous_corporate_action_type",
                "evidence_reference": None,
            },
        )

    def test_experimental_ticker_events_map_only_complete_transitions(self) -> None:
        transport = FixtureTransport(self.fixture)
        rows = self.provider(transport).ticker_events("META", CUTOFF)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].old_ticker, rows[0].new_ticker), ("FB", "META"))
        self.assertEqual(rows[0].effective_date, date(2022, 6, 9))
        self.assertTrue(rows[0].provenance["experimental"])

    def test_pagination_is_host_path_bounded_and_provenance_is_retained(self) -> None:
        first = dict(self.fixture["splits"])
        second = dict(self.fixture["splits"])
        first["results"] = [dict(first["results"][0], id="one", execution_date="2025-01-01")]
        first["next_url"] = "https://api.massive.com/stocks/v1/splits?cursor=safe"
        second["results"] = [dict(second["results"][0], id="two", execution_date="2025-02-01")]
        responses = [first, second, self.fixture["dividends"]]

        def transport(url: str, headers: object, timeout: float, maximum: int) -> HttpResult:
            return HttpResult(status=200, body=json.dumps(responses.pop(0)), headers={})

        rows = self.provider(transport).corporate_actions(
            "AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF
        )
        self.assertEqual(len([row for row in rows if row.event_type == "stock_split"]), 2)
        self.assertEqual(len(responses), 0)

        for unsafe in (
            "https://evil.example/stocks/v1/splits?cursor=x",
            "https://api.massive.com/v3/reference/tickers?cursor=x",
            "https://api.massive.com/stocks/v1/splits-escaped?cursor=x",
            "https://api.massive.com/stocks/v1/splits?apiKey=secret",
        ):
            bad = dict(self.fixture["splits"], next_url=unsafe)
            with self.subTest(unsafe=unsafe), self.assertRaises(PaginationError):
                self.provider(
                    lambda *_: HttpResult(status=200, body=json.dumps(bad), headers={})
                ).corporate_actions("AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF)

    def test_empty_results_are_valid(self) -> None:
        empty = {"status": "OK", "request_id": "empty", "results": []}
        provider = self.provider(
            lambda *_: HttpResult(status=200, body=json.dumps(empty), headers={})
        )
        self.assertEqual(
            provider.corporate_actions("AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF),
            [],
        )

    def test_malformed_missing_and_oversized_responses_fail_closed(self) -> None:
        cases = (
            HttpResult(status=200, body="{", headers={}),
            HttpResult(status=200, body=json.dumps({"status": "OK"}), headers={}),
            HttpResult(status=200, body="x" * 101, headers={}),
        )
        for index, response in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ProviderBoundaryError):
                self.provider(lambda *_: response, max_response_bytes=100).metadata("AAPL", CUTOFF)

    def test_http_failures_are_sanitized_and_never_include_key_or_body(self) -> None:
        for status in (401, 403, 404):
            with self.subTest(status=status):
                provider = self.provider(
                    lambda *_: HttpResult(
                        status=status,
                        body="test-secret-never-output provider diagnostic",
                        headers={},
                    )
                )
                with self.assertRaises(ProviderRequestError) as caught:
                    provider.metadata("AAPL", CUTOFF)
                message = str(caught.exception)
                self.assertNotIn("test-secret", message)
                self.assertNotIn("provider diagnostic", message)
                self.assertEqual(caught.exception.status, status)

    def test_http_400_inside_pagination_remains_a_provider_request_error(self) -> None:
        provider = self.provider(
            lambda *_: HttpResult(
                status=400,
                body=json.dumps({"error": "invalid corporate-action filter"}),
                headers={},
            )
        )
        with self.assertRaises(ProviderRequestError) as caught:
            provider.corporate_actions(
                "AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF
            )
        self.assertEqual(caught.exception.status, 400)
        self.assertNotIsInstance(caught.exception, PaginationError)

    def test_http_403_inside_pagination_remains_a_provider_request_error(self) -> None:
        provider = self.provider(
            lambda *_: HttpResult(
                status=403,
                body=json.dumps({"error": "not entitled to corporate actions"}),
                headers={},
            )
        )
        with self.assertRaises(ProviderRequestError) as caught:
            provider.corporate_actions(
                "AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF
            )
        self.assertEqual(caught.exception.status, 403)
        self.assertIn("not entitled", str(caught.exception))

    def test_http_429_inside_pagination_remains_rate_limit_error(self) -> None:
        attempts = 0

        def rate_limited(*_: object) -> HttpResult:
            nonlocal attempts
            attempts += 1
            return HttpResult(
                status=429,
                body=json.dumps({"error": "rate limit exceeded"}),
                headers={"Retry-After": "0"},
            )

        with self.assertRaises(ProviderRequestError) as caught:
            self.provider(rate_limited).corporate_actions(
                "AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF
            )
        self.assertEqual(attempts, 3)
        self.assertEqual(caught.exception.status, 429)
        self.assertIn("rate limit exceeded", str(caught.exception))

    def test_safe_json_error_detail_is_retained_and_bounded(self) -> None:
        provider = self.provider(
            lambda *_: HttpResult(
                status=400,
                body=json.dumps({
                    "status": "ERROR",
                    "error": "Invalid sort parameter: id.asc",
                    "request_id": "safe-request-123",
                }),
                headers={},
            )
        )
        with self.assertRaises(ProviderRequestError) as caught:
            provider.corporate_actions(
                "AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF
            )
        message = str(caught.exception)
        self.assertIn("Invalid sort parameter", message)
        self.assertIn("safe-request-123", message)
        self.assertLessEqual(
            len(message),
            len("provider rejected request with HTTP 400: ") + MASSIVE_ERROR_DETAIL_CHARS,
        )

    def test_malicious_or_oversized_error_body_is_not_echoed(self) -> None:
        bodies = (
            json.dumps({
                "error": "<html>Authorization: Bearer sk-fakeSecret123456789</html>",
                "message": "apiKey=test-secret-never-output",
            }),
            json.dumps({"error": "x" * 9000}),
        )
        for body in bodies:
            with self.subTest(size=len(body)):
                provider = self.provider(
                    lambda *_: HttpResult(status=400, body=body, headers={})
                )
                with self.assertRaises(ProviderRequestError) as caught:
                    provider.corporate_actions(
                        "AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF
                    )
                message = str(caught.exception)
                self.assertNotIn("sk-fakeSecret", message)
                self.assertNotIn("test-secret", message)
                self.assertNotIn("<html>", message)
                self.assertLessEqual(
                    len(message),
                    len("provider rejected request with HTTP 400: ")
                    + MASSIVE_ERROR_DETAIL_CHARS,
                )

    def test_rate_limit_and_server_errors_use_bounded_retries(self) -> None:
        for status in (429, 500):
            attempts = 0

            def transport(*_: object) -> HttpResult:
                nonlocal attempts
                attempts += 1
                return HttpResult(status=status, body="{}", headers={"Retry-After": "0"})

            with self.subTest(status=status):
                with self.assertRaises(ProviderRequestError):
                    self.provider(transport).metadata("AAPL", CUTOFF)
                self.assertEqual(attempts, 3)

    def test_timeout_is_sanitized_and_bounded(self) -> None:
        attempts = 0

        def timeout(*_: object) -> HttpResult:
            nonlocal attempts
            attempts += 1
            raise TimeoutError("test-secret-never-output")

        with self.assertRaises(ProviderRequestError) as caught:
            self.provider(timeout).metadata("AAPL", CUTOFF)
        self.assertEqual(attempts, 3)
        self.assertNotIn("test-secret", str(caught.exception))

    def test_pagination_loop_and_duplicate_page_are_rejected(self) -> None:
        looping = dict(
            self.fixture["splits"],
            next_url="https://api.massive.com/stocks/v1/splits?cursor=same",
        )
        with self.assertRaises(PaginationError):
            self.provider(
                lambda *_: HttpResult(status=200, body=json.dumps(looping), headers={})
            ).corporate_actions("AAPL", date(2025, 1, 1), date(2025, 12, 31), CUTOFF)

    def test_call_cap_counts_retries_and_stops_before_transport(self) -> None:
        attempts = 0

        def server_error(*_: object) -> HttpResult:
            nonlocal attempts
            attempts += 1
            return HttpResult(status=500, body="{}", headers={})

        provider = self.provider(server_error, max_calls=2)
        with self.assertRaisesRegex(ProviderRequestError, "quota exhausted"):
            provider.metadata("AAPL", CUTOFF)
        self.assertEqual(attempts, 2)
        self.assertEqual(provider.usage.estimated_billable_calls, 2)


if __name__ == "__main__":
    unittest.main()
