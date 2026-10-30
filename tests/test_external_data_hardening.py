from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from market_analytics.cache import CacheEntry, CacheMetadata, FileCache
from market_analytics.database import MarketDatabase, SCHEMA_VERSION
from market_analytics.database_provider import DatabaseProvider
from market_analytics.exchange_calendar import ExchangeCalendar
from market_analytics.models import (
    CompanyMetadataRecord, CorporateActionRecord, PriceObservation,
)
from market_analytics.provider_runtime import (
    DatasetCapability, HttpResult, PaginationError, ProviderBoundaryError,
    ProviderPage, ProviderRequestController, ProviderRequestError, ProviderUsage,
    RateLimitPolicy, StoragePolicy, UsageLedger, decode_json_object, paginate,
    preflight_range, validate_object, validate_record_array,
)
from market_analytics.providers import SyntheticProviderFixture


UTC = timezone.utc
D = Decimal
ROOT = Path(__file__).resolve().parents[1]
PROVIDER_FIXTURE = ROOT / "fixtures" / "provider-foundation-synthetic.json"
FAILURE_FIXTURE = ROOT / "fixtures" / "provider-failure-matrix-synthetic.json"


class FakeTime:
    def __init__(self) -> None:
        self.seconds = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.seconds

    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self.seconds)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.seconds += seconds


class PaginationTests(unittest.TestCase):
    def test_one_page_and_page_provenance(self) -> None:
        result = paginate(
            lambda cursor: ProviderPage(items=[{"id": 1}], provenance={"request_id": "a"}),
            max_pages=2,
        )
        self.assertEqual(result.items, [{"id": 1}])
        self.assertEqual(result.pages[0].source, {"request_id": "a"})

    def test_multi_page_preserves_stable_order(self) -> None:
        pages = {
            None: ProviderPage(items=[{"id": 1}], next_cursor="two"),
            "two": ProviderPage(items=[{"id": 2}], next_cursor=None),
        }
        result = paginate(lambda cursor: pages[cursor], max_pages=3, stable_key=lambda x: x["id"])
        self.assertEqual([item["id"] for item in result.items], [1, 2])

    def test_repeated_cursor_is_rejected(self) -> None:
        with self.assertRaisesRegex(PaginationError, "repeated"):
            paginate(
                lambda cursor: ProviderPage(items=[{"cursor": cursor}], next_cursor="same"),
                max_pages=3,
            )

    def test_duplicate_nonempty_page_is_rejected(self) -> None:
        with self.assertRaisesRegex(PaginationError, "duplicate page"):
            paginate(
                lambda cursor: ProviderPage(
                    items=[{"id": 1}], next_cursor="next" if cursor is None else None
                ),
                max_pages=3,
            )

    def test_empty_middle_page_with_continuation_is_rejected(self) -> None:
        pages = {
            None: ProviderPage(items=[{"id": 1}], next_cursor="two"),
            "two": ProviderPage(items=[], next_cursor="unexpected"),
        }
        with self.assertRaisesRegex(PaginationError, "unresolved continuation"):
            paginate(lambda cursor: pages[cursor], max_pages=4)

    def test_malformed_pagination_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(PaginationError, "malformed"):
            paginate(lambda cursor: {"items": [], "next_cursor": 17}, max_pages=2)  # type: ignore[arg-type,return-value]

    def test_hard_page_limit_and_unstable_order_are_rejected(self) -> None:
        with self.assertRaisesRegex(PaginationError, "hard maximum"):
            paginate(
                lambda cursor: ProviderPage(items=[{"id": cursor or "0"}], next_cursor=str(int(cursor or "0") + 1)),
                max_pages=2,
            )
        pages = {
            None: ProviderPage(items=[{"id": 2}], next_cursor="two"),
            "two": ProviderPage(items=[{"id": 1}]),
        }
        with self.assertRaisesRegex(PaginationError, "stable order"):
            paginate(lambda cursor: pages[cursor], max_pages=2, stable_key=lambda x: x["id"])


class RateLimitTests(unittest.TestCase):
    def controller(self, fake: FakeTime, **updates: object) -> ProviderRequestController:
        values = {
            "requests_per_minute": 5, "daily_quota": None,
            "max_retries": 2, "base_backoff_seconds": 1,
            "max_backoff_seconds": 10, "max_retry_after_seconds": 5,
        }
        values.update(updates)
        return ProviderRequestController(
            provider="synthetic", policy=RateLimitPolicy(**values),
            clock=fake.clock, utc_now=fake.now, sleeper=fake.sleep,
        )

    def test_429_honors_bounded_retry_after_then_succeeds(self) -> None:
        fake = FakeTime()
        controller = self.controller(fake)
        responses = iter([
            HttpResult(status=429, body="{}", headers={"Retry-After": "9"}),
            HttpResult(status=200, body="{}"),
        ])
        self.assertEqual(controller.request(lambda: next(responses)).status, 200)
        self.assertEqual(fake.sleeps, [5])
        self.assertEqual(controller.usage.retries, 1)

    def test_billable_calls_count_only_actual_send_attempts(self) -> None:
        successful = self.controller(FakeTime())
        successful.request(lambda: HttpResult(status=200, body="{}"))
        self.assertEqual(successful.usage.estimated_billable_calls, 1)

        retried = self.controller(FakeTime())
        responses = iter([
            HttpResult(status=500, body="{}"),
            HttpResult(status=200, body="{}"),
        ])
        retried.request(lambda: next(responses))
        self.assertEqual(retried.usage.estimated_billable_calls, 2)

        permanent = self.controller(FakeTime())
        with self.assertRaises(ProviderRequestError):
            permanent.request(lambda: HttpResult(status=404, body="{}"))
        self.assertEqual(permanent.usage.estimated_billable_calls, 1)

        rate_limited = self.controller(FakeTime())
        responses = iter([
            HttpResult(status=429, body="{}"),
            HttpResult(status=200, body="{}"),
        ])
        rate_limited.request(lambda: next(responses))
        self.assertEqual(rate_limited.usage.estimated_billable_calls, 2)

        quota_limited = self.controller(FakeTime(), daily_quota=1)
        quota_limited.request(lambda: HttpResult(status=200, body="{}"))
        with self.assertRaises(ProviderRequestError):
            quota_limited.request(lambda: HttpResult(status=200, body="{}"))
        self.assertEqual(quota_limited.usage.estimated_billable_calls, 1)

    def test_malformed_retry_after_uses_bounded_backoff(self) -> None:
        fake = FakeTime()
        controller = self.controller(fake)
        responses = iter([
            HttpResult(status=429, body="{}", headers={"retry-after": "nonsense"}),
            HttpResult(status=200, body="{}"),
        ])
        controller.request(lambda: next(responses))
        self.assertEqual(fake.sleeps, [1])

    def test_repeated_throttling_stops(self) -> None:
        fake = FakeTime()
        controller = self.controller(fake, max_retries=1)
        with self.assertRaisesRegex(ProviderRequestError, "remained rate limited"):
            controller.request(lambda: HttpResult(status=429, body="{}"))
        self.assertEqual(controller.usage.requests_attempted, 2)

    def test_daily_quota_exhaustion_is_a_hard_stop(self) -> None:
        fake = FakeTime()
        controller = self.controller(fake, daily_quota=1)
        controller.request(lambda: HttpResult(status=200, body="{}"))
        with self.assertRaisesRegex(ProviderRequestError, "quota exhausted"):
            controller.request(lambda: HttpResult(status=200, body="{}"))
        self.assertEqual(controller.usage.requests_attempted, 1)

    def test_permanent_4xx_errors_are_not_retried(self) -> None:
        for status in (401, 403, 404):
            fake = FakeTime()
            controller = self.controller(fake)
            with self.subTest(status=status), self.assertRaises(ProviderRequestError) as raised:
                controller.request(lambda: HttpResult(status=status, body="secret body"))
            self.assertEqual(raised.exception.status, status)
            self.assertNotIn("secret", str(raised.exception))
            self.assertEqual(controller.usage.requests_attempted, 1)

    def test_network_failure_retries_are_bounded_and_sanitized(self) -> None:
        fake = FakeTime()
        controller = self.controller(fake, max_retries=1)
        with self.assertRaisesRegex(ProviderRequestError, "bounded retries") as raised:
            controller.request(lambda: (_ for _ in ()).throw(RuntimeError("secret-token")))
        self.assertNotIn("secret-token", str(raised.exception))
        self.assertEqual(controller.usage.retries, 1)

    def test_requests_per_minute_paces_attempts(self) -> None:
        fake = FakeTime()
        controller = self.controller(fake, requests_per_minute=1, max_retries=0)
        controller.request(lambda: HttpResult(status=200, body="{}"))
        controller.request(lambda: HttpResult(status=200, body="{}"))
        self.assertEqual(fake.sleeps, [60])


class PayloadAndSchemaTests(unittest.TestCase):
    def test_empty_malformed_wrong_root_and_declared_error_fail_closed(self) -> None:
        cases = (
            HttpResult(status=200, body=""),
            HttpResult(status=200, body="{"),
            HttpResult(status=200, body="[]"),
            HttpResult(status=200, body='{"error":"bad key"}'),
        )
        for response in cases:
            with self.subTest(body=response.body), self.assertRaises(ProviderBoundaryError):
                decode_json_object(response)

    def test_unknown_fields_are_tolerated_but_required_drift_fails(self) -> None:
        required = {"ticker": str, "close": (int, float)}
        valid = validate_object({"ticker": "FICT", "close": 10.5, "new_field": 1}, required)
        self.assertEqual(valid["new_field"], 1)
        for payload in (
            {"ticker": "FICT"},
            {"symbol": "FICT", "close": 10.5},
            {"ticker": "FICT", "close": "10.5"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ProviderBoundaryError):
                validate_object(payload, required)

    def test_truncated_array_fails_whole_response(self) -> None:
        with self.assertRaisesRegex(ProviderBoundaryError, "truncated"):
            validate_record_array(
                {"results": [{"ticker": "A"}], "count": 2}, field_name="results",
                required={"ticker": str}, expected_count_field="count",
            )

    def test_malformed_records_are_isolated_when_envelope_is_complete(self) -> None:
        batch = validate_record_array(
            {"results": [{"ticker": "GOOD"}, {"symbol": "BAD"}], "count": 2},
            field_name="results", required={"ticker": str}, expected_count_field="count",
        )
        self.assertEqual(batch.valid, [{"ticker": "GOOD"}])
        self.assertEqual(len(batch.errors), 1)


class RevisionAndIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.database = MarketDatabase(Path(self.folder.name) / "revisions.sqlite3")
        self.database.migrate()
        self.fixture = SyntheticProviderFixture(PROVIDER_FIXTURE)

    def tearDown(self) -> None:
        self.database.close()
        self.folder.cleanup()

    def company(self, ticker: str, name: str, *, with_identity: bool = True) -> int:
        company_id = self.database.upsert_company(
            ticker=ticker, canonical_name=name, exchange=None, cik=None
        )
        if with_identity:
            self.database.insert_company_metadata(company_id, CompanyMetadataRecord(
                ticker=ticker, company_name=name, source="fixture-identity",
                source_record_id=f"{ticker.lower()}-identity-v1",
                effective_from="2020-01-01T00:00:00Z",
                observed_at="2020-01-01T00:00:00Z",
                available_at="2020-01-01T00:00:00Z",
                retrieved_at="2020-01-01T00:00:00Z",
            ))
        return company_id

    def test_metadata_correction_is_point_in_time_and_preserves_prior_row(self) -> None:
        company_id = self.company("META", "Metadata Test", with_identity=False)
        base = CompanyMetadataRecord(
            ticker="META", company_name="Metadata Test", sector="Old Sector",
            source="fixture", source_record_id="meta-1",
            effective_from="2025-01-01T00:00:00Z", observed_at="2025-01-01T01:00:00Z",
            available_at="2025-01-01T02:00:00Z", retrieved_at="2025-01-01T03:00:00Z",
        )
        corrected = CompanyMetadataRecord.model_validate({
            **base.model_dump(mode="json"), "sector": "Correct Sector",
            "observed_at": "2025-02-01T01:00:00Z", "available_at": "2025-02-01T02:00:00Z",
            "retrieved_at": "2025-02-01T03:00:00Z",
        })
        self.database.insert_company_metadata(company_id, base)
        self.database.insert_company_metadata(company_id, corrected)
        early = DatabaseProvider(self.database, datetime(2025, 1, 15, tzinfo=UTC)).company("META")
        late = DatabaseProvider(self.database, datetime(2025, 2, 2, tzinfo=UTC)).company("META")
        self.assertEqual((early.sector, late.sector), ("Old Sector", "Correct Sector"))
        self.assertEqual(self.database.connection.execute("SELECT COUNT(*) FROM company_metadata_history").fetchone()[0], 2)

    def test_price_correction_is_versioned_and_current_is_deterministic(self) -> None:
        company_id = self.company("FICN", "Fictional Compute")
        base = self.fixture.prices(
            "FICN", date(2025, 6, 2), date(2025, 6, 2),
            datetime(2025, 6, 3, tzinfo=UTC),
        )[0]
        corrected = PriceObservation.model_validate({
            **base.model_dump(mode="json"), "close": "103",
            "observed_at": "2025-06-04T20:00:00Z", "available_at": "2025-06-04T20:05:00Z",
            "retrieved_at": "2025-06-04T21:00:00Z",
        })
        self.assertEqual(self.database.insert_price(company_id, base), 1)
        self.assertEqual(self.database.insert_price(company_id, corrected), 1)
        self.assertEqual(self.database.insert_price(company_id, corrected), 0)
        early = DatabaseProvider(self.database, datetime(2025, 6, 3, tzinfo=UTC))
        late = DatabaseProvider(self.database, datetime(2025, 6, 5, tzinfo=UTC))
        self.assertEqual(early.get_price_observations("FICN", date(2025, 6, 2), date(2025, 6, 2))[0].close, D("102"))
        self.assertEqual(late.get_price_observations("FICN", date(2025, 6, 2), date(2025, 6, 2))[0].close, D("103"))
        self.assertEqual(len(self.database.price_revision_history(company_id, date(2025, 6, 2), "synthetic-prices", base.source_record_id)), 2)

    def test_price_series_preserves_every_trading_date(self) -> None:
        company_id = self.company("FICN", "Fictional Compute")
        records = self.fixture.prices(
            "FICN", date(2025, 6, 2), date(2025, 6, 3),
            datetime(2025, 6, 5, tzinfo=UTC),
        )
        for record in records:
            shared_series_record = PriceObservation.model_validate({
                **record.model_dump(mode="json"),
                "source_record_id": "FICN:daily",
            })
            self.assertEqual(self.database.insert_price(company_id, shared_series_record), 1)

        observations = DatabaseProvider(
            self.database, datetime(2025, 6, 5, tzinfo=UTC)
        ).get_price_observations("FICN", date(2025, 6, 2), date(2025, 6, 3))

        self.assertEqual(
            [(item.trading_date, item.close) for item in observations],
            [(date(2025, 6, 2), D("102")), (date(2025, 6, 3), D("101"))],
        )

    def test_price_series_keeps_independent_sources_for_the_same_date(self) -> None:
        company_id = self.company("FICN", "Fictional Compute")
        primary = self.fixture.prices(
            "FICN", date(2025, 6, 2), date(2025, 6, 2),
            datetime(2025, 6, 3, tzinfo=UTC),
        )[0]
        primary = PriceObservation.model_validate({
            **primary.model_dump(mode="json"),
            "source_record_id": "FICN:daily",
        })
        secondary = PriceObservation.model_validate({
            **primary.model_dump(mode="json"),
            "close": "101",
            "source": "synthetic-prices-secondary",
        })
        self.assertEqual(self.database.insert_price(company_id, primary), 1)
        self.assertEqual(self.database.insert_price(company_id, secondary), 1)

        observations = DatabaseProvider(
            self.database, datetime(2025, 6, 3, tzinfo=UTC)
        ).get_price_observations("FICN", date(2025, 6, 2), date(2025, 6, 2))

        self.assertEqual(
            [(item.source, item.close) for item in observations],
            [("synthetic-prices", D("102")), ("synthetic-prices-secondary", D("101"))],
        )

    def test_corporate_action_correction_is_point_in_time(self) -> None:
        company_id = self.company("FICN", "Fictional Compute")
        base = next(item for item in self.fixture._actions if item.source_record_id == "action-cash-dividend")
        corrected = CorporateActionRecord.model_validate({
            **base.model_dump(mode="json"), "amount": "0.30",
            "observed_at": "2025-03-20T15:00:00Z", "available_at": "2025-03-20T15:05:00Z",
            "retrieved_at": "2025-03-20T16:00:00Z",
        })
        self.database.insert_corporate_action(company_id, base)
        self.database.insert_corporate_action(company_id, corrected)
        early = DatabaseProvider(self.database, datetime(2025, 3, 15, tzinfo=UTC))
        late = DatabaseProvider(self.database, datetime(2025, 3, 25, tzinfo=UTC))
        self.assertEqual(early.get_corporate_actions("FICN", date(2025, 1, 1), date(2025, 12, 31))[0].amount, D("0.25"))
        self.assertEqual(late.get_corporate_actions("FICN", date(2025, 1, 1), date(2025, 12, 31))[0].amount, D("0.30"))
        self.assertEqual(len(self.database.corporate_action_revision_history(company_id, "synthetic-actions", "action-cash-dividend")), 2)

    def test_retrieval_history_is_separate_from_normalized_idempotency(self) -> None:
        for hour in (1, 2):
            self.assertEqual(self.database.record_provider_retrieval(
                provider="fixture", dataset="prices", request_identifier="AAPL:prices",
                retrieved_at=f"2026-01-01T0{hour}:00:00+00:00", content_hash="abc",
                http_status=200, cache_status="MISS", provenance={"page": 1},
            ), 1)
        self.assertEqual(self.database.connection.execute("SELECT COUNT(*) FROM provider_retrieval_events").fetchone()[0], 2)

    def test_ticker_reuse_resolves_by_identity_and_effective_time(self) -> None:
        old_id = self.company("NEW", "Original SAME Company", with_identity=False)
        new_id = self.company("SAME", "Reuser of SAME", with_identity=False)
        records = (
            (old_id, CompanyMetadataRecord(
                ticker="SAME", company_name="Original SAME Company", source="fixture",
                source_record_id="old-same", effective_from="2020-01-01T00:00:00Z",
                effective_to="2025-01-01T00:00:00Z", observed_at="2020-01-01T01:00:00Z",
                available_at="2020-01-01T02:00:00Z", retrieved_at="2020-01-01T03:00:00Z",
            )),
            (new_id, CompanyMetadataRecord(
                ticker="SAME", company_name="Reuser of SAME", source="fixture",
                source_record_id="new-same", effective_from="2025-01-01T00:00:00Z",
                observed_at="2025-01-01T01:00:00Z", available_at="2025-01-01T02:00:00Z",
                retrieved_at="2025-01-01T03:00:00Z",
            )),
        )
        for company_id, record in records:
            self.database.insert_company_metadata(company_id, record)
        self.assertEqual(DatabaseProvider(self.database, datetime(2024, 1, 1, tzinfo=UTC)).company("SAME").company, "Original SAME Company")
        self.assertEqual(DatabaseProvider(self.database, datetime(2026, 1, 1, tzinfo=UTC)).company("SAME").company, "Reuser of SAME")

    def test_exchange_change_inactive_period_relisting_and_historical_universe(self) -> None:
        company_id = self.company("LIFE", "Lifecycle Corp", with_identity=False)
        records = (
            CompanyMetadataRecord(
                ticker="LIFE", company_name="Lifecycle Corp", exchange="XNYS", active=True,
                source="fixture", source_record_id="life-1", effective_from="2020-01-01T00:00:00Z",
                effective_to="2024-01-01T00:00:00Z", observed_at="2020-01-01T01:00:00Z",
                available_at="2020-01-01T02:00:00Z", retrieved_at="2020-01-01T03:00:00Z",
            ),
            CompanyMetadataRecord(
                ticker="LIFE", company_name="Lifecycle Corp", exchange="XNAS", active=False,
                delisting_date="2024-01-01", source="fixture", source_record_id="life-2",
                effective_from="2024-01-01T00:00:00Z", effective_to="2025-01-01T00:00:00Z",
                observed_at="2024-01-01T01:00:00Z", available_at="2024-01-01T02:00:00Z",
                retrieved_at="2024-01-01T03:00:00Z",
            ),
            CompanyMetadataRecord(
                ticker="LIFE", company_name="Lifecycle Corp", exchange="XNAS", active=True,
                source="fixture", source_record_id="life-3", effective_from="2025-01-01T00:00:00Z",
                observed_at="2025-01-01T01:00:00Z", available_at="2025-01-01T02:00:00Z",
                retrieved_at="2025-01-01T03:00:00Z",
            ),
        )
        for record in records:
            self.database.insert_company_metadata(company_id, record)
        inactive = DatabaseProvider(self.database, datetime(2024, 6, 1, tzinfo=UTC))
        relisted = DatabaseProvider(self.database, datetime(2025, 6, 1, tzinfo=UTC))
        self.assertIn("LIFE", [item.ticker for item in inactive.list_companies()])
        self.assertNotIn("LIFE", [item.ticker for item in inactive.list_active_companies()])
        self.assertIn("LIFE", [item.ticker for item in relisted.list_active_companies()])
        self.assertEqual(relisted.company("LIFE").exchange, "XNAS")


class CalendarAndActionTests(unittest.TestCase):
    def calendar(self) -> ExchangeCalendar:
        return ExchangeCalendar(
            exchange="XFIC", timezone_name="America/New_York",
            regular_open=time(9, 30), regular_close=time(16),
            holidays=frozenset({date(2026, 1, 1)}),
            early_closes={date(2026, 11, 27): time(13)},
        )

    def test_weekend_holiday_full_and_early_close_sessions(self) -> None:
        calendar = self.calendar()
        self.assertEqual(calendar.session(date(2026, 10, 3)).kind, "CLOSED")
        self.assertEqual(calendar.session(date(2026, 1, 1)).kind, "CLOSED")
        self.assertEqual(calendar.session(date(2026, 10, 1)).kind, "FULL")
        early = calendar.session(date(2026, 11, 27))
        self.assertEqual(early.kind, "EARLY_CLOSE")
        self.assertEqual(early.closes_at.hour, 18)

    def test_exchange_timezone_controls_trading_date(self) -> None:
        tokyo = ExchangeCalendar(
            exchange="XTKS", timezone_name="Asia/Tokyo",
            regular_open=time(9), regular_close=time(15),
        )
        self.assertEqual(
            tokyo.trading_date_for(datetime(2026, 1, 1, 16, tzinfo=UTC)),
            date(2026, 1, 2),
        )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            tokyo.trading_date_for(datetime(2026, 1, 1, 16))

    def test_action_dates_remain_distinct_and_future_event_can_be_known(self) -> None:
        action = CorporateActionRecord(
            ticker="FICT", event_type="cash_dividend", announcement_date="2026-01-02",
            declaration_date="2026-01-02", record_date="2026-01-20",
            ex_date="2026-01-19", payment_date="2026-02-01", effective_date="2026-01-19",
            amount="0.25", currency="USD", source="fixture", source_record_id="div-1",
            observed_at="2026-01-02T15:00:00Z", available_at="2026-01-02T15:05:00Z",
            retrieved_at="2026-01-02T16:00:00Z",
        )
        self.assertNotEqual(action.announcement_date, action.payment_date)
        with tempfile.TemporaryDirectory() as folder, MarketDatabase(Path(folder) / "a.sqlite3") as db:
            db.migrate()
            company_id = db.upsert_company(ticker="FICT", canonical_name="Fictional", exchange=None, cik=None)
            db.insert_company_metadata(company_id, CompanyMetadataRecord(
                ticker="FICT", company_name="Fictional", source="fixture-identity",
                source_record_id="fict-identity-v1",
                effective_from="2020-01-01T00:00:00Z",
                observed_at="2020-01-01T00:00:00Z",
                available_at="2020-01-01T00:00:00Z",
                retrieved_at="2020-01-01T00:00:00Z",
            ))
            db.insert_corporate_action(company_id, action)
            before = DatabaseProvider(db, datetime(2026, 1, 2, 15, 4, tzinfo=UTC))
            known = DatabaseProvider(db, datetime(2026, 1, 3, tzinfo=UTC))
            self.assertEqual(before.get_corporate_actions("FICT", date(2026, 1, 1), date(2026, 2, 28)), [])
            self.assertEqual(len(known.get_corporate_actions("FICT", date(2026, 1, 1), date(2026, 2, 28))), 1)


class AdjustmentCapabilityAndCacheTests(unittest.TestCase):
    def base_price(self, **updates: object) -> dict[str, object]:
        values: dict[str, object] = {
            "ticker": "FICT", "trading_date": "2026-01-02", "close": "100",
            "currency": "USD", "source": "fixture", "source_record_id": "p1",
            "observed_at": "2026-01-02T21:00:00Z", "available_at": "2026-01-02T21:05:00Z",
            "retrieved_at": "2026-01-02T22:00:00Z",
        }
        values.update(updates)
        return values

    def test_ambiguous_or_unbound_adjustments_fail_closed(self) -> None:
        for updates in (
            {"provider_adjusted_close": "99"},
            {"split_adjusted_close": "50", "adjustment_origin": "internally_derived"},
            {"total_return_index": "1000", "adjustment_origin": "provider_supplied"},
            {"dividend_treatment": "cash_reinvested"},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                PriceObservation.model_validate(self.base_price(**updates))

    def test_explicit_adjustment_series_do_not_silently_convert(self) -> None:
        price = PriceObservation.model_validate(self.base_price(
            split_adjusted_close="50", adjustment_origin="internally_derived",
            split_event_ids=["split:1"],
        ))
        self.assertEqual(price.value_for("split_adjusted_close"), D("50"))
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            price.value_for("dividend_adjusted_close")

    def test_capability_preflight_fails_unverified_and_out_of_depth(self) -> None:
        policy = StoragePolicy(redistribution="REVIEW_REQUIRED", local_storage="UNVERIFIED")
        unverified = DatasetCapability(
            dataset="prices", status="UNVERIFIED", requires_auth=True,
            storage_policy=policy,
        )
        self.assertFalse(preflight_range(
            unverified, date(2026, 1, 1), date(2026, 1, 2), as_of=date(2026, 1, 2)
        ).supported)
        supported = DatasetCapability(
            dataset="prices", status="SUPPORTED", requires_auth=True,
            historical_depth_days=30, requests_per_minute=5,
            pagination_style="cursor", adjusted_price_semantics="split-only",
            correction_behavior="versioned", point_in_time_quality="retrieval-time only",
            storage_policy=policy,
        )
        decision = preflight_range(
            supported, date(2025, 1, 1), date(2026, 1, 2), as_of=date(2026, 1, 2)
        )
        self.assertEqual(decision.status, "REQUIRES_HIGHER_TIER")

    def test_usage_ledger_tracks_provider_and_run_totals_without_invented_cost(self) -> None:
        ledger = UsageLedger()
        ledger.record(ProviderUsage(
            provider="fixture", requests_attempted=2, requests_succeeded=1,
            requests_failed=1, retries=1, quota_units=2, quota_remaining=8,
        ))
        ledger.record(ProviderUsage(provider="other", requests_attempted=1, requests_succeeded=1))
        self.assertEqual(ledger.totals()["fixture"].quota_remaining, 8)
        self.assertEqual(ledger.run_total().requests_attempted, 3)
        self.assertEqual(ledger.run_total().estimated_cost, 0)

    def test_stale_cache_is_visible_but_never_returned_as_fresh(self) -> None:
        now = datetime(2026, 1, 2, tzinfo=UTC)
        payload = {"value": 1}
        packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        entry = CacheEntry(metadata=CacheMetadata(
            source="fixture", retrieval_timestamp=now - timedelta(days=2),
            expiry_timestamp=now - timedelta(days=1), request_identifier="fixture:stale",
            content_hash=hashlib.sha256(packed.encode()).hexdigest(),
        ), payload=payload)
        with tempfile.TemporaryDirectory() as folder:
            cache = FileCache(Path(folder))
            cache.put(entry)
            self.assertEqual(cache.lookup("fixture:stale", now).status, "STALE")
            self.assertIsNone(cache.get("fixture:stale", now))


class FailureMatrixTests(unittest.TestCase):
    def test_compact_matrix_is_labeled_and_covers_every_required_failure(self) -> None:
        payload = json.loads(FAILURE_FIXTURE.read_text(encoding="utf-8"))
        self.assertIn("SYNTHETIC", payload["fixture_notice"])
        names = {item["name"] for item in payload["cases"]}
        self.assertEqual(names, {
            "auth_failure", "rate_limit", "quota_exhausted", "timeout",
            "malformed_json", "partial_response", "pagination_loop", "schema_drift",
            "stale_cache", "corrected_data", "ticker_reuse", "delisting",
            "future_dated_record", "ambiguous_adjustment_basis",
            "unsupported_historical_range",
        })


if __name__ == "__main__":
    unittest.main()
