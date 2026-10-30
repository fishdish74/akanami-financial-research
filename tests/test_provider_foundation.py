from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from market_analytics.database import MarketDatabase, SCHEMA_VERSION
from market_analytics.database_provider import DatabaseProvider
from market_analytics.models import PriceObservation
from market_analytics.peers import compare_preferred_peers
from market_analytics.providers import SyntheticProviderFixture


D = Decimal
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "provider-foundation-synthetic.json"


class ProviderFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = SyntheticProviderFixture(FIXTURE)

    def test_fixture_is_explicitly_synthetic_and_covers_required_actions(self) -> None:
        self.assertIn("SYNTHETIC", self.fixture.fixture_notice)
        cutoff = datetime(2026, 12, 31, tzinfo=timezone.utc)
        actions = self.fixture.corporate_actions(
            "FICN", date(2025, 1, 1), date(2026, 12, 31), cutoff
        )
        self.assertEqual(
            {item.event_type for item in actions},
            {
                "stock_split", "reverse_split", "cash_dividend", "stock_dividend",
                "ticker_change", "merger", "acquisition", "spin_off", "delisting",
                "relisting",
            },
        )

    def test_metadata_history_and_ticker_change_resolve_at_exact_cutoffs(self) -> None:
        before = self.fixture.metadata(
            "FICO", datetime(2025, 5, 31, 23, 59, 59, tzinfo=timezone.utc)
        )
        at_change = self.fixture.metadata(
            "FICO", datetime(2025, 6, 1, tzinfo=timezone.utc)
        )
        self.assertEqual([item.ticker for item in before], ["FICO"])
        self.assertEqual([item.ticker for item in at_change], ["FICN"])
        self.assertEqual(at_change[0].prior_ticker, "FICO")

    def test_future_metadata_prices_and_actions_are_excluded(self) -> None:
        cutoff = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        self.assertEqual(self.fixture.metadata("FUTR", cutoff), [])
        self.assertEqual(
            [item.trading_date for item in self.fixture.prices(
                "FICN", date(2025, 1, 1), date(2026, 12, 31), cutoff
            )],
            [date(2025, 6, 2), date(2025, 6, 3)],
        )
        self.assertNotIn(
            "action-future",
            [item.source_record_id for item in self.fixture.corporate_actions(
                "FICN", date(2025, 1, 1), date(2026, 12, 31), cutoff
            )],
        )

    def test_missing_adjusted_close_fails_closed_for_adjusted_series(self) -> None:
        item = self.fixture.prices(
            "FICN", date(2025, 6, 3), date(2025, 6, 3),
            datetime(2025, 6, 4, tzinfo=timezone.utc),
        )[0]
        self.assertEqual(item.value_for("raw_close"), D("101"))
        with self.assertRaisesRegex(ValueError, "adjustment methods cannot be mixed"):
            item.value_for("split_adjusted_close")

    def test_provider_adjusted_close_requires_explicit_basis(self) -> None:
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))["prices"][0]
        raw["provider_adjustment_basis"] = None
        with self.assertRaises(ValidationError):
            PriceObservation.model_validate(raw)

    def test_naive_provider_cutoff_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.fixture.metadata("FICN", datetime(2025, 6, 2))


class ProviderStorageTests(unittest.TestCase):
    def test_v2_migration_preserves_legacy_price_without_guessing_adjustment(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v2.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE schema_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO schema_metadata VALUES ('schema_version', '2');
                CREATE TABLE companies(id INTEGER PRIMARY KEY, ticker TEXT NOT NULL UNIQUE);
                INSERT INTO companies VALUES (7, 'LEGC');
                CREATE TABLE prices(
                    id INTEGER PRIMARY KEY, company_id INTEGER NOT NULL,
                    price_date TEXT NOT NULL, open TEXT, high TEXT, low TEXT,
                    close TEXT NOT NULL, adjusted_close TEXT, volume TEXT,
                    currency TEXT NOT NULL, source TEXT NOT NULL,
                    available_at TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                    source_identifier TEXT NOT NULL,
                    UNIQUE(company_id, price_date, source, source_identifier)
                );
                INSERT INTO prices VALUES(
                    1, 7, '2025-01-02', '10', '11', '9', '10.5', '10.25',
                    '100', 'USD', 'legacy', '2025-01-02T21:00:00+00:00',
                    '2025-01-02T22:00:00+00:00', 'legacy-1'
                );
                """
            )
            connection.close()
            with MarketDatabase(path) as database:
                self.assertEqual(database.migrate(), SCHEMA_VERSION)
                row = database.connection.execute("SELECT * FROM prices").fetchone()
                self.assertEqual(row["ticker_at_observation"], "LEGC")
                self.assertEqual(row["observed_at"], row["available_at"])
                self.assertEqual(row["provider_adjustment_basis"], "unknown")
                self.assertEqual(row["split_adjusted_close"], None)

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "provider.sqlite3"
        self.database = MarketDatabase(self.path)
        self.assertEqual(self.database.migrate(), SCHEMA_VERSION)
        self.fixture = SyntheticProviderFixture(FIXTURE)
        self.company_id = self.database.upsert_company(
            ticker="FICN", canonical_name="Fictional Compute New, Inc.",
            exchange=None, cik="1234567",
        )

    def tearDown(self) -> None:
        self.database.close()
        self.folder.cleanup()

    def test_ingestion_is_idempotent_and_preserves_provenance(self) -> None:
        metadata = self.fixture.metadata(
            "FICO", datetime(2025, 5, 31, tzinfo=timezone.utc)
        )[0]
        price = self.fixture.prices(
            "FICN", date(2025, 6, 2), date(2025, 6, 2),
            datetime(2025, 6, 3, tzinfo=timezone.utc),
        )[0]
        action = self.fixture.corporate_actions(
            "FICN", date(2025, 2, 3), date(2025, 2, 3),
            datetime(2025, 2, 4, tzinfo=timezone.utc),
        )[0]
        for record, insert in (
            (metadata, self.database.insert_company_metadata),
            (price, self.database.insert_price),
            (action, self.database.insert_corporate_action),
        ):
            self.assertEqual(insert(self.company_id, record), 1)
            self.assertEqual(insert(self.company_id, record), 0)
        later_retrieval = metadata.model_copy(
            update={"retrieved_at": datetime(2025, 6, 1, tzinfo=timezone.utc)}
        )
        self.assertEqual(
            self.database.insert_company_metadata(self.company_id, later_retrieval), 0
        )
        self.assertEqual(
            json.loads(self.database.connection.execute(
                "SELECT provenance_json FROM company_metadata_history"
            ).fetchone()[0]),
            {"fixture": True},
        )

    def test_database_reads_exclude_future_records_and_resolve_old_ticker(self) -> None:
        for record in self.fixture._metadata:  # fixture records are the deterministic ingestion input
            if record.ticker in {"FICO", "FICN"}:
                self.database.insert_company_metadata(self.company_id, record)
        for record in self.fixture._prices:
            self.database.insert_price(self.company_id, record)
        for record in self.fixture._actions:
            self.database.insert_corporate_action(self.company_id, record)

        old = DatabaseProvider(
            self.database, datetime(2025, 5, 31, 23, 59, 59, tzinfo=timezone.utc)
        ).company("FICO")
        current_provider = DatabaseProvider(
            self.database, datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        )
        current = current_provider.company("FICO")
        self.assertEqual(old.ticker, "FICO")
        self.assertEqual(current.ticker, "FICN")
        self.assertEqual(
            [item.trading_date for item in current_provider.get_price_observations(
                "FICO", date(2025, 1, 1), date(2026, 12, 31)
            )],
            [date(2025, 6, 2), date(2025, 6, 3)],
        )
        self.assertNotIn(
            "action-future",
            [item.source_record_id for item in current_provider.get_corporate_actions(
                "FICO", date(2025, 1, 1), date(2026, 12, 31)
            )],
        )

    def test_conflicting_metadata_stays_unknown_and_is_auditable(self) -> None:
        conflict_id = self.database.upsert_company(
            ticker="CNFL", canonical_name="Fictional Conflict Corp.", exchange=None, cik=None
        )
        cutoff = datetime(2025, 2, 1, tzinfo=timezone.utc)
        for record in self.fixture.metadata("CNFL", cutoff):
            self.database.insert_company_metadata(conflict_id, record)
        company = DatabaseProvider(self.database, cutoff).company("CNFL")
        self.assertEqual(company.sector, "Unknown")
        self.assertEqual(company.industry, "Unknown")
        self.assertTrue(any("conflicting_fields" in item for item in company.metadata_provenance))

    def test_schema_v3_has_foreign_keys_and_integrity(self) -> None:
        self.assertEqual(self.database.connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertEqual(self.database.connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertEqual(
            {
                row[0] for row in self.database.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }.issuperset({"company_metadata_history", "prices", "corporate_actions"}),
            True,
        )


class ProviderPeerTests(unittest.TestCase):
    def test_industry_is_preferred_and_sector_fallback_is_explicit(self) -> None:
        values = {
            "FICN": {"ROIC": D("0.10")}, "PEER": {"ROIC": D("0.20")},
            "SECT": {"ROIC": D("0.90")},
        }
        metadata = {
            "FICN": {"sector": "Technology", "industry": "Application Software", "provenance": ["registry:ficn"]},
            "PEER": {"sector": "Technology", "industry": "Application Software", "provenance": ["registry:peer"]},
            "SECT": {"sector": "Technology", "industry": "Computer Hardware", "provenance": ["registry:sect"]},
        }
        preferred = compare_preferred_peers(values, {"ROIC": "higher"}, metadata)
        self.assertEqual(preferred["FICN"]["ROIC"].peer_scope, "industry")
        self.assertEqual(preferred["FICN"]["ROIC"].peer_group, "Application Software")
        self.assertEqual(preferred["FICN"]["ROIC"].peer_provenance, ["registry:ficn", "registry:peer"])
        self.assertEqual(preferred["SECT"]["ROIC"].status, "missing_data")

        fallback = compare_preferred_peers(
            values, {"ROIC": "higher"}, metadata, allow_sector_fallback=True
        )
        self.assertEqual(fallback["SECT"]["ROIC"].peer_scope, "sector")
        self.assertEqual(fallback["SECT"]["ROIC"].status, "valid")

    def test_unknown_metadata_never_creates_an_unknown_peer_bucket(self) -> None:
        values = {"AAA": {"ROIC": D("0.1")}, "BBB": {"ROIC": D("0.9")}}
        metadata = {
            "AAA": {"sector": "Unknown", "industry": "Unknown", "provenance": []},
            "BBB": {"sector": "Unknown", "industry": "Unknown", "provenance": []},
        }
        comparisons = compare_preferred_peers(
            values, {"ROIC": "higher"}, metadata, allow_sector_fallback=True
        )
        self.assertEqual(comparisons["AAA"]["ROIC"].status, "missing_data")
        self.assertIsNone(comparisons["AAA"]["ROIC"].peer_group)


if __name__ == "__main__":
    unittest.main()
