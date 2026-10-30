"""M1/F4: actual SEC CLI, offline clients, disposable schema versions only."""
import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from market_analytics.cli import main
from market_analytics.database import MarketDatabase
from market_analytics.paper_persistence import PaperPersistence, PAPER_WRITE_ENV
from market_analytics.sec_ingestion import ingest_company_facts_payload, sync_sec_company
from tests.test_database_sec_integration import ENTITY, FIXTURE, RETRIEVED


class SecSchemaCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "db.sqlite3"
        self.cache = Path(self.temp.name) / "cache"
        self.client = Mock()
        self.client.company_facts.return_value = SimpleNamespace(
            payload=FIXTURE.read_bytes(), retrieved_at=RETRIEVED,
            source_url="https://example.test/companyfacts", request_identifier="fixture",
            path=self.cache / "fixture.json", from_cache=True)

    def create(self, version):
        with MarketDatabase(self.path) as db:
            db.migrate()
            if version >= 5:
                PaperPersistence(db, environment={PAPER_WRITE_ENV: "1"}).migrate_v4_to_v5()
            if version >= 6:
                db.migrate_v5_to_v6()

    def run_cli(self):
        output = io.StringIO()
        with patch("sources.sec_fundamentals.SecCompanyFactsClient", return_value=self.client) as factory, \
             patch("entities.EntityRegistry.load") as registry, contextlib.redirect_stdout(output), \
             contextlib.redirect_stderr(io.StringIO()):
            registry.return_value.resolve.return_value = ENTITY
            code = main(["sec-sync", "TEST", "--database", str(self.path), "--cache-dir", str(self.cache)])
        return code, output.getvalue(), factory

    def test_actual_cli_schema6_ingests_without_migration_and_replay_is_idempotent(self):
        self.create(6)
        with patch.object(MarketDatabase, "migrate", side_effect=AssertionError("implicit migration")):
            code, output, factory = self.run_cli()
            self.assertEqual(code, 0)
            self.assertGreater(json.loads(output)["facts_inserted"], 0)
            factory.assert_called_once()
            code, output, _ = self.run_cli()
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["facts_inserted"], 0)
        with MarketDatabase(self.path, read_only=True) as db:
            self.assertEqual(db.schema_version(), 6)
            self.assertEqual(db.connection.execute("SELECT count(*) FROM forward_paper_evaluation_events").fetchone()[0], 0)

    def test_cli_existing_v4_v5_never_runs_migrations(self):
        for version in (4, 5):
            with self.subTest(version=version):
                self.path = Path(self.temp.name) / f"v{version}.sqlite3"
                self.create(version)
                with patch.object(MarketDatabase, "migrate", side_effect=AssertionError("migration")):
                    self.assertEqual(self.run_cli()[0], 0)
                with MarketDatabase(self.path, read_only=True) as db:
                    self.assertEqual(db.schema_version(), version)

    def test_cli_explicit_sync_initializes_missing_or_empty_database_only_to_v4(self):
        for empty in (False, True):
            self.path = Path(self.temp.name) / f"empty{empty}.sqlite3"
            if empty:
                sqlite3.connect(self.path).close()
            self.assertEqual(self.run_cli()[0], 0)
            with MarketDatabase(self.path, read_only=True) as db:
                self.assertEqual(db.schema_version(), 4)
                self.assertIsNone(db.connection.execute("SELECT name FROM sqlite_master WHERE name='forward_paper_decisions'").fetchone())

    def test_existing_v3_core_upgrade_remains_v4_not_v5_or_v6(self):
        self.create(4)
        with MarketDatabase(self.path) as db:
            db.connection.executescript("DROP TABLE provider_retrieval_events; UPDATE schema_metadata SET value='3' WHERE key='schema_version';")
        self.assertEqual(self.run_cli()[0], 0)
        with MarketDatabase(self.path, read_only=True) as db:
            self.assertEqual(db.schema_version(), 4)

    def test_newer_negative_and_malformed_versions_rejected_before_provider(self):
        self.create(6)
        for version in ("7", "-1", "not-a-version"):
            with self.subTest(version=version):
                with MarketDatabase(self.path) as db:
                    db.connection.execute("UPDATE schema_metadata SET value=? WHERE key='schema_version'", (version,))
                    db.connection.commit()
                    before = tuple(db.connection.iterdump())
                code, _, factory = self.run_cli()
                self.assertEqual(code, 2)
                factory.assert_not_called()
                with MarketDatabase(self.path, read_only=True) as db:
                    self.assertEqual(tuple(db.connection.iterdump()), before)

    def test_incomplete_schema6_and_mislabeled_schema_fail_before_provider(self):
        for change in ("DROP TABLE financial_metric_lifecycle_events", "ALTER TABLE financial_source_facts RENAME COLUMN taxonomy TO broken", "UPDATE schema_metadata SET value='4' WHERE key='schema_version'", "UPDATE schema_metadata SET value='3' WHERE key='schema_version'"):
            self.path = Path(self.temp.name) / f"bad{len(list(Path(self.temp.name).iterdir()))}.sqlite3"
            self.create(6)
            with MarketDatabase(self.path) as db:
                db.connection.executescript(change)
                before = tuple(db.connection.iterdump())
            code, _, factory = self.run_cli()
            self.assertEqual(code, 2)
            factory.assert_not_called()
            with MarketDatabase(self.path, read_only=True) as db:
                self.assertEqual(tuple(db.connection.iterdump()), before)

    def test_empty_version_metadata_retains_explicit_core_initialization(self):
        with sqlite3.connect(self.path) as c:
            c.execute("CREATE TABLE schema_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.assertEqual(self.run_cli()[0], 0)
        with MarketDatabase(self.path, read_only=True) as db:
            self.assertEqual(db.schema_version(), 4)

    def test_unversioned_nonempty_database_is_not_initialized(self):
        with sqlite3.connect(self.path) as c:
            c.execute("CREATE TABLE unrelated(value TEXT)")
        code, _, factory = self.run_cli()
        self.assertEqual(code, 2)
        factory.assert_not_called()
        with sqlite3.connect(self.path) as c:
            self.assertEqual(c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("unrelated",)])

    def test_direct_sync_rejects_unsupported_before_fetch_without_initialization(self):
        for version in (0, 3, 7):
            self.path = Path(self.temp.name) / f"direct{version}.sqlite3"
            with MarketDatabase(self.path) as db:
                if version:
                    db.migrate()
                    db.connection.execute("UPDATE schema_metadata SET value=? WHERE key='schema_version'", (str(version),))
                    db.connection.commit()
                before = tuple(db.connection.iterdump())
                with self.assertRaises(ValueError):
                    sync_sec_company(db, ENTITY, self.client, cache_root=self.cache)
                self.client.company_facts.assert_not_called()
                self.assertEqual(tuple(db.connection.iterdump()), before)

    def test_readonly_inspection_is_allowed_but_ingestion_and_preparation_denied(self):
        self.create(6)
        with MarketDatabase(self.path, read_only=True) as db:
            before = tuple(db.connection.iterdump())
            self.assertEqual(db.require_sec_schema(), 6)
            with self.assertRaises(PermissionError):
                db.prepare_sec_schema()
            with self.assertRaises(PermissionError):
                sync_sec_company(db, ENTITY, self.client, cache_root=self.cache)
            with self.assertRaises(PermissionError):
                ingest_company_facts_payload(db, ENTITY, FIXTURE.read_bytes(), retrieved_at=RETRIEVED,
                    source_url="fixture", request_identifier="fixture")
            self.client.company_facts.assert_not_called()
            self.assertEqual(tuple(db.connection.iterdump()), before)

    def test_core_migration_owner_still_rejects_v6_and_paper_migration_requires_gate(self):
        self.create(4)
        with MarketDatabase(self.path) as db:
            with self.assertRaises(PermissionError):
                PaperPersistence(db, environment={}).migrate_v4_to_v5()
            self.assertEqual(db.schema_version(), 4)
        self.path = Path(self.temp.name) / "six.sqlite3"
        self.create(6)
        with MarketDatabase(self.path) as db:
            with self.assertRaises(RuntimeError):
                db.migrate()
            self.assertEqual(db.schema_version(), 6)

    def legacy_core(self, version=3):
        self.create(4)
        with MarketDatabase(self.path) as db:
            db.connection.execute("DROP TABLE provider_retrieval_events")
            if version < 3:
                db.connection.executescript("""
                    DROP TABLE company_metadata_history;
                    DROP TABLE corporate_actions;
                    DROP TABLE prices;
                    CREATE TABLE prices (
                        id INTEGER PRIMARY KEY, company_id INTEGER NOT NULL,
                        price_date TEXT NOT NULL, open TEXT, high TEXT, low TEXT,
                        close TEXT NOT NULL, adjusted_close TEXT, volume TEXT,
                        currency TEXT NOT NULL, source TEXT NOT NULL,
                        available_at TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                        source_identifier TEXT NOT NULL,
                        UNIQUE(company_id,price_date,source,source_identifier));
                """)
            else:
                # These are v4 additions, not prerequisites for valid v3.
                for column in ("declaration_date", "record_date", "payment_date"):
                    db.connection.execute(f"ALTER TABLE corporate_actions DROP COLUMN {column}")
            if version == 1:
                db.connection.execute("DROP INDEX uq_financial_period_logical")
            db.connection.execute("UPDATE schema_metadata SET value=? WHERE key='schema_version'", (str(version),))
            db.connection.commit()

    def rejected_legacy(self, damage, version=3):
        self.legacy_core(version)
        with MarketDatabase(self.path) as db:
            db.connection.executescript(damage)
            before = tuple(db.connection.iterdump())
            schema = db.connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
            schema = [tuple(row) for row in schema]
        code, _, factory = self.run_cli()
        self.assertEqual(code, 2)
        factory.assert_not_called()
        self.client.company_facts.assert_not_called()
        with MarketDatabase(self.path, read_only=True) as db:
            self.assertEqual(tuple(db.connection.iterdump()), before)
            self.assertEqual(db.schema_version(), version)
            self.assertEqual([tuple(row) for row in db.connection.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")], schema)
            self.assertIsNone(db.connection.execute("SELECT name FROM sqlite_master WHERE name='provider_retrieval_events'").fetchone())

    def test_rejected_v3_missing_alias_table_leaves_logical_state_unchanged(self):
        self.rejected_legacy("DROP TABLE company_aliases;")

    def test_rejected_v3_missing_currency_leaves_logical_state_unchanged(self):
        self.rejected_legacy("ALTER TABLE companies RENAME COLUMN currency TO broken_currency;")

    def test_rejected_earlier_core_missing_required_storage_does_not_partially_upgrade(self):
        for version in (1, 2):
            self.path = Path(self.temp.name) / f"bad-old-{version}.sqlite3"
            with self.subTest(version=version):
                self.rejected_legacy("DROP TABLE company_aliases;", version)

    def test_rejected_v3_missing_price_input_or_actions_table_does_not_upgrade(self):
        for i, damage in enumerate(("ALTER TABLE prices RENAME COLUMN close TO broken_close;", "DROP TABLE corporate_actions;")):
            self.path = Path(self.temp.name) / f"bad-input-{i}.sqlite3"
            with self.subTest(damage=damage):
                self.rejected_legacy(damage)

    def test_valid_legacy_versions_preserve_existing_company_data_and_stop_at_v4(self):
        for version in (1, 2, 3):
            self.path = Path(self.temp.name) / f"valid-old-{version}.sqlite3"
            self.legacy_core(version)
            with MarketDatabase(self.path) as db:
                db.upsert_company(ticker="KEEP", canonical_name="Preserved", exchange="TEST", cik="2", aliases=["Keep Alias"])
                before = tuple(db.connection.execute("SELECT * FROM companies WHERE ticker='KEEP'").fetchone())
                aliases = [tuple(r) for r in db.connection.execute("SELECT * FROM company_aliases")]
            with self.subTest(version=version):
                code, output, _ = self.run_cli()
                self.assertEqual(code, 0)
                self.assertGreater(json.loads(output)["facts_inserted"], 0)
                with MarketDatabase(self.path, read_only=True) as db:
                    self.assertEqual(db.schema_version(), 4)
                    self.assertEqual(tuple(db.connection.execute("SELECT * FROM companies WHERE ticker='KEEP'").fetchone()), before)
                    self.assertEqual([tuple(r) for r in db.connection.execute("SELECT * FROM company_aliases WHERE alias='Keep Alias'")], aliases)
                    self.assertIsNone(db.connection.execute("SELECT name FROM sqlite_master WHERE name='forward_paper_decisions'").fetchone())

    def assert_path_conflict_rejected(self, version, conflict):
        self.path = Path(self.temp.name) / f"conflict-{len(list(Path(self.temp.name).iterdir()))}.sqlite3"
        self.legacy_core(version)
        with MarketDatabase(self.path) as db:
            db.upsert_company(ticker="KEEP", canonical_name="Preserved", exchange="TEST",
                              cik="2", aliases=["Keep Alias"])
            db.connection.executescript(conflict)
            before = tuple(db.connection.iterdump())
            schema = [tuple(r) for r in db.connection.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
        code, _, factory = self.run_cli()
        self.assertEqual(code, 2)
        factory.assert_not_called()
        self.client.company_facts.assert_not_called()
        with MarketDatabase(self.path, read_only=True) as db:
            self.assertEqual(db.schema_version(), version)
            self.assertEqual(tuple(db.connection.iterdump()), before)
            self.assertEqual([tuple(r) for r in db.connection.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")], schema)

    def test_premature_retrieval_table_rejects_before_any_upgrade(self):
        for version in (1, 2, 3):
            with self.subTest(version=version):
                self.assert_path_conflict_rejected(version,
                    "CREATE TABLE provider_retrieval_events(marker TEXT); "
                    "INSERT INTO provider_retrieval_events VALUES ('preserve');")

    def test_v4_rename_and_index_conflicts_reject_before_any_upgrade(self):
        for version in (1, 2, 3):
            for conflict in (
                "CREATE TABLE prices_v3(marker TEXT);",
                "CREATE VIEW provider_retrieval_events AS SELECT id FROM companies;",
                "CREATE INDEX idx_provider_retrieval_request ON companies(ticker);",
                "DROP INDEX IF EXISTS idx_prices_semantic_revision; CREATE INDEX idx_prices_semantic_revision ON companies(ticker);",
                "DROP INDEX IF EXISTS idx_prices_available; CREATE TABLE idx_prices_available(marker TEXT);",
            ):
                with self.subTest(version=version, conflict=conflict):
                    self.assert_path_conflict_rejected(version, conflict)

    def test_v2_v3_conditional_creation_prerequisites_checked_up_front(self):
        for version in (1, 2):
            for conflict in (
                "CREATE TABLE company_metadata_history(marker TEXT);",
                "CREATE TABLE corporate_actions(marker TEXT);",
                "CREATE TABLE idx_metadata_available(marker TEXT);",
                "CREATE TABLE idx_actions_available(marker TEXT);",
                "CREATE VIEW corporate_actions AS SELECT id FROM companies;",
            ):
                with self.subTest(version=version, conflict=conflict):
                    self.assert_path_conflict_rejected(version, conflict)

    def test_v1_index_name_collision_rejects_before_upgrade(self):
        self.assert_path_conflict_rejected(1,
            "CREATE TABLE uq_financial_period_logical(marker TEXT);")

    def test_compatible_optional_outputs_and_disappearing_price_index_are_allowed(self):
        for version in (1, 2):
            self.path = Path(self.temp.name) / f"compatible-{version}.sqlite3"
            # A historically supported superset: IF NOT EXISTS tables are already
            # compatible, and a price-owned index disappears with prices_v3.
            self.legacy_core(3)
            with MarketDatabase(self.path) as db:
                db.connection.execute("UPDATE schema_metadata SET value=? WHERE key='schema_version'", (str(version),))
                db.connection.commit()
            self.assertEqual(self.run_cli()[0], 0)
            with MarketDatabase(self.path, read_only=True) as db:
                self.assertEqual(db.schema_version(), 4)
                self.assertEqual(db.connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
