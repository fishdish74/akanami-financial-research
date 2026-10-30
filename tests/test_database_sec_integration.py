from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from config import SourceRequestConfig
from entities import Entity
from market_analytics.config import load_config
from market_analytics.database import MarketDatabase, SCHEMA_VERSION
from market_analytics.database_provider import DatabaseProvider
from market_analytics.models import CompanyMetadataRecord
from market_analytics.pipeline import run_provider_pipeline
from market_analytics.pipeline import run_fixture_pipeline
from market_analytics.paper_persistence import PAPER_WRITE_ENV, PaperPersistence
from market_analytics.reports import ranking_report
from market_analytics.sec_ingestion import ingest_company_facts_payload
from market_analytics.sec_xbrl import load_concept_mapping, parse_company_facts
from sources.sec_fundamentals import RawSecCache, SecCompanyFactsClient


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sec-companyfacts-synthetic.json"
ECONOMIC_PERIOD_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "sec-companyfacts-economic-periods.json"
)
RETRIEVED = datetime(2025, 3, 2, 12, tzinfo=timezone.utc)
ENTITY = Entity(
    ticker="TEST",
    canonical_name="Synthetic Filing Test Company",
    aliases=["Synthetic Test"],
    exchange="TEST",
    cik="1",
)


class DatabaseTests(unittest.TestCase):
    def test_schema_creation_and_version_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with MarketDatabase(Path(folder) / "market.sqlite3") as database:
                self.assertEqual(database.migrate(), SCHEMA_VERSION)
                self.assertEqual(database.migrate(), SCHEMA_VERSION)
                self.assertEqual(database.database_status()["counts"]["companies"], 0)

    def test_directory_is_not_a_valid_database_path(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                MarketDatabase(Path(folder))

    def test_company_constraints_and_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as folder, MarketDatabase(Path(folder) / "db.sqlite3") as database:
            database.migrate()
            company_id = database.upsert_company(
                ticker="AAA", canonical_name="A", exchange="NYSE", cik="1", aliases=["Alias"]
            )
            self.assertGreater(company_id, 0)
            with self.assertRaises(sqlite3.IntegrityError):
                database.upsert_company(
                    ticker="BBB", canonical_name="B", exchange="NYSE", cik="1"
                )

    def test_interrupted_transaction_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as folder, MarketDatabase(Path(folder) / "db.sqlite3") as database:
            database.migrate()
            with self.assertRaises(RuntimeError):
                with database.transaction() as connection:
                    connection.execute(
                        "INSERT INTO companies(ticker, canonical_name, currency, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        ("ROLL", "Rollback", "USD", "2025-01-01", "2025-01-01"),
                    )
                    raise RuntimeError("interrupted")
            self.assertIsNone(database.get_company("ROLL"))

    def test_read_snapshot_does_not_mix_concurrent_database_states(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snapshot.sqlite3"
            with MarketDatabase(path) as reader:
                reader.migrate()
                reader.upsert_company(ticker="OLD", canonical_name="Old State", exchange=None, cik=None)
                with MarketDatabase(path) as writer:
                    with reader.read_snapshot():
                        self.assertEqual([row["ticker"] for row in reader.list_companies()], ["OLD"])
                        writer.upsert_company(ticker="NEW", canonical_name="New State", exchange=None, cik=None)
                        self.assertEqual([row["ticker"] for row in reader.list_companies()], ["OLD"])
                    self.assertEqual(
                        [row["ticker"] for row in reader.list_companies()], ["NEW", "OLD"]
                    )

    def test_v1_migration_deduplicates_null_start_periods_and_rewires_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v1.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                PRAGMA foreign_keys=ON;
                CREATE TABLE schema_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO schema_metadata VALUES ('schema_version', '1');
                CREATE TABLE companies(
                    id INTEGER PRIMARY KEY, ticker TEXT NOT NULL UNIQUE
                );
                INSERT INTO companies VALUES (1, 'TEST');
                CREATE TABLE financial_periods(
                    id INTEGER PRIMARY KEY, company_id INTEGER NOT NULL,
                    accession_number TEXT NOT NULL, period_type TEXT NOT NULL,
                    start_date TEXT, end_date TEXT NOT NULL
                );
                CREATE TABLE financial_metrics(
                    id INTEGER PRIMARY KEY,
                    period_id INTEGER NOT NULL REFERENCES financial_periods(id)
                );
                CREATE TABLE prices(
                    id INTEGER PRIMARY KEY, company_id INTEGER NOT NULL,
                    price_date TEXT NOT NULL, open TEXT, high TEXT, low TEXT,
                    close TEXT NOT NULL, adjusted_close TEXT, volume TEXT,
                    currency TEXT NOT NULL, source TEXT NOT NULL,
                    available_at TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                    source_identifier TEXT NOT NULL,
                    UNIQUE(company_id, price_date, source, source_identifier)
                );
                INSERT INTO financial_periods VALUES (10, 1, 'a', 'POINT_IN_TIME', NULL, '2025-01-01');
                INSERT INTO financial_periods VALUES (11, 1, 'a', 'POINT_IN_TIME', NULL, '2025-01-01');
                INSERT INTO financial_metrics VALUES (20, 11);
                """
            )
            connection.close()
            with MarketDatabase(path) as database:
                self.assertEqual(database.migrate(), SCHEMA_VERSION)
                self.assertEqual(
                    database.connection.execute("SELECT COUNT(*) FROM financial_periods").fetchone()[0],
                    1,
                )
                self.assertEqual(
                    database.connection.execute("SELECT period_id FROM financial_metrics").fetchone()[0],
                    10,
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    database.connection.execute(
                        "INSERT INTO financial_periods VALUES (12, 1, 'a', 'POINT_IN_TIME', NULL, '2025-01-01')"
                    )


class SecNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = FIXTURE.read_bytes()

    def test_mapping_is_versioned_and_has_core_concepts(self) -> None:
        version, mapping = load_concept_mapping()
        self.assertEqual(version, "sec-xbrl-v2")
        self.assertEqual(mapping[("us-gaap", "Revenues")].canonical_metric, "revenue")
        self.assertEqual(mapping[("us-gaap", "NetIncomeLoss")].canonical_metric, "net_income")
        self.assertEqual(
            mapping[("dei", "EntityCommonStockSharesOutstanding")].canonical_metric,
            "shares_outstanding",
        )

    def test_custom_taxonomy_name_collision_is_not_mapped_as_us_gaap(self) -> None:
        raw = json.loads(self.payload)
        raw["facts"]["custom-company"] = {
            "Revenues": raw["facts"]["us-gaap"]["Revenues"]
        }
        _, _, facts, warnings = parse_company_facts(
            json.dumps(raw).encode(), retrieved_at=RETRIEVED
        )
        self.assertFalse(
            any(
                fact.taxonomy == "custom-company" and fact.source_concept == "Revenues"
                for fact in facts
            )
        )
        self.assertTrue(any("unsupported taxonomy coverage" in item for item in warnings))

    def test_parse_normalizes_units_periods_and_provenance(self) -> None:
        cik, name, facts, warnings = parse_company_facts(self.payload, retrieved_at=RETRIEVED)
        self.assertEqual(cik, "0000000001")
        self.assertEqual(name, "Synthetic Filing Test Company")
        revenue = [fact for fact in facts if fact.canonical_metric == "revenue"]
        self.assertTrue(any(fact.period_type == "FY" for fact in revenue))
        self.assertTrue(any(fact.period_type == "Q" for fact in revenue))
        self.assertFalse(any(fact.start_date == "2025-01-01" for fact in revenue))
        self.assertTrue(all(len(fact.source_fact_id) == 64 for fact in facts))
        self.assertTrue(any("ambiguous cumulative" in warning for warning in warnings))

    def test_unsupported_taxonomy_coverage_is_explicit(self) -> None:
        raw = json.loads(self.payload)
        raw["facts"]["us-gaap"]["UnsupportedFixtureConcept"] = {"units": {"USD": []}}
        _, _, _, warnings = parse_company_facts(
            json.dumps(raw).encode(), retrieved_at=RETRIEVED
        )
        self.assertTrue(any("unsupported taxonomy coverage" in warning for warning in warnings))

    def test_free_cash_flow_is_derived_only_from_traceable_inputs(self) -> None:
        _, _, facts, _ = parse_company_facts(self.payload, retrieved_at=RETRIEVED)
        fcf = next(fact for fact in facts if fact.canonical_metric == "free_cash_flow")
        self.assertEqual(fcf.normalized_value, Decimal("15000000"))
        self.assertEqual(fcf.formula_version, "fcf-v1")
        self.assertEqual(len(fcf.derivation["inputs"]), 2)  # type: ignore[index]

    def test_duplicate_source_fact_is_deduplicated(self) -> None:
        raw = json.loads(self.payload)
        rows = raw["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
        rows.append(dict(rows[0]))
        _, _, facts, _ = parse_company_facts(json.dumps(raw).encode(), retrieved_at=RETRIEVED)
        matching = [
            fact for fact in facts
            if fact.source_concept == "NetIncomeLoss" and fact.end_date == "2023-12-31"
        ]
        self.assertEqual(len(matching), 1)

    def test_conflicting_source_facts_are_preserved_and_flagged(self) -> None:
        raw = json.loads(self.payload)
        rows = raw["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
        conflict = dict(rows[0])
        conflict["val"] = conflict["val"] + 1
        rows.append(conflict)
        _, _, facts, warnings = parse_company_facts(json.dumps(raw).encode(), retrieved_at=RETRIEVED)
        conflicting = [fact for fact in facts if fact.quality_status == "conflict"]
        self.assertEqual(len(conflicting), 2)
        self.assertTrue(any("conflicting XBRL facts" in warning for warning in warnings))

    def test_malformed_response_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            parse_company_facts(b"not-json", retrieved_at=RETRIEVED)


class IngestionPointInTimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = MarketDatabase(Path(self.temporary.name) / "db.sqlite3")
        self.database.migrate()
        self.payload = FIXTURE.read_bytes()

    def tearDown(self) -> None:
        self.database.close()
        self.temporary.cleanup()

    def ingest(self) -> dict[str, object]:
        result = ingest_company_facts_payload(
            self.database,
            ENTITY,
            self.payload,
            retrieved_at=RETRIEVED,
            source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
            request_identifier="companyfacts:CIK0000000001",
        )
        company_id = int(self.database.get_company("TEST")["id"])
        self.database.insert_company_metadata(company_id, CompanyMetadataRecord(
            ticker="TEST", company_name=ENTITY.canonical_name, cik="0000000001",
            exchange="TEST", source="fixture-identity",
            source_record_id="test-identity-v1",
            effective_from="2020-01-01T00:00:00Z",
            observed_at="2020-01-01T00:00:00Z",
            available_at="2020-01-01T00:00:00Z",
            retrieved_at="2025-03-02T12:00:00Z",
        ))
        return result

    @staticmethod
    def persistence_state(database: MarketDatabase) -> dict[str, list[tuple[object, ...]]]:
        tables = (
            "companies",
            "company_aliases",
            "source_artifacts",
            "financial_periods",
            "financial_metrics",
        )
        return {
            table: [tuple(row) for row in database.connection.execute(f"SELECT * FROM {table}")]
            for table in tables
        }

    @classmethod
    def semantic_persistence_state(
        cls, database: MarketDatabase
    ) -> dict[str, list[tuple[object, ...]]]:
        state = cls.persistence_state(database)
        state["companies"] = [row[:8] for row in state["companies"]]
        return state

    def test_duplicate_ingestion_is_idempotent(self) -> None:
        first = self.ingest()
        second = self.ingest()
        self.assertGreater(first["facts_inserted"], 0)
        self.assertEqual(second["facts_inserted"], 0)
        self.assertEqual(second["duplicates_ignored"], second["facts_normalized"])

    def test_ingestion_rolls_back_company_artifact_and_facts_on_late_failure(self) -> None:
        self.database.connection.execute(
            """
            CREATE TRIGGER fail_sec_metric_insert
            BEFORE INSERT ON financial_metrics
            BEGIN
                SELECT RAISE(ABORT, 'injected late SEC persistence failure');
            END
            """
        )

        with self.assertRaisesRegex(
            sqlite3.IntegrityError, "injected late SEC persistence failure"
        ):
            self.ingest()

        for table in (
            "companies",
            "company_aliases",
            "source_artifacts",
            "financial_periods",
            "financial_metrics",
        ):
            with self.subTest(table=table):
                count = self.database.connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                self.assertEqual(count, 0)

    def test_ingestion_rolls_back_at_each_early_persistence_stage(self) -> None:
        _, _, facts, _ = parse_company_facts(self.payload, retrieved_at=RETRIEVED)
        late_fact_id = facts[1].source_fact_id
        failure_triggers = {
            "after_company": """
                CREATE TRIGGER injected_failure BEFORE INSERT ON source_artifacts
                BEGIN SELECT RAISE(ABORT, 'after company'); END
            """,
            "after_artifact": """
                CREATE TRIGGER injected_failure BEFORE INSERT ON financial_periods
                BEGIN SELECT RAISE(ABORT, 'after artifact'); END
            """,
            "during_normalization": """
                CREATE TRIGGER injected_failure BEFORE INSERT ON financial_metrics
                BEGIN SELECT RAISE(ABORT, 'during normalization'); END
            """,
            "after_first_normalized_fact": f"""
                CREATE TRIGGER injected_failure BEFORE INSERT ON financial_metrics
                WHEN NEW.source_fact_id='{late_fact_id}'
                BEGIN SELECT RAISE(ABORT, 'after first normalized fact'); END
            """,
        }

        for stage, trigger_sql in failure_triggers.items():
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                with MarketDatabase(Path(folder) / "stage.sqlite3") as database:
                    database.migrate()
                    database.connection.execute(trigger_sql)
                    with self.assertRaises(sqlite3.IntegrityError):
                        ingest_company_facts_payload(
                            database,
                            ENTITY,
                            self.payload,
                            retrieved_at=RETRIEVED,
                            source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
                            request_identifier="companyfacts:CIK0000000001",
                        )
                    self.assertEqual(
                        self.persistence_state(database),
                        {table: [] for table in self.persistence_state(database)},
                    )
                    database.connection.execute("DROP TRIGGER injected_failure")
                    result = ingest_company_facts_payload(
                        database,
                        ENTITY,
                        self.payload,
                        retrieved_at=RETRIEVED,
                        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
                        request_identifier="companyfacts:CIK0000000001",
                    )
                    self.assertGreater(result["facts_inserted"], 0)
                    repeated = ingest_company_facts_payload(
                        database,
                        ENTITY,
                        self.payload,
                        retrieved_at=RETRIEVED,
                        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
                        request_identifier="companyfacts:CIK0000000001",
                    )
                    self.assertEqual(repeated["facts_inserted"], 0)
                    with tempfile.TemporaryDirectory() as clean_folder:
                        with MarketDatabase(Path(clean_folder) / "clean.sqlite3") as clean:
                            clean.migrate()
                            ingest_company_facts_payload(
                                clean,
                                ENTITY,
                                self.payload,
                                retrieved_at=RETRIEVED,
                                source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
                                request_identifier="companyfacts:CIK0000000001",
                            )
                            self.assertEqual(
                                self.semantic_persistence_state(database),
                                self.semantic_persistence_state(clean),
                            )

    def test_ingestion_rolls_back_retirement_and_reruns_idempotently(self) -> None:
        self.ingest()
        before = self.persistence_state(self.database)
        reduced = json.loads(self.payload)
        del reduced["facts"]["us-gaap"]["NetIncomeLoss"]
        reduced_payload = json.dumps(reduced, sort_keys=True).encode()
        self.database.connection.execute(
            """
            CREATE TRIGGER fail_retirement
            BEFORE UPDATE ON financial_metrics
            WHEN NEW.quality_status='retired' AND OLD.quality_status<>'retired'
            BEGIN SELECT RAISE(ABORT, 'injected retirement failure'); END
            """
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected retirement failure"):
            ingest_company_facts_payload(
                self.database,
                ENTITY,
                reduced_payload,
                retrieved_at=RETRIEVED + timedelta(minutes=1),
                source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
                request_identifier="companyfacts:CIK0000000001",
            )
        self.assertEqual(self.persistence_state(self.database), before)

        self.database.connection.execute("DROP TRIGGER fail_retirement")
        result = ingest_company_facts_payload(
            self.database,
            ENTITY,
            reduced_payload,
            retrieved_at=RETRIEVED + timedelta(minutes=1),
            source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
            request_identifier="companyfacts:CIK0000000001",
        )
        self.assertGreater(result["facts_retired"], 0)
        after_success = self.persistence_state(self.database)
        repeated = ingest_company_facts_payload(
            self.database,
            ENTITY,
            reduced_payload,
            retrieved_at=RETRIEVED + timedelta(minutes=1),
            source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
            request_identifier="companyfacts:CIK0000000001",
        )
        self.assertEqual(repeated["facts_inserted"], 0)
        self.assertEqual(repeated["facts_retired"], 0)
        repeated_state = self.persistence_state(self.database)
        for table in (
            "company_aliases",
            "source_artifacts",
            "financial_periods",
            "financial_metrics",
        ):
            self.assertEqual(repeated_state[table], after_success[table])
        self.assertEqual(
            tuple(self.database.get_company("TEST"))[1:8],
            after_success["companies"][0][1:8],
        )
        self.assertEqual(self.database.connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_mapping_replacement_failure_rolls_back_and_can_rerun(self) -> None:
        self.ingest()
        _, _, facts, _ = parse_company_facts(self.payload, retrieved_at=RETRIEVED)
        replacement = replace(facts[0], mapping_version="sec-xbrl-atomicity-test")
        replacement_facts = [replacement, *facts[1:]]
        company_id = int(self.database.get_company("TEST")["id"])
        artifact_id = int(
            self.database.connection.execute("SELECT id FROM source_artifacts").fetchone()[0]
        )
        before = self.persistence_state(self.database)
        self.database.connection.execute(
            """
            CREATE TRIGGER fail_mapping_replacement
            BEFORE UPDATE ON financial_metrics
            WHEN NEW.mapping_version<>OLD.mapping_version
            BEGIN SELECT RAISE(ABORT, 'injected mapping replacement failure'); END
            """
        )

        with self.assertRaisesRegex(
            sqlite3.IntegrityError, "injected mapping replacement failure"
        ):
            self.database.reconcile_financial_facts(
                company_id, replacement_facts, source_artifact_id=artifact_id
            )
        self.assertEqual(self.persistence_state(self.database), before)

        self.database.connection.execute("DROP TRIGGER fail_mapping_replacement")
        changed, retired = self.database.reconcile_financial_facts(
            company_id, replacement_facts, source_artifact_id=artifact_id
        )
        self.assertEqual((changed, retired), (1, 0))
        self.assertEqual(
            self.database.reconcile_financial_facts(
                company_id, replacement_facts, source_artifact_id=artifact_id
            ),
            (0, 0),
        )
        row = self.database.connection.execute(
            "SELECT mapping_version FROM financial_metrics WHERE source_fact_id=?",
            (replacement.source_fact_id,),
        ).fetchone()
        self.assertEqual(row["mapping_version"], "sec-xbrl-atomicity-test")

    def test_filing_is_unavailable_before_filed_date(self) -> None:
        self.ingest()
        rows = self.database.get_metrics_as_of("TEST", "2025-01-15T23:59:59+00:00")
        latest_revenue = [row for row in rows if row["canonical_metric"] == "revenue"]
        self.assertEqual(latest_revenue[0]["end_date"], "2023-12-31")

    def test_exact_same_day_cutoff_rejects_end_of_day_filing_availability(self) -> None:
        self.ingest()
        provider = DatabaseProvider(
            self.database, datetime(2025, 3, 1, 12, tzinfo=timezone.utc)
        )
        rows = self.database.get_metrics_as_of("TEST", provider._boundary)
        self.assertFalse(
            any(row["available_at"].startswith("2025-03-01") for row in rows)
        )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            DatabaseProvider(self.database, datetime(2025, 3, 1, 12))

    def test_exact_cutoff_propagates_to_run_and_manifest(self) -> None:
        self.ingest()
        cutoff = datetime(2025, 3, 1, 12, 34, 56, tzinfo=timezone.utc)
        run = run_provider_pipeline(DatabaseProvider(self.database, cutoff))
        self.assertEqual(run.input_cutoff, cutoff.isoformat())
        self.assertEqual(run.manifest.input_cutoff, cutoff)

    def test_amendment_does_not_leak_backward(self) -> None:
        self.ingest()
        before = self.database.get_latest_metric(
            "TEST", "revenue", "2025-02-15T23:59:59+00:00"
        )
        after = self.database.get_latest_metric(
            "TEST", "revenue", "2025-03-02T23:59:59+00:00"
        )
        self.assertEqual(Decimal(before["normalized_value"]), Decimal("120000000"))
        self.assertEqual(Decimal(after["normalized_value"]), Decimal("121000000"))
        self.assertEqual(after["is_amendment"], 1)

    def test_normalized_fact_traces_to_artifact_concept_and_accession(self) -> None:
        result = self.ingest()
        row = self.database.get_latest_metric(
            "TEST", "net_income", "2025-03-02T23:59:59+00:00"
        )
        self.assertEqual(row["source_concept"], "NetIncomeLoss")
        self.assertEqual(row["accession_number"], "0000000001-25-000001")
        self.assertEqual(result["content_hash"], hashlib.sha256(self.payload).hexdigest())

    def test_database_provider_runs_existing_analytics_and_rankings(self) -> None:
        self.ingest()
        provider = DatabaseProvider(self.database, date(2025, 3, 2))
        run = run_provider_pipeline(provider)
        self.assertIn("TEST", run.companies)
        self.assertEqual(run.metrics["TEST"]["Net Margin"].status, "valid")
        self.assertEqual(len(run.rankings), 4)
        self.assertEqual(run.rankings["one_year"][0].ticker, "TEST")
        self.assertEqual(run.manifest.files_written, 0)
        report = ranking_report(run.rankings["one_year"], "Database Integration Ranking")
        self.assertIn("| 1 | TEST |", report)

    def test_database_provider_uses_fy_flows_adjacent_growth_and_metric_cagr_spans(self) -> None:
        self.ingest()
        company = DatabaseProvider(self.database, date(2025, 3, 2)).company("TEST")
        revenue = company.financials["revenue"]
        prior = company.financials["prior_revenue"]
        self.assertEqual(revenue.period.kind, "FY")
        self.assertEqual(prior.period.kind, "FY")
        self.assertEqual(prior.period.fiscal_year, revenue.period.fiscal_year - 1)
        self.assertNotIn("cagr_years", company.financials)
        for prefix in ("revenue", "eps", "fcf"):
            years = company.financials.get(f"{prefix}_cagr_years")
            start = company.financials.get(f"{prefix}_start")
            if years is not None and start is not None:
                self.assertEqual(
                    years.value,
                    Decimal(revenue.period.fiscal_year - start.period.fiscal_year),
                )

    def test_mapping_version_change_reprocesses_existing_fact_without_duplication(self) -> None:
        self.ingest()
        _, _, facts, _ = parse_company_facts(self.payload, retrieved_at=RETRIEVED)
        fact = facts[0]
        company_id = int(self.database.get_company("TEST")["id"])
        artifact_id = int(
            self.database.connection.execute("SELECT id FROM source_artifacts LIMIT 1").fetchone()[0]
        )
        changed = replace(fact, mapping_version="sec-xbrl-test-v2")
        self.assertEqual(
            self.database.insert_financial_facts(
                company_id, [changed], source_artifact_id=artifact_id
            ),
            1,
        )
        self.assertEqual(
            self.database.insert_financial_facts(
                company_id, [changed], source_artifact_id=artifact_id
            ),
            0,
        )
        row = self.database.connection.execute(
            "SELECT mapping_version FROM financial_metrics WHERE source_fact_id=?",
            (changed.source_fact_id,),
        ).fetchone()
        self.assertEqual(row["mapping_version"], "sec-xbrl-test-v2")

    def test_mapping_reconciliation_replaces_and_retires_obsolete_interpretations(self) -> None:
        mapping_a = """\
mapping_version: mapping-a
concepts:
  us-gaap:
    Revenues: {metric: revenue, priority: 1, units: [USD], kind: flow}
    NetIncomeLoss: {metric: net_income, priority: 1, units: [USD], kind: flow}
"""
        mapping_b = """\
mapping_version: mapping-b
concepts:
  us-gaap:
    Revenues: {metric: gross_profit, priority: 1, units: [USD], kind: flow}
"""
        first_path = Path(self.temporary.name) / "mapping-a.yaml"
        second_path = Path(self.temporary.name) / "mapping-b.yaml"
        first_path.write_text(mapping_a, encoding="utf-8")
        second_path.write_text(mapping_b, encoding="utf-8")
        company_id = self.database.upsert_company(
            ticker=ENTITY.ticker,
            canonical_name=ENTITY.canonical_name,
            exchange=ENTITY.exchange,
            cik=ENTITY.cik,
        )
        artifact_id = self.database.insert_source_artifact(
            provider="sec",
            request_identifier="companyfacts:CIK0000000001",
            source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
            retrieved_at=RETRIEVED.isoformat(),
            content_hash=hashlib.sha256(self.payload).hexdigest(),
            relative_path="immutable/raw.json",
            byte_count=len(self.payload),
        )
        _, _, first_facts, _ = parse_company_facts(
            self.payload, retrieved_at=RETRIEVED, mapping_path=first_path
        )
        self.database.reconcile_financial_facts(
            company_id, first_facts, source_artifact_id=artifact_id
        )
        self.assertTrue(self.database.get_metric_history("TEST", "revenue"))
        self.assertTrue(self.database.get_metric_history("TEST", "net_income"))

        _, _, second_facts, _ = parse_company_facts(
            self.payload, retrieved_at=RETRIEVED, mapping_path=second_path
        )
        _, retired = self.database.reconcile_financial_facts(
            company_id, second_facts, source_artifact_id=artifact_id
        )
        self.assertGreater(retired, 0)
        active = self.database.get_metrics_as_of("TEST", "2026-01-01T00:00:00+00:00")
        self.assertFalse(any(row["canonical_metric"] in {"revenue", "net_income"} for row in active))
        self.assertTrue(any(row["canonical_metric"] == "gross_profit" for row in active))
        self.assertTrue(
            all(
                row["quality_status"] == "retired"
                for row in self.database.get_metric_history("TEST", "net_income")
            )
        )
        self.assertEqual(
            self.database.connection.execute("SELECT COUNT(*) FROM source_artifacts").fetchone()[0],
            1,
        )

    def test_multi_company_reconciliation_is_atomic(self) -> None:
        company_id = self.database.upsert_company(
            ticker=ENTITY.ticker,
            canonical_name=ENTITY.canonical_name,
            exchange=ENTITY.exchange,
            cik=ENTITY.cik,
        )
        artifact_id = self.database.insert_source_artifact(
            provider="sec",
            request_identifier="companyfacts:CIK0000000001",
            source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
            retrieved_at=RETRIEVED.isoformat(),
            content_hash=hashlib.sha256(self.payload).hexdigest(),
            relative_path="immutable/raw.json",
            byte_count=len(self.payload),
        )
        _, _, facts, _ = parse_company_facts(self.payload, retrieved_at=RETRIEVED)
        invalid = replace(facts[1], source_fact_id="invalid-period", period_type="INVALID")
        with self.assertRaisesRegex(ValueError, "invalid or unsupported period"):
            self.database.reconcile_financial_fact_sets([
                (company_id, [facts[0]], artifact_id),
                (company_id, [invalid], artifact_id),
            ])
        self.assertEqual(
            self.database.connection.execute("SELECT COUNT(*) FROM financial_metrics").fetchone()[0],
            0,
        )

    def test_database_and_fixture_providers_match_for_equivalent_canonical_inputs(self) -> None:
        self.ingest()
        provider = DatabaseProvider(self.database, date(2025, 3, 2))
        company = provider.company("TEST")
        financials: dict[str, object] = {}
        for name, point in company.financials.items():
            financials[name] = {
                "value": str(point.value) if point.value is not None else None,
                "unit": point.unit,
                "currency": point.currency,
                "period_kind": point.period.kind,
                "period_label": point.period.label,
                "period_end": point.period.end_date.isoformat(),
                "period_start": point.period.start_date.isoformat() if point.period.start_date else None,
                "fiscal_year": point.period.fiscal_year,
                "fiscal_quarter": point.period.fiscal_quarter,
                "as_of": point.as_of.isoformat(),
            }
        fixture = {
            "analysis_date": "2025-03-02",
            "fixture_notice": "Equivalent canonical-input test only.",
            "defaults": {"currency": "USD", "monetary_unit": "raw", "shares_unit": "raw"},
            "companies": [{
                "ticker": company.ticker,
                "company": company.company,
                "sector": company.sector,
                "industry": company.industry,
                "as_of": company.as_of.isoformat(),
                "cik": company.cik,
                "source_quality": company.source_quality,
                "financials": financials,
            }],
        }
        fixture_path = Path(self.temporary.name) / "equivalent.json"
        fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
        database_run = run_provider_pipeline(provider)
        fixture_run = run_fixture_pipeline(fixture_path)
        self.assertEqual(database_run.metrics["TEST"], fixture_run.metrics["TEST"])
        self.assertIsNone(database_run.companies["TEST"].news_score)
        self.assertTrue(
            all(
                result.ranking_status == "insufficient_evidence"
                for values in database_run.rankings.values()
                for result in values
            )
        )

    def test_ranking_snapshot_and_components_are_reproducible(self) -> None:
        self.ingest()
        PaperPersistence(
            self.database, environment={PAPER_WRITE_ENV: "1"}
        ).migrate_v4_to_v5()
        self.database.migrate_v5_to_v6()
        provider = DatabaseProvider(self.database, date(2025, 3, 2))
        run = run_provider_pipeline(provider)
        rows = self.database.get_metrics_as_of("TEST", "2025-03-02T23:59:59+00:00")
        snapshot = self.database.create_snapshot(
            as_of="2025-03-02T23:59:59+00:00",
            metric_ids=[row["id"] for row in rows],
            description="integration test",
        )
        inserted = self.database.store_analytics_run(
            run,
            snapshot_id=snapshot,
            weights=load_config().ranking_weights,
            source_versions={"sec_xbrl_mapping": "sec-xbrl-v2"},
        )
        repeated = self.database.store_analytics_run(
            run,
            snapshot_id=snapshot,
            weights=load_config().ranking_weights,
            source_versions={"sec_xbrl_mapping": "sec-xbrl-v2"},
        )
        self.assertEqual(inserted, 0)
        self.assertEqual(repeated, 0)
        history = self.database.get_ranking_history("TEST", "one_year")
        self.assertEqual(history, [])


class EconomicPeriodRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = MarketDatabase(Path(self.temporary.name) / "db.sqlite3")
        self.database.migrate()
        entity = Entity(
            ticker="ECON",
            canonical_name="Economic Period Regression Company",
            exchange="TEST",
            cik="2",
        )
        ingest_company_facts_payload(
            self.database,
            entity,
            ECONOMIC_PERIOD_FIXTURE.read_bytes(),
            retrieved_at=datetime(2026, 2, 16, 12, tzinfo=timezone.utc),
            source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000002.json",
            request_identifier="companyfacts:CIK0000000002",
        )
        company_id = int(self.database.get_company("ECON")["id"])
        self.database.insert_company_metadata(company_id, CompanyMetadataRecord(
            ticker="ECON", company_name=entity.canonical_name, cik="0000000002",
            exchange="TEST", source="fixture-identity",
            source_record_id="econ-identity-v1",
            effective_from="2020-01-01T00:00:00Z",
            observed_at="2020-01-01T00:00:00Z",
            available_at="2020-01-01T00:00:00Z",
            retrieved_at="2026-02-16T12:00:00Z",
        ))
        self.company = DatabaseProvider(self.database, date(2026, 3, 1)).company("ECON")

    def tearDown(self) -> None:
        self.database.close()
        self.temporary.cleanup()

    def test_sec_comparative_fy_does_not_control_prior_or_cagr(self) -> None:
        financials = self.company.financials
        self.assertEqual(financials["revenue"].period.end_date, date(2025, 12, 31))
        self.assertEqual(financials["prior_revenue"].period.end_date, date(2024, 12, 31))
        self.assertEqual(financials["revenue_start"].period.end_date, date(2022, 12, 31))
        self.assertEqual(financials["revenue_cagr_years"].value, Decimal("3"))
        self.assertEqual(financials["prior_eps"].period.end_date, date(2024, 12, 31))
        self.assertEqual(financials["eps_start"].period.end_date, date(2022, 12, 31))
        self.assertEqual(financials["eps_cagr_years"].value, Decimal("3"))
        calculated = run_provider_pipeline(
            DatabaseProvider(self.database, date(2026, 3, 1))
        ).metrics["ECON"]
        self.assertEqual(calculated["Revenue Growth"].value, Decimal("160") / Decimal("140") - 1)
        self.assertEqual(calculated["EPS Growth"].value, Decimal("4") / Decimal("3.5") - 1)

    def test_roe_uses_annual_opening_and_closing_equity(self) -> None:
        average_equity = self.company.financials["average_equity"]
        self.assertEqual(average_equity.value, Decimal("70"))
        self.assertEqual(average_equity.period.end_date, date(2025, 12, 31))
        self.assertIn("2024-12-31:2025-12-31", average_equity.source)
        calculated = run_provider_pipeline(
            DatabaseProvider(self.database, date(2026, 3, 1))
        ).metrics["ECON"]
        self.assertEqual(calculated["ROE"].value, Decimal("16") / Decimal("70"))


class RawSecCacheTests(unittest.TestCase):
    def test_cache_round_trip_avoids_repeat_http_request(self) -> None:
        payload = FIXTURE.read_bytes()

        class FakeHttp:
            calls = 0

            def get(self, url: str) -> bytes:
                self.calls += 1
                return payload

        with tempfile.TemporaryDirectory() as folder:
            now = datetime(2025, 3, 2, tzinfo=timezone.utc)
            http = FakeHttp()
            client = SecCompanyFactsClient(
                SourceRequestConfig(
                    delay_seconds=0, timeout_seconds=1, retries=0,
                    headers={"User-Agent": "TestResearch contact@example.com"},
                ),
                RawSecCache(Path(folder)),
                http=http,  # type: ignore[arg-type]
                now=lambda: now,
                max_age=timedelta(days=1),
            )
            first = client.company_facts(ENTITY)
            second = client.company_facts(ENTITY)
            self.assertFalse(first.from_cache)
            self.assertTrue(second.from_cache)
            self.assertEqual(http.calls, 1)
            self.assertEqual(first.content_hash, second.content_hash)

    def test_client_requires_identified_user_agent(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            config = SourceRequestConfig(
                delay_seconds=0, timeout_seconds=1, retries=0,
                headers={"User-Agent": ""},
            )
            with self.assertRaises(ValueError):
                SecCompanyFactsClient(config, RawSecCache(Path(folder)))

    def test_client_requires_cik(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            client = SecCompanyFactsClient(
                SourceRequestConfig(
                    delay_seconds=0, timeout_seconds=1, retries=0,
                    headers={"User-Agent": "TestResearch contact@example.com"},
                ),
                RawSecCache(Path(folder)),
            )
            with self.assertRaises(ValueError):
                client.company_facts(Entity(ticker="NONE", canonical_name="No CIK"))

    def test_future_sec_cache_timestamp_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            cache = RawSecCache(Path(folder))
            future = RETRIEVED + timedelta(hours=1)
            document = cache.put(
                payload=FIXTURE.read_bytes(), source_url="https://data.sec.gov/fixture",
                request_identifier="companyfacts:future", retrieved_at=future,
            )
            self.assertTrue(document.path.is_file())
            with self.assertRaisesRegex(ValueError, "future"):
                cache.latest(
                    "companyfacts:future", now=RETRIEVED, max_age=timedelta(days=1)
                )


if __name__ == "__main__":
    unittest.main()
