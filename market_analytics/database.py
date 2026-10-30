"""Versioned SQLite storage and repository queries for point-in-time market data."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence


SCHEMA_VERSION = 4
NORMALIZED_INTERPRETATION_SCHEMA_VERSION = 6

_AWARE_INSTANT = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:[0-9]{2})"
)


def _absolute_time(value: str | datetime) -> str:
    """Exact, fixed-width UTC comparison key; never rewrite persisted evidence.

    Restrict parsing before fromisoformat, which otherwise silently truncates
    sub-microsecond precision. Errors deliberately exclude supplied values.
    """
    try:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError
            value = value.isoformat()
        if not isinstance(value, str):
            raise ValueError
        return _canonical_instant(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("timestamp must be aware ISO-8601 with at most six fractional digits") from None


@lru_cache(maxsize=4096)
def _canonical_instant(value: str) -> str:
    """Bounded pure parsing cache: filings share timestamps across many facts."""
    if not _AWARE_INSTANT.fullmatch(value):
        raise ValueError
    if value[-1] != "Z" and (int(value[-5:-3]) > 23 or int(value[-2:]) > 59):
        raise ValueError
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    ).isoformat(timespec="microseconds")


SCHEMA_V1 = """
CREATE TABLE companies (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL UNIQUE COLLATE NOCASE,
    canonical_name TEXT NOT NULL,
    exchange TEXT,
    cik TEXT UNIQUE,
    sector TEXT,
    industry TEXT,
    currency TEXT NOT NULL DEFAULT 'USD',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (cik IS NULL OR (length(cik) = 10 AND cik NOT GLOB '*[^0-9]*'))
);
CREATE INDEX idx_companies_cik ON companies(cik);

CREATE TABLE company_aliases (
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    alias TEXT NOT NULL COLLATE NOCASE,
    PRIMARY KEY(company_id, alias)
);

CREATE TABLE source_artifacts (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    request_identifier TEXT NOT NULL,
    source_url TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    relative_path TEXT,
    byte_count INTEGER NOT NULL CHECK(byte_count >= 0),
    UNIQUE(provider, request_identifier, content_hash)
);

CREATE TABLE financial_periods (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    period_type TEXT NOT NULL CHECK(period_type IN ('FY','Q','POINT_IN_TIME')),
    start_date TEXT,
    end_date TEXT NOT NULL,
    filed_at TEXT NOT NULL,
    available_at TEXT NOT NULL,
    accession_number TEXT NOT NULL,
    form TEXT NOT NULL,
    source TEXT NOT NULL,
    is_amendment INTEGER NOT NULL DEFAULT 0 CHECK(is_amendment IN (0,1)),
    UNIQUE(company_id, accession_number, period_type, start_date, end_date)
);
CREATE INDEX idx_period_company_end ON financial_periods(company_id, end_date);
CREATE INDEX idx_period_available ON financial_periods(available_at);
CREATE UNIQUE INDEX uq_financial_period_logical ON financial_periods(
    company_id, accession_number, period_type, COALESCE(start_date, ''), end_date
);

CREATE TABLE financial_metrics (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
    period_id INTEGER NOT NULL REFERENCES financial_periods(id) ON DELETE RESTRICT,
    canonical_metric TEXT NOT NULL,
    normalized_value TEXT NOT NULL,
    unit TEXT NOT NULL,
    currency TEXT,
    taxonomy TEXT NOT NULL,
    source_concept TEXT NOT NULL,
    source_unit TEXT NOT NULL,
    source_value TEXT NOT NULL,
    source_fact_id TEXT NOT NULL UNIQUE,
    mapping_version TEXT NOT NULL,
    mapping_priority INTEGER NOT NULL,
    formula_version TEXT,
    derivation_json TEXT,
    quality_status TEXT NOT NULL,
    warning TEXT,
    filed_at TEXT NOT NULL,
    available_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    source TEXT NOT NULL,
    source_artifact_id INTEGER REFERENCES source_artifacts(id) ON DELETE RESTRICT
);
CREATE INDEX idx_metric_company_period ON financial_metrics(company_id, canonical_metric, period_id);
CREATE INDEX idx_metric_available ON financial_metrics(available_at);
CREATE INDEX idx_metric_lookup ON financial_metrics(company_id, canonical_metric, available_at, mapping_priority);

CREATE TABLE prices (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
    price_date TEXT NOT NULL,
    open TEXT,
    high TEXT,
    low TEXT,
    close TEXT NOT NULL,
    adjusted_close TEXT,
    split_adjusted_close TEXT,
    dividend_adjusted_close TEXT,
    provider_adjustment_basis TEXT,
    total_return_index TEXT,
    adjustment_origin TEXT,
    split_event_ids_json TEXT NOT NULL DEFAULT '[]',
    dividend_treatment TEXT,
    volume TEXT,
    currency TEXT NOT NULL,
    ticker_at_observation TEXT NOT NULL,
    exchange TEXT,
    source TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    available_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    price_identity TEXT NOT NULL UNIQUE
);
CREATE INDEX idx_prices_company_date ON prices(company_id, price_date);
CREATE INDEX idx_prices_available ON prices(available_at);
CREATE INDEX idx_prices_semantic_revision ON prices(company_id, price_date, source, source_identifier, available_at);

CREATE TABLE company_metadata_history (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
    ticker TEXT NOT NULL COLLATE NOCASE,
    company_name TEXT NOT NULL,
    cik TEXT,
    exchange TEXT,
    sector TEXT,
    industry TEXT,
    sic TEXT,
    listing_date TEXT,
    delisting_date TEXT,
    is_active INTEGER CHECK(is_active IS NULL OR is_active IN (0,1)),
    listing_type TEXT,
    prior_ticker TEXT COLLATE NOCASE,
    source TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    effective_to TEXT,
    observed_at TEXT NOT NULL,
    available_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    metadata_identity TEXT NOT NULL UNIQUE
);
CREATE INDEX idx_metadata_company_effective ON company_metadata_history(company_id, effective_from, effective_to);
CREATE INDEX idx_metadata_ticker_available ON company_metadata_history(ticker, available_at);
CREATE INDEX idx_metadata_available ON company_metadata_history(available_at);

CREATE TABLE corporate_actions (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
    ticker TEXT NOT NULL COLLATE NOCASE,
    event_type TEXT NOT NULL,
    announcement_date TEXT,
    declaration_date TEXT,
    record_date TEXT,
    ex_date TEXT,
    payment_date TEXT,
    effective_date TEXT,
    ratio TEXT,
    amount TEXT,
    currency TEXT,
    old_ticker TEXT COLLATE NOCASE,
    new_ticker TEXT COLLATE NOCASE,
    source TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    available_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    action_identity TEXT NOT NULL UNIQUE
);
CREATE INDEX idx_actions_company_effective ON corporate_actions(company_id, effective_date);
CREATE INDEX idx_actions_available ON corporate_actions(available_at);
CREATE INDEX idx_actions_ticker ON corporate_actions(ticker, effective_date);

CREATE TABLE provider_retrieval_events (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    dataset TEXT NOT NULL,
    request_identifier TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    http_status INTEGER,
    cache_status TEXT,
    provenance_json TEXT NOT NULL,
    retrieval_identity TEXT NOT NULL UNIQUE
);
CREATE INDEX idx_provider_retrieval_request ON provider_retrieval_events(provider, request_identifier, retrieved_at);

CREATE TABLE catalysts (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
    catalyst_type TEXT NOT NULL,
    description TEXT NOT NULL,
    expected_date TEXT,
    importance TEXT NOT NULL,
    confidence TEXT NOT NULL,
    status TEXT NOT NULL,
    source TEXT NOT NULL,
    source_date TEXT NOT NULL,
    available_at TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    UNIQUE(company_id, source, source_identifier)
);
CREATE INDEX idx_catalyst_event_date ON catalysts(company_id, expected_date);

CREATE TABLE risks (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence TEXT NOT NULL,
    evidence_reference TEXT NOT NULL,
    source TEXT NOT NULL,
    observed_date TEXT NOT NULL,
    available_at TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    UNIQUE(company_id, source, source_identifier)
);

CREATE TABLE data_snapshots (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    as_of TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    description TEXT
);

CREATE TABLE snapshot_inputs (
    snapshot_id TEXT NOT NULL REFERENCES data_snapshots(id) ON DELETE RESTRICT,
    financial_metric_id INTEGER NOT NULL REFERENCES financial_metrics(id) ON DELETE RESTRICT,
    PRIMARY KEY(snapshot_id, financial_metric_id)
);

CREATE TABLE run_manifests (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    ranking_version TEXT NOT NULL,
    calculation_version TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    source_versions_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    errors_json TEXT NOT NULL,
    input_snapshot_id TEXT REFERENCES data_snapshots(id) ON DELETE RESTRICT
);

CREATE TABLE rankings (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES run_manifests(run_id) ON DELETE RESTRICT,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
    ranking_timestamp TEXT NOT NULL,
    horizon TEXT NOT NULL,
    opportunity_score TEXT NOT NULL,
    opportunity_rank INTEGER NOT NULL,
    research_priority_score TEXT NOT NULL,
    research_priority TEXT NOT NULL,
    research_priority_rank INTEGER NOT NULL,
    confidence TEXT NOT NULL,
    data_quality_json TEXT NOT NULL,
    ranking_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    input_snapshot_id TEXT REFERENCES data_snapshots(id) ON DELETE RESTRICT,
    UNIQUE(run_id, company_id, horizon)
);
CREATE INDEX idx_rankings_date_horizon ON rankings(ranking_timestamp, horizon);
CREATE INDEX idx_rankings_company ON rankings(company_id, horizon, ranking_timestamp);

CREATE TABLE ranking_components (
    ranking_id INTEGER NOT NULL REFERENCES rankings(id) ON DELETE CASCADE,
    component TEXT NOT NULL,
    score TEXT NOT NULL,
    weight TEXT NOT NULL,
    contribution TEXT NOT NULL,
    PRIMARY KEY(ranking_id, component)
);

CREATE TABLE articles (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_url TEXT NOT NULL,
    normalized_url TEXT,
    content_hash TEXT,
    published_at TEXT,
    available_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    event_classification TEXT
);

CREATE TABLE article_companies (
    article_id TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
    PRIMARY KEY(article_id, company_id)
);

CREATE TABLE backtest_results (
    id INTEGER PRIMARY KEY,
    ranking_id INTEGER NOT NULL REFERENCES rankings(id) ON DELETE RESTRICT,
    horizon TEXT NOT NULL,
    evaluation_date TEXT,
    return_value TEXT,
    benchmark_return TEXT,
    metadata_json TEXT NOT NULL,
    UNIQUE(ranking_id, horizon)
);

CREATE TABLE macro_series (
    series_id TEXT PRIMARY KEY,
    title TEXT,
    unit TEXT,
    source TEXT NOT NULL
);

CREATE TABLE macro_observations (
    series_id TEXT NOT NULL REFERENCES macro_series(series_id) ON DELETE RESTRICT,
    observation_date TEXT NOT NULL,
    value TEXT,
    available_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    PRIMARY KEY(series_id, observation_date, available_at)
);
CREATE INDEX idx_macro_available ON macro_observations(series_id, available_at);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MarketDatabase:
    """A small repository layer; analytics code never issues SQL directly."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        if path.exists() and path.is_dir():
            raise ValueError("database path points to a directory")
        if read_only and not path.is_file():
            raise ValueError("read-only database path does not exist")
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.connection = sqlite3.connect(
                f"file:{path.resolve()}?mode=ro" if read_only else path,
                uri=read_only,
            )
        except sqlite3.Error as exc:
            raise ValueError("unable to open market database") from exc
        self.connection.row_factory = sqlite3.Row
        self.connection.create_function("financial_instant", 1, _absolute_time, deterministic=True)
        self.connection.execute("PRAGMA foreign_keys = ON")
        if not read_only:
            self.connection.execute("PRAGMA journal_mode = WAL")
        self.read_only = read_only

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MarketDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.connection.in_transaction:
            raise RuntimeError("nested database transactions are not supported")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    @contextmanager
    def read_snapshot(self) -> Iterator[sqlite3.Connection]:
        """Hold one coherent deferred SQLite snapshot for a complete read run."""

        if self.connection.in_transaction:
            raise RuntimeError("nested database transactions are not supported")
        try:
            self.connection.execute("BEGIN")
            # Establish the snapshot immediately rather than on an arbitrary
            # later query, while retaining read-only/deferred semantics.
            self.connection.execute("SELECT value FROM schema_metadata WHERE key = ?", ("schema_version",)).fetchone()
            yield self.connection
        finally:
            self.connection.rollback()

    def _sec_schema_version(self) -> int:
        """Reject mislabeled later schemas before considering core migration."""
        version = self.schema_version()
        tables = {row[0] for row in self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        if version < 5 and any(name.startswith("forward_paper_") for name in tables):
            raise ValueError("inconsistent SEC database schema")
        if version < NORMALIZED_INTERPRETATION_SCHEMA_VERSION and tables & {
            "financial_source_facts", "financial_metric_lifecycle_events"
        }:
            raise ValueError("inconsistent SEC database schema")
        if version == 0 and tables - {"schema_metadata"}:
            raise ValueError("unversioned nonempty database cannot be initialized by SEC ingestion")
        return version

    def require_sec_schema(self) -> int:
        """Check the SEC storage contract without initializing or migrating anything."""
        version = self._sec_schema_version()
        if version not in (SCHEMA_VERSION, 5, NORMALIZED_INTERPRETATION_SCHEMA_VERSION):
            raise ValueError("SEC ingestion requires an initialized schema v4, v5 or v6")
        self._require_sec_storage(version)
        return version

    def _require_sec_storage(self, version: int) -> None:
        """Validate existing SEC storage and source-version upgrade inputs.

        Core SEC columns predate v2. v2->v3 adds provider columns/tables;
        v3->v4 adds price identity, action date types and retrieval events.
        Do not demand those migration outputs from an older source database.
        """
        required = {
            "companies": "id ticker canonical_name exchange cik sector industry currency created_at updated_at",
            "company_aliases": "company_id alias",
            "source_artifacts": "id provider request_identifier source_url retrieved_at content_hash relative_path byte_count",
            "financial_periods": "id company_id period_type fiscal_year fiscal_quarter start_date end_date filed_at available_at accession_number form source is_amendment",
            "financial_metrics": "id company_id period_id canonical_metric normalized_value unit currency taxonomy source_concept source_unit source_value source_fact_id mapping_version mapping_priority formula_version derivation_json quality_status warning filed_at available_at retrieved_at source source_artifact_id",
        }
        if version in (1, 2, 3):
            required["prices"] = "id company_id price_date open high low close adjusted_close volume currency source available_at retrieved_at source_identifier"
            if version == 3:
                required["prices"] += " split_adjusted_close dividend_adjusted_close provider_adjustment_basis total_return_index ticker_at_observation exchange observed_at provenance_json"
                required["corporate_actions"] = "id company_id ticker event_type announcement_date ex_date effective_date ratio amount currency old_ticker new_ticker source source_record_id observed_at available_at retrieved_at provenance_json action_identity"
        if version >= 5:
            required.update({table: column for table, column in (
                ("forward_paper_decisions", "decision_id"),
                ("forward_paper_horizons", "decision_id"),
                ("forward_paper_evaluation_events", "evaluation_event_id"),
                ("forward_paper_event_provenance", "evaluation_event_id"),
            )})
        if version == NORMALIZED_INTERPRETATION_SCHEMA_VERSION:
            required["financial_metrics"] += " interpretation_identity"
            required["financial_source_facts"] = "source_fact_id company_id taxonomy source_concept source_unit source_value filed_at available_at retrieved_at source source_artifact_id"
            required["financial_metric_lifecycle_events"] = "id financial_metric_id status recorded_at mapping_version reason lifecycle_identity"
        if version in (1, 2):
            # IF NOT EXISTS permits compatible early copies of these outputs.
            # They need not exist yet, but if present the later index/ALTER
            # operations must be valid before v1->v2 can commit.
            optional = {
                "company_metadata_history": "company_id effective_from effective_to ticker available_at",
                "corporate_actions": "id company_id ticker event_type announcement_date ex_date effective_date ratio amount currency old_ticker new_ticker source source_record_id observed_at available_at retrieved_at provenance_json action_identity",
            }
            for table, columns in optional.items():
                if self.connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name=? COLLATE NOCASE AND type IN ('table','view','index')",
                    (table,),
                ).fetchone():
                    required[table] = columns
        try:
            for table, columns in required.items():
                kind = self.connection.execute(
                    "SELECT type FROM sqlite_master WHERE name=? COLLATE NOCASE AND type IN ('table','view','index')",
                    (table,),
                ).fetchone()
                if kind is None or kind[0] != "table":
                    raise ValueError("inconsistent SEC database schema")
                actual = {row["name"] for row in self.connection.execute(f"PRAGMA table_info({table})")}
                if not set(columns.split()) <= actual:
                    raise ValueError("inconsistent SEC database schema")
            # Do not interpret a mislabeled v6 database with the mutable v4/v5 path.
            has_identity = any(row["name"] == "interpretation_identity" for row in
                               self.connection.execute("PRAGMA table_info(financial_metrics)"))
            if has_identity != (version == NORMALIZED_INTERPRETATION_SCHEMA_VERSION):
                raise ValueError("inconsistent SEC database schema")
        except sqlite3.DatabaseError:
            raise ValueError("inconsistent SEC database schema") from None

    def _require_sec_migration_path(self, version: int) -> None:
        """Preflight the structural DDL prerequisites of the whole v1/2/3->4 path.

        Migrations commit separately, so checking only the immediate step is
        insufficient. Do not require future outputs or reject compatible objects
        skipped by IF NOT EXISTS / removed with the old price table.
        """
        objects = {row["name"].lower(): (row["type"], row["tbl_name"].lower())
                   for row in self.connection.execute(
                       "SELECT name,type,tbl_name FROM sqlite_master "
                       "WHERE type IN ('table','view','index')")}
        # Unconditional table creation / rename destinations in v3->v4.
        for name in ("prices_v3", "provider_retrieval_events"):
            if name in objects:
                raise ValueError("conflicting SEC core migration storage")
        # These two indices are dropped explicitly before rebuilding prices.
        for name in ("idx_prices_company_date", "idx_prices_available"):
            if name in objects and objects[name] != ("index", "prices"):
                raise ValueError("conflicting SEC core migration index")
        # Unconditional index creations. Old price-owned indices disappear when
        # prices_v3 is dropped; all other pre-existing objects survive and clash.
        for name in ("idx_prices_semantic_revision", "idx_provider_retrieval_request"):
            if name in objects and objects[name] != ("index", "prices"):
                raise ValueError("conflicting SEC core migration index")
        conditional_indices = []
        if version == 1:
            conditional_indices.append("uq_financial_period_logical")
        if version in (1, 2):
            conditional_indices.extend((
                "idx_metadata_company_effective", "idx_metadata_ticker_available",
                "idx_metadata_available", "idx_actions_company_effective",
                "idx_actions_available", "idx_actions_ticker",
            ))
        # CREATE INDEX IF NOT EXISTS tolerates an existing index, not a table or
        # view with that name. Preserve the migration's existing reuse semantics.
        for name in conditional_indices:
            if name in objects and objects[name][0] != "index":
                raise ValueError("conflicting SEC core migration index")

    def prepare_sec_schema(self) -> int:
        """SEC CLI's existing core initialization authority stops at v4.

        v5/v6 are installed only by their separately authorized migration owners.
        Existing supported databases are inspected, never migrated here.
        """
        if self.read_only:
            raise PermissionError("cannot prepare SEC writes on a read-only database")
        version = self._sec_schema_version()
        if version in (1, 2, 3):
            # Each existing core migration commits independently. Validate the
            # source before the first upgrade can change evidence or version.
            self._require_sec_storage(version)
            self._require_sec_migration_path(version)
        if version in (0, 1, 2, 3):
            self.migrate()
        return self.require_sec_schema()

    def migrate(self) -> int:
        if self.read_only:
            raise PermissionError("cannot migrate a read-only database")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT value FROM schema_metadata WHERE key = ?", ("schema_version",)
        ).fetchone()
        version = int(row["value"]) if row else 0
        if version > SCHEMA_VERSION:
            raise RuntimeError("database schema is newer than this application")
        if version == 0:
            version_sql = (
                "INSERT INTO schema_metadata(key, value) VALUES "
                f"('schema_version', '{SCHEMA_VERSION}');"
            )
            try:
                self.connection.executescript(
                    "BEGIN IMMEDIATE;\n" + SCHEMA_V1 + "\n" + version_sql + "\nCOMMIT;"
                )
            except Exception:
                self.connection.rollback()
                raise
            version = SCHEMA_VERSION
        if version == 1:
            self._migrate_v1_to_v2()
            version = 2
        if version == 2:
            self._migrate_v2_to_v3()
            version = 3
        if version == 3:
            self._migrate_v3_to_v4()
            version = 4
        return version

    def _migrate_v1_to_v2(self) -> None:
        """Deduplicate NULL-start logical periods before enforcing expression uniqueness."""

        with self.transaction() as connection:
            groups = connection.execute(
                """
                SELECT company_id, accession_number, period_type,
                       COALESCE(start_date, '') AS start_key, end_date,
                       MIN(id) AS keep_id, COUNT(*) AS row_count
                FROM financial_periods
                GROUP BY company_id, accession_number, period_type,
                         COALESCE(start_date, ''), end_date
                HAVING COUNT(*) > 1
                """
            ).fetchall()
            for group in groups:
                duplicates = connection.execute(
                    """
                    SELECT id FROM financial_periods
                    WHERE company_id=? AND accession_number=? AND period_type=?
                      AND COALESCE(start_date, '')=? AND end_date=? AND id<>?
                    """,
                    (
                        group["company_id"], group["accession_number"],
                        group["period_type"], group["start_key"], group["end_date"],
                        group["keep_id"],
                    ),
                ).fetchall()
                duplicate_ids = [int(row["id"]) for row in duplicates]
                if duplicate_ids:
                    placeholders = ",".join("?" for _ in duplicate_ids)
                    connection.execute(
                        f"UPDATE financial_metrics SET period_id=? WHERE period_id IN ({placeholders})",
                        [int(group["keep_id"]), *duplicate_ids],
                    )
                    connection.execute(
                        f"DELETE FROM financial_periods WHERE id IN ({placeholders})",
                        duplicate_ids,
                    )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_financial_period_logical
                ON financial_periods(
                    company_id, accession_number, period_type,
                    COALESCE(start_date, ''), end_date
                )
                """
            )
            connection.execute(
                "UPDATE schema_metadata SET value='2' WHERE key='schema_version'"
            )

    def _migrate_v2_to_v3(self) -> None:
        """Add append-only provider metadata, explicit prices, and actions."""

        with self.transaction() as connection:
            existing = {
                row["name"] for row in connection.execute("PRAGMA table_info(prices)")
            }
            additions = {
                "split_adjusted_close": "TEXT",
                "dividend_adjusted_close": "TEXT",
                "provider_adjustment_basis": "TEXT",
                "total_return_index": "TEXT",
                "ticker_at_observation": "TEXT",
                "exchange": "TEXT",
                "observed_at": "TEXT",
                "provenance_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for column, declaration in additions.items():
                if column not in existing:
                    connection.execute(f"ALTER TABLE prices ADD COLUMN {column} {declaration}")
            # Preserve legacy rows without inventing adjustment semantics. The old
            # table did not distinguish observation time or adjusted-close basis.
            connection.execute(
                "UPDATE prices SET ticker_at_observation=(SELECT ticker FROM companies "
                "WHERE companies.id=prices.company_id) WHERE ticker_at_observation IS NULL"
            )
            connection.execute(
                "UPDATE prices SET observed_at=available_at WHERE observed_at IS NULL"
            )
            connection.execute(
                "UPDATE prices SET provider_adjustment_basis='unknown' "
                "WHERE adjusted_close IS NOT NULL AND provider_adjustment_basis IS NULL"
            )
            migration_sql = """
                CREATE TABLE IF NOT EXISTS company_metadata_history (
                    id INTEGER PRIMARY KEY,
                    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
                    ticker TEXT NOT NULL COLLATE NOCASE, company_name TEXT NOT NULL,
                    cik TEXT, exchange TEXT, sector TEXT, industry TEXT, sic TEXT,
                    listing_date TEXT, delisting_date TEXT,
                    is_active INTEGER CHECK(is_active IS NULL OR is_active IN (0,1)),
                    listing_type TEXT, prior_ticker TEXT COLLATE NOCASE,
                    source TEXT NOT NULL, source_record_id TEXT NOT NULL,
                    effective_from TEXT NOT NULL, effective_to TEXT,
                    observed_at TEXT NOT NULL, available_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL, provenance_json TEXT NOT NULL,
                    metadata_identity TEXT NOT NULL UNIQUE
                );
                CREATE INDEX IF NOT EXISTS idx_metadata_company_effective ON company_metadata_history(company_id, effective_from, effective_to);
                CREATE INDEX IF NOT EXISTS idx_metadata_ticker_available ON company_metadata_history(ticker, available_at);
                CREATE INDEX IF NOT EXISTS idx_metadata_available ON company_metadata_history(available_at);
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    id INTEGER PRIMARY KEY,
                    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
                    ticker TEXT NOT NULL COLLATE NOCASE, event_type TEXT NOT NULL,
                    announcement_date TEXT, ex_date TEXT, effective_date TEXT,
                    ratio TEXT, amount TEXT, currency TEXT,
                    old_ticker TEXT COLLATE NOCASE, new_ticker TEXT COLLATE NOCASE,
                    source TEXT NOT NULL, source_record_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL, available_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL, provenance_json TEXT NOT NULL,
                    action_identity TEXT NOT NULL UNIQUE
                );
                CREATE INDEX IF NOT EXISTS idx_actions_company_effective ON corporate_actions(company_id, effective_date);
                CREATE INDEX IF NOT EXISTS idx_actions_available ON corporate_actions(available_at);
                CREATE INDEX IF NOT EXISTS idx_actions_ticker ON corporate_actions(ticker, effective_date);
                UPDATE schema_metadata SET value='3' WHERE key='schema_version';
                """
            for statement in migration_sql.split(";"):
                if statement.strip():
                    connection.execute(statement)

    def _migrate_v3_to_v4(self) -> None:
        """Version price corrections and preserve provider retrieval events."""

        with self.transaction() as connection:
            legacy_prices = list(connection.execute("SELECT * FROM prices"))
            connection.execute("DROP INDEX IF EXISTS idx_prices_company_date")
            connection.execute("DROP INDEX IF EXISTS idx_prices_available")
            connection.execute("ALTER TABLE prices RENAME TO prices_v3")
            connection.execute(
                """
                CREATE TABLE prices (
                    id INTEGER PRIMARY KEY,
                    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
                    price_date TEXT NOT NULL, open TEXT, high TEXT, low TEXT,
                    close TEXT NOT NULL, adjusted_close TEXT,
                    split_adjusted_close TEXT, dividend_adjusted_close TEXT,
                    provider_adjustment_basis TEXT, total_return_index TEXT,
                    adjustment_origin TEXT,
                    split_event_ids_json TEXT NOT NULL DEFAULT '[]',
                    dividend_treatment TEXT, volume TEXT, currency TEXT NOT NULL,
                    ticker_at_observation TEXT NOT NULL, exchange TEXT,
                    source TEXT NOT NULL, observed_at TEXT NOT NULL,
                    available_at TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                    source_identifier TEXT NOT NULL,
                    provenance_json TEXT NOT NULL DEFAULT '{}',
                    price_identity TEXT NOT NULL UNIQUE
                )
                """
            )
            for row in legacy_prices:
                semantic = {
                    key: row[key]
                    for key in (
                        "company_id", "price_date", "open", "high", "low", "close",
                        "adjusted_close", "split_adjusted_close", "dividend_adjusted_close",
                        "provider_adjustment_basis", "total_return_index", "volume",
                        "currency", "ticker_at_observation", "exchange", "source",
                        "observed_at", "available_at", "source_identifier",
                    )
                }
                identity = self._provider_identity("price-observation-v1", semantic)
                has_adjustment = any(
                    row[key] is not None
                    for key in (
                        "adjusted_close", "split_adjusted_close",
                        "dividend_adjusted_close", "total_return_index",
                    )
                )
                connection.execute(
                    """
                    INSERT INTO prices(
                        id, company_id, price_date, open, high, low, close,
                        adjusted_close, split_adjusted_close, dividend_adjusted_close,
                        provider_adjustment_basis, total_return_index, adjustment_origin,
                        split_event_ids_json, dividend_treatment, volume, currency,
                        ticker_at_observation, exchange, source, observed_at, available_at,
                        retrieved_at, source_identifier, provenance_json, price_identity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"], row["company_id"], row["price_date"], row["open"],
                        row["high"], row["low"], row["close"], row["adjusted_close"],
                        row["split_adjusted_close"], row["dividend_adjusted_close"],
                        row["provider_adjustment_basis"], row["total_return_index"],
                        "provider_supplied" if has_adjustment else None, "[]",
                        "provider_defined" if row["total_return_index"] is not None else None,
                        row["volume"], row["currency"], row["ticker_at_observation"],
                        row["exchange"], row["source"], row["observed_at"],
                        row["available_at"], row["retrieved_at"], row["source_identifier"],
                        row["provenance_json"], identity,
                    ),
                )
            connection.execute("DROP TABLE prices_v3")
            connection.execute(
                "CREATE INDEX idx_prices_company_date ON prices(company_id, price_date)"
            )
            connection.execute("CREATE INDEX idx_prices_available ON prices(available_at)")
            connection.execute(
                "CREATE INDEX idx_prices_semantic_revision ON prices"
                "(company_id, price_date, source, source_identifier, available_at)"
            )
            action_columns = {
                row["name"] for row in connection.execute(
                    "PRAGMA table_info(corporate_actions)"
                )
            }
            for column in ("declaration_date", "record_date", "payment_date"):
                if column not in action_columns:
                    connection.execute(
                        f"ALTER TABLE corporate_actions ADD COLUMN {column} TEXT"
                    )
            connection.execute(
                """
                CREATE TABLE provider_retrieval_events (
                    id INTEGER PRIMARY KEY,
                    provider TEXT NOT NULL, dataset TEXT NOT NULL,
                    request_identifier TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL, http_status INTEGER, cache_status TEXT,
                    provenance_json TEXT NOT NULL,
                    retrieval_identity TEXT NOT NULL UNIQUE
                )
                """
            )
            connection.execute(
                "CREATE INDEX idx_provider_retrieval_request ON provider_retrieval_events"
                "(provider, request_identifier, retrieved_at)"
            )
            connection.execute(
                "UPDATE schema_metadata SET value='4' WHERE key='schema_version'"
            )

    @staticmethod
    def _interpretation_identity(values: dict[str, Any]) -> str:
        payload = json.dumps(
            values, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(
            f"normalized-interpretation-v1\0{payload}".encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _lifecycle_identity(
        metric_id: int, status: str, recorded_at: str, mapping_version: str
    ) -> str:
        payload = json.dumps(
            {
                "financial_metric_id": metric_id,
                "mapping_version": mapping_version,
                "recorded_at": recorded_at,
                "status": status,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(
            f"normalized-lifecycle-v1\0{payload}".encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _interpretation_values(row: Any) -> dict[str, Any]:
        return {
            key: row[key]
            for key in (
                "source_fact_id", "period_id", "canonical_metric",
                "normalized_value", "unit", "currency", "mapping_version",
                "mapping_priority", "formula_version", "derivation_json",
                "quality_status", "warning",
            )
        }

    def migrate_v5_to_v6(
        self, *, failure_injector: Any | None = None
    ) -> int:
        """Create immutable normalized interpretations in a disposable v5 database."""

        if self.read_only:
            raise PermissionError("cannot migrate a read-only database")
        version = self.schema_version()
        if version == NORMALIZED_INTERPRETATION_SCHEMA_VERSION:
            return version
        if version != 5:
            raise RuntimeError("normalized-interpretation migration requires schema v5")

        def inject(stage: str) -> None:
            if failure_injector is not None:
                failure_injector(stage)

        foreign_keys = int(self.connection.execute("PRAGMA foreign_keys").fetchone()[0])
        self.connection.execute("PRAGMA foreign_keys=OFF")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            rows = list(self.connection.execute("SELECT * FROM financial_metrics ORDER BY id"))
            self.connection.execute(
                """
                CREATE TABLE financial_source_facts (
                    source_fact_id TEXT PRIMARY KEY,
                    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
                    taxonomy TEXT NOT NULL, source_concept TEXT NOT NULL,
                    source_unit TEXT NOT NULL, source_value TEXT NOT NULL,
                    filed_at TEXT NOT NULL, available_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL, source TEXT NOT NULL,
                    source_artifact_id INTEGER REFERENCES source_artifacts(id) ON DELETE RESTRICT
                )
                """
            )
            for row in rows:
                self.connection.execute(
                    """
                    INSERT INTO financial_source_facts(
                        source_fact_id,company_id,taxonomy,source_concept,
                        source_unit,source_value,filed_at,available_at,retrieved_at,
                        source,source_artifact_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row["source_fact_id"], row["company_id"], row["taxonomy"],
                        row["source_concept"], row["source_unit"],
                        row["source_value"], row["filed_at"], row["available_at"],
                        row["retrieved_at"], row["source"], row["source_artifact_id"],
                    ),
                )
            inject("after_source_facts")
            for name in (
                "idx_metric_company_period", "idx_metric_available", "idx_metric_lookup",
            ):
                self.connection.execute(f"DROP INDEX IF EXISTS {name}")
            self.connection.execute("ALTER TABLE snapshot_inputs RENAME TO snapshot_inputs_v5")
            self.connection.execute("ALTER TABLE financial_metrics RENAME TO financial_metrics_v5")
            self.connection.execute(
                """
                CREATE TABLE financial_metrics (
                    id INTEGER PRIMARY KEY,
                    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
                    period_id INTEGER NOT NULL REFERENCES financial_periods(id) ON DELETE RESTRICT,
                    canonical_metric TEXT NOT NULL, normalized_value TEXT NOT NULL,
                    unit TEXT NOT NULL, currency TEXT, taxonomy TEXT NOT NULL,
                    source_concept TEXT NOT NULL, source_unit TEXT NOT NULL,
                    source_value TEXT NOT NULL,
                    source_fact_id TEXT NOT NULL REFERENCES financial_source_facts(source_fact_id) ON DELETE RESTRICT,
                    interpretation_identity TEXT NOT NULL UNIQUE,
                    mapping_version TEXT NOT NULL, mapping_priority INTEGER NOT NULL,
                    formula_version TEXT, derivation_json TEXT,
                    quality_status TEXT NOT NULL, warning TEXT,
                    filed_at TEXT NOT NULL, available_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL, source TEXT NOT NULL,
                    source_artifact_id INTEGER REFERENCES source_artifacts(id) ON DELETE RESTRICT
                )
                """
            )
            for row in rows:
                identity = self._interpretation_identity(
                    self._interpretation_values(row)
                )
                self.connection.execute(
                    """
                    INSERT INTO financial_metrics(
                        id,company_id,period_id,canonical_metric,normalized_value,unit,
                        currency,taxonomy,source_concept,source_unit,source_value,
                        source_fact_id,interpretation_identity,mapping_version,
                        mapping_priority,formula_version,derivation_json,quality_status,
                        warning,filed_at,available_at,retrieved_at,source,source_artifact_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row["id"], row["company_id"], row["period_id"],
                        row["canonical_metric"], row["normalized_value"], row["unit"],
                        row["currency"], row["taxonomy"], row["source_concept"],
                        row["source_unit"], row["source_value"], row["source_fact_id"],
                        identity, row["mapping_version"], row["mapping_priority"],
                        row["formula_version"], row["derivation_json"],
                        row["quality_status"], row["warning"], row["filed_at"],
                        row["available_at"], row["retrieved_at"], row["source"],
                        row["source_artifact_id"],
                    ),
                )
            self.connection.execute(
                "CREATE INDEX idx_metric_company_period ON financial_metrics(company_id, canonical_metric, period_id)"
            )
            self.connection.execute(
                "CREATE INDEX idx_metric_available ON financial_metrics(available_at)"
            )
            self.connection.execute(
                "CREATE INDEX idx_metric_lookup ON financial_metrics(company_id, canonical_metric, available_at, mapping_priority)"
            )
            self.connection.execute(
                "CREATE INDEX idx_metric_source_fact ON financial_metrics(source_fact_id)"
            )
            self.connection.execute(
                """
                CREATE TABLE financial_metric_lifecycle_events (
                    id INTEGER PRIMARY KEY,
                    financial_metric_id INTEGER NOT NULL REFERENCES financial_metrics(id) ON DELETE RESTRICT,
                    status TEXT NOT NULL CHECK(status IN ('active','retired')),
                    recorded_at TEXT NOT NULL, mapping_version TEXT NOT NULL,
                    reason TEXT NOT NULL, lifecycle_identity TEXT NOT NULL UNIQUE
                )
                """
            )
            self.connection.execute(
                "CREATE INDEX idx_metric_lifecycle ON financial_metric_lifecycle_events(financial_metric_id, recorded_at, id)"
            )
            for row in rows:
                status = "retired" if row["quality_status"] == "retired" else "active"
                event_identity = self._lifecycle_identity(
                    int(row["id"]), status, row["retrieved_at"], row["mapping_version"]
                )
                self.connection.execute(
                    """INSERT INTO financial_metric_lifecycle_events(
                           financial_metric_id,status,recorded_at,mapping_version,
                           reason,lifecycle_identity
                       ) VALUES (?,?,?,?,?,?)""",
                    (
                        row["id"], status, row["retrieved_at"], row["mapping_version"],
                        "schema-v6 migration of current v5 state", event_identity,
                    ),
                )
            self.connection.execute(
                """
                CREATE TABLE snapshot_inputs (
                    snapshot_id TEXT NOT NULL REFERENCES data_snapshots(id) ON DELETE RESTRICT,
                    financial_metric_id INTEGER NOT NULL REFERENCES financial_metrics(id) ON DELETE RESTRICT,
                    PRIMARY KEY(snapshot_id, financial_metric_id)
                )
                """
            )
            self.connection.execute(
                """INSERT INTO snapshot_inputs(snapshot_id,financial_metric_id)
                   SELECT snapshot_id,financial_metric_id FROM snapshot_inputs_v5"""
            )
            inject("before_legacy_drop")
            self.connection.execute("DROP TABLE snapshot_inputs_v5")
            self.connection.execute("DROP TABLE financial_metrics_v5")
            self.connection.execute(
                "UPDATE schema_metadata SET value='6' WHERE key='schema_version'"
            )
            inject("before_v6_commit")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            self.connection.execute(f"PRAGMA foreign_keys={foreign_keys}")
        return self.schema_version()

    def schema_version(self) -> int:
        try:
            row = self.connection.execute(
                "SELECT value FROM schema_metadata WHERE key = ?", ("schema_version",)
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["value"]) if row else 0

    def upsert_company(
        self,
        *,
        ticker: str,
        canonical_name: str,
        exchange: str | None,
        cik: str | None,
        sector: str | None = None,
        industry: str | None = None,
        currency: str = "USD",
        aliases: Sequence[str] = (),
    ) -> int:
        with self.transaction() as connection:
            return self._upsert_company(
                connection,
                ticker=ticker,
                canonical_name=canonical_name,
                exchange=exchange,
                cik=cik,
                sector=sector,
                industry=industry,
                currency=currency,
                aliases=aliases,
            )

    def _upsert_company(
        self,
        connection: sqlite3.Connection,
        *,
        ticker: str,
        canonical_name: str,
        exchange: str | None,
        cik: str | None,
        sector: str | None = None,
        industry: str | None = None,
        currency: str = "USD",
        aliases: Sequence[str] = (),
    ) -> int:
        now = utc_now_iso()
        normalized_cik = str(cik).zfill(10) if cik else None
        connection.execute(
            """
            INSERT INTO companies(
                ticker, canonical_name, exchange, cik, sector, industry, currency,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                canonical_name=excluded.canonical_name,
                exchange=excluded.exchange,
                cik=excluded.cik,
                sector=COALESCE(excluded.sector, companies.sector),
                industry=COALESCE(excluded.industry, companies.industry),
                currency=excluded.currency,
                updated_at=excluded.updated_at
            """,
            (
                ticker.upper(), canonical_name, exchange, normalized_cik, sector,
                industry, currency.upper(), now, now,
            ),
        )
        row = connection.execute(
            "SELECT id FROM companies WHERE ticker = ?", (ticker.upper(),)
        ).fetchone()
        assert row is not None
        company_id = int(row["id"])
        for alias in sorted({value.strip() for value in aliases if value.strip()}):
            connection.execute(
                "INSERT OR IGNORE INTO company_aliases(company_id, alias) VALUES (?, ?)",
                (company_id, alias),
            )
        return company_id

    def get_company(self, ticker: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM companies WHERE ticker = ?", (ticker.upper(),)
        ).fetchone()

    def get_company_by_id(self, company_id: int) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM companies WHERE id = ?", (company_id,)
        ).fetchone()

    def list_companies(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM companies ORDER BY ticker"))

    @staticmethod
    def _provider_identity(kind: str, payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(f"{kind}\0{encoded}".encode("utf-8")).hexdigest()

    def insert_company_metadata(self, company_id: int, record: Any) -> int:
        payload = record.model_dump(
            mode="json", exclude={"company_id", "retrieved_at", "provenance"}
        )
        identity = self._provider_identity("company-metadata-v1", payload)
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO company_metadata_history(
                    company_id, ticker, company_name, cik, exchange, sector, industry,
                    sic, listing_date, delisting_date, is_active, listing_type,
                    prior_ticker, source, source_record_id, effective_from, effective_to,
                    observed_at, available_at, retrieved_at, provenance_json,
                    metadata_identity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    company_id, record.ticker, record.company_name, record.cik,
                    record.exchange, record.sector, record.industry, record.sic,
                    record.listing_date.isoformat() if record.listing_date else None,
                    record.delisting_date.isoformat() if record.delisting_date else None,
                    None if record.active is None else int(record.active), record.listing_type,
                    record.prior_ticker, record.source, record.source_record_id,
                    record.effective_from.isoformat(),
                    record.effective_to.isoformat() if record.effective_to else None,
                    record.observed_at.isoformat(), record.available_at.isoformat(),
                    record.retrieved_at.isoformat(),
                    json.dumps(record.provenance, sort_keys=True), identity,
                ),
            )
        return cursor.rowcount

    def metadata_as_of(
        self, ticker: str, cutoff: str, *, company_id: int | None = None
    ) -> list[sqlite3.Row]:
        parameters: list[Any] = [cutoff, cutoff]
        identity_clause = (
            "cmh.company_id=?" if company_id is not None
            else "(cmh.ticker=? OR cmh.prior_ticker=?)"
        )
        if company_id is not None:
            parameters.append(company_id)
        else:
            parameters.extend([ticker.upper()] * 2)
        return list(
            self.connection.execute(
                f"""
                SELECT cmh.* FROM company_metadata_history cmh
                WHERE cmh.available_at<=? AND cmh.effective_from<=?
                  AND (cmh.effective_to IS NULL OR ?<cmh.effective_to)
                  AND {identity_clause}
                ORDER BY cmh.available_at DESC, cmh.observed_at DESC,
                         cmh.effective_from DESC, cmh.metadata_identity
                """,
                [parameters[0], parameters[1], parameters[1], *parameters[2:]],
            )
        )

    def metadata_company_ids_as_of(self, cutoff: str) -> list[int]:
        """Return only identities explicitly known and effective by the cutoff."""

        return [
            int(row["company_id"])
            for row in self.connection.execute(
                """
                SELECT DISTINCT company_id
                FROM company_metadata_history
                WHERE available_at<=? AND effective_from<=?
                  AND (effective_to IS NULL OR ?<effective_to)
                ORDER BY company_id
                """,
                (cutoff, cutoff, cutoff),
            )
        ]

    def insert_price(self, company_id: int, record: Any) -> int:
        semantic = record.model_dump(
            mode="json", exclude={"company_id", "retrieved_at", "provenance"}
        )
        identity = self._provider_identity("price-observation-v1", semantic)
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO prices(
                    company_id, price_date, open, high, low, close, adjusted_close,
                    split_adjusted_close, dividend_adjusted_close,
                    provider_adjustment_basis, total_return_index, volume, currency,
                    adjustment_origin, split_event_ids_json, dividend_treatment,
                    ticker_at_observation, exchange, source, observed_at, available_at,
                    retrieved_at, source_identifier, provenance_json, price_identity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    company_id, record.trading_date.isoformat(),
                    str(record.open) if record.open is not None else None,
                    str(record.high) if record.high is not None else None,
                    str(record.low) if record.low is not None else None,
                    str(record.close),
                    str(record.provider_adjusted_close) if record.provider_adjusted_close is not None else None,
                    str(record.split_adjusted_close) if record.split_adjusted_close is not None else None,
                    str(record.dividend_adjusted_close) if record.dividend_adjusted_close is not None else None,
                    record.provider_adjustment_basis,
                    str(record.total_return_index) if record.total_return_index is not None else None,
                    str(record.volume) if record.volume is not None else None,
                    record.currency, record.adjustment_origin,
                    json.dumps(record.split_event_ids, sort_keys=True),
                    record.dividend_treatment, record.ticker, record.exchange, record.source,
                    record.observed_at.isoformat(), record.available_at.isoformat(),
                    record.retrieved_at.isoformat(), record.source_record_id,
                    json.dumps(record.provenance, sort_keys=True), identity,
                ),
            )
        return cursor.rowcount

    def prices_as_of(
        self, company_id: int, start: date, end: date, cutoff: str
    ) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                WITH eligible AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY company_id, price_date, source, source_identifier
                        ORDER BY available_at DESC, id DESC
                    ) AS revision_rank
                    FROM prices
                    WHERE company_id=? AND available_at<=?
                )
                SELECT * FROM eligible
                WHERE revision_rank=1 AND price_date BETWEEN ? AND ?
                ORDER BY price_date, available_at, source, source_identifier
                """,
                (company_id, cutoff, start.isoformat(), end.isoformat()),
            )
        )

    def price_revision_history(
        self, company_id: int, trading_date: date, source: str, source_identifier: str
    ) -> list[sqlite3.Row]:
        return list(self.connection.execute(
            """
            SELECT * FROM prices
            WHERE company_id=? AND price_date=? AND source=? AND source_identifier=?
            ORDER BY available_at, id
            """,
            (company_id, trading_date.isoformat(), source, source_identifier),
        ))

    def insert_corporate_action(self, company_id: int, record: Any) -> int:
        payload = record.model_dump(
            mode="json", exclude={"company_id", "retrieved_at", "provenance"}
        )
        identity = self._provider_identity("corporate-action-v1", payload)
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO corporate_actions(
                    company_id, ticker, event_type, announcement_date, ex_date,
                    declaration_date, record_date, payment_date, effective_date,
                    ratio, amount, currency, old_ticker, new_ticker,
                    source, source_record_id, observed_at, available_at, retrieved_at,
                    provenance_json, action_identity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    company_id, record.ticker, record.event_type,
                    record.announcement_date.isoformat() if record.announcement_date else None,
                    record.ex_date.isoformat() if record.ex_date else None,
                    record.declaration_date.isoformat() if record.declaration_date else None,
                    record.record_date.isoformat() if record.record_date else None,
                    record.payment_date.isoformat() if record.payment_date else None,
                    record.effective_date.isoformat() if record.effective_date else None,
                    str(record.ratio) if record.ratio is not None else None,
                    str(record.amount) if record.amount is not None else None,
                    record.currency, record.old_ticker, record.new_ticker,
                    record.source, record.source_record_id, record.observed_at.isoformat(),
                    record.available_at.isoformat(), record.retrieved_at.isoformat(),
                    json.dumps(record.provenance, sort_keys=True), identity,
                ),
            )
        return cursor.rowcount

    def corporate_actions_as_of(
        self, company_id: int, start: date, end: date, cutoff: str
    ) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                WITH eligible AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY company_id, source, source_record_id
                        ORDER BY available_at DESC, id DESC
                    ) AS revision_rank
                    FROM corporate_actions
                    WHERE company_id=? AND available_at<=?
                )
                SELECT * FROM eligible
                WHERE revision_rank=1
                  AND COALESCE(
                      effective_date, ex_date, record_date, declaration_date,
                      announcement_date, payment_date
                  ) BETWEEN ? AND ?
                ORDER BY COALESCE(
                             effective_date, ex_date, record_date, declaration_date,
                             announcement_date, payment_date
                         ),
                         available_at, action_identity
                """,
                (company_id, cutoff, start.isoformat(), end.isoformat()),
            )
        )

    def corporate_action_revision_history(
        self, company_id: int, source: str, source_record_id: str
    ) -> list[sqlite3.Row]:
        return list(self.connection.execute(
            """
            SELECT * FROM corporate_actions
            WHERE company_id=? AND source=? AND source_record_id=?
            ORDER BY available_at, id
            """,
            (company_id, source, source_record_id),
        ))

    def record_provider_retrieval(
        self, *, provider: str, dataset: str, request_identifier: str,
        retrieved_at: str, content_hash: str, http_status: int | None,
        cache_status: str | None, provenance: dict[str, Any] | None = None,
    ) -> int:
        payload = {
            "provider": provider, "dataset": dataset,
            "request_identifier": request_identifier, "retrieved_at": retrieved_at,
            "content_hash": content_hash,
        }
        identity = self._provider_identity("provider-retrieval-v1", payload)
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO provider_retrieval_events(
                    provider, dataset, request_identifier, retrieved_at, content_hash,
                    http_status, cache_status, provenance_json, retrieval_identity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider, dataset, request_identifier, retrieved_at, content_hash,
                    http_status, cache_status,
                    json.dumps(provenance or {}, sort_keys=True), identity,
                ),
            )
        return cursor.rowcount

    def insert_source_artifact(
        self, *, provider: str, request_identifier: str, source_url: str,
        retrieved_at: str, content_hash: str, relative_path: str | None, byte_count: int,
    ) -> int:
        with self.transaction() as connection:
            return self._insert_source_artifact(
                connection,
                provider=provider,
                request_identifier=request_identifier,
                source_url=source_url,
                retrieved_at=retrieved_at,
                content_hash=content_hash,
                relative_path=relative_path,
                byte_count=byte_count,
            )

    def _insert_source_artifact(
        self,
        connection: sqlite3.Connection,
        *,
        provider: str,
        request_identifier: str,
        source_url: str,
        retrieved_at: str,
        content_hash: str,
        relative_path: str | None,
        byte_count: int,
    ) -> int:
        connection.execute(
            """
            INSERT OR IGNORE INTO source_artifacts(
                provider, request_identifier, source_url, retrieved_at, content_hash,
                relative_path, byte_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (provider, request_identifier, source_url, retrieved_at, content_hash, relative_path, byte_count),
        )
        row = connection.execute(
            "SELECT id FROM source_artifacts WHERE provider=? AND request_identifier=? AND content_hash=?",
            (provider, request_identifier, content_hash),
        ).fetchone()
        assert row is not None
        return int(row["id"])

    def _period_id(self, connection: sqlite3.Connection, company_id: int, fact: Any) -> int:
        connection.execute(
            """
            INSERT OR IGNORE INTO financial_periods(
                company_id, fiscal_year, fiscal_quarter, period_type, start_date, end_date,
                filed_at, available_at, accession_number, form, source, is_amendment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company_id, fact.fiscal_year, fact.fiscal_quarter, fact.period_type,
                fact.start_date, fact.end_date, fact.filed_at, fact.available_at,
                fact.accession_number, fact.form, fact.source, int(fact.is_amendment),
            ),
        )
        row = connection.execute(
            """
            SELECT id FROM financial_periods
            WHERE company_id=? AND accession_number=? AND period_type=?
              AND start_date IS ? AND end_date=?
            """,
            (company_id, fact.accession_number, fact.period_type, fact.start_date, fact.end_date),
        ).fetchone()
        if row is None:
            raise ValueError("financial fact has an invalid or unsupported period")
        return int(row["id"])

    def insert_financial_facts(
        self, company_id: int, facts: Sequence[Any], *, source_artifact_id: int | None = None,
        processed_at: datetime | None = None,
    ) -> int:
        with self.transaction() as connection:
            return self._upsert_financial_facts(
                connection, company_id, facts, source_artifact_id=source_artifact_id,
                recorded_at=self._processing_timestamp(processed_at),
            )

    def _upsert_financial_facts(
        self,
        connection: sqlite3.Connection,
        company_id: int,
        facts: Sequence[Any],
        *,
        source_artifact_id: int | None,
        recorded_at: str,
    ) -> int:
        if self.schema_version() >= NORMALIZED_INTERPRETATION_SCHEMA_VERSION:
            return self._insert_versioned_financial_facts(
                connection, company_id, facts, source_artifact_id=source_artifact_id,
                recorded_at=recorded_at,
            )
        inserted = 0
        for fact in facts:
            period_id = self._period_id(connection, company_id, fact)
            cursor = connection.execute(
                    """
                    INSERT INTO financial_metrics(
                        company_id, period_id, canonical_metric, normalized_value, unit,
                        currency, taxonomy, source_concept, source_unit, source_value,
                        source_fact_id, mapping_version, mapping_priority, formula_version,
                        derivation_json, quality_status, warning, filed_at, available_at,
                        retrieved_at, source, source_artifact_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_fact_id) DO UPDATE SET
                        period_id=excluded.period_id,
                        canonical_metric=excluded.canonical_metric,
                        normalized_value=excluded.normalized_value,
                        unit=excluded.unit,
                        currency=excluded.currency,
                        mapping_version=excluded.mapping_version,
                        mapping_priority=excluded.mapping_priority,
                        formula_version=excluded.formula_version,
                        derivation_json=excluded.derivation_json,
                        quality_status=excluded.quality_status,
                        warning=excluded.warning,
                        source_artifact_id=excluded.source_artifact_id
                    WHERE financial_metrics.period_id IS NOT excluded.period_id
                       OR financial_metrics.canonical_metric IS NOT excluded.canonical_metric
                       OR financial_metrics.normalized_value IS NOT excluded.normalized_value
                       OR financial_metrics.unit IS NOT excluded.unit
                       OR financial_metrics.currency IS NOT excluded.currency
                       OR financial_metrics.mapping_version IS NOT excluded.mapping_version
                       OR financial_metrics.mapping_priority IS NOT excluded.mapping_priority
                       OR financial_metrics.formula_version IS NOT excluded.formula_version
                       OR financial_metrics.derivation_json IS NOT excluded.derivation_json
                       OR financial_metrics.quality_status IS NOT excluded.quality_status
                       OR financial_metrics.warning IS NOT excluded.warning
                       OR financial_metrics.source_artifact_id IS NOT excluded.source_artifact_id
                    """,
                (
                    company_id, period_id, fact.canonical_metric,
                    str(fact.normalized_value), fact.unit, fact.currency, fact.taxonomy,
                    fact.source_concept, fact.source_unit, str(fact.source_value),
                    fact.source_fact_id, fact.mapping_version, fact.mapping_priority,
                    fact.formula_version, json.dumps(fact.derivation, sort_keys=True)
                    if fact.derivation else None,
                    fact.quality_status, fact.warning, fact.filed_at, fact.available_at,
                    fact.retrieved_at, fact.source, source_artifact_id,
                ),
            )
            inserted += cursor.rowcount
        return inserted

    def _insert_versioned_financial_facts(
        self,
        connection: sqlite3.Connection,
        company_id: int,
        facts: Sequence[Any],
        *,
        source_artifact_id: int | None,
        recorded_at: str,
    ) -> int:
        for source in {fact.source for fact in facts}:
            self._validate_lifecycle_time(connection, company_id, source, recorded_at)
        changed = 0
        for fact in facts:
            period_id = self._period_id(connection, company_id, fact)
            source_values = {
                "source_fact_id": str(fact.source_fact_id),
                "company_id": company_id,
                "taxonomy": fact.taxonomy,
                "source_concept": fact.source_concept,
                "source_unit": fact.source_unit,
                "source_value": str(fact.source_value),
                "filed_at": fact.filed_at,
                "available_at": fact.available_at,
                "retrieved_at": fact.retrieved_at,
                "source": fact.source,
                "source_artifact_id": source_artifact_id,
            }
            connection.execute(
                """
                INSERT OR IGNORE INTO financial_source_facts(
                    source_fact_id,company_id,taxonomy,source_concept,
                    source_unit,source_value,filed_at,available_at,retrieved_at,
                    source,source_artifact_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                tuple(source_values.values()),
            )
            stored_source = connection.execute(
                "SELECT * FROM financial_source_facts WHERE source_fact_id=?",
                (fact.source_fact_id,),
            ).fetchone()
            immutable_source_keys = (
                "source_fact_id", "company_id", "taxonomy",
                "source_concept", "source_unit", "source_value", "filed_at",
                "available_at", "source",
            )
            if stored_source is None or any(
                stored_source[key] != source_values[key]
                for key in immutable_source_keys
            ):
                raise ValueError("source fact identity conflicts with immutable raw content")

            interpretation_values = {
                "source_fact_id": str(fact.source_fact_id),
                "period_id": period_id,
                "canonical_metric": fact.canonical_metric,
                "normalized_value": str(fact.normalized_value),
                "unit": fact.unit,
                "currency": fact.currency,
                "mapping_version": fact.mapping_version,
                "mapping_priority": fact.mapping_priority,
                "formula_version": fact.formula_version,
                "derivation_json": (
                    json.dumps(fact.derivation, sort_keys=True)
                    if fact.derivation else None
                ),
                "quality_status": fact.quality_status,
                "warning": fact.warning,
            }
            identity = self._interpretation_identity(interpretation_values)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO financial_metrics(
                    company_id,period_id,canonical_metric,normalized_value,unit,
                    currency,taxonomy,source_concept,source_unit,source_value,
                    source_fact_id,interpretation_identity,mapping_version,
                    mapping_priority,formula_version,derivation_json,quality_status,
                    warning,filed_at,available_at,retrieved_at,source,source_artifact_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    company_id, period_id, fact.canonical_metric,
                    str(fact.normalized_value), fact.unit, fact.currency, fact.taxonomy,
                    fact.source_concept, fact.source_unit, str(fact.source_value),
                    fact.source_fact_id, identity, fact.mapping_version,
                    fact.mapping_priority, fact.formula_version,
                    interpretation_values["derivation_json"], fact.quality_status,
                    fact.warning, fact.filed_at, fact.available_at, fact.retrieved_at,
                    fact.source, source_artifact_id,
                ),
            )
            metric = connection.execute(
                "SELECT id FROM financial_metrics WHERE interpretation_identity=?",
                (identity,),
            ).fetchone()
            assert metric is not None
            metric_id = int(metric["id"])
            current = connection.execute(
                """SELECT status FROM financial_metric_lifecycle_events
                   WHERE financial_metric_id=?
                   ORDER BY recorded_at DESC,id DESC LIMIT 1""",
                (metric_id,),
            ).fetchone()
            activated = current is None or current["status"] != "active"
            if activated:
                lifecycle_identity = self._lifecycle_identity(
                    metric_id, "active", recorded_at, fact.mapping_version
                )
                connection.execute(
                    """INSERT OR IGNORE INTO financial_metric_lifecycle_events(
                           financial_metric_id,status,recorded_at,mapping_version,
                           reason,lifecycle_identity
                       ) VALUES (?,?,?,?,?,?)""",
                    (
                        metric_id, "active", recorded_at, fact.mapping_version,
                        "emitted by authoritative mapping reconciliation",
                        lifecycle_identity,
                    ),
                )
            changed += int(cursor.rowcount > 0 or activated)
        return changed

    @staticmethod
    def _validate_lifecycle_time(
        connection: sqlite3.Connection, company_id: int, source: str, recorded_at: str,
    ) -> None:
        # Authoritative reconciliation retires across company/source, not just one
        # concept. Use that same scope for insertions, including retired history.
        # Equal boundaries are rejected: lifecycle identity dedup plus id ordering
        # cannot safely represent a second transition at the same instant.
        if not connection.in_transaction:
            raise RuntimeError("lifecycle_ordering_requires_transaction")
        proposed = datetime.fromisoformat(recorded_at)
        for row in connection.execute(
            """SELECT le.recorded_at FROM financial_metric_lifecycle_events le
               JOIN financial_metrics fm ON fm.id=le.financial_metric_id
               WHERE fm.company_id=? AND fm.source=?""", (company_id, source)
        ):
            if proposed <= datetime.fromisoformat(row[0]):
                raise ValueError("stale_interpretation_processing_time")

    @staticmethod
    def _processing_timestamp(processed_at: datetime | None) -> str:
        """One lifecycle clock per operation, independent of cached source time."""
        value = processed_at if processed_at is not None else datetime.now(timezone.utc)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("interpretation processing time must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat()

    def reconcile_financial_facts(
        self,
        company_id: int,
        facts: Sequence[Any],
        *,
        source_artifact_id: int,
        source: str = "SEC Company Facts",
        processed_at: datetime | None = None,
    ) -> tuple[int, int]:
        """Upsert one authoritative normalization and retire interpretations it omits.

        Raw source artifacts and old normalized rows remain immutable/auditable. Retired
        rows are excluded from active point-in-time reads and can be reactivated by a
        later mapping that emits the same source-fact identity.
        """

        with self.transaction() as connection:
            return self._reconcile_financial_facts(
                connection, company_id, facts, source_artifact_id, source,
                recorded_at=self._processing_timestamp(processed_at),
            )

    def reconcile_financial_fact_sets(
        self,
        batches: Sequence[tuple[int, Sequence[Any], int]],
        *,
        source: str = "SEC Company Facts",
        processed_at: datetime | None = None,
    ) -> list[tuple[int, int]]:
        """Reconcile several companies atomically for one authoritative provider."""

        with self.transaction() as connection:
            recorded_at = self._processing_timestamp(processed_at)
            return [
                self._reconcile_financial_facts(
                    connection, company_id, facts, source_artifact_id, source,
                    recorded_at=recorded_at,
                )
                for company_id, facts, source_artifact_id in batches
            ]

    def _reconcile_financial_facts(
        self,
        connection: sqlite3.Connection,
        company_id: int,
        facts: Sequence[Any],
        source_artifact_id: int,
        source: str,
        *,
        recorded_at: str,
    ) -> tuple[int, int]:
        if self.schema_version() >= NORMALIZED_INTERPRETATION_SCHEMA_VERSION:
            return self._reconcile_versioned_financial_facts(
                connection, company_id, facts, source_artifact_id, source,
                recorded_at=recorded_at,
            )
        active_ids = sorted({str(fact.source_fact_id) for fact in facts})
        changed = self._upsert_financial_facts(
            connection, company_id, facts, source_artifact_id=source_artifact_id,
            recorded_at=recorded_at,
        )
        connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS active_normalized_fact_ids "
            "(source_fact_id TEXT PRIMARY KEY)"
        )
        connection.execute("DELETE FROM active_normalized_fact_ids")
        connection.executemany(
            "INSERT INTO active_normalized_fact_ids(source_fact_id) VALUES (?)",
            ((value,) for value in active_ids),
        )
        retired = connection.execute(
            """
            UPDATE financial_metrics
            SET quality_status='retired',
                warning='retired by authoritative mapping reconciliation'
            WHERE company_id=? AND source=? AND quality_status<>'retired'
              AND NOT EXISTS (
                  SELECT 1 FROM active_normalized_fact_ids active
                  WHERE active.source_fact_id=financial_metrics.source_fact_id
              )
            """,
            (company_id, source),
        ).rowcount
        connection.execute("DELETE FROM active_normalized_fact_ids")
        return changed, retired

    def _reconcile_versioned_financial_facts(
        self,
        connection: sqlite3.Connection,
        company_id: int,
        facts: Sequence[Any],
        source_artifact_id: int,
        source: str,
        *,
        recorded_at: str,
    ) -> tuple[int, int]:
        self._validate_lifecycle_time(connection, company_id, source, recorded_at)
        changed = self._insert_versioned_financial_facts(
            connection, company_id, facts, source_artifact_id=source_artifact_id,
            recorded_at=recorded_at,
        )
        emitted: set[str] = set()
        for fact in facts:
            period_id = self._period_id(connection, company_id, fact)
            emitted.add(self._interpretation_identity({
                "source_fact_id": str(fact.source_fact_id),
                "period_id": period_id,
                "canonical_metric": fact.canonical_metric,
                "normalized_value": str(fact.normalized_value),
                "unit": fact.unit,
                "currency": fact.currency,
                "mapping_version": fact.mapping_version,
                "mapping_priority": fact.mapping_priority,
                "formula_version": fact.formula_version,
                "derivation_json": (
                    json.dumps(fact.derivation, sort_keys=True)
                    if fact.derivation else None
                ),
                "quality_status": fact.quality_status,
                "warning": fact.warning,
            }))
        artifact = connection.execute(
            "SELECT retrieved_at FROM source_artifacts WHERE id=?",
            (source_artifact_id,),
        ).fetchone()
        if artifact is None:
            raise ValueError("normalization reconciliation requires a source artifact")
        active_rows = list(connection.execute(
            """
            SELECT fm.id,fm.interpretation_identity,fm.mapping_version
            FROM financial_metrics fm
            WHERE fm.company_id=? AND fm.source=?
              AND COALESCE((
                  SELECT le.status FROM financial_metric_lifecycle_events le
                  WHERE le.financial_metric_id=fm.id
                  ORDER BY le.recorded_at DESC,le.id DESC LIMIT 1
              ),'active')='active'
            """,
            (company_id, source),
        ))
        retired = 0
        for row in active_rows:
            if row["interpretation_identity"] in emitted:
                continue
            lifecycle_identity = self._lifecycle_identity(
                int(row["id"]), "retired", recorded_at, row["mapping_version"]
            )
            cursor = connection.execute(
                """INSERT OR IGNORE INTO financial_metric_lifecycle_events(
                       financial_metric_id,status,recorded_at,mapping_version,
                       reason,lifecycle_identity
                   ) VALUES (?,?,?,?,?,?)""",
                (
                    row["id"], "retired", recorded_at, row["mapping_version"],
                    "retired by authoritative mapping reconciliation",
                    lifecycle_identity,
                ),
            )
            retired += cursor.rowcount
        return changed, retired

    def _financial_eligibility(self, as_of: str | datetime) -> tuple[str, tuple[Any, ...]]:
        """SQL for source-available, active-as-known interpretations (alias fm).

        Shared by revision history and latest-revision reads; revision selection
        remains the caller's responsibility. Lifecycle time is processing time,
        not the economic period, filing time, or cached source retrieval time.
        """
        as_of = _absolute_time(as_of)
        predicate = " AND financial_instant(fm.available_at) <= ? AND fm.quality_status <> 'retired'"
        parameters: tuple[Any, ...] = (as_of,)
        if self.schema_version() >= NORMALIZED_INTERPRETATION_SCHEMA_VERSION:
            predicate += """
              AND COALESCE((
                  SELECT le.status FROM financial_metric_lifecycle_events le
                  WHERE le.financial_metric_id=fm.id AND financial_instant(le.recorded_at)<=?
                  ORDER BY financial_instant(le.recorded_at) DESC,le.id DESC LIMIT 1
              ),'retired')='active'
            """
            parameters += (as_of,)
        return predicate, parameters

    def get_metric_history(
        self, ticker: str, metric: str, *, as_of: str | datetime | None = None
    ) -> list[sqlite3.Row]:
        """Revision-preserving history, optionally limited to active-as-known rows.

        Without a cutoff this is an audit history, including retired interpretations.
        With a cutoff it shares eligibility with ranking reads, but deliberately
        retains all eligible revisions rather than selecting only the latest.
        """
        parameters: list[Any] = [ticker.upper(), metric]
        boundary = ""
        temporal_order = "fm.available_at"  # Unrestricted audit history is unchanged.
        if as_of is not None:
            boundary, eligibility_parameters = self._financial_eligibility(as_of)
            parameters.extend(eligibility_parameters)
            temporal_order = "financial_instant(fm.available_at)"
        return list(
            self.connection.execute(
                """
                SELECT fm.*, fp.period_type, fp.start_date, fp.end_date, fp.fiscal_year,
                       fp.fiscal_quarter, fp.accession_number, fp.form, fp.is_amendment,
                       c.ticker, c.canonical_name
                FROM financial_metrics fm
                JOIN financial_periods fp ON fp.id = fm.period_id
                JOIN companies c ON c.id = fm.company_id
                WHERE c.ticker = ? AND fm.canonical_metric = ?
                """ + boundary +
                f" ORDER BY fp.end_date DESC, {temporal_order} DESC, fm.mapping_priority ASC, fm.id DESC",
                parameters,
            )
        )

    def get_metrics_as_of(self, ticker: str, as_of: str | datetime) -> list[sqlite3.Row]:
        """Return the latest known revision for every metric and fiscal period."""

        as_of = _absolute_time(as_of)
        company = self.get_company(ticker)
        if company is None:
            return []
        return self.get_metrics_as_of_company_id(int(company["id"]), as_of)

    def get_metrics_as_of_company_id(
        self, company_id: int, as_of: str | datetime
    ) -> list[sqlite3.Row]:
        """Return point-in-time metrics without resolving through a mutable ticker."""

        eligibility, eligibility_parameters = self._financial_eligibility(as_of)
        parameters = (company_id,) + eligibility_parameters

        return list(
            self.connection.execute(
                """
                WITH eligible AS (
                    SELECT fm.*, fp.period_type, fp.start_date, fp.end_date,
                           fp.fiscal_year, fp.fiscal_quarter, fp.accession_number,
                           fp.form, fp.is_amendment,
                           ROW_NUMBER() OVER (
                               PARTITION BY fm.canonical_metric, fp.period_type,
                                            COALESCE(fp.start_date, ''), fp.end_date
                               ORDER BY financial_instant(fm.available_at) DESC, fm.mapping_priority ASC, fm.id DESC
                           ) AS revision_rank
                    FROM financial_metrics fm
                    JOIN financial_periods fp ON fp.id = fm.period_id
                    WHERE fm.company_id = ?
                """ + eligibility + """
                )
                SELECT * FROM eligible WHERE revision_rank = 1
                ORDER BY canonical_metric, end_date DESC
                """,
                parameters,
            )
        )

    def get_latest_metric(self, ticker: str, metric: str, as_of: str | datetime) -> sqlite3.Row | None:
        return next(
            (
                row for row in self.get_metrics_as_of(ticker, as_of)
                if row["canonical_metric"] == metric
            ),
            None,
        )

    def snapshot_metrics(self, snapshot_id: str) -> list[sqlite3.Row]:
        """Replay the exact immutable normalized interpretations frozen by a snapshot."""

        return list(self.connection.execute(
            """SELECT fm.* FROM snapshot_inputs si
               JOIN financial_metrics fm ON fm.id=si.financial_metric_id
               WHERE si.snapshot_id=? ORDER BY fm.id""",
            (snapshot_id,),
        ))

    def get_catalysts(self, ticker: str, as_of: str) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT ca.* FROM catalysts ca JOIN companies c ON c.id=ca.company_id
                WHERE c.ticker=? AND ca.available_at<=?
                ORDER BY COALESCE(ca.expected_date, ca.source_date), ca.id
                """,
                (ticker.upper(), as_of),
            )
        )

    def get_risks(self, ticker: str, as_of: str) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT r.* FROM risks r JOIN companies c ON c.id=r.company_id
                WHERE c.ticker=? AND r.available_at<=?
                ORDER BY r.observed_date, r.id
                """,
                (ticker.upper(), as_of),
            )
        )

    def create_snapshot(self, *, as_of: str, metric_ids: Sequence[int], description: str) -> str:
        if self.schema_version() < NORMALIZED_INTERPRETATION_SCHEMA_VERSION:
            raise RuntimeError(
                "new analytics snapshots require immutable normalized interpretations "
                "from schema v6"
            )
        ordered = sorted(set(metric_ids))
        digest = hashlib.sha256(
            json.dumps({"as_of": as_of, "metric_ids": ordered}, separators=(",", ":")).encode()
        ).hexdigest()
        snapshot_id = digest[:24]
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO data_snapshots(id, created_at, as_of, input_hash, description) "
                "VALUES (?, ?, ?, ?, ?)",
                (snapshot_id, utc_now_iso(), as_of, digest, description),
            )
            for metric_id in ordered:
                connection.execute(
                    "INSERT OR IGNORE INTO snapshot_inputs(snapshot_id, financial_metric_id) VALUES (?, ?)",
                    (snapshot_id, metric_id),
                )
        return snapshot_id

    def database_status(self) -> dict[str, Any]:
        tables = (
            "companies", "financial_periods", "financial_metrics", "prices",
            "company_metadata_history", "corporate_actions",
            "provider_retrieval_events",
            "catalysts", "risks", "rankings", "ranking_components", "run_manifests",
            "source_artifacts", "macro_observations",
        )
        counts = {
            table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
        return {
            "schema_version": self.schema_version(),
            "path": str(self.path),
            "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "counts": counts,
        }

    def data_status(self, *, as_of_date: date | None = None) -> list[dict[str, Any]]:
        effective_date = as_of_date or datetime.now(timezone.utc).date()
        output: list[dict[str, Any]] = []
        current_interpretation = ""
        if self.schema_version() >= NORMALIZED_INTERPRETATION_SCHEMA_VERSION:
            current_interpretation = """
              AND COALESCE((
                  SELECT le.status FROM financial_metric_lifecycle_events le
                  WHERE le.financial_metric_id=fm.id
                  ORDER BY le.recorded_at DESC,le.id DESC LIMIT 1
              ),'retired')='active'
            """
        for company in self.list_companies():
            periods = self.connection.execute(
                """
                SELECT MAX(fp.filed_at) last_filing, MAX(fp.end_date) latest_period,
                       COUNT(DISTINCT CASE WHEN fp.period_type='FY' THEN fp.end_date END) annual_periods,
                       COUNT(DISTINCT CASE WHEN fp.period_type='Q' THEN fp.end_date END) quarterly_periods,
                       MAX(fm.retrieved_at) last_sync
                FROM financial_periods fp
                JOIN financial_metrics fm
                  ON fm.period_id=fp.id AND fm.quality_status<>'retired'
                WHERE fp.company_id=?
                """ + current_interpretation,
                (company["id"],),
            ).fetchone()
            quality_issue_count = int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM financial_metrics fm "
                    "WHERE fm.company_id=? AND fm.quality_status NOT IN ('valid', 'retired') "
                    + current_interpretation,
                    (company["id"],),
                ).fetchone()[0]
            )
            available = {
                row[0]
                for row in self.connection.execute(
                    "SELECT DISTINCT fm.canonical_metric FROM financial_metrics fm "
                    "WHERE fm.company_id=? AND fm.quality_status<>'retired' "
                    + current_interpretation,
                    (company["id"],),
                )
            }
            important = {"revenue", "net_income", "cash", "total_debt", "equity", "operating_cash_flow", "shares_outstanding"}
            output.append(
                {
                    "ticker": company["ticker"],
                    "last_filing": periods["last_filing"],
                    "latest_financial_period": periods["latest_period"],
                    "annual_periods": periods["annual_periods"],
                    "quarterly_periods": periods["quarterly_periods"],
                    "data_freshness_days": (
                        (effective_date - date.fromisoformat(periods["last_filing"][:10])).days
                        if periods["last_filing"] else None
                    ),
                    "missing_important_metrics": sorted(important - available),
                    "data_quality_issues": quality_issue_count,
                    "last_sync": periods["last_sync"],
                }
            )
        return output

    def store_analytics_run(
        self,
        run: Any,
        *,
        snapshot_id: str,
        weights: dict[str, dict[str, Decimal]],
        source_versions: dict[str, str],
    ) -> int:
        inserted = 0
        with self.transaction() as connection:
            manifest = run.manifest
            connection.execute(
                """
                INSERT OR IGNORE INTO run_manifests(
                    run_id, timestamp, pipeline_version, ranking_version,
                    calculation_version, input_hash, config_hash, source_versions_json,
                    warnings_json, errors_json, input_snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.run_id, manifest.timestamp.isoformat(), manifest.pipeline_version,
                    manifest.ranking_version, manifest.calculation_version, manifest.input_hash,
                    manifest.config_hash, json.dumps(source_versions, sort_keys=True),
                    json.dumps(manifest.warnings), json.dumps(manifest.errors), snapshot_id,
                ),
            )
            for horizon, results in run.rankings.items():
                ranked_results = [
                    result for result in results if result.ranking_status == "ranked"
                ]
                rankable_tickers = {result.ticker for result in ranked_results}
                ranked_priorities = [
                    item
                    for item in run.research_priorities[horizon]
                    if item.ticker in rankable_tickers
                ]
                priority_order = {
                    item.ticker: index
                    for index, item in enumerate(ranked_priorities, 1)
                }
                for opportunity_rank, result in enumerate(ranked_results, 1):
                    company = connection.execute(
                        "SELECT id FROM companies WHERE ticker=?", (result.ticker,)
                    ).fetchone()
                    if company is None:
                        raise ValueError(f"ranking company is absent from database: {result.ticker}")
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO rankings(
                            run_id, company_id, ranking_timestamp, horizon,
                            opportunity_score, opportunity_rank, research_priority_score,
                            research_priority, research_priority_rank, confidence,
                            data_quality_json, ranking_version, config_hash, input_snapshot_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            manifest.run_id, company["id"], manifest.timestamp.isoformat(), horizon,
                            str(result.opportunity_score), opportunity_rank,
                            str(result.research_priority_score), result.research_priority,
                            priority_order[result.ticker], result.confidence,
                            json.dumps(
                                {
                                    "missing_inputs": result.missing_inputs,
                                    "unavailable_inputs": result.unavailable_inputs,
                                    "warnings": result.warnings,
                                },
                                sort_keys=True,
                            ),
                            manifest.ranking_version, manifest.config_hash, snapshot_id,
                        ),
                    )
                    inserted += cursor.rowcount
                    ranking_row = connection.execute(
                        "SELECT id FROM rankings WHERE run_id=? AND company_id=? AND horizon=?",
                        (manifest.run_id, company["id"], horizon),
                    ).fetchone()
                    assert ranking_row is not None
                    for component, score in result.components.model_dump().items():
                        if score is None:
                            continue
                        weight = weights[horizon][component]
                        connection.execute(
                            """
                            INSERT OR IGNORE INTO ranking_components(
                                ranking_id, component, score, weight, contribution
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (ranking_row["id"], component, str(score), str(weight), str(Decimal(score) * weight)),
                        )
        return inserted

    def get_ranking_history(self, ticker: str, horizon: str | None = None) -> list[sqlite3.Row]:
        parameters: list[Any] = [ticker.upper()]
        condition = ""
        if horizon:
            condition = " AND r.horizon=?"
            parameters.append(horizon)
        return list(
            self.connection.execute(
                """
                SELECT r.*, c.ticker, c.canonical_name
                FROM rankings r JOIN companies c ON c.id=r.company_id
                WHERE c.ticker=?
                """ + condition + " ORDER BY r.ranking_timestamp DESC, r.horizon",
                parameters,
            )
        )

    def get_rankings(self, run_id: str, horizon: str | None = None) -> list[sqlite3.Row]:
        parameters: list[Any] = [run_id]
        condition = ""
        if horizon:
            condition = " AND r.horizon=?"
            parameters.append(horizon)
        return list(
            self.connection.execute(
                """
                SELECT r.*, c.ticker, c.canonical_name
                FROM rankings r JOIN companies c ON c.id=r.company_id
                WHERE r.run_id=?
                """ + condition + " ORDER BY r.horizon, r.opportunity_rank",
                parameters,
            )
        )
