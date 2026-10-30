"""Useful offline analytics commands; all output goes to stdout and no notes are written."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from .dcf import sensitivity_matrix
from .config import load_config
from .database import MarketDatabase, SCHEMA_VERSION, _absolute_time
from .database_provider import DatabaseProvider
from .pipeline import run_fixture_pipeline, run_provider_pipeline
from .providers import FixtureProvider
from .quality import validate_company
from .reports import (
    comparison_report, dcf_report, full_demo_report, ranking_report,
    sensitivity_table, watchlist_summary,
)
from security import safe_error


DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "analytics-demo.json"
HORIZONS = {"1d": "one_day", "1w": "one_week", "1m": "one_month", "1y": "one_year"}
DEFAULT_DATABASE = Path(__file__).resolve().parents[1] / "data" / "market_data.sqlite3"
DEFAULT_SEC_CACHE = Path(__file__).resolve().parents[1] / "data" / "sec"


def aware_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include an explicit timezone offset")
    return parsed.astimezone(timezone.utc)


def financial_cutoff(value: str) -> datetime:
    """Apply the database contract before parsing can truncate/normalize raw input."""
    try:
        return datetime.fromisoformat(_absolute_time(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _cutoff(args: argparse.Namespace, *, optional: bool = False) -> date | datetime | None:
    if getattr(args, "input_cutoff", None) is not None:
        return args.input_cutoff
    if getattr(args, "as_of", None) is not None:
        return args.as_of
    return None if optional else datetime.now(timezone.utc).date()


def _boundary(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return f"{value.isoformat()}T23:59:59.999999+00:00"


def _fixture_banner(notice: str) -> str:
    return f"FICTIONAL / SYNTHETIC FIXTURE DATA — {notice}\n\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic offline financial analytics")
    subparsers = parser.add_subparsers(dest="analytics_command", required=True)

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        value = subparsers.add_parser(name, help=help_text)
        value.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="Structured fictional fixture file")
        return value

    command("demo", "Run the complete offline analytics pipeline")
    rank = command("rank", "Show horizon-specific opportunity ranking")
    rank.add_argument("--horizon", choices=HORIZONS, default="1y")
    rank.add_argument("--real-data", action="store_true", help="Read point-in-time data from SQLite")
    rank.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    rank_cutoff = rank.add_mutually_exclusive_group()
    rank_cutoff.add_argument("--as-of", type=date.fromisoformat)
    rank_cutoff.add_argument("--input-cutoff", type=financial_cutoff)
    rank.add_argument(
        "--persist", action="store_true",
        help="Explicitly persist the real-data run manifest and rankings",
    )

    compare = command("compare", "Compare two fixture companies without declaring a winner")
    compare.add_argument("first")
    compare.add_argument("second")
    compare.add_argument("--horizon", choices=HORIZONS, default="1y")

    command("validate", "Report deterministic fixture data-quality issues")
    dcf = command("dcf", "Show DCF bear/base/bull scenarios and sensitivity")
    dcf.add_argument("ticker")
    command("watchlist-summary", "Show deterministic fictional watchlist analytics")

    def data_command(name: str, help_text: str) -> argparse.ArgumentParser:
        value = subparsers.add_parser(name, help=help_text)
        value.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
        return value

    data_command("database-status", "Show local database schema, size, and row counts")
    data_command("data-status", "Show per-company filing coverage and missing metrics")
    sync = data_command("sec-sync", "Cache, normalize, and ingest SEC Company Facts")
    sync.add_argument("ticker")
    sync.add_argument("--cache-dir", type=Path, default=DEFAULT_SEC_CACHE)
    fundamentals = data_command("fundamentals", "Show canonical point-in-time fundamentals")
    fundamentals.add_argument("ticker")
    fundamentals_cutoff = fundamentals.add_mutually_exclusive_group()
    fundamentals_cutoff.add_argument("--as-of", type=date.fromisoformat)
    fundamentals_cutoff.add_argument("--input-cutoff", type=financial_cutoff)
    history = data_command("history", "Show one canonical metric's revision-preserving history")
    history.add_argument("ticker")
    history.add_argument("metric")
    history_cutoff = history.add_mutually_exclusive_group()
    history_cutoff.add_argument("--as-of", type=date.fromisoformat)
    history_cutoff.add_argument("--input-cutoff", type=financial_cutoff)
    validate_data = data_command("validate-data", "Run deterministic quality checks on database data")
    validate_data.add_argument("ticker")
    validate_cutoff = validate_data.add_mutually_exclusive_group()
    validate_cutoff.add_argument("--as-of", type=date.fromisoformat)
    validate_cutoff.add_argument("--input-cutoff", type=financial_cutoff)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        command = args.analytics_command
        if command in {"database-status", "data-status", "sec-sync", "fundamentals", "history", "validate-data"}:
            mutable = command == "sec-sync"
            with MarketDatabase(args.database, read_only=not mutable) as database:
                if command == "sec-sync":
                    database.prepare_sec_schema()
                elif database.schema_version() == 0:
                    raise ValueError("database is not initialized; run sec-sync to initialize it")
                if command == "database-status":
                    schema_version = database.schema_version()
                    if schema_version == SCHEMA_VERSION:
                        status = database.database_status()
                    else:
                        status = {
                            "schema_version": schema_version,
                            "path": str(database.path),
                            "size_bytes": database.path.stat().st_size,
                        }
                    status["current_schema_version"] = SCHEMA_VERSION
                    status["migration_needed"] = schema_version < SCHEMA_VERSION
                    print(json.dumps(status, indent=2, sort_keys=True))
                elif command == "data-status":
                    print(json.dumps(database.data_status(), indent=2, sort_keys=True))
                elif command == "fundamentals":
                    boundary = _boundary(_cutoff(args))
                    rows = database.get_metrics_as_of(args.ticker, boundary)
                    print(json.dumps([dict(row) for row in rows], indent=2, sort_keys=True))
                elif command == "history":
                    boundary = _boundary(_cutoff(args, optional=True))
                    rows = database.get_metric_history(args.ticker, args.metric, as_of=boundary)
                    print(json.dumps([dict(row) for row in rows], indent=2, sort_keys=True))
                elif command == "validate-data":
                    cutoff = _cutoff(args)
                    provider = DatabaseProvider(database, cutoff)
                    company = provider.company(args.ticker)
                    issues = validate_company(company, provider.analysis_date, load_config().freshness_days)
                    print(json.dumps([issue.model_dump(mode="json") for issue in issues], indent=2, sort_keys=True))
                elif command == "sec-sync":
                    from config import CollectorConfig, source_configs
                    from entities import EntityRegistry
                    from sources.sec_fundamentals import RawSecCache, SecCompanyFactsClient
                    from .sec_ingestion import sync_sec_company

                    registry = EntityRegistry.load(CollectorConfig().registry_path)
                    entity = registry.resolve(args.ticker)
                    if entity is None:
                        raise ValueError(f"ticker is not present in entity registry: {args.ticker.upper()}")
                    client = SecCompanyFactsClient(
                        source_configs()["sec"], RawSecCache(args.cache_dir)
                    )
                    result = sync_sec_company(
                        database, entity, client, cache_root=args.cache_dir
                    )
                    print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        if command == "rank" and args.real_data:
            with MarketDatabase(args.database, read_only=not args.persist) as database:
                if database.schema_version() == 0:
                    raise ValueError("database is not initialized; run database-status or sec-sync")
                cutoff = _cutoff(args)
                provider = DatabaseProvider(database, cutoff)
                run = run_provider_pipeline(provider)
                if not run.companies:
                    raise ValueError("database contains no companies to rank")
                boundary = provider.input_cutoff.isoformat()
                if args.persist:
                    metric_ids = [
                        int(row["id"])
                        for company in database.list_companies()
                        for row in database.get_metrics_as_of(company["ticker"], boundary)
                    ]
                    snapshot_id = database.create_snapshot(
                        as_of=boundary,
                        metric_ids=metric_ids,
                        description="Point-in-time inputs for deterministic real-data ranking",
                    )
                    configuration = load_config()
                    from .sec_xbrl import load_concept_mapping

                    mapping_version, _ = load_concept_mapping()
                    database.store_analytics_run(
                        run,
                        snapshot_id=snapshot_id,
                        weights=configuration.ranking_weights,
                        source_versions={"sec_xbrl_mapping": mapping_version},
                    )
                horizon = HORIZONS[args.horizon]
                print(ranking_report(run.rankings[horizon], f"{args.horizon} Opportunity Ranking"), end="")
                print("\n" + ranking_report(run.research_priorities[horizon], f"{args.horizon} Research-Priority Ranking"), end="")
            return 0

        run = run_fixture_pipeline(args.fixture)
        if command == "demo":
            print(full_demo_report(run), end="")
        elif command == "rank":
            horizon = HORIZONS[args.horizon]
            print(_fixture_banner(run.fixture_notice) + ranking_report(run.rankings[horizon], f"{args.horizon} Opportunity Ranking"), end="")
        elif command == "compare":
            print(_fixture_banner(run.fixture_notice) + comparison_report(run, args.first, args.second, HORIZONS[args.horizon]), end="")
        elif command == "validate":
            payload = {
                "run_id": run.manifest.run_id,
                "fictional_fixture": True,
                "fixture_notice": run.fixture_notice,
                "files_written": 0,
                "issues": {
                    ticker: [issue.model_dump(mode="json") for issue in issues]
                    for ticker, issues in sorted(run.quality_issues.items())
                },
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif command == "dcf":
            provider = FixtureProvider(args.fixture)
            scenarios = provider.dcf_scenarios(args.ticker)
            if not scenarios:
                raise ValueError(f"no DCF assumptions exist for {args.ticker.upper()}")
            print(_fixture_banner(run.fixture_notice) + dcf_report(run.dcf_results[args.ticker.upper()]), end="")
            base = next((scenario for scenario in scenarios if scenario.name.casefold() == "base"), scenarios[0])
            matrix = sensitivity_matrix(
                base,
                [Decimal("0.07"), Decimal("0.08"), Decimal("0.09"), Decimal("0.10"), Decimal("0.11")],
                [Decimal("0.01"), Decimal("0.02"), Decimal("0.03"), Decimal("0.04")],
            )
            print("\n" + sensitivity_table(matrix), end="")
        elif command == "watchlist-summary":
            print(watchlist_summary(run), end="")
        return 0
    except (ValueError, KeyError, sqlite3.Error) as exc:
        error = safe_error(exc, category="analytics")
        print(f"analytics error: {error['message']}", file=sys.stderr)
        return 2
