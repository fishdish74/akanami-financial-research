from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sqlite3
import unittest
import tempfile
from decimal import Decimal
from pathlib import Path

import yaml

import collector
from market_analytics import ANALYTICS_COMMANDS
from market_analytics.cli import build_parser
from market_analytics.database import MarketDatabase, SCHEMA_VERSION
from market_analytics.dcf import sensitivity_matrix
from market_analytics.pipeline import run_fixture_pipeline
from market_analytics.providers import FixtureProvider
from market_analytics.reports import (
    company_snapshot, comparison_report, data_quality_report, dcf_report,
    full_demo_report, ranking_report, sensitivity_table, watchlist_summary,
)
from tests.helpers import ANALYTICS_FIXTURE


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analytics_run = run_fixture_pipeline(ANALYTICS_FIXTURE)

    def test_full_report_is_deterministic_with_valid_yaml(self) -> None:
        first = full_demo_report(self.analytics_run)
        second = full_demo_report(self.analytics_run)
        self.assertEqual(first, second)
        metadata = yaml.safe_load(first.split("---", 2)[1])
        self.assertTrue(metadata["fictional_fixture"])
        self.assertEqual(metadata["files_written"], 0)

    def test_ranking_report_has_stable_ticker_order(self) -> None:
        report = ranking_report(self.analytics_run.rankings["one_year"], "One Year")
        positions = [report.index(f"| {index} | {item.ticker} |") for index, item in enumerate(self.analytics_run.rankings["one_year"], 1)]
        self.assertEqual(positions, sorted(positions))

    def test_comparison_does_not_declare_winner(self) -> None:
        report = comparison_report(self.analytics_run, "QLTY", "GRWX")
        self.assertIn("does not declare a winner", report)
        self.assertIn("## Financial Metrics", report)

    def test_company_snapshot_contains_all_structured_sections(self) -> None:
        report = company_snapshot(self.analytics_run, "QLTY")
        for section in ("## Snapshot", "## Component Scores", "## Key Metrics", "## Catalysts", "## Risks", "## Data Quality"):
            self.assertIn(section, report)

    def test_dcf_and_sensitivity_reports_are_markdown_tables(self) -> None:
        provider = FixtureProvider(ANALYTICS_FIXTURE)
        scenarios = provider.dcf_scenarios("QLTY")
        report = dcf_report(self.analytics_run.dcf_results["QLTY"])
        matrix = sensitivity_matrix(
            scenarios[1], [Decimal("0.08"), Decimal("0.09")], [Decimal("0.02"), Decimal("0.03")]
        )
        self.assertIn("| Bear |", report)
        self.assertIn("No center cell is designated as correct", sensitivity_table(matrix))

    def test_watchlist_summary_is_not_recommendation(self) -> None:
        report = watchlist_summary(self.analytics_run)
        self.assertIn("not a price forecast or recommendation", report)
        self.assertIn("Highest research priority", report)

    def test_quality_report_orders_issues_deterministically(self) -> None:
        first = data_quality_report(self.analytics_run.quality_issues["LEVR"])
        second = data_quality_report(self.analytics_run.quality_issues["LEVR"])
        self.assertEqual(first, second)


class AnalyticsCliTests(unittest.TestCase):
    def invoke(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = collector.main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_command_registry_matches_cli_parser(self) -> None:
        parser = build_parser()
        command_action = next(
            action for action in parser._actions if action.dest == "analytics_command"
        )
        self.assertEqual(set(command_action.choices), set(ANALYTICS_COMMANDS))

    def test_demo_command_is_offline_and_reports_zero_writes(self) -> None:
        code, output, error = self.invoke(["demo"])
        self.assertEqual(code, 0)
        self.assertEqual(error, "")
        self.assertIn("files_written: 0", output)

    def test_rank_command_accepts_horizon(self) -> None:
        code, output, _ = self.invoke(["rank", "--horizon", "1d"])
        self.assertEqual(code, 0)
        self.assertIn("| 1 | CATX |", output)
        self.assertIn("FICTIONAL / SYNTHETIC FIXTURE DATA", output)

    def test_exact_cutoff_requires_timezone_and_is_preserved(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "rank", "--real-data", "--input-cutoff", "2026-09-30T14:45:12-07:00"
        ])
        self.assertEqual(args.input_cutoff.isoformat(), "2026-09-30T21:45:12+00:00")
        with self.assertRaises(SystemExit):
            parser.parse_args(["rank", "--real-data", "--input-cutoff", "2026-09-30T14:45:12"])

    def test_every_fixture_cli_output_is_visibly_labeled(self) -> None:
        for arguments in (["rank"], ["compare", "QLTY", "GRWX"], ["dcf", "QLTY"]):
            with self.subTest(arguments=arguments):
                code, output, error = self.invoke(arguments)
                self.assertEqual((code, error), (0, ""))
                self.assertIn("FICTIONAL / SYNTHETIC FIXTURE DATA", output)
        code, output, error = self.invoke(["validate"])
        self.assertEqual((code, error), (0, ""))
        payload = __import__("json").loads(output)
        self.assertTrue(payload["fictional_fixture"])
        self.assertIn("fictional", payload["fixture_notice"].casefold())

    def test_read_only_command_does_not_create_mistyped_database(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mistyped.sqlite3"
            code, _, error = self.invoke(["data-status", "--database", str(path)])
            self.assertEqual(code, 2)
            self.assertIn("does not exist", error)
            self.assertFalse(path.exists())

    def test_database_status_does_not_create_missing_database(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.sqlite3"
            code, _, error = self.invoke(["database-status", "--database", str(path)])
            self.assertEqual(code, 2)
            self.assertIn("does not exist", error)
            self.assertFalse(path.exists())

    def test_database_status_reports_old_schema_without_migrating(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE schema_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO schema_metadata VALUES ('schema_version', '2');
                CREATE TABLE companies(id INTEGER PRIMARY KEY, ticker TEXT NOT NULL UNIQUE);
                CREATE TABLE prices(
                    id INTEGER PRIMARY KEY, company_id INTEGER NOT NULL,
                    price_date TEXT NOT NULL, open TEXT, high TEXT, low TEXT,
                    close TEXT NOT NULL, adjusted_close TEXT, volume TEXT,
                    currency TEXT NOT NULL, source TEXT NOT NULL,
                    available_at TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                    source_identifier TEXT NOT NULL,
                    UNIQUE(company_id, price_date, source, source_identifier)
                );
                """
            )
            connection.close()
            before = hashlib.sha256(path.read_bytes()).hexdigest()

            code, output, error = self.invoke(["database-status", "--database", str(path)])

            self.assertEqual((code, error), (0, ""))
            status = json.loads(output)
            self.assertEqual(status["schema_version"], 2)
            self.assertEqual(status["current_schema_version"], SCHEMA_VERSION)
            self.assertTrue(status["migration_needed"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            with sqlite3.connect(path) as check:
                self.assertEqual(
                    check.execute(
                        "SELECT value FROM schema_metadata WHERE key='schema_version'"
                    ).fetchone()[0],
                    "2",
                )

    def test_database_status_reads_current_schema_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "current.sqlite3"
            with MarketDatabase(path) as database:
                database.migrate()
            before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            before_mtime = path.stat().st_mtime_ns

            code, output, error = self.invoke(["database-status", "--database", str(path)])

            self.assertEqual((code, error), (0, ""))
            status = json.loads(output)
            self.assertEqual(status["schema_version"], SCHEMA_VERSION)
            self.assertFalse(status["migration_needed"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before_hash)
            self.assertEqual(path.stat().st_mtime_ns, before_mtime)

    def test_real_data_rank_persistence_is_opt_in(self) -> None:
        args = build_parser().parse_args(["rank", "--real-data"])
        self.assertFalse(args.persist)

    def test_compare_command_requires_real_fixture_tickers(self) -> None:
        code, _, error = self.invoke(["compare", "QLTY", "NOPE"])
        self.assertEqual(code, 2)
        self.assertIn("not present", error)

    def test_dcf_command_fails_when_assumptions_are_absent(self) -> None:
        code, _, error = self.invoke(["dcf", "VALX"])
        self.assertEqual(code, 2)
        self.assertIn("no DCF assumptions", error)

    def test_validate_command_returns_structured_output(self) -> None:
        code, output, _ = self.invoke(["validate"])
        self.assertEqual(code, 0)
        self.assertIn('"files_written": 0', output)
        self.assertIn('"LEVR"', output)

    def test_legacy_collector_cli_still_parses(self) -> None:
        args = collector.parse_args(["--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo"])
        self.assertEqual(args.ticker, "NVDA")


if __name__ == "__main__":
    unittest.main()
