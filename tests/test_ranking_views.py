from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from market_analytics.pipeline import run_fixture_pipeline
from market_analytics.ranking_views import (
    HISTORY_DIRECTORY_NAMES,
    HORIZON_FILE_NAMES,
    current_ranking_path,
    historical_filename,
    historical_ranking_path,
    proposed_ranking_paths,
    ranking_view_writes_enabled,
    refresh_current_ranking_view,
    render_ranking_markdown,
    write_historical_ranking_snapshot,
)
from obsidian import write_new_note_atomic
from tests.helpers import ANALYTICS_FIXTURE


class RankingViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analytics_run = run_fixture_pipeline(ANALYTICS_FIXTURE)

    def test_horizon_to_current_file_mapping(self) -> None:
        root = Path("/vault")
        expected = {
            "one_day": root / "Rankings/Current/1 Day Ranking.md",
            "one_week": root / "Rankings/Current/1 Week Ranking.md",
            "one_month": root / "Rankings/Current/1 Month Ranking.md",
            "one_year": root / "Rankings/Current/1 Year Ranking.md",
        }
        self.assertEqual(
            {horizon: current_ranking_path(root, horizon) for horizon in HORIZON_FILE_NAMES},
            expected,
        )

    def test_horizon_to_history_directory_mapping(self) -> None:
        self.assertEqual(
            HISTORY_DIRECTORY_NAMES,
            {
                "one_day": "Daily",
                "one_week": "Weekly",
                "one_month": "Monthly",
                "one_year": "Yearly",
            },
        )

    def test_historical_name_is_date_plus_run_id_and_deterministic(self) -> None:
        first = historical_filename(self.analytics_run, "one_day")
        second = historical_filename(self.analytics_run, "one_day")
        self.assertEqual(first, second)
        self.assertEqual(first, f"2026-09-29 - {self.analytics_run.manifest.run_id}.md")

    def test_unknown_horizon_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            current_ranking_path(Path("/vault"), "quarter")

    def test_render_is_deterministic(self) -> None:
        first = render_ranking_markdown(self.analytics_run, "one_year")
        second = render_ranking_markdown(self.analytics_run, "one_year")
        self.assertEqual(first, second)

    def test_current_frontmatter_has_required_metadata(self) -> None:
        markdown = render_ranking_markdown(self.analytics_run, "one_month")
        metadata = yaml.safe_load(markdown.split("---", 2)[1])
        self.assertEqual(metadata["type"], "generated-ranking-view")
        self.assertTrue(metadata["generated"])
        self.assertFalse(metadata["immutable"])
        self.assertEqual(metadata["ranking_horizon"], "one_month")
        self.assertEqual(metadata["companies_ranked"], 6)
        self.assertEqual(metadata["files_written"], 0)
        for field in (
            "generated_at", "ranking_version", "pipeline_version", "calculation_version",
            "run_id", "input_hash", "config_hash",
        ):
            self.assertTrue(metadata[field])

    def test_snapshot_frontmatter_is_immutable(self) -> None:
        markdown = render_ranking_markdown(self.analytics_run, "one_week", snapshot=True)
        metadata = yaml.safe_load(markdown.split("---", 2)[1])
        self.assertEqual(metadata["type"], "generated-ranking-snapshot")
        self.assertTrue(metadata["immutable"])

    def test_render_includes_both_orders_components_and_warnings(self) -> None:
        markdown = render_ranking_markdown(self.analytics_run, "one_day")
        for text in (
            "## Opportunity Ranking",
            "## Research-Priority Ranking",
            "Financial Strength",
            "## Major Catalysts",
            "## Major Risks",
            "## Important Warnings",
            "## Insufficient-Data Flags",
            "FICTIONAL DEMO DATA",
        ):
            self.assertIn(text, markdown)

    def test_proposed_paths_has_no_filesystem_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "vault"
            paths = proposed_ranking_paths(root, self.analytics_run)
            self.assertEqual(len(paths), 4)
            self.assertFalse(root.exists())

    def test_both_write_gates_are_required(self) -> None:
        combinations = (("0", "0"), ("1", "0"), ("0", "1"))
        for collector, ranking in combinations:
            with self.subTest(collector=collector, ranking=ranking), patch.dict(
                os.environ,
                {
                    "MARKET_COLLECTOR_ENABLE_WRITES": collector,
                    "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES": ranking,
                },
            ):
                self.assertFalse(ranking_view_writes_enabled())

    def test_disabled_refresh_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {
                "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES": "0",
            },
        ):
            root = Path(folder) / "vault"
            with self.assertRaises(PermissionError):
                refresh_current_ranking_view(root, self.analytics_run, "one_day")
            self.assertFalse(root.exists())

    def test_current_generated_view_can_refresh_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {
                "MARKET_COLLECTOR_ENABLE_WRITES": "1",
                "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES": "1",
            },
        ):
            root = Path(folder)
            destination = refresh_current_ranking_view(root, self.analytics_run, "one_day")
            original = destination.read_text(encoding="utf-8")
            refresh_current_ranking_view(root, self.analytics_run, "one_day")
            self.assertEqual(destination.read_text(encoding="utf-8"), original)

    def test_current_refresh_refuses_modified_generated_body(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {
                "MARKET_COLLECTOR_ENABLE_WRITES": "1",
                "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES": "1",
            },
        ):
            root = Path(folder)
            destination = refresh_current_ranking_view(root, self.analytics_run, "one_day")
            modified = destination.read_text(encoding="utf-8").replace(
                "# 1 Day Ranking", "# 1 Day Ranking\n\nHuman addition"
            )
            destination.write_text(modified, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "modified"):
                refresh_current_ranking_view(root, self.analytics_run, "one_day")
            self.assertEqual(destination.read_text(encoding="utf-8"), modified)

    def test_current_refresh_refuses_frontmatter_edit(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {"MARKET_COLLECTOR_ENABLE_WRITES": "1", "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES": "1"},
        ):
            root = Path(folder)
            destination = refresh_current_ranking_view(root, self.analytics_run, "one_day")
            modified = destination.read_text(encoding="utf-8").replace(
                "companies_ranked: 6", "companies_ranked: 7"
            )
            destination.write_text(modified, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "modified"):
                refresh_current_ranking_view(root, self.analytics_run, "one_day")
            self.assertEqual(destination.read_text(encoding="utf-8"), modified)

    def test_current_refresh_detects_concurrent_change_before_replace(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {"MARKET_COLLECTOR_ENABLE_WRITES": "1", "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES": "1"},
        ):
            root = Path(folder)
            destination = refresh_current_ranking_view(root, self.analytics_run, "one_day")
            original = destination.read_bytes()
            real_read_bytes = Path.read_bytes
            reads = 0

            def changed_on_compare(path: Path) -> bytes:
                nonlocal reads
                if path == destination:
                    reads += 1
                    if reads == 2:
                        return original + b"\nconcurrent human edit"
                return real_read_bytes(path)

            with patch.object(Path, "read_bytes", changed_on_compare):
                with self.assertRaisesRegex(RuntimeError, "concurrently"):
                    refresh_current_ranking_view(root, self.analytics_run, "one_day")
            self.assertEqual(destination.read_bytes(), original)

    def test_current_refresh_refuses_human_authored_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {
                "MARKET_COLLECTOR_ENABLE_WRITES": "1",
                "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES": "1",
            },
        ):
            root = Path(folder)
            destination = current_ranking_path(root, "one_day")
            destination.parent.mkdir(parents=True)
            destination.write_text("# Human note\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                refresh_current_ranking_view(root, self.analytics_run, "one_day")
            self.assertEqual(destination.read_text(encoding="utf-8"), "# Human note\n")

    def test_current_refresh_refuses_redirected_ranking_directory(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {
                "MARKET_COLLECTOR_ENABLE_WRITES": "1",
                "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES": "1",
            },
        ):
            root = Path(folder) / "vault"
            outside = Path(folder) / "outside"
            outside.mkdir()
            (root / "Rankings").mkdir(parents=True)
            (root / "Rankings" / "Current").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                refresh_current_ranking_view(root, self.analytics_run, "one_day")
            self.assertEqual(list(outside.iterdir()), [])

    def test_history_snapshot_is_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {
                "MARKET_COLLECTOR_ENABLE_WRITES": "1",
                "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES": "1",
            },
        ):
            root = Path(folder)
            destination = write_historical_ranking_snapshot(root, self.analytics_run, "one_year")
            original = destination.read_text(encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_historical_ranking_snapshot(root, self.analytics_run, "one_year")
            self.assertEqual(destination.read_text(encoding="utf-8"), original)
            self.assertEqual(destination, historical_ranking_path(root, self.analytics_run, "one_year"))

    def test_ranking_writes_touch_only_rankings_tree(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {
                "MARKET_COLLECTOR_ENABLE_WRITES": "1",
                "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES": "1",
            },
        ):
            root = Path(folder)
            refresh_current_ranking_view(root, self.analytics_run, "one_month")
            write_historical_ranking_snapshot(root, self.analytics_run, "one_month")
            files = [path.relative_to(root) for path in root.rglob("*.md")]
            self.assertEqual(len(files), 2)
            self.assertTrue(all(path.parts[0] == "Rankings" for path in files))
            self.assertFalse((root / "03 Articles").exists())

    def test_research_note_writer_still_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {
                "MARKET_COLLECTOR_ENABLE_WRITES": "1",
                "MARKET_ANALYTICS_ENABLE_RANKING_VIEW_WRITES": "1",
            },
        ):
            root = Path(folder)
            note = root / "03 Articles" / "existing.md"
            note.parent.mkdir(parents=True)
            note.write_text("human research", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_new_note_atomic(note, "replacement", allowed_root=root)
            self.assertEqual(note.read_text(encoding="utf-8"), "human research")


if __name__ == "__main__":
    unittest.main()
