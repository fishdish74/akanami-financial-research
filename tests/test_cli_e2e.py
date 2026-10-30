from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from analysis import (
    AnalysisResult, ModelResponseValidationError, ResponseTelemetry,
    maximum_analysis_request_cost_usd,
)
from collector import build_sources, parse_args, run, validate_args
from config import CollectorConfig, ModelConfig
from tests.helpers import analysis as make_fixture_analysis, candidate as make_fixture_candidate


class _FakeSource:
    def __init__(
        self, name: str, candidates: list[object] | None = None, *,
        error: Exception | None = None, warnings: list[str] | None = None,
    ) -> None:
        self.name = name
        self.candidates = list(candidates or [])
        self.error = error
        self.last_warnings = list(warnings or [])

    def discover(self, entity, limit):
        if self.error is not None:
            raise self.error
        return self.candidates[:limit]


def _source_candidates(source: str, count: int) -> list[object]:
    return [
        make_fixture_candidate(
            title=f"NVIDIA reports material {source} event {index}",
            source=source,
            source_url=f"https://{source.casefold()}.example.com/event-{index}",
            content=(
                f"NVIDIA Corporation reported material {source} event {index} with "
                "enough distinct fixture text for deterministic collector testing."
            ),
            snippet=f"Material {source} event {index}.",
        )
        for index in range(count)
    ]


class CliValidationTests(unittest.TestCase):
    def test_nonpositive_candidate_limit_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--ticker", "NVDA", "--dry-run", "--max-candidates", "0"])

    def test_write_requires_live_ai(self) -> None:
        args = parse_args(["--ticker", "NVDA", "--write", "--offline-analysis"])
        with self.assertRaisesRegex(ValueError, "requires --enable-ai"):
            validate_args(args)

    def test_write_gate_is_fail_closed(self) -> None:
        args = parse_args(["--ticker", "NVDA", "--write", "--enable-ai"])
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "present-but-never-used", "MARKET_COLLECTOR_ENABLE_WRITES": "0"},
        ):
            with self.assertRaises(PermissionError):
                validate_args(args)

    def test_unknown_source_is_rejected(self) -> None:
        args = parse_args(["--ticker", "NVDA", "--dry-run", "--sources", "unknown"])
        validate_args(args)
        with self.assertRaisesRegex(ValueError, "unsupported source"):
            build_sources(args)

    def test_fixture_forces_manual_source(self) -> None:
        args = parse_args(["--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo"])
        validate_args(args)
        self.assertEqual(args.sources, "manual")
        self.assertTrue(str(args.candidate_file).endswith("fixtures/offline-demo.json"))


class OfflineEndToEndTests(unittest.TestCase):
    def _config(self, root: Path) -> CollectorConfig:
        registry = root / "entities.json"
        registry.write_text(
            json.dumps(
                {
                    "entities": [
                        {
                            "ticker": "NVDA",
                            "canonical_name": "NVIDIA Corporation",
                            "aliases": ["NVIDIA"],
                            "exchange": "NASDAQ",
                            "alternate_listings": [],
                            "cik": "0001045810",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return CollectorConfig(
            vault_dir=root,
            article_dir=root / "03 Articles",
            registry_path=registry,
            max_deep_calls=3,
            max_ai_budget_usd=0.25,
            min_content_chars=80,
            model=ModelConfig(),
        )

    def test_fixture_run_exercises_pipeline_without_writes_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args(
                [
                    "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                    "--offline-analysis", "--max-candidates", "4", "--max-deep-calls", "3",
                ]
            )
            with patch.dict(os.environ, {"MARKET_COLLECTOR_ENABLE_WRITES": "0"}):
                validate_args(args)
                summary, exit_code = run(args, self._config(root))
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["candidates_discovered"], 4)
            self.assertGreaterEqual(summary["candidates_rejected_before_analysis"], 1)
            self.assertEqual(summary["model_requests"], 0)
            self.assertEqual(summary["files_created"], [])
            self.assertFalse((root / "03 Articles").exists())
            self.assertEqual(summary["processing_failures"], [])
            self.assertEqual(summary["files_would_create"], [])
            smoke = [item for item in summary["outcomes"] if item["decision"] == "SMOKE TEST ONLY"]
            self.assertTrue(smoke)
            self.assertTrue(all(not item["eligible_for_research"] for item in smoke))
            self.assertTrue(
                all("candidate_id" in outcome for outcome in summary["outcomes"])
            )

    def test_discovery_only_reports_deferred_without_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args(
                ["--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo", "--max-candidates", "1"]
            )
            validate_args(args)
            summary, exit_code = run(args, self._config(root))
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["files_would_create"], [])
            self.assertIn("DEFER", {item["decision"] for item in summary["outcomes"]})

    def test_missing_fixture_is_a_sanitized_source_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args(
                ["--ticker", "NVDA", "--dry-run", "--fixture", "does-not-exist", "--offline-analysis"]
            )
            validate_args(args)
            summary, exit_code = run(args, self._config(root))
            self.assertEqual(exit_code, 1)
            self.assertEqual(len(summary["source_errors"]), 1)
            self.assertEqual(summary["source_errors"][0]["category"], "source")

    def test_successful_model_usage_survives_renderer_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                "--enable-ai", "--max-candidates", "1",
            ])
            analyzer = Mock()
            analyzer.analyze.return_value = AnalysisResult(
                payload=make_fixture_analysis(),
                input_tokens=1000,
                output_tokens=500,
                approximate_cost_usd=0.00123,
                usage_available=True,
                request_id="resp_fixture",
                response_model="fixture-model",
            )
            with (
                patch.dict(os.environ, {
                    "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                    "OPENAI_API_KEY": "fixture-key-never-used",
                }),
                patch("collector.OpenAIAnalyzer", return_value=analyzer),
                patch("collector.render_article", side_effect=RuntimeError("renderer failed")),
            ):
                validate_args(args)
                summary, exit_code = run(args, self._config(root))

            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["model_attempts"], 1)
            self.assertEqual(summary["model_requests"], 1)
            self.assertEqual(summary["model_responses"], 1)
            self.assertEqual(summary["model_successes"], 1)
            self.assertEqual(summary["model_failures"], 0)
            self.assertEqual(summary["ai_input_tokens"], 1000)
            self.assertEqual(summary["ai_output_tokens"], 500)
            self.assertEqual(summary["approximate_ai_cost_usd"], 0.00123)
            self.assertEqual(len(summary["processing_failures"]), 1)

    def test_successful_model_usage_survives_payload_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                "--enable-ai", "--max-candidates", "1",
            ])
            analyzer = Mock()
            analyzer.analyze.return_value = AnalysisResult(
                payload=make_fixture_analysis(), input_tokens=1000, output_tokens=500,
                approximate_cost_usd=0.00123, usage_available=True,
                request_id="resp_fixture", response_model="fixture-model",
            )
            with (
                patch.dict(os.environ, {
                    "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                    "OPENAI_API_KEY": "fixture-key-never-used",
                }),
                patch("collector.OpenAIAnalyzer", return_value=analyzer),
                patch("collector.AnalysisPayload.model_validate", side_effect=ValueError("validation failed")),
            ):
                validate_args(args)
                summary, exit_code = run(args, self._config(root))

            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["model_successes"], 1)
            self.assertEqual(summary["model_failures"], 0)
            self.assertEqual(summary["ai_input_tokens"], 1000)
            self.assertEqual(summary["ai_output_tokens"], 500)
            self.assertEqual(summary["approximate_ai_cost_usd"], 0.00123)
            self.assertEqual(len(summary["processing_failures"]), 1)

    def test_failed_model_response_with_usage_is_accounted_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                "--enable-ai", "--max-candidates", "1",
            ])
            analyzer = Mock()
            analyzer.analyze.side_effect = ModelResponseValidationError(
                "model response failed validation",
                ResponseTelemetry(
                    request_id="resp_fixture", response_model="fixture-model",
                    latency_ms=1, refusal=None, input_tokens=1000,
                    cached_input_tokens=0, output_tokens=500, reasoning_tokens=0,
                    total_tokens=1500, approximate_cost_usd=0.00123,
                    usage_available=True,
                ),
            )
            with (
                patch.dict(os.environ, {
                    "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                    "OPENAI_API_KEY": "fixture-key-never-used",
                }),
                patch("collector.OpenAIAnalyzer", return_value=analyzer),
            ):
                validate_args(args)
                summary, exit_code = run(args, self._config(root))

            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["model_attempts"], 1)
            self.assertEqual(summary["model_requests"], 1)
            self.assertEqual(summary["model_responses"], 1)
            self.assertEqual(summary["model_successes"], 0)
            self.assertEqual(summary["model_failures"], 1)
            self.assertEqual(summary["ai_input_tokens"], 1000)
            self.assertEqual(summary["ai_output_tokens"], 500)
            self.assertEqual(summary["approximate_ai_cost_usd"], 0.00123)
            self.assertEqual(summary["failed_attempt_usage_unknown"], 0)

    def test_failed_model_response_without_usage_stops_later_ai_work(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                "--enable-ai", "--max-candidates", "4", "--max-deep-calls", "3",
            ])
            analyzer = Mock()
            analyzer.analyze.side_effect = ModelResponseValidationError(
                "model response failed without usage",
                ResponseTelemetry(
                    request_id="resp_fixture", response_model="fixture-model",
                    latency_ms=1, refusal=None, input_tokens=0,
                    cached_input_tokens=0, output_tokens=0, reasoning_tokens=0,
                    total_tokens=0, approximate_cost_usd=0.0,
                    usage_available=False,
                ),
            )
            with (
                patch.dict(os.environ, {
                    "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                    "OPENAI_API_KEY": "fixture-key-never-used",
                }),
                patch("collector.OpenAIAnalyzer", return_value=analyzer),
            ):
                validate_args(args)
                summary, exit_code = run(args, self._config(root))

            self.assertEqual(exit_code, 1)
            self.assertEqual(analyzer.analyze.call_count, 1)
            self.assertEqual(summary["model_failures"], 1)
            self.assertEqual(summary["failed_attempt_usage_unknown"], 1)
            self.assertTrue(any(
                item.get("decision") == "DEFER" and "usage" in item.get("reason", "")
                for item in summary["outcomes"]
            ))

    def test_successful_response_without_usage_stops_later_ai_work(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                "--enable-ai", "--max-candidates", "4", "--max-deep-calls", "3",
            ])
            analyzer = Mock()
            analyzer.analyze.return_value = AnalysisResult(
                payload=make_fixture_analysis(), usage_available=False,
                request_id="resp_fixture", response_model="fixture-model",
            )
            with (
                patch.dict(os.environ, {
                    "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                    "OPENAI_API_KEY": "fixture-key-never-used",
                }),
                patch("collector.OpenAIAnalyzer", return_value=analyzer),
            ):
                validate_args(args)
                summary, exit_code = run(args, self._config(root))

            self.assertEqual(exit_code, 0)
            self.assertEqual(analyzer.analyze.call_count, 1)
            self.assertEqual(summary["model_successes"], 1)
            self.assertEqual(summary["model_failures"], 0)
            self.assertEqual(summary["model_usage_missing"], 1)
            self.assertTrue(any(
                item.get("decision") == "DEFER" and "usage" in item.get("reason", "")
                for item in summary["outcomes"]
            ))

    def test_budget_blocks_first_request_when_worst_case_cannot_fit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                "--enable-ai", "--max-candidates", "1",
            ])
            analyzer = Mock()
            analyzer.analyze.return_value = AnalysisResult(payload=make_fixture_analysis())
            config = replace(self._config(root), max_ai_budget_usd=0.000001)
            with (
                patch.dict(os.environ, {
                    "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                    "OPENAI_API_KEY": "fixture-key-never-used",
                }),
                patch("collector.OpenAIAnalyzer", return_value=analyzer),
            ):
                validate_args(args)
                summary, exit_code = run(args, config)

            self.assertEqual(exit_code, 0)
            self.assertEqual(analyzer.analyze.call_count, 0)
            self.assertEqual(summary["model_attempts"], 0)
            self.assertTrue(any(
                item.get("decision") == "DEFER" and "budget" in item.get("reason", "")
                for item in summary["outcomes"]
            ))

    def test_budget_exact_boundary_allows_request(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                "--enable-ai", "--max-candidates", "1",
            ])
            base_config = self._config(root)
            maximum = maximum_analysis_request_cost_usd(base_config.model)
            config = replace(base_config, max_ai_budget_usd=maximum)
            analyzer = Mock()
            analyzer.analyze.return_value = AnalysisResult(payload=make_fixture_analysis())
            with (
                patch.dict(os.environ, {
                    "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                    "OPENAI_API_KEY": "fixture-key-never-used",
                }),
                patch("collector.OpenAIAnalyzer", return_value=analyzer),
            ):
                validate_args(args)
                summary, exit_code = run(args, config)

            self.assertEqual(exit_code, 0)
            self.assertEqual(analyzer.analyze.call_count, 1)
            self.assertEqual(summary["model_attempts"], 1)
            self.assertEqual(summary["maximum_ai_request_cost_usd"], maximum)

    def test_budget_exact_boundary_after_known_spend_allows_next_request(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                "--enable-ai", "--max-candidates", "4", "--max-deep-calls", "3",
            ])
            base_config = self._config(root)
            maximum = maximum_analysis_request_cost_usd(base_config.model)
            known_cost = 0.001
            config = replace(base_config, max_ai_budget_usd=maximum + known_cost)
            analyzer = Mock()
            analyzer.analyze.return_value = AnalysisResult(
                payload=make_fixture_analysis(), approximate_cost_usd=known_cost,
            )
            with (
                patch.dict(os.environ, {
                    "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                    "OPENAI_API_KEY": "fixture-key-never-used",
                }),
                patch("collector.OpenAIAnalyzer", return_value=analyzer),
            ):
                validate_args(args)
                summary, exit_code = run(args, config)

            self.assertEqual(exit_code, 0)
            self.assertEqual(analyzer.analyze.call_count, 2)
            self.assertEqual(summary["model_attempts"], 2)

    def test_budget_well_below_ceiling_records_actual_usage(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                "--enable-ai", "--max-candidates", "1",
            ])
            base_config = self._config(root)
            maximum = maximum_analysis_request_cost_usd(base_config.model)
            config = replace(base_config, max_ai_budget_usd=maximum * 2)
            analyzer = Mock()
            analyzer.analyze.return_value = AnalysisResult(
                payload=make_fixture_analysis(), input_tokens=1000, output_tokens=500,
                approximate_cost_usd=0.00123,
            )
            with (
                patch.dict(os.environ, {
                    "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                    "OPENAI_API_KEY": "fixture-key-never-used",
                }),
                patch("collector.OpenAIAnalyzer", return_value=analyzer),
            ):
                validate_args(args)
                summary, exit_code = run(args, config)

            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["ai_input_tokens"], 1000)
            self.assertEqual(summary["ai_output_tokens"], 500)
            self.assertEqual(summary["approximate_ai_cost_usd"], 0.00123)

    def test_budget_allows_some_candidates_then_defers_later_work(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                "--enable-ai", "--max-candidates", "4", "--max-deep-calls", "3",
            ])
            base_config = self._config(root)
            maximum = maximum_analysis_request_cost_usd(base_config.model)
            config = replace(base_config, max_ai_budget_usd=maximum)
            analyzer = Mock()
            analyzer.analyze.return_value = AnalysisResult(
                payload=make_fixture_analysis(), approximate_cost_usd=0.001,
            )
            with (
                patch.dict(os.environ, {
                    "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                    "OPENAI_API_KEY": "fixture-key-never-used",
                }),
                patch("collector.OpenAIAnalyzer", return_value=analyzer),
            ):
                validate_args(args)
                summary, exit_code = run(args, config)

            self.assertEqual(exit_code, 0)
            self.assertEqual(analyzer.analyze.call_count, 1)
            self.assertEqual(summary["model_attempts"], 1)
            self.assertTrue(any(
                item.get("decision") == "DEFER" and "budget" in item.get("reason", "")
                for item in summary["outcomes"]
            ))

    def test_downstream_failure_spend_blocks_next_request(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--fixture", "offline-demo",
                "--enable-ai", "--max-candidates", "4", "--max-deep-calls", "3",
            ])
            base_config = self._config(root)
            maximum = maximum_analysis_request_cost_usd(base_config.model)
            config = replace(base_config, max_ai_budget_usd=maximum)
            analyzer = Mock()
            analyzer.analyze.return_value = AnalysisResult(
                payload=make_fixture_analysis(), input_tokens=1000, output_tokens=500,
                approximate_cost_usd=0.00123,
            )
            with (
                patch.dict(os.environ, {
                    "MARKET_COLLECTOR_ENABLE_WRITES": "0",
                    "OPENAI_API_KEY": "fixture-key-never-used",
                }),
                patch("collector.OpenAIAnalyzer", return_value=analyzer),
                patch("collector.render_article", side_effect=RuntimeError("renderer failed")),
            ):
                validate_args(args)
                summary, exit_code = run(args, config)

            self.assertEqual(exit_code, 1)
            self.assertEqual(analyzer.analyze.call_count, 1)
            self.assertEqual(summary["ai_input_tokens"], 1000)
            self.assertEqual(summary["ai_output_tokens"], 500)
            self.assertEqual(summary["approximate_ai_cost_usd"], 0.00123)
            self.assertEqual(len(summary["processing_failures"]), 1)
            self.assertTrue(any(
                item.get("decision") == "DEFER" and "budget" in item.get("reason", "")
                for item in summary["outcomes"]
            ))

    def test_partial_source_failure_has_distinct_status_and_exit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--sources", "good,bad",
                "--max-candidates", "2",
            ])
            sources = [
                _FakeSource("good", _source_candidates("Good", 1)),
                _FakeSource("bad", error=RuntimeError("fixture failure")),
            ]
            with patch("collector.build_sources", return_value=sources):
                summary, exit_code = run(args, self._config(root))

            self.assertEqual(summary["run_status"], "partial")
            self.assertEqual(exit_code, 3)
            self.assertEqual(len(summary["source_errors"]), 1)

    def test_all_requested_sources_succeed_with_success_status(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--sources", "a,b",
                "--max-candidates", "2",
            ])
            sources = [
                _FakeSource("a", _source_candidates("A", 1)),
                _FakeSource("b", _source_candidates("B", 1)),
            ]
            with patch("collector.build_sources", return_value=sources):
                summary, exit_code = run(args, self._config(root))
            self.assertEqual((summary["run_status"], exit_code), ("success", 0))

    def test_all_requested_sources_fail_with_failed_status(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--sources", "a,b",
                "--max-candidates", "2",
            ])
            sources = [
                _FakeSource("a", error=RuntimeError("fixture a failure")),
                _FakeSource("b", error=RuntimeError("fixture b failure")),
            ]
            with patch("collector.build_sources", return_value=sources):
                summary, exit_code = run(args, self._config(root))
            self.assertEqual((summary["run_status"], exit_code), ("failed", 1))
            self.assertEqual(len(summary["source_errors"]), 2)

    def test_empty_source_and_successful_source_are_not_partial(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--sources", "empty,good",
                "--max-candidates", "2",
            ])
            sources = [
                _FakeSource("empty"),
                _FakeSource("good", _source_candidates("Good", 1)),
            ]
            with patch("collector.build_sources", return_value=sources):
                summary, exit_code = run(args, self._config(root))
            self.assertEqual((summary["run_status"], exit_code), ("success", 0))
            self.assertEqual(summary["source_errors"], [])

    def test_source_warning_does_not_create_partial_status(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--sources", "warning",
                "--max-candidates", "1",
            ])
            sources = [
                _FakeSource(
                    "warning", _source_candidates("Warning", 1),
                    warnings=["fixture warning"],
                )
            ]
            with patch("collector.build_sources", return_value=sources):
                summary, exit_code = run(args, self._config(root))
            self.assertEqual((summary["run_status"], exit_code), ("success", 0))
            self.assertEqual(len(summary["source_warnings"]), 1)

    def test_discovery_summary_separates_raw_and_admitted_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--sources", "a,b,c",
                "--max-candidates", "10",
            ])
            sources = [
                _FakeSource("a", _source_candidates("A", 10)),
                _FakeSource("b", _source_candidates("B", 10)),
                _FakeSource("c", _source_candidates("C", 10)),
            ]
            with patch("collector.build_sources", return_value=sources):
                summary, exit_code = run(args, self._config(root))

            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["candidates_discovered_raw"], 30)
            self.assertEqual(summary["candidates_admitted"], 10)
            self.assertEqual(summary["source_discovery_distribution"], {"a": 10, "b": 10, "c": 10})
            self.assertEqual(summary["source_admitted_distribution"], {"a": 4, "b": 3, "c": 3})
            self.assertEqual(summary["candidates_discovered"], 10)
            self.assertEqual(summary["source_distribution"], {"a": 4, "b": 3, "c": 3})

    def test_discovery_accounting_includes_successful_empty_source(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--sources", "empty,full",
                "--max-candidates", "4",
            ])
            sources = [
                _FakeSource("empty"),
                _FakeSource("full", _source_candidates("Full", 2)),
            ]
            with patch("collector.build_sources", return_value=sources):
                summary, _ = run(args, self._config(root))
            self.assertEqual(summary["candidates_discovered_raw"], 2)
            self.assertEqual(summary["candidates_admitted"], 2)
            self.assertEqual(summary["source_discovery_distribution"], {"empty": 0, "full": 2})
            self.assertEqual(summary["source_admitted_distribution"], {"full": 2})

    def test_discovery_accounting_excludes_failed_source_counts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--sources", "failed,good",
                "--max-candidates", "4",
            ])
            sources = [
                _FakeSource("failed", error=RuntimeError("fixture failure")),
                _FakeSource("good", _source_candidates("Good", 2)),
            ]
            with patch("collector.build_sources", return_value=sources):
                summary, exit_code = run(args, self._config(root))
            self.assertEqual((summary["run_status"], exit_code), ("partial", 3))
            self.assertEqual(summary["source_discovery_distribution"], {"good": 2})
            self.assertEqual(summary["candidates_discovered_raw"], 2)
            self.assertEqual(len(summary["source_errors"]), 1)

    def test_single_source_raw_and_admitted_counts_match_under_cap(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = parse_args([
                "--ticker", "NVDA", "--dry-run", "--sources", "only",
                "--max-candidates", "4",
            ])
            sources = [_FakeSource("only", _source_candidates("Only", 3))]
            with patch("collector.build_sources", return_value=sources):
                summary, exit_code = run(args, self._config(root))
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["candidates_discovered_raw"], 3)
            self.assertEqual(summary["candidates_admitted"], 3)
            self.assertEqual(summary["source_discovery_distribution"], {"only": 3})
            self.assertEqual(summary["source_admitted_distribution"], {"only": 3})


if __name__ == "__main__":
    unittest.main()
