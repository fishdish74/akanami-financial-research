#!/usr/bin/env python3
"""Manually triggered, append-only V1 market-research collector."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from analysis import (
    OfflineSmokeAnalyzer, OpenAIAnalyzer, maximum_analysis_request_cost_usd,
)
from config import CollectorConfig, api_key_is_configured, source_configs, writes_enabled
from dedup import ExistingResearchIndex, candidate_id, classify_duplicate
from entities import Entity, EntityRegistry
from filters import FilterResult, cheap_filter
from market_analytics import ANALYTICS_COMMANDS
from obsidian import write_new_note_atomic
from renderer import deterministic_filename, render_article, validate_rendered_markdown
from schemas import AnalysisPayload, Candidate, ProcessingMetadata
from security import safe_error
from sources import GoogleNewsRssSource, ManualJsonSource, SecSubmissionsSource


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
HIGH_SIGNAL_TERMS = {
    "earnings", "guidance", "acquisition", "merger", "filing", "investigation",
    "lawsuit", "financing", "bankruptcy", "restructuring", "contract", "approval",
}
PARTIAL_SOURCE_EXIT_CODE = 3


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Analytics/data commands: demo, rank, compare, validate, dcf, "
            "watchlist-summary, database-status, data-status, sec-sync, "
            "fundamentals, history, validate-data"
        ),
    )
    parser.add_argument("--ticker", required=True, help="Target ticker")
    parser.add_argument("--company", help="Canonical company name when not in registry")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Never write research files")
    mode.add_argument("--write", action="store_true", help="Explicit append-only live write")
    analysis = parser.add_mutually_exclusive_group()
    analysis.add_argument("--enable-ai", action="store_true", help="Call the configured OpenAI model")
    analysis.add_argument(
        "--offline-analysis", action="store_true",
        help="Deterministic schema/render smoke test; never deep research",
    )
    parser.add_argument("--sources", default="google-news", help="Comma-separated: google-news,sec,manual")
    parser.add_argument("--candidate-file", type=Path, help="JSON input for the manual source")
    parser.add_argument(
        "--fixture", metavar="NAME",
        help="Bundled manual fixture name (for example: offline-demo)",
    )
    parser.add_argument("--max-candidates", type=positive_int, default=10)
    parser.add_argument("--max-deep-calls", type=positive_int)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.fixture and args.candidate_file:
        raise ValueError("use either --fixture or --candidate-file, not both")
    if args.fixture:
        if Path(args.fixture).name != args.fixture:
            raise ValueError("fixture names cannot contain path components")
        args.candidate_file = FIXTURE_DIR / f"{args.fixture}.json"
        args.sources = "manual"
    if args.write and not args.enable_ai:
        raise ValueError("--write requires --enable-ai; offline smoke output cannot be saved")
    if args.write and not writes_enabled():
        raise PermissionError(
            "Real writes are disabled. Keep MARKET_COLLECTOR_ENABLE_WRITES=0 for dry runs."
        )
    if args.enable_ai and not api_key_is_configured():
        raise ValueError("OPENAI_API_KEY is required with --enable-ai")


def resolve_target(args: argparse.Namespace, registry: EntityRegistry) -> Entity:
    ticker_entity = registry.resolve(args.ticker)
    company_entity = registry.resolve(args.company) if args.company else None
    if args.company:
        if ticker_entity is not None and company_entity is None:
            raise ValueError(
                f"company identifier does not match registered ticker {ticker_entity.ticker}"
            )
        if ticker_entity is None and company_entity is not None:
            raise ValueError(
                f"ticker identifier does not match registered company {company_entity.canonical_name}"
            )
        if ticker_entity is not None and company_entity is not None and ticker_entity.ticker != company_entity.ticker:
            raise ValueError(
                f"contradictory target identifiers: {ticker_entity.ticker} and {company_entity.ticker}"
            )
    entity = ticker_entity or company_entity
    if entity is not None:
        return entity
    return Entity(
        ticker=args.ticker.upper(),
        canonical_name=args.company or args.ticker.upper(),
        aliases=[],
        alternate_listings=[],
    )


def build_sources(args: argparse.Namespace):
    configs = source_configs()
    requested = [value.strip() for value in args.sources.split(",") if value.strip()]
    if not requested:
        raise ValueError("at least one source is required")
    if len(requested) != len(set(requested)):
        raise ValueError("source names must not be repeated")
    if len(requested) > 3:
        raise ValueError("V1 supports at most three source types")
    result = []
    for name in requested:
        if name == "google-news":
            result.append(GoogleNewsRssSource(configs[name]))
        elif name == "sec":
            result.append(SecSubmissionsSource(configs[name]))
        elif name == "manual":
            if not args.candidate_file:
                raise ValueError("--candidate-file or --fixture is required for the manual source")
            result.append(ManualJsonSource(args.candidate_file))
        else:
            raise ValueError(f"unsupported source: {name}")
    return result


def provisional_priority(candidate: Candidate, result: FilterResult) -> tuple[int, int, float]:
    title_tokens = set(candidate.title.casefold().split())
    signal = 1 if title_tokens & HIGH_SIGNAL_TERMS else 0
    published = candidate.published.timestamp() if candidate.published else 0.0
    return (signal + result.priority_adjustment, 5 - candidate.source_tier, published)


def fair_merge_candidate_records(
    discovered: list[tuple[str, list[Candidate]]], limit: int
) -> list[tuple[str, Candidate]]:
    """Round-robin source results so configured source order cannot starve later sources."""

    iterators = [(name, iter(candidates)) for name, candidates in discovered]
    merged: list[tuple[str, Candidate]] = []
    while len(merged) < limit:
        added = False
        for name, iterator in iterators:
            if len(merged) >= limit:
                break
            try:
                merged.append((name, next(iterator)))
                added = True
            except StopIteration:
                continue
        if not added:
            break
    return merged


def _empty_summary(args: argparse.Namespace, config: CollectorConfig, target: Entity) -> dict[str, object]:
    return {
        "run_id": uuid.uuid4().hex,
        "mode": "dry-run" if args.dry_run else "write",
        "analysis_mode": "live_ai" if args.enable_ai else "offline_smoke" if args.offline_analysis else "none",
        "ticker": target.ticker,
        "company": target.canonical_name,
        "writes_enabled": writes_enabled(),
        "model": config.model.model if args.enable_ai else None,
        "prompt_version": config.model.prompt_version,
        "pipeline_version": config.model.pipeline_version,
        "run_status": "success",
        "source_distribution": {},
        "source_discovery_distribution": {},
        "source_admitted_distribution": {},
        "source_warnings": [],
        "source_errors": [],
        "candidates_discovered": 0,
        "candidates_discovered_raw": 0,
        "candidates_admitted": 0,
        "candidates_rejected_before_analysis": 0,
        "rejection_reasons": {},
        "duplicate_classifications": {},
        "selected_for_analysis": 0,
        "would_reach_openai": 0,
        "model_requests": 0,
        "model_attempts": 0,
        "model_responses": 0,
        "model_successes": 0,
        "model_failures": 0,
        "analysis_inputs_truncated": 0,
        "model_usage_missing": 0,
        "failed_attempt_usage_unknown": 0,
        "ai_budget_mode": "conservative_hard_pre_request_ceiling",
        "approximate_ai_budget_usd": config.max_ai_budget_usd,
        "maximum_ai_request_cost_usd": (
            maximum_analysis_request_cost_usd(config.model) if args.enable_ai else 0.0
        ),
        "processing_failures": [],
        "outcomes": [],
        "files_would_create": [],
        "files_created": [],
        "files_written": 0,
        "ai_input_tokens": 0,
        "ai_output_tokens": 0,
        "approximate_ai_cost_usd": 0.0,
    }


def run(args: argparse.Namespace, config: CollectorConfig) -> tuple[dict[str, object], int]:
    started_at = time.monotonic()
    registry = EntityRegistry.load(config.registry_path)
    target = resolve_target(args, registry)
    summary = _empty_summary(args, config, target)
    sources = build_sources(args)

    discovered_by_source: list[tuple[str, list[Candidate]]] = []
    for source in sources:
        try:
            discovered = list(source.discover(target, args.max_candidates))
            discovered_by_source.append((source.name, discovered))
            for warning in getattr(source, "last_warnings", []):
                summary["source_warnings"].append({"source": source.name, "message": warning})
        except Exception as exc:
            summary["source_errors"].append({"source": source.name, **safe_error(exc, category="source")})
    candidate_records = fair_merge_candidate_records(discovered_by_source, args.max_candidates)
    candidates = [candidate for _, candidate in candidate_records]
    raw_source_counts = {
        source_name: len(discovered)
        for source_name, discovered in discovered_by_source
    }
    source_counts = Counter(source_name for source_name, _ in candidate_records)
    summary["source_discovery_distribution"] = dict(sorted(raw_source_counts.items()))
    summary["source_admitted_distribution"] = dict(sorted(source_counts.items()))
    summary["source_distribution"] = dict(sorted(source_counts.items()))
    summary["candidates_discovered_raw"] = sum(raw_source_counts.values())
    summary["candidates_admitted"] = len(candidates)
    summary["candidates_discovered"] = len(candidates)

    index = ExistingResearchIndex.scan(config.vault_dir)
    run_candidates: list[Candidate] = []
    selected: list[tuple[Candidate, str, FilterResult]] = []
    rejection_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()

    for candidate in candidates:
        result = cheap_filter(candidate, target, index, config.min_content_chars)
        if not result.accepted:
            label = result.rejection_reason or "other"
            rejection_counts[label] += 1
            summary["candidates_rejected_before_analysis"] += 1
            summary["outcomes"].append(
                {
                    "candidate_id": result.candidate_id,
                    "title": candidate.title,
                    "source": candidate.source,
                    "decision": "REJECT",
                    "reason": result.reason,
                    "rejection_reason": result.rejection_reason,
                }
            )
            if result.rejection_reason == "duplicate":
                duplicate_counts["EXACT_DUPLICATE"] += 1
            continue

        duplicate_status = classify_duplicate(candidate, index, run_candidates)
        duplicate_counts[duplicate_status] += 1
        if duplicate_status in {"EXACT_DUPLICATE", "SYNDICATED_DUPLICATE"}:
            rejection_counts["duplicate"] += 1
            summary["candidates_rejected_before_analysis"] += 1
            decision = "REJECT"
        elif duplicate_status in {"SAME_EVENT_NO_NEW_INFO", "SAME_EVENT_CONFIRMATION"}:
            decision = "LINK ONLY"
        else:
            selected.append((candidate, duplicate_status, result))
            run_candidates.append(candidate)
            continue
        summary["outcomes"].append(
            {
                "candidate_id": candidate_id(candidate),
                "title": candidate.title,
                "source": candidate.source,
                "decision": decision,
                "reason": duplicate_status,
            }
        )
        run_candidates.append(candidate)

    selected.sort(key=lambda item: provisional_priority(item[0], item[2]), reverse=True)
    max_calls = min(args.max_deep_calls or config.max_deep_calls, config.max_deep_calls)
    deferred = selected[max_calls:]
    selected = selected[:max_calls]
    for candidate, status, filter_result in deferred:
        summary["outcomes"].append(
            {
                "candidate_id": candidate_id(candidate),
                "title": candidate.title,
                "source": candidate.source,
                "decision": "DEFER",
                "reason": f"deep-analysis cap reached; {status}",
                "filter_reason": filter_result.reason,
            }
        )
    summary["rejection_reasons"] = dict(sorted(rejection_counts.items()))
    summary["duplicate_classifications"] = dict(sorted(duplicate_counts.items()))
    summary["selected_for_analysis"] = len(selected)
    summary["would_reach_openai"] = len(selected)

    analyzer = OpenAIAnalyzer(config.model) if args.enable_ai else OfflineSmokeAnalyzer() if args.offline_analysis else None
    if analyzer is None:
        for candidate, duplicate_status, filter_result in selected:
            summary["outcomes"].append(
                {
                    "candidate_id": candidate_id(candidate),
                    "title": candidate.title,
                    "source": candidate.source,
                    "decision": "DEFER",
                    "reason": f"{duplicate_status}; analysis not enabled",
                    "filter_reason": filter_result.reason,
                }
            )
    else:
        total_cost = 0.0
        maximum_request_cost = maximum_analysis_request_cost_usd(config.model)
        stop_ai_for_missing_usage = False
        metadata = ProcessingMetadata(
            model_id=config.model.model if args.enable_ai else "offline-smoke",
            prompt_version=config.model.prompt_version,
            pipeline_version=config.model.pipeline_version,
            analysis_mode="live_ai" if args.enable_ai else "offline_smoke",
            analysis_completed=args.enable_ai,
        )
        for candidate, duplicate_status, filter_result in selected:
            if args.enable_ai and stop_ai_for_missing_usage:
                summary["outcomes"].append(
                    {
                        "candidate_id": candidate_id(candidate),
                        "title": candidate.title,
                        "source": candidate.source,
                        "decision": "DEFER",
                        "reason": "prior model response omitted usage; budget accounting is incomplete",
                        "filter_reason": filter_result.reason,
                    }
                )
                continue
            if (
                args.enable_ai
                and total_cost + maximum_request_cost > config.max_ai_budget_usd
            ):
                summary["outcomes"].append(
                    {
                        "candidate_id": candidate_id(candidate),
                        "title": candidate.title,
                        "source": candidate.source,
                        "decision": "DEFER",
                        "reason": (
                            "conservative hard AI budget cannot accommodate the next "
                            "maximum-size request"
                        ),
                        "filter_reason": filter_result.reason,
                    }
                )
                continue
            model_response_succeeded = False
            try:
                if args.enable_ai:
                    summary["model_attempts"] += 1
                result = analyzer.analyze(candidate)
                if args.enable_ai:
                    model_response_succeeded = True
                    summary["model_successes"] += 1
                    summary["model_requests"] += 1
                    summary["model_responses"] += 1
                summary["ai_input_tokens"] += result.input_tokens
                summary["ai_output_tokens"] += result.output_tokens
                total_cost += result.approximate_cost_usd
                if result.input_truncated:
                    summary["analysis_inputs_truncated"] += 1
                if args.enable_ai and not result.usage_available:
                    summary["model_usage_missing"] += 1
                    stop_ai_for_missing_usage = True
                validated = AnalysisPayload.model_validate(result.payload.model_dump())
                markdown = render_article(validated, candidate, duplicate_status, metadata)
                validate_rendered_markdown(markdown)
                filename = deterministic_filename(validated, candidate)
                destination = config.article_dir / filename
                decision = (
                    "SAVE + REVIEW" if validated.needs_review else "SAVE"
                ) if args.enable_ai else "SMOKE TEST ONLY"
                summary["outcomes"].append(
                    {
                        "candidate_id": candidate_id(candidate),
                        "title": candidate.title,
                        "source": candidate.source,
                        "decision": decision,
                        "reason": duplicate_status,
                        "filter_reason": filter_result.reason,
                        "proposed_filename": str(destination.relative_to(config.vault_dir)),
                        "analysis": validated.model_dump(mode="json"),
                        "analysis_completed": args.enable_ai,
                        "eligible_for_research": args.enable_ai,
                        "input_truncated": result.input_truncated,
                    }
                )
                if args.enable_ai:
                    summary["files_would_create"].append(str(destination.relative_to(config.vault_dir)))
                if args.write:
                    write_new_note_atomic(
                        destination, markdown, allowed_root=config.vault_dir
                    )
                    summary["files_created"].append(str(destination.relative_to(config.vault_dir)))
                    summary["files_written"] += 1
            except Exception as exc:
                if args.enable_ai and not model_response_succeeded:
                    summary["model_failures"] += 1
                    if getattr(exc, "request_id", None) or getattr(exc, "response_model", None):
                        summary["model_requests"] += 1
                        summary["model_responses"] += 1
                    if bool(getattr(exc, "usage_available", False)):
                        summary["ai_input_tokens"] += int(getattr(exc, "input_tokens", 0) or 0)
                        summary["ai_output_tokens"] += int(getattr(exc, "output_tokens", 0) or 0)
                        total_cost += float(getattr(exc, "approximate_cost_usd", 0.0) or 0.0)
                    else:
                        summary["failed_attempt_usage_unknown"] += 1
                        stop_ai_for_missing_usage = True
                failure = {
                    "candidate_id": candidate_id(candidate),
                    "title": candidate.title,
                    **safe_error(exc, category="analysis-render-write"),
                }
                for field_name in (
                    "request_id", "response_model", "latency_ms", "refusal",
                    "input_tokens", "output_tokens", "total_tokens",
                    "approximate_cost_usd", "usage_available",
                ):
                    if hasattr(exc, field_name):
                        failure[field_name] = getattr(exc, field_name)
                summary["processing_failures"].append(failure)
        summary["approximate_ai_cost_usd"] = round(total_cost, 6)

    summary["duplicates"] = sum(
        count
        for status, count in duplicate_counts.items()
        if status in {"EXACT_DUPLICATE", "SYNDICATED_DUPLICATE"}
    )
    summary["elapsed_ms"] = round((time.monotonic() - started_at) * 1000, 3)
    if summary["processing_failures"] or (summary["source_errors"] and not candidates):
        summary["run_status"] = "failed"
        exit_code = 1
    elif summary["source_errors"]:
        summary["run_status"] = "partial"
        exit_code = PARTIAL_SOURCE_EXIT_CODE
    else:
        summary["run_status"] = "success"
        exit_code = 0
    return summary, exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    actual_argv = list(argv) if argv is not None else sys.argv[1:]
    if actual_argv and actual_argv[0] in ANALYTICS_COMMANDS:
        from market_analytics.cli import main as analytics_main

        return analytics_main(actual_argv)
    try:
        args = parse_args(actual_argv)
        validate_args(args)
        summary, exit_code = run(args, CollectorConfig())
    except (ValueError, PermissionError) as exc:
        print(json.dumps({"error": safe_error(exc, category="configuration")}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
