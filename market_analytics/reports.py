"""Stable, Obsidian-friendly Markdown renderers for analytics results."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import statistics

import yaml

from .dcf import DCFResult, SensitivityResult
from .models import Catalyst, MetricResult, QualityIssue, RankingResult, Risk
from .pipeline import AnalyticsRun


def _number(value: Decimal | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    quantum = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)
    return f"{value.quantize(quantum, rounding=ROUND_HALF_UP):,}"


def _metric(result: MetricResult) -> str:
    if result.value is None:
        return f"— ({result.status})"
    if result.unit == "decimal_ratio":
        return f"{_number(result.value * 100, 1)}%"
    if result.unit == "percent":
        return f"{_number(result.value, 1)}%"
    return f"{_number(result.value)} {result.unit}".strip()


def company_snapshot(run: AnalyticsRun, ticker: str, horizon: str = "one_year") -> str:
    ticker = ticker.upper()
    company = run.companies[ticker]
    ranking = next(item for item in run.rankings[horizon] if item.ticker == ticker)
    lines = [
        f"# {ticker} — {company.company}", "",
        f"> Fictional/offline status: {run.fixture_notice}", "",
        "## Snapshot", "",
        f"- Sector: {company.sector}", f"- Industry: {company.industry}",
        f"- Analysis date: {run.analysis_date}", f"- Confidence: {ranking.confidence}", "",
        "## Component Scores", "",
        "| Component | Score |", "|---|---:|",
    ]
    for name, value in ranking.components.model_dump().items():
        rendered = "—" if value is None else _number(Decimal(value), 0)
        lines.append(f"| {name.replace('_', ' ').title()} | {rendered} |")
    lines.extend(["", "## Key Metrics", "", "| Metric | Value | Status |", "|---|---:|---|"])
    for name in sorted(run.metrics[ticker]):
        result = run.metrics[ticker][name]
        lines.append(f"| {name} | {_metric(result)} | {result.status} |")
    lines.extend(["", "## Unavailable Ranking Inputs", ""])
    lines.extend(
        f"- {name}: {reason}"
        for name, reason in ranking.unavailable_inputs.items()
    )
    if not ranking.unavailable_inputs:
        lines.append("- None")
    lines.extend(["", catalyst_summary(company.catalysts), "", risk_summary(company.risks), "", data_quality_report(run.quality_issues[ticker])])
    return "\n".join(lines).rstrip() + "\n"


def ranking_report(results: list[RankingResult], title: str) -> str:
    lines = [f"# {title}", "", "Opportunity score supports research comparison; it is not a price forecast or recommendation.", "", "| Rank | Ticker | Opportunity | Research Priority | Confidence | Status |", "|---:|---|---:|---|---|---|"]
    for rank, result in enumerate(results, 1):
        lines.append(f"| {rank} | {result.ticker} | {_number(result.opportunity_score, 0)} | {result.research_priority} ({_number(result.research_priority_score, 0)}) | {result.confidence} | {result.ranking_status} |")
    lines.extend(["", "## Deterministic Explanations", ""])
    for result in results:
        lines.extend([f"### {result.ticker}", "", "Positive contributors:"])
        lines.extend(f"- {value}" for value in result.positive_contributors or ["None above the configured threshold."])
        lines.append("\nNegative contributors:")
        lines.extend(f"- {value}" for value in result.negative_contributors or ["None below the configured threshold."])
        if result.unavailable_inputs:
            lines.append("\nUnavailable inputs:")
            lines.extend(
                f"- {name}: {reason}"
                for name, reason in result.unavailable_inputs.items()
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def comparison_report(run: AnalyticsRun, first: str, second: str, horizon: str = "one_year") -> str:
    tickers = [first.upper(), second.upper()]
    for ticker in tickers:
        if ticker not in run.companies:
            raise ValueError(f"ticker not present in analytics run: {ticker}")
    lines = [
        f"# Peer Comparison — {tickers[0]} vs {tickers[1]}", "",
        "This report shows differences and does not declare a winner.", "",
        "## Component Scores", "",
        f"| Component | {tickers[0]} | {tickers[1]} | Difference |", "|---|---:|---:|---:|",
    ]
    rankings = {
        ticker: next(item for item in run.rankings[horizon] if item.ticker == ticker)
        for ticker in tickers
    }
    for name in rankings[tickers[0]].components.model_dump():
        first_raw = rankings[tickers[0]].components.model_dump()[name]
        second_raw = rankings[tickers[1]].components.model_dump()[name]
        first_value = Decimal(first_raw) if first_raw is not None else None
        second_value = Decimal(second_raw) if second_raw is not None else None
        left = "—" if first_value is None else _number(first_value, 0)
        right = "—" if second_value is None else _number(second_value, 0)
        difference = (
            "—" if first_value is None or second_value is None
            else _number(first_value - second_value, 0)
        )
        lines.append(
            f"| {name.replace('_', ' ').title()} | {left} | {right} | {difference} |"
        )
    lines.extend(["", "## Financial Metrics", "", f"| Metric | {tickers[0]} | {tickers[1]} |", "|---|---:|---:|"])
    names = sorted(set(run.metrics[tickers[0]]) | set(run.metrics[tickers[1]]))
    for name in names:
        left = run.metrics[tickers[0]].get(name)
        right = run.metrics[tickers[1]].get(name)
        lines.append(f"| {name} | {_metric(left) if left else '—'} | {_metric(right) if right else '—'} |")
    lines.extend(["", "## Confidence and Freshness", ""])
    for ticker in tickers:
        lines.append(f"- {ticker}: confidence {rankings[ticker].confidence}; {len(run.quality_issues[ticker])} data-quality issue(s).")
        for name, reason in rankings[ticker].unavailable_inputs.items():
            lines.append(f"  - {name}: {reason}")
    return "\n".join(lines).rstrip() + "\n"


def dcf_report(results: list[DCFResult]) -> str:
    lines = ["# DCF Scenario Summary", "", "DCF values are assumption-sensitive scenarios, not a uniquely correct value.", "", "| Scenario | Complete | Enterprise Value | Equity Value | Value / Share |", "|---|---|---:|---:|---:|"]
    for result in results:
        lines.append(f"| {result.scenario} | {'yes' if result.complete else 'no'} | {_number(result.enterprise_value)} | {_number(result.equity_value)} | {_number(result.value_per_share)} |")
    for result in results:
        if result.warnings:
            lines.extend(["", f"## {result.scenario} Warnings", *[f"- {warning}" for warning in result.warnings]])
    return "\n".join(lines).rstrip() + "\n"


def sensitivity_table(result: SensitivityResult) -> str:
    headers = " | ".join(f"{_number(rate * 100, 1)}%" for rate in result.discount_rates)
    lines = ["## DCF Sensitivity", "", "Rows are terminal growth; columns are discount rates. No center cell is designated as correct.", "", f"| Terminal Growth \\ Discount Rate | {headers} |", "|---|" + "---:|" * len(result.discount_rates)]
    for growth, row in zip(result.terminal_growth_rates, result.values):
        lines.append(f"| {_number(growth * 100, 1)}% | " + " | ".join(_number(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def catalyst_summary(catalysts: list[Catalyst]) -> str:
    lines = ["## Catalysts", "", "| Date | Type | Catalyst | Importance | Confidence | Status |", "|---|---|---|---|---|---|"]
    for item in sorted(catalysts, key=lambda value: (value.expected_date or value.source_date, value.name)):
        lines.append(f"| {item.expected_date or 'unknown'} | {item.catalyst_type} | {item.name} | {item.importance} | {item.confidence} | {item.status} |")
    if not catalysts:
        lines.append("| — | — | No structured catalysts | — | — | — |")
    return "\n".join(lines)


def risk_summary(risks: list[Risk]) -> str:
    lines = ["## Risks", "", "| Category | Severity | Confidence | Evidence | Freshness |", "|---|---|---|---|---|"]
    for item in sorted(risks, key=lambda value: (value.category, value.severity, value.evidence)):
        lines.append(f"| {item.category} | {item.severity} | {item.confidence} | {item.evidence} | {item.freshness} |")
    if not risks:
        lines.append("| — | — | — | No structured risks | — |")
    return "\n".join(lines)


def data_quality_report(issues: list[QualityIssue]) -> str:
    lines = ["## Data Quality", "", "| Severity | Code | Message | Fields |", "|---|---|---|---|"]
    for issue in issues:
        lines.append(f"| {issue.severity} | {issue.code} | {issue.message} | {', '.join(issue.fields)} |")
    if not issues:
        lines.append("| INFO | none | No deterministic issues detected | — |")
    return "\n".join(lines)


def watchlist_summary(run: AnalyticsRun, horizon: str = "one_year") -> str:
    results = run.rankings[horizon]
    priorities = run.research_priorities[horizon]
    sectors = Counter(company.sector for company in run.companies.values())
    industries = Counter(company.industry for company in run.companies.values())
    upcoming = {
        ticker: sum(item.status in {"upcoming", "uncertain"} for item in company.catalysts)
        for ticker, company in run.companies.items()
    }
    missing = {
        item.ticker: len(item.missing_inputs)
        for item in results
    }
    stale = {
        ticker: sum(issue.code in {"stale_input", "stale_news"} for issue in issues)
        for ticker, issues in run.quality_issues.items()
    }
    lines = [
        "# Fictional Watchlist Analytics", "", run.fixture_notice, "",
        "## Overview", "", f"- Companies: {len(run.companies)}",
        f"- Highest opportunity score: {results[0].ticker} ({_number(results[0].opportunity_score, 0)})",
        f"- Highest research priority: {priorities[0].ticker} ({priorities[0].research_priority})",
        "- Sector exposure: " + ", ".join(f"{name} {count}" for name, count in sorted(sectors.items())),
        "- Industry exposure: " + ", ".join(f"{name} {count}" for name, count in sorted(industries.items())),
        f"- Most upcoming catalysts: {sorted(upcoming, key=lambda ticker: (-upcoming[ticker], ticker))[0]} ({max(upcoming.values())})",
        f"- Most missing ranking inputs: {sorted(missing, key=lambda ticker: (-missing[ticker], ticker))[0]} ({max(missing.values())})",
        f"- Stalest structured research inputs: {sorted(stale, key=lambda ticker: (-stale[ticker], ticker))[0]} ({max(stale.values())} stale flags)",
        "", "## Metric Leaders", "", "| Metric | Leader | Value |", "|---|---|---:|",
    ]
    for metric in ("ROIC", "FCF Yield", "Revenue Growth", "Debt / Equity", "P/E"):
        values = [(ticker, result.value) for ticker, metrics_map in run.metrics.items() if (result := metrics_map.get(metric)) and result.value is not None and result.applicable]
        if not values:
            continue
        reverse = metric not in {"Debt / Equity", "P/E"}
        ticker, value = sorted(values, key=lambda item: (item[1], item[0]), reverse=reverse)[0]
        lines.append(f"| {metric} | {ticker} | {_number(value)} |")
    lines.extend(["", "## Watchlist Medians", "", "| Metric | Median |", "|---|---:|"])
    for metric in ("P/E", "EV / EBITDA", "Revenue Growth", "FCF Yield"):
        values = [result.value for metrics_map in run.metrics.values() if (result := metrics_map.get(metric)) and result.value is not None and result.applicable]
        lines.append(f"| {metric} | {_number(Decimal(str(statistics.median(values)))) if values else '—'} |")
    lines.extend(["", ranking_report(results, f"{horizon.replace('_', ' ').title()} Opportunity Ranking")])
    return "\n".join(lines).rstrip() + "\n"


def full_demo_report(run: AnalyticsRun) -> str:
    frontmatter = {
        "type": "analytics-demo",
        "pipeline_version": run.manifest.pipeline_version,
        "ranking_version": run.manifest.ranking_version,
        "calculation_version": run.manifest.calculation_version,
        "run_id": run.manifest.run_id,
        "analysis_date": run.analysis_date,
        "fictional_fixture": True,
        "files_written": 0,
    }
    yaml_text = yaml.safe_dump(frontmatter, sort_keys=False).strip()
    sections = [
        f"---\n{yaml_text}\n---\n\n# Deterministic Analytics Demo",
        f"> {run.fixture_notice}",
        ranking_report(run.rankings["one_day"], "1-Day Opportunity Ranking"),
        ranking_report(run.rankings["one_year"], "1-Year Opportunity Ranking"),
        watchlist_summary(run, "one_year"),
        "## Reproducibility\n\n"
        f"- Input hash: `{run.manifest.input_hash}`\n"
        f"- Config hash: `{run.manifest.config_hash}`\n"
        f"- Run ID: `{run.manifest.run_id}`\n"
        f"- Files written: {run.manifest.files_written}",
    ]
    return "\n\n".join(section.strip() for section in sections) + "\n"
