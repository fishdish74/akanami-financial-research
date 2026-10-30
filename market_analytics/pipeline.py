"""Complete deterministic analytics pipeline over provider-supplied structured data."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from datetime import datetime, time, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from . import frameworks, metrics
from .config import AnalyticsConfig, config_hash, load_config, metric_relevance
from .dcf import DCFResult, run_scenarios
from .models import CompanyData, Horizon, MetricResult, PeerComparison, QualityIssue, RankingResult, RunManifest
from .normalization import normalized_value, periods_compatible
from .financial_periods import period_issue
from .peers import compare_preferred_peers, compare_universe
from .providers import FixtureProvider
from .quality import exclude_unusable_stale_inputs, validate_company
from .ranking import rank_company, rank_universe, research_priority_order


EXPECTED_METRICS = (
    "P/E", "Forward P/E", "PEG", "EV / EBITDA", "Price / Sales", "Price / Book",
    "Earnings Yield", "FCF Yield", "ROE", "ROIC", "Gross Margin", "Operating Margin",
    "Net Margin", "FCF Margin", "Revenue Growth", "EPS Growth", "FCF Growth",
    "Revenue CAGR", "EPS CAGR", "FCF CAGR", "Debt / Equity", "Net Debt",
    "Net Debt / EBITDA", "Interest Coverage", "Current Ratio", "Quick Ratio", "Cash / Debt",
)


class AnalyticsRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_date: str
    input_cutoff: str
    fixture_notice: str
    companies: dict[str, CompanyData]
    metrics: dict[str, dict[str, MetricResult]]
    quality_issues: dict[str, list[QualityIssue]]
    comparisons: dict[str, dict[str, PeerComparison]]
    rankings: dict[Horizon, list[RankingResult]]
    research_priorities: dict[Horizon, list[RankingResult]]
    dcf_results: dict[str, list[DCFResult]]
    manifest: RunManifest


def _value(company: CompanyData, name: str) -> Decimal | None:
    point = company.financials.get(name)
    return normalized_value(point, name) if point else None


def _growth_with_period_check(company: CompanyData, name: str, current: str, prior: str) -> MetricResult:
    function = {"Revenue Growth": metrics.revenue_growth, "EPS Growth": metrics.eps_growth, "FCF Growth": metrics.fcf_growth}[name]
    result = function(_value(company, current), _value(company, prior))
    current_point = company.financials.get(current)
    prior_point = company.financials.get(prior)
    if current_point and prior_point:
        compatible, warning = periods_compatible(current_point.period, prior_point.period)
        if not compatible:
            return result.model_copy(update={
                "status": "invalid_input", "confidence": "low",
                "warning": warning, "applicable": False,
            })
    return result


def calculate_company_metrics(company: CompanyData, config: AnalyticsConfig) -> dict[str, MetricResult]:
    v = lambda name: _value(company, name)
    def checked(name, function, *values):
        issue = period_issue(company.financials, name)
        if issue:
            # Call only the missing-input branch to retain result units/fields;
            # incompatible numeric values never reach the formula.
            return function(*([None] * len(values))).model_copy(update={
                "value": None, "status": "invalid_input", "applicable": False,
                "confidence": "low", "warning": issue,
            })
        return function(*values)

    # ROA retains closing-assets methodology; historical inputs are not a selector.
    net_debt_result = checked("Net Debt", metrics.net_debt, v("total_debt"), v("cash"))
    roa_result = checked("ROA", metrics.roa, v("net_income"), v("total_assets"))
    pe_result = checked("P/E", metrics.pe, v("price_per_share"), v("eps"))
    revenue_growth_result = _growth_with_period_check(
        company, "Revenue Growth", "revenue", "prior_revenue"
    )
    eps_growth_result = _growth_with_period_check(
        company, "EPS Growth", "eps", "prior_eps"
    )
    fcf_growth_result = _growth_with_period_check(
        company, "FCF Growth", "free_cash_flow", "prior_free_cash_flow"
    )
    results = [
        pe_result,
        checked("Forward P/E", metrics.forward_pe, v("price_per_share"), v("forward_eps")),
        metrics.peg(pe_result.value if pe_result.status == "valid" else None, v("forward_eps_growth")),
        checked("EV / EBITDA", metrics.ev_to_ebitda, v("enterprise_value"), v("ebitda")),
        checked("Price / Sales", metrics.price_to_sales, v("market_cap"), v("revenue")),
        checked("Price / Book", metrics.price_to_book, v("market_cap"), v("equity")),
        checked("Earnings Yield", metrics.earnings_yield, v("net_income"), v("market_cap")),
        checked("FCF Yield", metrics.fcf_yield, v("free_cash_flow"), v("market_cap")),
        roa_result,
        checked("ROE", metrics.roe, v("net_income"), v("average_equity")),
        checked("ROIC", metrics.roic, v("nopat"), v("invested_capital")),
        checked("Gross Margin", metrics.gross_margin, v("gross_profit"), v("revenue")),
        checked("Operating Margin", metrics.operating_margin, v("operating_income"), v("revenue")),
        checked("Net Margin", metrics.net_margin, v("net_income"), v("revenue")),
        checked("FCF Margin", metrics.fcf_margin, v("free_cash_flow"), v("revenue")),
        revenue_growth_result,
        eps_growth_result,
        fcf_growth_result,
        metrics.revenue_cagr(v("revenue"), v("revenue_start"), v("revenue_cagr_years") or v("cagr_years")),
        metrics.eps_cagr(v("eps"), v("eps_start"), v("eps_cagr_years") or v("cagr_years")),
        metrics.fcf_cagr(v("free_cash_flow"), v("fcf_start"), v("fcf_cagr_years") or v("cagr_years")),
        checked("Debt / Equity", metrics.debt_to_equity, v("total_debt"), v("equity")),
        net_debt_result,
        checked("Net Debt / EBITDA", metrics.net_debt_to_ebitda, net_debt_result.value, v("ebitda")),
        checked("Interest Coverage", metrics.interest_coverage, v("ebit"), v("interest_expense")),
        checked("Current Ratio", metrics.current_ratio, v("current_assets"), v("current_liabilities")),
        checked("Quick Ratio", metrics.quick_ratio, v("cash"), v("receivables"), v("current_liabilities")),
        checked("Cash / Debt", metrics.cash_to_debt, v("cash"), v("total_debt")),
        checked("Altman Z-Score", frameworks.altman_z_score, v("working_capital"), v("total_assets"), v("retained_earnings"), v("ebit"), v("market_cap"), v("total_liabilities"), v("revenue")),
        checked("Beneish M-Score", frameworks.beneish_m_score, v("dsri"), v("gmi"), v("aqi"), v("sgi"), v("depi"), v("sgai"), v("tata"), v("lvgi")),
        frameworks.rule_of_40(
            revenue_growth_result.value
            if revenue_growth_result.status in {"valid", "warning"}
            and revenue_growth_result.applicable else None,
            checked("FCF Margin", metrics.fcf_margin, v("free_cash_flow"), v("revenue")).value,
        ),
        checked("Graham Number", frameworks.graham_number, v("eps"), v("book_value_per_share")),
        checked("Magic Formula Earnings Yield", frameworks.magic_formula_earnings_yield, v("ebit"), v("enterprise_value")),
        checked("Magic Formula Return on Capital", frameworks.magic_formula_return_on_capital, v("ebit"), v("net_working_capital"), v("net_fixed_assets")),
    ]
    output = {result.name: result for result in results}
    net_income_value = v("net_income")
    total_assets_value = v("total_assets")
    operating_cash_flow_value = v("operating_cash_flow")
    current_ratio_value = output["Current Ratio"].value
    gross_margin_value = output["Gross Margin"].value
    signals = {
        "positive_roa": (
            roa_result.value > 0 if roa_result.status == "valid" else None
        ),
        "positive_cfo": None if operating_cash_flow_value is None else operating_cash_flow_value > 0,
        "roa_improved": (
            None if total_assets_value == 0 or v("prior_roa") is None or v("roa") is None
            else v("roa") > v("prior_roa")
        ),
        "cfo_exceeds_income": None if operating_cash_flow_value is None or net_income_value is None or period_issue(company.financials, "CFO vs Income") else operating_cash_flow_value > net_income_value,
        "leverage_declined": None if v("prior_debt_to_assets") is None or v("debt_to_assets") is None else v("debt_to_assets") < v("prior_debt_to_assets"),
        "current_ratio_improved": None if v("prior_current_ratio") is None or current_ratio_value is None else current_ratio_value > v("prior_current_ratio"),
        "no_new_shares": None if v("prior_shares_outstanding") is None or v("shares_outstanding") is None else v("shares_outstanding") <= v("prior_shares_outstanding"),
        "gross_margin_improved": None if v("prior_gross_margin") is None or gross_margin_value is None else gross_margin_value > v("prior_gross_margin"),
        "asset_turnover_improved": None if v("prior_asset_turnover") is None or v("asset_turnover") is None else v("asset_turnover") > v("prior_asset_turnover"),
    }
    output["Piotroski F-Score"] = frameworks.piotroski_f_score(
        signals,
        unavailable_reasons=(
            {"positive_roa": "total_assets_zero"}
            if roa_result.status == "invalid_input" and total_assets_value == 0
            else None
        ),
    )
    for name, result in list(output.items()):
        relevance = metric_relevance(config, company.industry, name)
        if relevance == "not_applicable":
            output[name] = result.model_copy(update={"status": "not_applicable", "applicable": False, "confidence": "low", "warning": f"{name} is configured as not applicable for {company.industry}"})
        elif relevance == "weak" and result.status == "valid":
            output[name] = result.model_copy(update={"status": "warning", "confidence": "medium", "warning": f"{name} has weak relevance for {company.industry}"})
    return output


def _hash_payload(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_provider_pipeline(provider: object, config: AnalyticsConfig | None = None) -> AnalyticsRun:
    configuration = config or load_config()
    source_companies = provider.list_companies()  # type: ignore[attr-defined]
    issues = {
        company.ticker: validate_company(company, provider.analysis_date, configuration.freshness_days)  # type: ignore[attr-defined]
        for company in source_companies
    }
    companies_list = [
        exclude_unusable_stale_inputs(company, issues[company.ticker])
        for company in source_companies
    ]
    companies = {company.ticker: company for company in companies_list}
    metric_results = {company.ticker: calculate_company_metrics(company, configuration) for company in companies_list}
    peer_values = {
        ticker: {
            name: result.value if result.status in {"valid", "warning"} and result.applicable else None
            for name, result in results.items()
        }
        for ticker, results in metric_results.items()
    }
    growth_inputs = {
        "Revenue Growth": ("revenue", "prior_revenue"),
        "EPS Growth": ("eps", "prior_eps"),
        "FCF Growth": ("free_cash_flow", "prior_free_cash_flow"),
    }
    metric_bases = {
        company.ticker: {
            metric: (
                company.financials[current].period.kind
                if current in company.financials
                and prior in company.financials
                and periods_compatible(
                    company.financials[current].period,
                    company.financials[prior].period,
                )[0]
                else None
            )
            for metric, (current, prior) in growth_inputs.items()
        }
        for company in companies_list
    }
    peer_metadata = {
        company.ticker: {
            "sector": company.sector, "industry": company.industry,
            "provenance": company.metadata_provenance,
        }
        for company in companies_list
    }
    # Fictional fixtures intentionally compare the whole demonstration universe.
    # Real providers prefer point-in-time industry peers. Sector fallback is an
    # explicit configuration choice; unknown metadata never forms a peer bucket.
    if getattr(provider, "fixture_notice", ""):
        comparisons = compare_universe(
            peer_values, configuration.metric_directions, metric_bases
        )
    else:
        comparisons = compare_preferred_peers(
            peer_values, configuration.metric_directions, peer_metadata,
            primary_scope=configuration.peer_selection.primary_scope,
            allow_sector_fallback=configuration.peer_selection.allow_sector_fallback,
            minimum_companies=configuration.peer_selection.minimum_companies,
            metric_bases=metric_bases,
        )
    # Preserve why an individual metric was unavailable. Peer-set failure stays
    # distinct, while metric-level structural/adverse/stale states override only
    # that company's comparison record.
    for ticker, results in metric_results.items():
        stale_fields = {
            field
            for issue in issues[ticker]
            if issue.code == "stale_input"
            for field in issue.fields
        }
        for name, result in results.items():
            if result.status in {"not_applicable", "invalid_input"}:
                comparisons[ticker][name] = comparisons[ticker][name].model_copy(
                    update={"status": result.status, "warning": result.warning}
                )
            elif result.status == "missing_data" and stale_fields.intersection(result.input_fields):
                comparisons[ticker][name] = comparisons[ticker][name].model_copy(
                    update={"warning": "stale data removed before metric calculation"}
                )
    rankings: dict[Horizon, list[RankingResult]] = {}
    priorities: dict[Horizon, list[RankingResult]] = {}
    for horizon in ("one_day", "one_week", "one_month", "one_year"):
        values = [
            rank_company(company, comparisons[company.ticker], issues[company.ticker], configuration, horizon, provider.analysis_date)  # type: ignore[attr-defined]
            for company in companies_list
        ]
        rankings[horizon] = rank_universe(values)
        priorities[horizon] = research_priority_order(values)
    dcf_results = {ticker: run_scenarios(provider.dcf_scenarios(ticker)) for ticker in companies}  # type: ignore[attr-defined]
    input_hash = _hash_payload(provider.provenance_payload())  # type: ignore[attr-defined]
    cfg_hash = config_hash(configuration)
    input_cutoff = getattr(
        provider,
        "input_cutoff",
        datetime.combine(provider.analysis_date, time.max, tzinfo=timezone.utc),  # type: ignore[attr-defined]
    )
    timestamp = (
        input_cutoff if getattr(provider, "fixture_notice", "")
        else datetime.now(timezone.utc)
    )
    run_identity = (
        f"{input_hash}:{cfg_hash}:{input_cutoff.isoformat()}:"
        f"{configuration.versions.pipeline}:{configuration.versions.ranking}:"
        f"{configuration.versions.calculation}"
    )
    run_id = hashlib.sha256(run_identity.encode("utf-8")).hexdigest()[:20]
    warnings = sorted({issue.message for values in issues.values() for issue in values})
    manifest = RunManifest(
        run_id=run_id, timestamp=timestamp, input_cutoff=input_cutoff,
        pipeline_version=configuration.versions.pipeline,
        ranking_version=configuration.versions.ranking,
        calculation_version=configuration.versions.calculation,
        input_hash=input_hash, config_hash=cfg_hash,
        tickers=sorted(companies), warnings=warnings, errors=[],
        candidates=len(companies), files_proposed=["Analytics Demo Report.md"], files_written=0,
    )
    return AnalyticsRun(
        analysis_date=provider.analysis_date.isoformat(),  # type: ignore[attr-defined]
        input_cutoff=input_cutoff.isoformat(),
        fixture_notice=provider.fixture_notice,  # type: ignore[attr-defined]
        companies=companies, metrics=metric_results, quality_issues=issues,
        comparisons=comparisons, rankings=rankings, research_priorities=priorities,
        dcf_results=dcf_results, manifest=manifest,
    )


def run_provider_pipeline(provider: object, config: AnalyticsConfig | None = None) -> AnalyticsRun:
    """Run every provider read inside one coherent snapshot when supported."""

    snapshot = getattr(provider, "read_snapshot", None)
    context = snapshot() if snapshot is not None else nullcontext()
    with context:
        return _run_provider_pipeline(provider, config)


def run_fixture_pipeline(path: Path, config: AnalyticsConfig | None = None) -> AnalyticsRun:
    return run_provider_pipeline(FixtureProvider(path), config)
