"""Horizon-specific opportunity and independent research-priority rankings."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from .config import AnalyticsConfig
from .models import (
    Catalyst, CompanyData, ComponentScores, Horizon, PeerComparison,
    QualityIssue, RankingResult, Risk,
)
from .quality import confidence_from_quality, data_quality_score


HORIZON_DAYS: dict[Horizon, int] = {
    "one_day": 1,
    "one_week": 7,
    "one_month": 31,
    "one_year": 366,
}
IMPORTANCE_POINTS = {"low": Decimal("10"), "medium": Decimal("25"), "high": Decimal("45"), "critical": Decimal("65")}
RISK_DEDUCTIONS = {"low": Decimal("5"), "medium": Decimal("15"), "high": Decimal("30"), "critical": Decimal("50")}
COMPONENT_METRICS = {
    "valuation": ["P/E", "Forward P/E", "EV / EBITDA", "Price / Sales", "Earnings Yield", "FCF Yield"],
    "quality": ["ROE", "ROIC", "Gross Margin", "Operating Margin", "Net Margin", "FCF Margin"],
    "growth": ["Revenue Growth", "EPS Growth", "FCF Growth"],
    "financial_strength": ["Debt / Equity", "Net Debt / EBITDA", "Interest Coverage", "Current Ratio"],
}


def _round(value: Decimal) -> Decimal:
    return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)


def _bounded(value: Decimal) -> Decimal:
    return max(Decimal("0"), min(Decimal("100"), value))


def catalyst_score(catalysts: list[Catalyst], horizon: Horizon, as_of: date) -> Decimal | None:
    window = HORIZON_DAYS[horizon]
    eligible = [
        catalyst
        for catalyst in catalysts
        if catalyst.status in {"upcoming", "uncertain"}
        and catalyst.expected_date is not None
        and 0 <= (catalyst.expected_date - as_of).days <= window
    ]
    if not eligible:
        return None
    score = Decimal("25")
    for catalyst in eligible:
        days = (catalyst.expected_date - as_of).days
        timing = Decimal("1") - (Decimal(days) / Decimal(max(window, 1))) * Decimal("0.5")
        confidence_factor = {"low": Decimal("0.5"), "medium": Decimal("0.75"), "high": Decimal("1")}[catalyst.confidence]
        score += IMPORTANCE_POINTS[catalyst.importance] * timing * confidence_factor
    return _round(_bounded(score))


def risk_resilience_score(risks: list[Risk], as_of: date) -> Decimal | None:
    if not risks:
        return None
    score = Decimal("100")
    for risk in risks:
        age = max((as_of - risk.freshness).days, 0)
        freshness_factor = Decimal("1") if age <= 90 else Decimal("0.75")
        confidence_factor = {"low": Decimal("0.5"), "medium": Decimal("0.75"), "high": Decimal("1")}[risk.confidence]
        score -= RISK_DEDUCTIONS[risk.severity] * freshness_factor * confidence_factor
    return _round(_bounded(score))


def _peer_component(
    comparisons: dict[str, PeerComparison], metric_names: list[str]
) -> tuple[Decimal | None, dict[str, str]]:
    values: list[Decimal] = []
    unavailable: dict[str, str] = {}
    applicable_count = len(metric_names)
    for metric in metric_names:
        comparison = comparisons.get(metric)
        if comparison and comparison.percentile is not None and comparison.status == "valid":
            values.append(comparison.percentile if comparison.direction != "context" else Decimal("50"))
        else:
            if comparison is None:
                reason = "ordinary missing data"
            elif comparison.status == "not_applicable":
                # The pipeline emits this marker only for configured structural
                # relevance. Formula-level N/A (including zero denominators)
                # must not shed applicable weight.
                if (comparison.warning or "").startswith(
                    f"{metric} is configured as not applicable for "
                ):
                    reason = "structural not-applicable"
                    applicable_count -= 1
                else:
                    reason = "adverse or economically invalid input"
            elif comparison.status == "invalid_input":
                reason = "adverse or economically invalid input"
            elif comparison.warning and "insufficient_peer_set" in comparison.warning:
                reason = "insufficient peer set"
            elif comparison.warning and "stale" in comparison.warning.casefold():
                reason = "stale data"
            else:
                reason = "ordinary missing data"
            unavailable[metric] = reason
    if not values:
        return None, unavailable
    # Equal constituent weights remain fixed over applicable metrics. Missing
    # evidence supplies no points, not a fabricated observation or percentile.
    return _round(sum(values) / Decimal(applicable_count)), unavailable


def build_components(
    company: CompanyData,
    comparisons: dict[str, PeerComparison],
    issues: list[QualityIssue],
    horizon: Horizon,
    as_of: date,
) -> tuple[ComponentScores, list[str], dict[str, str]]:
    component_values: dict[str, Decimal | None] = {}
    missing: list[str] = []
    unavailable: dict[str, str] = {}
    for component, metric_names in COMPONENT_METRICS.items():
        component_values[component], absent = _peer_component(comparisons, metric_names)
        missing.extend(absent)
        unavailable.update(absent)
    component_values["catalysts"] = catalyst_score(company.catalysts, horizon, as_of)
    if component_values["catalysts"] is None:
        missing.append("Catalyst coverage")
        unavailable["Catalyst coverage"] = "ordinary missing data"
    component_values["news"] = _round(company.news_score) if company.news_score is not None else None
    if component_values["news"] is None:
        missing.append("News coverage")
        unavailable["News coverage"] = "ordinary missing data"
    component_values["risk"] = risk_resilience_score(company.risks, as_of)
    if component_values["risk"] is None:
        missing.append("Risk coverage")
        unavailable["Risk coverage"] = "ordinary missing data"
    component_values["data_quality"] = _round(
        data_quality_score(issues, len(set(missing)), sum(len(v) for v in COMPONENT_METRICS.values()))
    )
    if any(issue.severity == "BLOCKING" for issue in issues):
        unavailable["Ranking"] = "blocking data-quality issue"
    return ComponentScores(**component_values), sorted(set(missing)), dict(sorted(unavailable.items()))


def research_priority_score(
    company: CompanyData,
    issues: list[QualityIssue],
    missing_count: int,
    as_of: date,
) -> Decimal:
    score = Decimal("10")
    for catalyst in company.catalysts:
        if catalyst.status in {"upcoming", "uncertain"} and catalyst.expected_date:
            days = (catalyst.expected_date - as_of).days
            if 0 <= days <= 31:
                score += IMPORTANCE_POINTS[catalyst.importance] * Decimal("0.7")
    for risk in company.risks:
        score += RISK_DEDUCTIONS[risk.severity] * Decimal("0.6")
    score += sum(
        {"INFO": Decimal("1"), "WARNING": Decimal("4"), "ERROR": Decimal("12"), "BLOCKING": Decimal("30")}[issue.severity]
        for issue in issues
    )
    score += min(Decimal(missing_count * 2), Decimal("20"))
    return _round(_bounded(score))


def priority_label(score: Decimal) -> str:
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def rank_company(
    company: CompanyData,
    comparisons: dict[str, PeerComparison],
    issues: list[QualityIssue],
    config: AnalyticsConfig,
    horizon: Horizon,
    as_of: date,
) -> RankingResult:
    components, missing, unavailable = build_components(company, comparisons, issues, horizon, as_of)
    weights = config.ranking_weights[horizon]
    values = components.model_dump()
    blocking = any(issue.severity == "BLOCKING" for issue in issues)
    capability_blocked = not company.capability.ranking_eligible
    available_weight = sum(
        weights[name] for name in weights if values[name] is not None
    )
    ranking_status = (
        "blocked" if blocking or capability_blocked else
        "insufficient_evidence" if available_weight < Decimal("0.50") else
        "ranked"
    )
    opportunity = None
    if ranking_status == "ranked":
        # Deliberately conservative missing-weight policy: configured weights are
        # never redistributed to the remaining evidence. An unavailable
        # component contributes zero; data_quality separately measures evidence
        # breadth/reliability and confidence communicates uncertainty.
        opportunity = _round(
            sum(Decimal(values[name]) * weights[name] for name in weights if values[name] is not None)
        )
    priority_score = research_priority_score(company, issues, len(missing), as_of)
    confidence = confidence_from_quality(
        issues, len(missing), sum(len(v) for v in COMPONENT_METRICS.values()), company.source_quality
    )
    names = {
        "valuation": "Valuation", "quality": "Business quality", "growth": "Growth",
        "financial_strength": "Financial strength", "catalysts": "Catalysts",
        "news": "Recent news", "risk": "Risk resilience", "data_quality": "Data quality",
    }
    positive = [f"{names[name]} score is {values[name]:.0f}/100" for name in weights if values[name] is not None and Decimal(values[name]) >= 65]
    negative = [f"{names[name]} score is {values[name]:.0f}/100" for name in weights if values[name] is not None and Decimal(values[name]) <= 35]
    warnings = [issue.message for issue in issues]
    if capability_blocked:
        unavailable["Security capability"] = ",".join(
            company.capability.reason_codes
        ) or "unsupported security capability"
        warnings.extend(company.capability.explanations)
        warnings.append(
            "Ranking blocked by the explicit provider/security capability gate."
        )
    if confidence == "low":
        warnings.append("Low confidence prevents treating the ranking as a strong comparison.")
    if ranking_status == "blocked":
        warnings.append("Ranking blocked by a BLOCKING data-quality issue.")
    elif ranking_status == "insufficient_evidence":
        warnings.append("Not rankable: less than 50% of configured component weight has evidence.")
    return RankingResult(
        ticker=company.ticker, company=company.company, horizon=horizon,
        components=components, opportunity_score=opportunity, ranking_status=ranking_status,
        research_priority_score=priority_score,
        research_priority=priority_label(priority_score),
        confidence=confidence,
        positive_contributors=positive,
        negative_contributors=negative,
        missing_inputs=missing,
        unavailable_inputs=unavailable,
        capability_reason_codes=company.capability.reason_codes,
        warnings=sorted(set(warnings)),
        ranking_version=config.versions.ranking,
    )


def rank_universe(results: list[RankingResult]) -> list[RankingResult]:
    """Stable opportunity order; research priority remains a separate field."""
    return sorted(
        results,
        key=lambda item: (
            item.opportunity_score is None,
            -(item.opportunity_score or Decimal("0")),
            item.ticker,
        ),
    )


def research_priority_order(results: list[RankingResult]) -> list[RankingResult]:
    return sorted(results, key=lambda item: (-item.research_priority_score, item.ticker))
