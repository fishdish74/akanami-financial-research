"""Deterministic data-quality, freshness, and categorical confidence rules."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal

from .config import FreshnessThresholds
from .models import CompanyData, Confidence, DataPoint, FreshnessReport, QualityIssue
from .normalization import normalize_data_point
from .financial_periods import RULES, period_issue


MONETARY_FIELDS = {
    "market_cap", "enterprise_value", "revenue", "prior_revenue", "gross_profit",
    "operating_income", "net_income", "free_cash_flow", "prior_free_cash_flow",
    "total_debt", "cash", "equity", "ebit", "ebitda", "nopat", "invested_capital",
    "current_assets", "current_liabilities", "receivables", "interest_expense",
}

def is_historical_support_field(field: str) -> bool:
    return (
        field == "cagr_years"
        or field.startswith("prior_")
        or field.startswith("opening_")
        or field.endswith("_start")
    )


def validate_records(
    records: list[tuple[str, str, str, Decimal | None, str, str | None]],
) -> list[QualityIssue]:
    """Detect duplicate/conflicting flat rows before they become a keyed snapshot.

    Each tuple is (ticker, metric, period_label, value, unit, currency).
    """
    seen: dict[tuple[str, str, str], tuple[Decimal | None, str, str | None]] = {}
    issues: list[QualityIssue] = []
    for ticker, metric, period, value, unit, currency in records:
        key = (ticker.upper(), metric, period)
        current = (value, unit, currency)
        if key not in seen:
            seen[key] = current
            continue
        prior = seen[key]
        if prior == current:
            issues.append(QualityIssue(code="duplicate_row", severity="WARNING", message=f"duplicate row for {ticker} {metric} {period}", fields=[metric]))
        else:
            changed = []
            if prior[0] != value:
                changed.append("value")
            if prior[1] != unit:
                changed.append("unit")
            if prior[2] != currency:
                changed.append("currency")
            issues.append(QualityIssue(code="conflicting_values", severity="BLOCKING" if "currency" in changed else "ERROR", message=f"conflicting {'/'.join(changed)} for {ticker} {metric} {period}", fields=[metric]))
    return issues


def freshness_category(field: str) -> str:
    if field in {"price_per_share", "market_cap", "enterprise_value"}:
        return "price"
    if field.startswith("forward_") or "estimate" in field:
        return "estimate"
    return "fundamentals"


def freshness_threshold(
    field: str, point: DataPoint, thresholds: FreshnessThresholds
) -> tuple[str, int] | None:
    """Return a cadence-aware policy for a current input.

    Historical comparison inputs are deliberately old and are not claims about
    current freshness. Their current counterpart is assessed instead.
    """

    if is_historical_support_field(field):
        return None
    category = freshness_category(field)
    if category == "price":
        return "price", thresholds.price
    if category == "estimate":
        return "estimate", thresholds.estimate
    if point.period.kind == "FY":
        return "annual fundamentals", thresholds.annual_fundamentals
    return "quarterly/instant fundamentals", thresholds.fundamentals


def validate_company(
    company: CompanyData,
    as_of: date,
    thresholds: FreshnessThresholds,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    if company.as_of > as_of:
        issues.append(QualityIssue(code="future_company_date", severity="BLOCKING", message="company snapshot is dated after the analysis date", fields=["as_of"]))

    for metric, groups in RULES.items():
        reason = period_issue(company.financials, metric)
        if reason:
            fields = list(sum(groups, ()))
            issues.append(QualityIssue(code=reason, severity="ERROR",
                message=f"{metric}: {reason}", fields=fields))

    currencies = {
        point.currency for name, point in company.financials.items()
        if name in MONETARY_FIELDS and point.value is not None and point.currency
    }
    if len(currencies) > 1:
        issues.append(QualityIssue(code="mixed_currency", severity="BLOCKING", message="monetary inputs contain inconsistent currencies", fields=sorted(MONETARY_FIELDS & company.financials.keys())))

    period_rows: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for name, point in company.financials.items():
        base_name = name.removeprefix("prior_").removesuffix("_start")
        key = (base_name, point.period.kind, point.period.label)
        period_rows[key].append(name)
        normalized = normalize_data_point(point, name)
        if normalized.status == "invalid_input":
            issues.append(QualityIssue(code="invalid_unit_value", severity="ERROR", message=normalized.warning or "invalid unit value", fields=[name]))
        if point.as_of > as_of:
            issues.append(QualityIssue(code="future_input", severity="BLOCKING", message=f"{name} is dated after the analysis date", fields=[name]))
        policy = freshness_threshold(name, point, thresholds)
        if policy is not None:
            policy_name, threshold = policy
            age = (as_of - point.as_of).days
            if age > threshold:
                severity = "ERROR" if age > threshold * 2 else "WARNING"
                issues.append(
                    QualityIssue(
                        code="stale_input",
                        severity=severity,
                        message=(
                            f"{name} is {age} days old; {policy_name} threshold is {threshold}"
                        ),
                        fields=[name],
                    )
                )

    for fields in period_rows.values():
        if len(fields) > 1 and not ({"revenue", "prior_revenue"} & set(fields)):
            issues.append(QualityIssue(code="duplicate_period", severity="ERROR", message="multiple fields map to the same metric and fiscal period", fields=sorted(fields)))

    for field in ("market_cap", "enterprise_value", "shares_outstanding"):
        point = company.financials.get(field)
        if point and point.value is not None and point.value < 0:
            issues.append(QualityIssue(code="negative_valuation_input", severity="BLOCKING", message=f"{field} cannot be negative", fields=[field]))
    shares = company.financials.get("shares_outstanding")
    if shares and shares.value is not None and shares.value == 0:
        issues.append(QualityIssue(code="zero_shares", severity="BLOCKING", message="shares_outstanding must be positive", fields=["shares_outstanding"]))

    share_fields = [name for name in company.financials if name.startswith("shares_outstanding")]
    if len(share_fields) > 1:
        values = [company.financials[name].value for name in share_fields if company.financials[name].value is not None]
        if values and min(values) > 0 and max(values) / min(values) > Decimal("1.25"):
            issues.append(QualityIssue(code="shares_discontinuity", severity="WARNING", message="shares outstanding changed by more than 25%; verify split/issuance treatment", fields=share_fields))
    for catalyst in company.catalysts:
        age = (as_of - catalyst.source_date).days
        if age < 0:
            issues.append(QualityIssue(code="future_catalyst_source", severity="BLOCKING", message=f"catalyst source date is after the analysis date: {catalyst.name}", fields=["catalysts"]))
        elif age > thresholds.news:
            issues.append(QualityIssue(code="stale_news", severity="WARNING", message=f"catalyst source is {age} days old; news threshold is {thresholds.news}", fields=["catalysts"]))
    return sorted(issues, key=lambda item: ({"BLOCKING": 0, "ERROR": 1, "WARNING": 2, "INFO": 3}[item.severity], item.code, item.message))


def freshness_report(company: CompanyData, as_of: date) -> FreshnessReport:
    def oldest_age(names: list[str]) -> int | None:
        dates = [company.financials[name].as_of for name in names if name in company.financials]
        return max((as_of - value).days for value in dates) if dates else None

    current_names = [
        name for name in company.financials
        if not is_historical_support_field(name)
    ]
    price_names = [name for name in current_names if freshness_category(name) == "price"]
    estimate_names = [name for name in current_names if freshness_category(name) == "estimate"]
    fundamental_names = [name for name in current_names if freshness_category(name) == "fundamentals"]
    filing_dates = [company.financials[name].as_of for name in fundamental_names]
    news_dates = [item.source_date for item in company.catalysts]
    return FreshnessReport(
        price_age=oldest_age(price_names),
        filing_age=min(((as_of - value).days for value in filing_dates), default=None),
        fundamentals_age=oldest_age(fundamental_names),
        estimate_age=oldest_age(estimate_names),
        news_age=max((as_of - value).days for value in news_dates) if news_dates else None,
    )


def exclude_unusable_stale_inputs(
    company: CompanyData, issues: list[QualityIssue]
) -> CompanyData:
    """Prevent severely obsolete current facts from entering calculations."""

    unusable = {
        field
        for issue in issues
        if issue.code == "stale_input" and issue.severity in {"ERROR", "BLOCKING"}
        for field in issue.fields
    }
    if not unusable:
        return company
    financials = {
        name: point for name, point in company.financials.items() if name not in unusable
    }
    return company.model_copy(update={"financials": financials})


def data_quality_score(issues: list[QualityIssue], missing_count: int, total_expected: int) -> Decimal:
    deductions = {"INFO": 1, "WARNING": 5, "ERROR": 15, "BLOCKING": 40}
    score = Decimal("100") - sum(Decimal(deductions[issue.severity]) for issue in issues)
    if total_expected:
        score -= Decimal("30") * Decimal(missing_count) / Decimal(total_expected)
    return max(Decimal("0"), min(Decimal("100"), score))


def confidence_from_quality(
    issues: list[QualityIssue],
    missing_count: int,
    total_expected: int,
    source_quality: Confidence,
) -> Confidence:
    if any(issue.severity == "BLOCKING" for issue in issues):
        return "low"
    missing_ratio = Decimal(missing_count) / Decimal(total_expected) if total_expected else Decimal("1")
    if any(issue.severity == "ERROR" for issue in issues) or missing_ratio > Decimal("0.30") or source_quality == "low":
        return "low"
    if any(issue.severity == "WARNING" for issue in issues) or missing_ratio > Decimal("0.10") or source_quality == "medium":
        return "medium"
    return "high"
