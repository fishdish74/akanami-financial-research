"""Financial formulas with explicit missing/applicability behavior."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Optional

from .models import MetricResult


PERCENT = Decimal("100")


def _missing(name: str, unit: str, fields: list[str]) -> MetricResult:
    return MetricResult(
        name=name, unit=unit, status="missing_data", confidence="low",
        input_fields=fields, warning="one or more required inputs are missing",
    )


def _not_applicable(name: str, unit: str, fields: list[str], warning: str) -> MetricResult:
    return MetricResult(
        name=name, unit=unit, status="not_applicable", confidence="low",
        input_fields=fields, warning=warning, applicable=False,
    )


def _ratio(
    name: str,
    numerator: Optional[Decimal],
    denominator: Optional[Decimal],
    numerator_field: str,
    denominator_field: str,
    *,
    unit: str = "x",
    denominator_must_be_positive: bool = False,
    negative_warning: Optional[str] = None,
) -> MetricResult:
    fields = [numerator_field, denominator_field]
    if numerator is None or denominator is None:
        return _missing(name, unit, fields)
    if denominator == 0:
        return _not_applicable(name, unit, fields, "denominator is zero")
    if denominator_must_be_positive and denominator < 0:
        return _not_applicable(name, unit, fields, "negative denominator makes this metric misleading")
    try:
        value = numerator / denominator
    except (InvalidOperation, ZeroDivisionError):
        return MetricResult(
            name=name, unit=unit, status="invalid_input", confidence="low",
            input_fields=fields, warning="inputs could not be divided",
        )
    status = "warning" if negative_warning and value < 0 else "valid"
    return MetricResult(
        name=name, value=value, unit=unit, status=status,
        confidence="medium" if status == "warning" else "high",
        input_fields=fields, warning=negative_warning if status == "warning" else None,
    )


# Valuation
def pe(price_per_share: Optional[Decimal], eps: Optional[Decimal]) -> MetricResult:
    if eps is not None and eps <= 0:
        return _not_applicable("P/E", "x", ["price_per_share", "eps"], "P/E is not meaningful for non-positive earnings")
    return _ratio("P/E", price_per_share, eps, "price_per_share", "eps", denominator_must_be_positive=True)


def forward_pe(price_per_share: Optional[Decimal], forward_eps: Optional[Decimal]) -> MetricResult:
    result = pe(price_per_share, forward_eps)
    return result.model_copy(update={"name": "Forward P/E", "input_fields": ["price_per_share", "forward_eps"]})


def peg(pe_ratio: Optional[Decimal], eps_growth_ratio: Optional[Decimal]) -> MetricResult:
    if eps_growth_ratio is not None and eps_growth_ratio <= 0:
        return _not_applicable("PEG", "x", ["pe", "eps_growth"], "PEG is not meaningful with non-positive expected growth")
    growth_percent = eps_growth_ratio * PERCENT if eps_growth_ratio is not None else None
    return _ratio("PEG", pe_ratio, growth_percent, "pe", "eps_growth", denominator_must_be_positive=True)


def ev_to_ebitda(enterprise_value: Optional[Decimal], ebitda: Optional[Decimal]) -> MetricResult:
    if ebitda is not None and ebitda <= 0:
        return _not_applicable("EV / EBITDA", "x", ["enterprise_value", "ebitda"], "EV/EBITDA is not meaningful for non-positive EBITDA")
    return _ratio("EV / EBITDA", enterprise_value, ebitda, "enterprise_value", "ebitda", denominator_must_be_positive=True)


def price_to_sales(market_cap: Optional[Decimal], revenue: Optional[Decimal]) -> MetricResult:
    return _ratio("Price / Sales", market_cap, revenue, "market_cap", "revenue", denominator_must_be_positive=True)


def price_to_book(market_cap: Optional[Decimal], equity: Optional[Decimal]) -> MetricResult:
    return _ratio("Price / Book", market_cap, equity, "market_cap", "equity", denominator_must_be_positive=True)


def earnings_yield(net_income: Optional[Decimal], market_cap: Optional[Decimal]) -> MetricResult:
    return _ratio("Earnings Yield", net_income, market_cap, "net_income", "market_cap", unit="decimal_ratio", denominator_must_be_positive=True, negative_warning="negative earnings produce a negative yield")


def fcf_yield(fcf: Optional[Decimal], market_cap: Optional[Decimal]) -> MetricResult:
    return _ratio("FCF Yield", fcf, market_cap, "free_cash_flow", "market_cap", unit="decimal_ratio", denominator_must_be_positive=True, negative_warning="negative free cash flow produces a negative yield")


# Quality and margins
def roa(net_income: Optional[Decimal], total_assets: Optional[Decimal]) -> MetricResult:
    """Keep an undefined asset denominator distinct from a genuine zero return."""
    fields = ["net_income", "total_assets"]
    if net_income is not None and total_assets == 0:
        return MetricResult(
            name="ROA", unit="decimal_ratio", status="invalid_input",
            confidence="low", input_fields=fields,
            warning="total_assets_zero: ROA is undefined because total assets are zero",
        )
    return _ratio(
        "ROA", net_income, total_assets, *fields, unit="decimal_ratio"
    )


def roe(net_income: Optional[Decimal], average_equity: Optional[Decimal]) -> MetricResult:
    return _ratio("ROE", net_income, average_equity, "net_income", "average_equity", unit="decimal_ratio", denominator_must_be_positive=True, negative_warning="negative ROE indicates a loss")


def roic(nopat: Optional[Decimal], invested_capital: Optional[Decimal]) -> MetricResult:
    return _ratio("ROIC", nopat, invested_capital, "nopat", "invested_capital", unit="decimal_ratio", denominator_must_be_positive=True, negative_warning="negative ROIC indicates value destruction for the period")


def gross_margin(gross_profit: Optional[Decimal], revenue: Optional[Decimal]) -> MetricResult:
    return _ratio("Gross Margin", gross_profit, revenue, "gross_profit", "revenue", unit="decimal_ratio", denominator_must_be_positive=True, negative_warning="gross margin is negative")


def operating_margin(operating_income: Optional[Decimal], revenue: Optional[Decimal]) -> MetricResult:
    return _ratio("Operating Margin", operating_income, revenue, "operating_income", "revenue", unit="decimal_ratio", denominator_must_be_positive=True, negative_warning="operating margin is negative")


def net_margin(net_income: Optional[Decimal], revenue: Optional[Decimal]) -> MetricResult:
    return _ratio("Net Margin", net_income, revenue, "net_income", "revenue", unit="decimal_ratio", denominator_must_be_positive=True, negative_warning="net margin is negative")


def fcf_margin(fcf: Optional[Decimal], revenue: Optional[Decimal]) -> MetricResult:
    return _ratio("FCF Margin", fcf, revenue, "free_cash_flow", "revenue", unit="decimal_ratio", denominator_must_be_positive=True, negative_warning="FCF margin is negative")


# Growth
def growth_rate(name: str, current: Optional[Decimal], prior: Optional[Decimal], current_field: str, prior_field: str) -> MetricResult:
    fields = [current_field, prior_field]
    if current is None or prior is None:
        return _missing(name, "decimal_ratio", fields)
    if prior <= 0:
        return _not_applicable(name, "decimal_ratio", fields, "growth from a non-positive base is not comparable")
    return MetricResult(name=name, value=(current / prior) - 1, unit="decimal_ratio", status="valid", input_fields=fields)


def revenue_growth(current: Optional[Decimal], prior: Optional[Decimal]) -> MetricResult:
    return growth_rate("Revenue Growth", current, prior, "revenue", "prior_revenue")


def eps_growth(current: Optional[Decimal], prior: Optional[Decimal]) -> MetricResult:
    return growth_rate("EPS Growth", current, prior, "eps", "prior_eps")


def fcf_growth(current: Optional[Decimal], prior: Optional[Decimal]) -> MetricResult:
    return growth_rate("FCF Growth", current, prior, "free_cash_flow", "prior_free_cash_flow")


def cagr(name: str, ending: Optional[Decimal], beginning: Optional[Decimal], years: Optional[Decimal], fields: list[str]) -> MetricResult:
    if ending is None or beginning is None or years is None:
        return _missing(name, "decimal_ratio", fields)
    if ending <= 0 or beginning <= 0 or years <= 0:
        return _not_applicable(name, "decimal_ratio", fields, "CAGR requires positive endpoints and years")
    value = Decimal(str((float(ending / beginning) ** (1 / float(years))) - 1))
    return MetricResult(name=name, value=value, unit="decimal_ratio", status="valid", input_fields=fields)


def revenue_cagr(ending: Optional[Decimal], beginning: Optional[Decimal], years: Optional[Decimal]) -> MetricResult:
    return cagr("Revenue CAGR", ending, beginning, years, ["revenue", "revenue_start", "years"])


def eps_cagr(ending: Optional[Decimal], beginning: Optional[Decimal], years: Optional[Decimal]) -> MetricResult:
    return cagr("EPS CAGR", ending, beginning, years, ["eps", "eps_start", "years"])


def fcf_cagr(ending: Optional[Decimal], beginning: Optional[Decimal], years: Optional[Decimal]) -> MetricResult:
    return cagr("FCF CAGR", ending, beginning, years, ["free_cash_flow", "fcf_start", "years"])


# Financial strength
def debt_to_equity(debt: Optional[Decimal], equity: Optional[Decimal]) -> MetricResult:
    return _ratio("Debt / Equity", debt, equity, "total_debt", "equity", denominator_must_be_positive=True)


def net_debt(debt: Optional[Decimal], cash: Optional[Decimal]) -> MetricResult:
    if debt is None or cash is None:
        return _missing("Net Debt", "currency", ["total_debt", "cash"])
    return MetricResult(name="Net Debt", value=debt - cash, unit="currency", status="valid", input_fields=["total_debt", "cash"])


def net_debt_to_ebitda(net_debt_value: Optional[Decimal], ebitda: Optional[Decimal]) -> MetricResult:
    return _ratio("Net Debt / EBITDA", net_debt_value, ebitda, "net_debt", "ebitda", denominator_must_be_positive=True)


def interest_coverage(ebit: Optional[Decimal], interest_expense: Optional[Decimal]) -> MetricResult:
    return _ratio("Interest Coverage", ebit, interest_expense, "ebit", "interest_expense", denominator_must_be_positive=True, negative_warning="negative interest coverage indicates operating losses")


def current_ratio(current_assets: Optional[Decimal], current_liabilities: Optional[Decimal]) -> MetricResult:
    return _ratio("Current Ratio", current_assets, current_liabilities, "current_assets", "current_liabilities", denominator_must_be_positive=True)


def quick_ratio(cash: Optional[Decimal], receivables: Optional[Decimal], current_liabilities: Optional[Decimal]) -> MetricResult:
    if cash is None or receivables is None:
        return _missing("Quick Ratio", "x", ["cash", "receivables", "current_liabilities"])
    return _ratio("Quick Ratio", cash + receivables, current_liabilities, "cash_plus_receivables", "current_liabilities", denominator_must_be_positive=True)


def cash_to_debt(cash: Optional[Decimal], debt: Optional[Decimal]) -> MetricResult:
    return _ratio("Cash / Debt", cash, debt, "cash", "total_debt", denominator_must_be_positive=True)
