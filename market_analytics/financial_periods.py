"""Formula-specific economic-period eligibility, independent of filing FY labels.

Selection remains latest-input-only. A mismatch is not repaired by silently
substituting an older observation. Market multiples intentionally use a current
market observation with historical fundamentals; accounting ratios do not.
"""
from .models import DataPoint

# (duration inputs, balance inputs, current market inputs). Average balances
# carry the duration they summarize, established from endpoints by the provider.
RULES = {
    "P/E": (("eps",), (), ("price_per_share",)),
    "Forward P/E": (("forward_eps",), (), ("price_per_share",)),
    "EV / EBITDA": (("ebitda",), (), ("enterprise_value",)),
    "Price / Sales": (("revenue",), (), ("market_cap",)),
    "Price / Book": ((), ("equity",), ("market_cap",)),
    "Earnings Yield": (("net_income",), (), ("market_cap",)),
    "FCF Yield": (("free_cash_flow",), (), ("market_cap",)),
    "ROA": (("net_income",), ("total_assets",), ()),
    "ROE": (("net_income", "average_equity"), (), ()),
    "ROIC": (("nopat",), ("invested_capital",), ()),
    "Gross Margin": (("gross_profit", "revenue"), (), ()),
    "Operating Margin": (("operating_income", "revenue"), (), ()),
    "Net Margin": (("net_income", "revenue"), (), ()),
    "FCF Margin": (("free_cash_flow", "revenue"), (), ()),
    "Debt / Equity": ((), ("total_debt", "equity"), ()),
    "Net Debt": ((), ("total_debt", "cash"), ()),
    "Net Debt / EBITDA": (("ebitda",), ("total_debt", "cash"), ()),
    "Interest Coverage": (("ebit", "interest_expense"), (), ()),
    "Current Ratio": ((), ("current_assets", "current_liabilities"), ()),
    "Quick Ratio": ((), ("cash", "receivables", "current_liabilities"), ()),
    "Cash / Debt": ((), ("cash", "total_debt"), ()),
    "Altman Z-Score": (("ebit", "revenue"), ("working_capital", "total_assets", "retained_earnings", "total_liabilities"), ("market_cap",)),
    "Graham Number": (("eps",), ("book_value_per_share",), ()),
    "Magic Formula Earnings Yield": (("ebit",), (), ("enterprise_value",)),
    "Magic Formula Return on Capital": (("ebit",), ("net_working_capital", "net_fixed_assets"), ()),
    "CFO vs Income": (("operating_cash_flow", "net_income"), (), ()),
    "Beneish M-Score": (("dsri", "gmi", "aqi", "sgi", "depi", "sgai", "tata", "lvgi"), (), ()),
}


def period_issue(financials: dict[str, DataPoint], name: str) -> str | None:
    """None means no period objection; missing values retain numeric missing-data rules."""
    flows, stocks, markets = RULES[name]
    names = flows + stocks + markets
    if any(key not in financials or financials[key].value is None for key in names):
        return None
    periods = {key: financials[key].period for key in names}
    duration = [periods[key] for key in flows]
    balances = [periods[key] for key in stocks]
    if any(p.start_date is None for p in duration):
        return "insufficient_financial_period_metadata"
    allowed = {"FTM"} if name == "Forward P/E" else {"FY", "TTM"}
    if any(p.kind not in allowed or p.start_date > p.end_date for p in duration):
        return "incompatible_financial_periods"
    if duration and any((p.start_date, p.end_date) != (duration[0].start_date, duration[0].end_date) for p in duration):
        return "incompatible_financial_periods"
    if any(p.kind != "POINT_IN_TIME" for p in balances):
        return "incompatible_financial_periods"
    if balances and any(p.end_date != balances[0].end_date for p in balances):
        return "incompatible_financial_periods"
    if duration and any(p.end_date != duration[0].end_date for p in balances):
        return "incompatible_financial_periods"
    for key in markets:
        market = periods[key]
        if market.kind != "POINT_IN_TIME":
            return "incompatible_financial_periods"
        reference = duration + balances
        if name != "Forward P/E" and any(p.end_date > market.end_date for p in reference):
            return "incompatible_financial_periods"
    return None
