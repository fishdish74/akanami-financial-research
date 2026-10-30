"""Established accounting and valuation frameworks with applicability warnings."""

from __future__ import annotations

from decimal import Decimal
from math import sqrt
from typing import Optional

from .metrics import _missing, _not_applicable, _ratio
from .models import MetricResult


def piotroski_f_score(
    signals: dict[str, Optional[bool]], *,
    unavailable_reasons: dict[str, str] | None = None,
) -> MetricResult:
    """Sum the nine standard binary Piotroski profitability/leverage/efficiency signals."""
    required = {
        "positive_roa", "positive_cfo", "roa_improved", "cfo_exceeds_income",
        "leverage_declined", "current_ratio_improved", "no_new_shares",
        "gross_margin_improved", "asset_turnover_improved",
    }
    missing = sorted(name for name in required if signals.get(name) is None)
    if missing:
        reasons = unavailable_reasons or {}
        invalid = sorted(
            f"{name}: {reasons[name]}" for name in missing if name in reasons
        )
        if invalid:
            return MetricResult(
                name="Piotroski F-Score", unit="score_0_9",
                status="invalid_input", confidence="low",
                input_fields=sorted(required),
                warning="unavailable signals: " + "; ".join(invalid),
            )
        return _missing("Piotroski F-Score", "score_0_9", sorted(required))
    return MetricResult(
        name="Piotroski F-Score",
        value=Decimal(sum(bool(signals[name]) for name in required)),
        unit="score_0_9",
        status="valid",
        input_fields=sorted(required),
        warning="Most useful for established non-financial companies; not a universal quality score.",
        confidence="medium",
    )


def altman_z_score(
    working_capital: Optional[Decimal],
    total_assets: Optional[Decimal],
    retained_earnings: Optional[Decimal],
    ebit: Optional[Decimal],
    market_value_equity: Optional[Decimal],
    total_liabilities: Optional[Decimal],
    sales: Optional[Decimal],
) -> MetricResult:
    """Original public-manufacturing Altman Z-score; sector applicability is limited."""
    values = [working_capital, total_assets, retained_earnings, ebit, market_value_equity, total_liabilities, sales]
    fields = ["working_capital", "total_assets", "retained_earnings", "ebit", "market_value_equity", "total_liabilities", "sales"]
    if any(value is None for value in values):
        return _missing("Altman Z-Score", "score", fields)
    assert all(value is not None for value in values)
    if total_assets <= 0 or total_liabilities <= 0:
        return _not_applicable("Altman Z-Score", "score", fields, "positive assets and liabilities are required")
    score = (
        Decimal("1.2") * working_capital / total_assets
        + Decimal("1.4") * retained_earnings / total_assets
        + Decimal("3.3") * ebit / total_assets
        + Decimal("0.6") * market_value_equity / total_liabilities
        + sales / total_assets
    )
    return MetricResult(
        name="Altman Z-Score", value=score, unit="score", status="warning",
        confidence="medium", input_fields=fields,
        warning="Original model was designed for publicly traded manufacturers; thresholds may not transfer across sectors.",
    )


def beneish_m_score(
    dsri: Optional[Decimal], gmi: Optional[Decimal], aqi: Optional[Decimal],
    sgi: Optional[Decimal], depi: Optional[Decimal], sgai: Optional[Decimal],
    tata: Optional[Decimal], lvgi: Optional[Decimal],
) -> MetricResult:
    """Eight-variable Beneish M-score from precomputed, period-consistent indices."""
    values = [dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi]
    fields = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]
    if any(value is None for value in values):
        return _missing("Beneish M-Score", "score", fields)
    assert all(value is not None for value in values)
    score = (
        Decimal("-4.84") + Decimal("0.920") * dsri + Decimal("0.528") * gmi
        + Decimal("0.404") * aqi + Decimal("0.892") * sgi
        + Decimal("0.115") * depi - Decimal("0.172") * sgai
        + Decimal("4.679") * tata - Decimal("0.327") * lvgi
    )
    return MetricResult(
        name="Beneish M-Score", value=score, unit="score", status="warning",
        confidence="medium", input_fields=fields,
        warning="A screening model, not proof of manipulation; financial firms and unusual accounting structures need separate treatment.",
    )


def rule_of_40(revenue_growth_ratio: Optional[Decimal], margin_ratio: Optional[Decimal]) -> MetricResult:
    if revenue_growth_ratio is None or margin_ratio is None:
        return _missing("Rule of 40", "percent", ["revenue_growth", "profit_or_fcf_margin"])
    return MetricResult(
        name="Rule of 40", value=(revenue_growth_ratio + margin_ratio) * Decimal("100"),
        unit="percent", status="warning", confidence="medium",
        input_fields=["revenue_growth", "profit_or_fcf_margin"],
        warning="Primarily a software/SaaS heuristic; not automatically applicable to other industries.",
    )


def graham_number(eps: Optional[Decimal], book_value_per_share: Optional[Decimal]) -> MetricResult:
    fields = ["eps", "book_value_per_share"]
    if eps is None or book_value_per_share is None:
        return _missing("Graham Number", "currency_per_share", fields)
    if not eps.is_finite() or not book_value_per_share.is_finite():
        return _not_applicable(
            "Graham Number", "currency_per_share", fields,
            "finite positive EPS and book value per share are required",
        )
    if eps <= 0 or book_value_per_share <= 0:
        return _not_applicable("Graham Number", "currency_per_share", fields, "positive EPS and book value per share are required")
    product = Decimal("22.5") * eps * book_value_per_share
    return MetricResult(
        name="Graham Number", value=Decimal(str(sqrt(float(product)))),
        unit="currency_per_share", status="warning", confidence="medium",
        input_fields=fields,
        warning="A conservative historical heuristic; often misleading for asset-light or rapidly changing businesses.",
    )


def magic_formula_earnings_yield(ebit: Optional[Decimal], enterprise_value: Optional[Decimal]) -> MetricResult:
    result = _ratio("Magic Formula Earnings Yield", ebit, enterprise_value, "ebit", "enterprise_value", unit="decimal_ratio", denominator_must_be_positive=True)
    return result.model_copy(update={"warning": "Use with Magic Formula return on capital and sector-aware peer context.", "confidence": "medium"})


def magic_formula_return_on_capital(
    ebit: Optional[Decimal], net_working_capital: Optional[Decimal], net_fixed_assets: Optional[Decimal]
) -> MetricResult:
    if net_working_capital is None or net_fixed_assets is None:
        return _missing("Magic Formula Return on Capital", "decimal_ratio", ["ebit", "net_working_capital", "net_fixed_assets"])
    result = _ratio(
        "Magic Formula Return on Capital", ebit,
        net_working_capital + net_fixed_assets,
        "ebit", "net_working_capital_plus_net_fixed_assets",
        unit="decimal_ratio", denominator_must_be_positive=True,
    )
    return result.model_copy(update={"warning": "Capital definitions require consistent accounting treatment across peers.", "confidence": "medium"})
