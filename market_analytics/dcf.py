"""Explicit-assumption discounted cash-flow calculations and sensitivity tables."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DCFScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    starting_fcf: Optional[Decimal]
    growth_rates: list[Decimal]
    projection_period: int = Field(ge=1, le=30)
    discount_rate: Optional[Decimal]
    terminal_growth_rate: Optional[Decimal]
    net_debt: Optional[Decimal]
    shares_outstanding: Optional[Decimal]
    currency: str = "USD"


class DCFProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int
    growth_rate: Decimal
    projected_fcf: Decimal
    discount_factor: Decimal
    present_value: Decimal


class DCFResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str
    complete: bool
    warnings: list[str]
    projections: list[DCFProjection]
    terminal_value: Optional[Decimal] = None
    present_value_terminal: Optional[Decimal] = None
    enterprise_value: Optional[Decimal] = None
    equity_value: Optional[Decimal] = None
    value_per_share: Optional[Decimal] = None
    currency: str = "USD"
    calculation_version: str = "dcf-v1"


class SensitivityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discount_rates: list[Decimal]
    terminal_growth_rates: list[Decimal]
    values: list[list[Optional[Decimal]]]
    warnings: list[str]


def calculate_dcf(scenario: DCFScenario) -> DCFResult:
    warnings: list[str] = []
    required = {
        "starting_fcf": scenario.starting_fcf,
        "discount_rate": scenario.discount_rate,
        "terminal_growth_rate": scenario.terminal_growth_rate,
        "net_debt": scenario.net_debt,
        "shares_outstanding": scenario.shares_outstanding,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        return DCFResult(
            scenario=scenario.name, complete=False,
            warnings=[f"missing required input: {name}" for name in missing],
            projections=[], currency=scenario.currency,
        )
    assert all(value is not None for value in required.values())
    if len(scenario.growth_rates) != scenario.projection_period:
        return DCFResult(
            scenario=scenario.name, complete=False,
            warnings=["growth_rates length must equal projection_period"],
            projections=[], currency=scenario.currency,
        )
    if scenario.discount_rate <= Decimal("-1"):
        warnings.append("discount rate must be greater than -100%")
    if scenario.discount_rate <= scenario.terminal_growth_rate:
        warnings.append("discount rate must be greater than terminal growth rate")
    if scenario.shares_outstanding <= 0:
        warnings.append("shares outstanding must be positive")
    if scenario.starting_fcf < 0:
        warnings.append("starting free cash flow is negative; DCF interpretation is unusually uncertain")
    if warnings and any("must" in warning for warning in warnings):
        return DCFResult(
            scenario=scenario.name, complete=False, warnings=warnings,
            projections=[], currency=scenario.currency,
        )

    projections: list[DCFProjection] = []
    fcf = scenario.starting_fcf
    pv_sum = Decimal("0")
    for year, growth in enumerate(scenario.growth_rates, 1):
        fcf *= 1 + growth
        factor = Decimal("1") / ((1 + scenario.discount_rate) ** year)
        present = fcf * factor
        pv_sum += present
        projections.append(
            DCFProjection(
                year=year, growth_rate=growth, projected_fcf=fcf,
                discount_factor=factor, present_value=present,
            )
        )
    terminal_value = fcf * (1 + scenario.terminal_growth_rate) / (
        scenario.discount_rate - scenario.terminal_growth_rate
    )
    terminal_pv = terminal_value / ((1 + scenario.discount_rate) ** scenario.projection_period)
    enterprise_value = pv_sum + terminal_pv
    equity_value = enterprise_value - scenario.net_debt
    value_per_share = equity_value / scenario.shares_outstanding
    return DCFResult(
        scenario=scenario.name, complete=True, warnings=warnings,
        projections=projections, terminal_value=terminal_value,
        present_value_terminal=terminal_pv, enterprise_value=enterprise_value,
        equity_value=equity_value, value_per_share=value_per_share,
        currency=scenario.currency,
    )


def run_scenarios(scenarios: list[DCFScenario]) -> list[DCFResult]:
    return [calculate_dcf(scenario) for scenario in scenarios]


def sensitivity_matrix(
    scenario: DCFScenario,
    discount_rates: list[Decimal],
    terminal_growth_rates: list[Decimal],
) -> SensitivityResult:
    values: list[list[Optional[Decimal]]] = []
    warnings = ["Sensitivity values are scenarios, not a uniquely correct valuation."]
    for terminal_growth in terminal_growth_rates:
        row: list[Optional[Decimal]] = []
        for discount_rate in discount_rates:
            varied = scenario.model_copy(
                update={"discount_rate": discount_rate, "terminal_growth_rate": terminal_growth}
            )
            result = calculate_dcf(varied)
            row.append(result.value_per_share if result.complete else None)
        values.append(row)
    return SensitivityResult(
        discount_rates=discount_rates,
        terminal_growth_rates=terminal_growth_rates,
        values=values,
        warnings=warnings,
    )
