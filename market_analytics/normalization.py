"""Explicit unit and fiscal-period normalization."""

from __future__ import annotations

from decimal import Decimal

from .models import DataPoint, FiscalPeriod, MetricResult


SCALARS = {
    "raw": Decimal("1"),
    "thousands": Decimal("1000"),
    "millions": Decimal("1000000"),
    "billions": Decimal("1000000000"),
}


def normalize_data_point(point: DataPoint, name: str) -> MetricResult:
    """Normalize amounts to base units and percentages/bps to decimal ratios."""
    if point.value is None:
        return MetricResult(
            name=name, unit="unknown", status="missing_data", confidence="low",
            input_fields=[name], warning="value is missing",
        )
    if point.unit in SCALARS:
        return MetricResult(
            name=name,
            value=point.value * SCALARS[point.unit],
            unit=point.currency or "base_units",
            status="valid",
            input_fields=[name],
        )
    if point.unit == "percent":
        if point.value < Decimal("-100") or point.value > Decimal("10000"):
            return MetricResult(
                name=name, unit="decimal_ratio", status="invalid_input", confidence="low",
                input_fields=[name], warning="percentage is outside the supported range",
            )
        value = point.value / Decimal("100")
    elif point.unit == "basis_points":
        value = point.value / Decimal("10000")
    else:
        value = point.value
    return MetricResult(
        name=name, value=value, unit="decimal_ratio", status="valid", input_fields=[name]
    )


def periods_compatible(first: FiscalPeriod, second: FiscalPeriod) -> tuple[bool, str | None]:
    """Prevent FY/Q/TTM/forward/point-in-time comparisons from silently mixing."""
    if first.kind != second.kind:
        return False, f"period mismatch: {first.kind} versus {second.kind}"
    if first.kind == "Q" and first.fiscal_quarter != second.fiscal_quarter:
        return False, "quarter-over-quarter comparison requested across unlike fiscal quarters"
    return True, None


def normalized_value(point: DataPoint, name: str) -> Decimal | None:
    result = normalize_data_point(point, name)
    return result.value if result.status == "valid" else None
