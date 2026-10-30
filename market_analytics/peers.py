"""Dependency-free peer statistics with direction-aware percentiles."""

from __future__ import annotations

import math
import statistics
from decimal import Decimal
from typing import Any, Literal, Optional

from .models import PeerComparison


Direction = Literal["higher", "lower", "context"]


def percentile_rank(value: Decimal, population: list[Decimal], direction: Direction) -> Decimal:
    """Tie-aware percentile. Higher percentile always means more favorable unless context."""
    if not population:
        raise ValueError("population cannot be empty")
    below = sum(item < value for item in population)
    equal = sum(item == value for item in population)
    raw = (Decimal(below) + Decimal(equal) / 2) / Decimal(len(population)) * Decimal("100")
    if direction == "lower":
        return Decimal("100") - raw
    return raw


def compare_metric(
    ticker: str,
    metric: str,
    value: Optional[Decimal],
    peer_values: dict[str, Optional[Decimal]],
    direction: Direction,
    *,
    minimum_companies: int = 2,
    peer_scope: Literal["watchlist", "sector", "industry"] | None = None,
    peer_group: str | None = None,
    peer_provenance: list[str] | None = None,
) -> PeerComparison:
    population = [item for item in peer_values.values() if item is not None]
    if value is None or len(population) < minimum_companies:
        return PeerComparison(
            ticker=ticker, metric=metric, value=value, peer_median=None,
            percentile=None, z_score=None, relative_spread=None,
            direction=direction, status="missing_data",
            warning=(
                "insufficient_peer_set: at least "
                f"{minimum_companies} comparable companies are required"
            ),
            peer_scope=peer_scope, peer_group=peer_group,
            peer_provenance=peer_provenance or [],
        )
    median = Decimal(str(statistics.median(population)))
    percentile = percentile_rank(value, population, direction)
    mean = Decimal(str(statistics.fmean(population)))
    variance = sum((item - mean) ** 2 for item in population) / Decimal(len(population))
    standard_deviation = Decimal(str(math.sqrt(float(variance))))
    z_score = None if standard_deviation == 0 else (value - mean) / standard_deviation
    if direction == "lower" and z_score is not None:
        z_score = -z_score
    spread = None if median == 0 else (value - median) / abs(median)
    return PeerComparison(
        ticker=ticker, metric=metric, value=value, peer_median=median,
        percentile=percentile, z_score=z_score, relative_spread=spread,
        direction=direction, status="valid",
        warning="Direction is context-dependent; percentile is descriptive." if direction == "context" else None,
        peer_scope=peer_scope, peer_group=peer_group,
        peer_provenance=peer_provenance or [],
    )


def compare_universe(
    metric_values: dict[str, dict[str, Optional[Decimal]]],
    directions: dict[str, Direction],
    metric_bases: dict[str, dict[str, str | None]] | None = None,
) -> dict[str, dict[str, PeerComparison]]:
    result: dict[str, dict[str, PeerComparison]] = {ticker: {} for ticker in metric_values}
    metric_names = sorted({name for values in metric_values.values() for name in values})
    for metric in metric_names:
        peers = {ticker: values.get(metric) for ticker, values in metric_values.items()}
        direction = directions.get(metric, "context")
        for ticker in sorted(metric_values):
            basis = (metric_bases or {}).get(ticker, {}).get(metric)
            comparable = {
                peer: value for peer, value in peers.items()
                if basis is None
                or (metric_bases or {}).get(peer, {}).get(metric) == basis
            }
            result[ticker][metric] = compare_metric(
                ticker, metric, peers[ticker], comparable, direction
            )
    return result


def compare_scoped(
    metric_values: dict[str, dict[str, Optional[Decimal]]],
    directions: dict[str, Direction],
    metadata: dict[str, dict[str, Any]],
    scope: Literal["watchlist", "sector", "industry"],
    *,
    minimum_companies: int = 2,
    metric_bases: dict[str, dict[str, str | None]] | None = None,
) -> dict[str, dict[str, PeerComparison]]:
    """Compare against the whole watchlist or each company's matching peer group."""
    if scope == "watchlist":
        return compare_universe(metric_values, directions, metric_bases)
    output: dict[str, dict[str, PeerComparison]] = {}
    field = scope
    metric_names = sorted({name for values in metric_values.values() for name in values})
    for ticker in sorted(metric_values):
        group_value = str(metadata[ticker].get(field) or "")
        known_scope = bool(group_value.strip()) and group_value.strip().casefold() not in {
            "unknown", "n/a", "none", "unclassified",
        }
        group = (
            {
                peer: values for peer, values in metric_values.items()
                if metadata[peer][field] == group_value
            }
            if known_scope
            else {ticker: metric_values[ticker]}
        )
        provenance = sorted(
            {
                item
                for peer in group
                for item in metadata[peer].get("provenance", [])
            }
        )
        output[ticker] = {
            metric: compare_metric(
                ticker, metric, metric_values[ticker].get(metric),
                {
                    peer: peer_values.get(metric)
                    for peer, peer_values in group.items()
                    if (metric_bases or {}).get(ticker, {}).get(metric) is None
                    or (metric_bases or {}).get(peer, {}).get(metric)
                    == (metric_bases or {}).get(ticker, {}).get(metric)
                },
                directions.get(metric, "context"),
                minimum_companies=minimum_companies,
                peer_scope=scope, peer_group=group_value if known_scope else None,
                peer_provenance=provenance,
            )
            for metric in metric_names
        }
    return output


def compare_preferred_peers(
    metric_values: dict[str, dict[str, Optional[Decimal]]],
    directions: dict[str, Direction],
    metadata: dict[str, dict[str, Any]],
    *,
    primary_scope: Literal["industry", "sector"] = "industry",
    allow_sector_fallback: bool = False,
    minimum_companies: int = 2,
    metric_bases: dict[str, dict[str, str | None]] | None = None,
) -> dict[str, dict[str, PeerComparison]]:
    """Select point-in-time industry peers, with an explicitly enabled sector fallback."""

    unknown = {"", "unknown", "n/a", "none", "unclassified"}
    metric_names = sorted({name for values in metric_values.values() for name in values})
    output: dict[str, dict[str, PeerComparison]] = {}
    for ticker in sorted(metric_values):
        selected_scope: Literal["industry", "sector"] = primary_scope
        selected_value = str(metadata[ticker].get(primary_scope) or "").strip()
        group: dict[str, dict[str, Optional[Decimal]]] = {}
        if selected_value.casefold() not in unknown:
            group = {
                peer: values for peer, values in metric_values.items()
                if str(metadata[peer].get(primary_scope) or "").strip() == selected_value
            }
        if len(group) < minimum_companies and primary_scope == "industry" and allow_sector_fallback:
            sector = str(metadata[ticker].get("sector") or "").strip()
            if sector.casefold() not in unknown:
                fallback = {
                    peer: values for peer, values in metric_values.items()
                    if str(metadata[peer].get("sector") or "").strip() == sector
                }
                if len(fallback) >= minimum_companies:
                    selected_scope, selected_value, group = "sector", sector, fallback
        if len(group) < minimum_companies:
            group = {ticker: metric_values[ticker]}
            if selected_value.casefold() in unknown:
                selected_value = ""
        provenance = sorted(
            {
                item
                for peer in group
                for item in metadata[peer].get("provenance", [])
            }
        )
        output[ticker] = {
            metric: compare_metric(
                ticker, metric, metric_values[ticker].get(metric),
                {
                    peer: values.get(metric)
                    for peer, values in group.items()
                    if (metric_bases or {}).get(ticker, {}).get(metric) is None
                    or (metric_bases or {}).get(peer, {}).get(metric)
                    == (metric_bases or {}).get(ticker, {}).get(metric)
                },
                directions.get(metric, "context"),
                minimum_companies=minimum_companies,
                peer_scope=selected_scope,
                peer_group=selected_value or None,
                peer_provenance=provenance,
            )
            for metric in metric_names
        }
    return output
