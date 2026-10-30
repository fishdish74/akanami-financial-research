"""Deterministic financial analytics for the portfolio demonstration."""

from .models import MetricResult

ANALYTICS_COMMANDS = frozenset({
    "demo", "rank", "compare", "validate", "dcf", "watchlist-summary",
    "database-status", "data-status", "sec-sync", "fundamentals", "history",
    "validate-data",
})

__all__ = ["ANALYTICS_COMMANDS", "MetricResult"]
