"""Canonical validation for identifiers shared across collector boundaries."""

from __future__ import annotations

import re


TICKER_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9.-]{0,14}")


def normalize_ticker(value: object) -> str:
    ticker = str(value).strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        raise ValueError("ticker contains unsupported characters")
    return ticker
