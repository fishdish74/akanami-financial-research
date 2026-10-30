"""Dependency-free exchange-session ownership for US forward-paper evaluation."""

from __future__ import annotations

import calendar as month_calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TradingSession(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    exchange: str
    trading_date: date
    kind: Literal["FULL", "EARLY_CLOSE", "CLOSED"]
    timezone_name: str
    opens_at: datetime | None = None
    closes_at: datetime | None = None


class ExchangeCalendar(BaseModel):
    """Explicit calendar input; it does not claim to know every exchange holiday."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    exchange: str
    timezone_name: str
    regular_open: time
    regular_close: time
    weekend_days: frozenset[int] = Field(default_factory=lambda: frozenset({5, 6}))
    holidays: frozenset[date] = Field(default_factory=frozenset)
    early_closes: dict[date, time] = Field(default_factory=dict)
    policy_version: str = "explicit-exchange-calendar-v1"
    supported_start: date | None = None
    supported_end: date | None = None

    @field_validator("exchange", mode="before")
    @classmethod
    def normalize_exchange(cls, value: object) -> str:
        result = str(value).strip().upper()
        if not result:
            raise ValueError("exchange is required")
        return result

    @field_validator("timezone_name")
    @classmethod
    def known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("unknown exchange timezone") from exc
        return value

    @model_validator(mode="after")
    def valid_hours(self) -> "ExchangeCalendar":
        if self.regular_close <= self.regular_open:
            raise ValueError("regular close must follow regular open")
        if any(value <= self.regular_open or value >= self.regular_close for value in self.early_closes.values()):
            raise ValueError("early close must fall inside the regular session")
        if any(day < 0 or day > 6 for day in self.weekend_days):
            raise ValueError("weekend days must use Python weekday numbers")
        if (
            self.supported_start is not None
            and self.supported_end is not None
            and self.supported_end < self.supported_start
        ):
            raise ValueError("calendar support end precedes support start")
        return self

    def session(self, trading_date: date) -> TradingSession:
        if self.supported_start is not None and trading_date < self.supported_start:
            raise ValueError("date is before the validated calendar range")
        if self.supported_end is not None and trading_date > self.supported_end:
            raise ValueError("date is after the validated calendar range")
        if trading_date.weekday() in self.weekend_days or trading_date in self.holidays:
            return TradingSession(
                exchange=self.exchange, trading_date=trading_date, kind="CLOSED",
                timezone_name=self.timezone_name,
            )
        zone = ZoneInfo(self.timezone_name)
        close = self.early_closes.get(trading_date, self.regular_close)
        return TradingSession(
            exchange=self.exchange, trading_date=trading_date,
            kind="EARLY_CLOSE" if trading_date in self.early_closes else "FULL",
            timezone_name=self.timezone_name,
            opens_at=datetime.combine(trading_date, self.regular_open, tzinfo=zone).astimezone(timezone.utc),
            closes_at=datetime.combine(trading_date, close, tzinfo=zone).astimezone(timezone.utc),
        )

    def next_session_on_or_after(
        self, target: date, *, max_calendar_days: int = 7
    ) -> TradingSession:
        """Resolve by calendar only; provider bars never define whether a session existed."""

        if max_calendar_days < 0 or max_calendar_days > 14:
            raise ValueError("calendar resolution bound must be between 0 and 14 days")
        for offset in range(max_calendar_days + 1):
            candidate = self.session(target + timedelta(days=offset))
            if candidate.kind != "CLOSED":
                return candidate
        raise ValueError("no validated exchange session exists inside the resolution bound")

    def previous_session_on_or_before(
        self, target: date, *, max_calendar_days: int = 7
    ) -> TradingSession:
        """Resolve the latest preceding session from calendar truth alone."""

        if max_calendar_days < 0 or max_calendar_days > 14:
            raise ValueError("calendar resolution bound must be between 0 and 14 days")
        for offset in range(max_calendar_days + 1):
            candidate = self.session(target - timedelta(days=offset))
            if candidate.kind != "CLOSED":
                return candidate
        raise ValueError("no validated exchange session exists inside the resolution bound")

    def trading_date_for(self, timestamp: datetime) -> date:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("market timestamp must be timezone-aware")
        return timestamp.astimezone(ZoneInfo(self.timezone_name)).date()


US_EQUITY_CALENDAR_POLICY_VERSION = "us-equity-xnas-xnys-2025-2030-v2"
SUPPORTED_US_EQUITY_EXCHANGES = frozenset({"XNAS", "XNYS"})
_EXCHANGE_ALIASES = {"NASDAQ": "XNAS", "NYSE": "XNYS"}


def _observed(value: date) -> date:
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(
        days=(weekday - first.weekday()) % 7 + 7 * (occurrence - 1)
    )


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month, month_calendar.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian computus, used only to identify Good Friday."""

    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _us_equity_holidays(year: int) -> set[date]:
    return {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),       # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),       # Washington's Birthday
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),         # Memorial Day
        _observed(date(year, 6, 19)),      # Juneteenth
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),       # Labor Day
        _nth_weekday(year, 11, 3, 4),      # Thanksgiving
        _observed(date(year, 12, 25)),
    }


def _us_equity_early_closes(year: int) -> dict[date, time]:
    values: dict[date, time] = {}
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    values[thanksgiving + timedelta(days=1)] = time(13)
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 5 and christmas_eve not in _us_equity_holidays(year):
        values[christmas_eve] = time(13)
    independence_closure = _observed(date(year, 7, 4))
    preceding = independence_closure - timedelta(days=1)
    while preceding.weekday() >= 5 or preceding in _us_equity_holidays(year):
        preceding -= timedelta(days=1)
    values[preceding] = time(13)
    return values


def us_equity_calendar(exchange: str) -> ExchangeCalendar:
    """Return the bounded v1 XNAS/XNYS calendar; no international support is implied."""

    canonical = _EXCHANGE_ALIASES.get(exchange.strip().upper(), exchange.strip().upper())
    if canonical not in SUPPORTED_US_EQUITY_EXCHANGES:
        raise ValueError("forward-paper v1 supports only XNAS/NASDAQ and XNYS/NYSE")
    years = range(2025, 2031)
    holidays = frozenset(day for year in years for day in _us_equity_holidays(year))
    early_closes = {
        day: close for year in years for day, close in _us_equity_early_closes(year).items()
        if day not in holidays
    }
    return ExchangeCalendar(
        exchange=canonical,
        timezone_name="America/New_York",
        regular_open=time(9, 30),
        regular_close=time(16),
        holidays=holidays,
        early_closes=early_closes,
        policy_version=US_EQUITY_CALENDAR_POLICY_VERSION,
        supported_start=date(2025, 1, 1),
        supported_end=date(2030, 12, 31),
    )
