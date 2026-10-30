"""Offline-testable controls shared by external-data provider adapters."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CapabilityStatus = Literal["SUPPORTED", "UNVERIFIED", "UNAVAILABLE", "REQUIRES_HIGHER_TIER"]
PolicyStatus = Literal["ALLOWED", "PROHIBITED", "UNVERIFIED", "REVIEW_REQUIRED"]


class ProviderBoundaryError(ValueError):
    """A sanitized failure at the external-provider boundary."""


class PaginationError(ProviderBoundaryError):
    pass


class ProviderRequestError(ProviderBoundaryError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ProviderPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[dict[str, Any]]
    next_cursor: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("next_cursor", mode="before")
    @classmethod
    def valid_cursor(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("next-page token must be a non-empty string")
        return value.strip()


class PageProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    page_number: int = Field(ge=1)
    requested_cursor: str | None
    returned_cursor: str | None
    item_count: int = Field(ge=0)
    content_hash: str
    source: dict[str, Any] = Field(default_factory=dict)


class PaginatedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[dict[str, Any]]
    pages: list[PageProvenance]
    termination: Literal["complete", "empty_page"]

    @property
    def complete(self) -> bool:
        """Successful termination must retain proof that no cursor remains."""
        return bool(self.pages) and self.pages[-1].returned_cursor is None


def paginate(
    fetch_page: Callable[[str | None], ProviderPage],
    *,
    max_pages: int,
    stable_key: Callable[[dict[str, Any]], object] | None = None,
) -> PaginatedResult:
    """Collect bounded cursor pages while rejecting loops and duplicate content."""

    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    cursor: str | None = None
    seen_cursors: set[str] = set()
    seen_page_hashes: set[str] = set()
    items: list[dict[str, Any]] = []
    pages: list[PageProvenance] = []
    previous_key: object | None = None
    for page_number in range(1, max_pages + 1):
        try:
            page = ProviderPage.model_validate(fetch_page(cursor))
        except ProviderBoundaryError:
            # Retrieval/HTTP ownership stays with the provider boundary. Only a
            # successfully retrieved page with invalid pagination metadata is
            # classified as a pagination failure.
            raise
        except Exception as exc:
            raise PaginationError(f"malformed pagination metadata on page {page_number}") from exc
        packed = json.dumps(page.items, sort_keys=True, separators=(",", ":"), default=str)
        content_hash = hashlib.sha256(packed.encode("utf-8")).hexdigest()
        if content_hash in seen_page_hashes and page.items:
            raise PaginationError(f"duplicate page content detected on page {page_number}")
        seen_page_hashes.add(content_hash)
        pages.append(PageProvenance(
            page_number=page_number, requested_cursor=cursor,
            returned_cursor=page.next_cursor, item_count=len(page.items),
            content_hash=content_hash, source=page.provenance,
        ))
        if not page.items:
            if page.next_cursor is not None:
                raise PaginationError("empty page has unresolved continuation")
            return PaginatedResult(items=items, pages=pages, termination="empty_page")
        if stable_key is not None:
            for item in page.items:
                current_key = stable_key(item)
                if previous_key is not None and current_key < previous_key:  # type: ignore[operator]
                    raise PaginationError("provider pages are not in stable order")
                previous_key = current_key
        items.extend(page.items)
        if page.next_cursor is None:
            return PaginatedResult(items=items, pages=pages, termination="complete")
        if page.next_cursor == cursor or page.next_cursor in seen_cursors:
            raise PaginationError("repeated pagination cursor detected")
        seen_cursors.add(page.next_cursor)
        cursor = page.next_cursor
    raise PaginationError(f"pagination exceeded hard maximum of {max_pages} pages")


class HttpResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: int = Field(ge=100, le=599)
    body: bytes | str
    headers: dict[str, str] = Field(default_factory=dict)


class RateLimitPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requests_per_minute: int = Field(gt=0)
    daily_quota: int | None = Field(default=None, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    base_backoff_seconds: float = Field(default=1.0, gt=0)
    max_backoff_seconds: float = Field(default=30.0, gt=0)
    max_retry_after_seconds: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def valid_bounds(self) -> "RateLimitPolicy":
        if self.max_backoff_seconds < self.base_backoff_seconds:
            raise ValueError("maximum backoff cannot be below base backoff")
        return self


class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    requests_attempted: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    retries: int = 0
    quota_units: int = 0
    estimated_billable_calls: int = 0
    estimated_cost: float = 0.0
    quota_remaining: int | None = None


def parse_retry_after(value: str | None, now: datetime) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value.strip())
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                return None
            return max(0.0, (target.astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


@dataclass
class ProviderRequestController:
    provider: str
    policy: RateLimitPolicy
    clock: Callable[[], float]
    utc_now: Callable[[], datetime]
    sleeper: Callable[[float], None]
    usage: ProviderUsage = field(init=False)
    _attempt_times: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.usage = ProviderUsage(
            provider=self.provider, quota_remaining=self.policy.daily_quota
        )

    def _pace(self) -> None:
        now = self.clock()
        self._attempt_times = [value for value in self._attempt_times if now - value < 60]
        if len(self._attempt_times) >= self.policy.requests_per_minute:
            delay = 60 - (now - self._attempt_times[0])
            if delay > 0:
                self.sleeper(delay)
                now = self.clock()
                self._attempt_times = [value for value in self._attempt_times if now - value < 60]
        self._attempt_times.append(self.clock())

    def request(
        self,
        send: Callable[[], HttpResult],
        *,
        quota_units: int = 1,
        error_detail: Callable[[HttpResult], str | None] | None = None,
    ) -> HttpResult:
        def failure_message(base: str, result: HttpResult) -> str:
            if error_detail is None:
                return base
            try:
                detail = error_detail(result)
            except Exception:
                detail = None
            return f"{base}: {detail}" if detail else base

        for attempt in range(self.policy.max_retries + 1):
            if (
                self.policy.daily_quota is not None
                and self.usage.quota_units + quota_units > self.policy.daily_quota
            ):
                raise ProviderRequestError("provider daily quota exhausted")
            self._pace()
            self.usage.requests_attempted += 1
            # Without provider-specific pricing, conservatively treat every actual
            # send invocation as one potentially billable call. Preflight/quota stops
            # and cache hits never reach this point and therefore are not counted.
            self.usage.estimated_billable_calls += 1
            self.usage.quota_units += quota_units
            if self.policy.daily_quota is not None:
                self.usage.quota_remaining = self.policy.daily_quota - self.usage.quota_units
            try:
                result = HttpResult.model_validate(send())
            except Exception as exc:
                self.usage.requests_failed += 1
                if attempt >= self.policy.max_retries:
                    raise ProviderRequestError("provider request failed after bounded retries") from exc
                self.usage.retries += 1
                self.sleeper(min(
                    self.policy.base_backoff_seconds * (2 ** attempt),
                    self.policy.max_backoff_seconds,
                ))
                continue
            if 200 <= result.status < 300:
                self.usage.requests_succeeded += 1
                return result
            self.usage.requests_failed += 1
            if 300 <= result.status < 400:
                raise ProviderRequestError("unexpected_redirect", status=result.status)
            if result.status == 429:
                if attempt >= self.policy.max_retries:
                    raise ProviderRequestError(
                        failure_message(
                            "provider remained rate limited after bounded retries", result
                        ),
                        status=429,
                    )
                retry_after = parse_retry_after(
                    next((value for key, value in result.headers.items() if key.casefold() == "retry-after"), None),
                    self.utc_now(),
                )
                delay = retry_after if retry_after is not None else self.policy.base_backoff_seconds * (2 ** attempt)
                self.usage.retries += 1
                self.sleeper(min(delay, self.policy.max_retry_after_seconds, self.policy.max_backoff_seconds))
                continue
            if 400 <= result.status < 500:
                raise ProviderRequestError(
                    failure_message(
                        f"provider rejected request with HTTP {result.status}", result
                    ),
                    status=result.status,
                )
            if attempt >= self.policy.max_retries:
                raise ProviderRequestError(
                    failure_message(
                        f"provider failed with HTTP {result.status} after bounded retries",
                        result,
                    ),
                    status=result.status,
                )
            self.usage.retries += 1
            self.sleeper(min(
                self.policy.base_backoff_seconds * (2 ** attempt),
                self.policy.max_backoff_seconds,
            ))
        raise AssertionError("unreachable")


def decode_json_object(
    response: HttpResult,
    *,
    provider_error_fields: Sequence[str] = ("error", "error_message", "message"),
) -> dict[str, Any]:
    if not 200 <= response.status < 300:
        raise ProviderBoundaryError(f"cannot decode unsuccessful HTTP {response.status}")
    if response.body in (b"", ""):
        raise ProviderBoundaryError("provider returned an empty response body")
    try:
        payload = json.loads(response.body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ProviderBoundaryError("provider returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ProviderBoundaryError("provider response root must be an object")
    for field_name in provider_error_fields:
        if field_name in payload and payload[field_name] not in (None, "", False):
            raise ProviderBoundaryError("provider declared an error in an HTTP 200 response")
    return payload


def validate_object(
    payload: Mapping[str, Any], required: Mapping[str, type | tuple[type, ...]]
) -> dict[str, Any]:
    output = dict(payload)
    for field_name, expected_type in required.items():
        if field_name not in payload:
            raise ProviderBoundaryError(f"provider object is missing required field: {field_name}")
        value = payload[field_name]
        accepted_types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
        if isinstance(value, bool) and any(
            candidate in {int, float} for candidate in accepted_types
        ):
            raise ProviderBoundaryError(f"provider field has wrong type: {field_name}")
        if not isinstance(value, expected_type):
            raise ProviderBoundaryError(f"provider field has wrong type: {field_name}")
    return output


class RecordBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: list[dict[str, Any]]
    errors: list[str]


def validate_record_array(
    payload: Mapping[str, Any], *, field_name: str,
    required: Mapping[str, type | tuple[type, ...]],
    expected_count_field: str | None = None,
) -> RecordBatch:
    records = payload.get(field_name)
    if not isinstance(records, list):
        raise ProviderBoundaryError(f"provider field must be an array: {field_name}")
    if expected_count_field is not None:
        expected = payload.get(expected_count_field)
        if not isinstance(expected, int) or isinstance(expected, bool):
            raise ProviderBoundaryError("provider result count is missing or malformed")
        if expected != len(records):
            raise ProviderBoundaryError("provider response appears truncated")
    valid: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {index}: expected object")
            continue
        try:
            valid.append(validate_object(record, required))
        except ProviderBoundaryError as exc:
            errors.append(f"record {index}: {exc}")
    return RecordBatch(valid=valid, errors=errors)


class StoragePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    local_storage: PolicyStatus = "UNVERIFIED"
    long_term_retention: PolicyStatus = "UNVERIFIED"
    redistribution: PolicyStatus = "UNVERIFIED"
    internal_use_only: PolicyStatus = "UNVERIFIED"
    attribution: PolicyStatus = "UNVERIFIED"
    notes: str = ""


class DatasetCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset: str
    status: CapabilityStatus
    requires_auth: bool
    entitlement: str | None = None
    historical_depth_days: int | None = Field(default=None, gt=0)
    earliest_supported_date: date | None = None
    requests_per_minute: int | None = Field(default=None, gt=0)
    daily_quota: int | None = Field(default=None, gt=0)
    pagination_style: str | None = None
    corporate_action_coverage: list[str] = Field(default_factory=list)
    metadata_coverage: list[str] = Field(default_factory=list)
    adjusted_price_semantics: str | None = None
    correction_behavior: str | None = None
    point_in_time_quality: str | None = None
    storage_policy: StoragePolicy = Field(default_factory=StoragePolicy)


class CapabilityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    supported: bool
    status: CapabilityStatus
    reason: str


def preflight_range(
    capability: DatasetCapability, start: date, end: date, *, as_of: date
) -> CapabilityDecision:
    if end < start:
        raise ValueError("requested range end precedes start")
    if capability.status != "SUPPORTED":
        return CapabilityDecision(
            supported=False, status=capability.status,
            reason=f"dataset capability is {capability.status.lower()}",
        )
    if capability.earliest_supported_date and start < capability.earliest_supported_date:
        return CapabilityDecision(
            supported=False, status="REQUIRES_HIGHER_TIER",
            reason="requested start predates verified entitlement",
        )
    if capability.historical_depth_days is not None:
        earliest = as_of.toordinal() - capability.historical_depth_days
        if start.toordinal() < earliest:
            return CapabilityDecision(
                supported=False, status="REQUIRES_HIGHER_TIER",
                reason="requested range exceeds verified historical depth",
            )
    return CapabilityDecision(supported=True, status="SUPPORTED", reason="range is plausibly supported")


class UsageLedger:
    """Provider usage accounting; cost remains unknown unless explicitly supplied."""

    def __init__(self) -> None:
        self._totals: dict[str, ProviderUsage] = {}

    def record(self, usage: ProviderUsage) -> None:
        total = self._totals.setdefault(usage.provider, ProviderUsage(provider=usage.provider))
        for field_name in (
            "requests_attempted", "requests_succeeded", "requests_failed", "retries",
            "quota_units", "estimated_billable_calls",
        ):
            setattr(total, field_name, getattr(total, field_name) + getattr(usage, field_name))
        total.estimated_cost += usage.estimated_cost
        total.quota_remaining = usage.quota_remaining

    def totals(self) -> dict[str, ProviderUsage]:
        return {name: value.model_copy(deep=True) for name, value in self._totals.items()}

    def run_total(self) -> ProviderUsage:
        result = ProviderUsage(provider="ALL")
        for usage in self._totals.values():
            result.requests_attempted += usage.requests_attempted
            result.requests_succeeded += usage.requests_succeeded
            result.requests_failed += usage.requests_failed
            result.retries += usage.retries
            result.quota_units += usage.quota_units
            result.estimated_billable_calls += usage.estimated_billable_calls
            result.estimated_cost += usage.estimated_cost
        return result
