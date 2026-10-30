"""Minimal validation-only Massive adapter; no database or filesystem persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

from .capabilities import classify_corporate_action
from .models import CompanyMetadataRecord, CorporateActionRecord, PriceObservation
from .provider_runtime import (
    DatasetCapability,
    HttpResult,
    PaginatedResult,
    PaginationError,
    ProviderBoundaryError,
    ProviderPage,
    ProviderRequestController,
    RateLimitPolicy,
    StoragePolicy,
    decode_json_object,
    paginate,
)


MASSIVE_API_KEY_ENV = "MASSIVE_API_KEY"
MASSIVE_BASE_URL = "https://api.massive.com"
MASSIVE_HOST = "api.massive.com"
MASSIVE_VALIDATION_MAX_CALLS = 20
MASSIVE_VALIDATION_MAX_PAGES = 3
MASSIVE_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MASSIVE_ERROR_BODY_INSPECTION_BYTES = 8 * 1024
MASSIVE_ERROR_DETAIL_CHARS = 240

Transport = Callable[[str, Mapping[str, str], float, int], HttpResult]


@dataclass(frozen=True)
class SplitActionBatch:
    """One complete split-endpoint retrieval, including empty-result provenance."""

    requested_start: date
    requested_end: date
    retrieved_at: datetime
    actions: tuple[CorporateActionRecord, ...]
    page_source_ids: tuple[str, ...]
    pagination_complete: bool


def _safe_massive_error_detail(
    response: HttpResult, *, secrets: tuple[str, ...] = ()
) -> str | None:
    """Extract a small diagnostic from a JSON error envelope without echoing bodies."""

    body = response.body
    raw = body if isinstance(body, bytes) else body.encode("utf-8", errors="replace")
    if not raw or len(raw) > MASSIVE_ERROR_BODY_INSPECTION_BYTES:
        return None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    parts: list[str] = []
    for field in ("error", "message", "status", "request_id"):
        value = payload.get(field)
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            continue
        text = " ".join(str(value).split())
        if not text or re.search(r"<\s*/?\s*[A-Za-z][^>]*>", text):
            continue
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[redacted]")
        text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", text)
        text = re.sub(
            r"(?i)\b(authorization|bearer|api[_-]?key)\b\s*[:=]?\s*[^\s,;]+",
            r"\1 [redacted]",
            text,
        )
        text = re.sub(
            r"(?i)([?&](?:api[_-]?key|token|key)=)[^&\s]+",
            r"\1[redacted]",
            text,
        )
        remaining = MASSIVE_ERROR_DETAIL_CHARS - len("; ".join(parts))
        if remaining <= len(field) + 1:
            break
        parts.append(f"{field}={text}"[:remaining])
    detail = "; ".join(parts)
    return detail[:MASSIVE_ERROR_DETAIL_CHARS] or None


def massive_capabilities() -> dict[str, DatasetCapability]:
    """Return only documentation-verified validation capabilities.

    Account-specific entitlement, rate limits, correction history, PIT availability,
    and storage rights remain unverified until a live account is inspected.
    """

    unknown_storage = StoragePolicy(
        notes="Massive account terms and intended use must be reviewed before persistence"
    )
    retrieval_only = "retrieval-only; provider first-known/correction time unavailable"
    return {
        "metadata": DatasetCapability(
            dataset="Massive ticker overview",
            status="SUPPORTED",
            requires_auth=True,
            entitlement="documented for all Stocks plans; live account not yet verified",
            metadata_coverage=[
                "ticker", "name", "active", "primary_exchange", "cik", "sic",
                "list_date", "currency", "provider identifiers",
            ],
            point_in_time_quality=retrieval_only,
            storage_policy=unknown_storage,
        ),
        "daily_prices": DatasetCapability(
            dataset="Massive daily custom bars",
            status="SUPPORTED",
            requires_auth=True,
            entitlement="Stocks Basic documents two years; higher tiers extend history",
            historical_depth_days=730,
            pagination_style="bounded next_url on api.massive.com",
            adjusted_price_semantics="adjusted=false raw; adjusted=true split-adjusted only",
            correction_behavior="UNVERIFIED",
            point_in_time_quality=retrieval_only,
            storage_policy=unknown_storage,
        ),
        "splits": DatasetCapability(
            dataset="Massive splits",
            status="SUPPORTED",
            requires_auth=True,
            entitlement="Stocks Basic documents two years; paid tiers document all history",
            historical_depth_days=730,
            pagination_style="bounded next_url on api.massive.com",
            corporate_action_coverage=[
                "forward_split", "reverse_split", "stock_dividend"
            ],
            point_in_time_quality=retrieval_only,
            storage_policy=unknown_storage,
        ),
        "dividends": DatasetCapability(
            dataset="Massive dividends",
            status="SUPPORTED",
            requires_auth=True,
            entitlement="Stocks Basic documents two years; paid tiers document all history",
            historical_depth_days=730,
            pagination_style="bounded next_url on api.massive.com",
            corporate_action_coverage=["cash_dividend"],
            point_in_time_quality=retrieval_only,
            storage_policy=unknown_storage,
        ),
        "ticker_events": DatasetCapability(
            dataset="Massive ticker events",
            status="UNVERIFIED",
            requires_auth=True,
            entitlement="documented as experimental; live account access unverified",
            corporate_action_coverage=["ticker_change"],
            point_in_time_quality=retrieval_only,
            storage_policy=unknown_storage,
        ),
    }


def massive_api_key_is_configured() -> bool:
    return bool(os.getenv(MASSIVE_API_KEY_ENV, "").strip())


def _read_bounded(response: Any, maximum: int) -> bytes:
    payload = response.read(maximum + 1)
    if len(payload) > maximum:
        raise ProviderBoundaryError("Massive response exceeded the validation size limit")
    return payload


class _RejectRedirects(HTTPRedirectHandler):
    """Never issue a hidden second send or forward bearer credentials."""

    def http_error_302(self, req, fp, code, msg, headers):
        # Reject before urllib parses Location or constructs another Request.
        raise HTTPError(req.full_url, code, "unexpected_redirect", headers, fp)

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


def _default_transport(
    url: str, headers: Mapping[str, str], timeout: float, maximum: int
) -> HttpResult:
    request = Request(url, headers=dict(headers))
    try:
        with build_opener(_RejectRedirects()).open(request, timeout=timeout) as response:
            return HttpResult(
                status=int(response.status),
                body=_read_bounded(response, maximum),
                headers=dict(response.headers.items()),
            )
    except HTTPError as exc:
        try:
            if 300 <= exc.code < 400:
                # Neither Location nor an arbitrary redirect body belongs in logs.
                return HttpResult(status=exc.code, body=b"", headers={})
            body = _read_bounded(exc, maximum)
            headers_out = dict(exc.headers.items()) if exc.headers else {}
            return HttpResult(status=exc.code, body=body, headers=headers_out)
        finally:
            exc.close()
    except (URLError, TimeoutError, OSError) as exc:
        raise ProviderBoundaryError("Massive transport request failed") from exc


def _aware_utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _optional_date(value: object, field: str) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ProviderBoundaryError(f"Massive {field} is malformed") from exc


def _required_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ProviderBoundaryError(f"Massive record is missing required field: {field}")
    return value.strip()


def _required_number(record: Mapping[str, Any], field: str) -> Decimal:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ProviderBoundaryError(f"Massive record has invalid numeric field: {field}")
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ProviderBoundaryError(
            f"Massive record has invalid numeric field: {field}"
        ) from exc
    if not result.is_finite():
        raise ProviderBoundaryError(f"Massive record has invalid numeric field: {field}")
    return result


class MassiveValidationProvider:
    """Thin live adapter for controlled validation, never production persistence."""

    def __init__(
        self,
        api_key: str,
        *,
        transport: Transport = _default_transport,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        max_calls: int = MASSIVE_VALIDATION_MAX_CALLS,
        max_pages: int = MASSIVE_VALIDATION_MAX_PAGES,
        timeout_seconds: float = 15.0,
        max_response_bytes: int = MASSIVE_MAX_RESPONSE_BYTES,
        requests_per_minute: int = 5,
    ) -> None:
        key = api_key.strip()
        if not key:
            raise ValueError(f"{MASSIVE_API_KEY_ENV} is required")
        if not 1 <= max_calls <= MASSIVE_VALIDATION_MAX_CALLS:
            raise ValueError("Massive validation call cap must be between 1 and 20")
        if not 1 <= max_pages <= MASSIVE_VALIDATION_MAX_PAGES:
            raise ValueError("Massive validation page cap must be between 1 and 3")
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ValueError("Massive validation transport limits must be positive")
        self._api_key = key
        self._transport = transport
        self._now = now
        self._timeout = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._max_pages = max_pages
        self._metadata_cache: dict[str, CompanyMetadataRecord] = {}
        self.controller = ProviderRequestController(
            provider="massive-validation",
            policy=RateLimitPolicy(
                requests_per_minute=requests_per_minute,
                daily_quota=max_calls,
                max_retries=2,
                base_backoff_seconds=1.0,
                max_backoff_seconds=10.0,
                max_retry_after_seconds=10.0,
            ),
            clock=clock,
            utc_now=now,
            sleeper=sleeper,
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "MassiveValidationProvider":
        return cls(os.getenv(MASSIVE_API_KEY_ENV, ""), **kwargs)

    @property
    def usage(self) -> Any:
        return self.controller.usage.model_copy(deep=True)

    @staticmethod
    def _url(path: str, parameters: Mapping[str, object] | None = None) -> str:
        query = urlencode(
            [(key, str(value).lower() if isinstance(value, bool) else str(value))
             for key, value in (parameters or {}).items() if value is not None]
        )
        return urlunsplit(("https", MASSIVE_HOST, path, query, ""))

    @staticmethod
    def _validated_next_url(value: object, allowed_path_prefix: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise PaginationError("Massive next_url must be a non-empty URL")
        parsed = urlsplit(value.strip())
        path_allowed = parsed.path == allowed_path_prefix or (
            allowed_path_prefix.endswith("/")
            and parsed.path.startswith(allowed_path_prefix)
        )
        if (
            parsed.scheme != "https"
            or parsed.hostname != MASSIVE_HOST
            or parsed.port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or not path_allowed
        ):
            raise PaginationError("Massive pagination URL escaped its allowed endpoint")
        if any(key.casefold() == "apikey" for key, _ in parse_qsl(parsed.query)):
            raise PaginationError("Massive pagination URL unexpectedly contained a credential")
        return parsed.geturl()

    def _request_json(
        self, url: str, *, allowed_statuses: frozenset[str] = frozenset({"OK"})
    ) -> tuple[dict[str, Any], datetime]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "MarketResearchCollector/validation-only",
        }
        response = self.controller.request(
            lambda: self._transport(
                url, headers, self._timeout, self._max_response_bytes
            ),
            error_detail=lambda result: _safe_massive_error_detail(
                result, secrets=(self._api_key,)
            ),
        )
        body_size = len(response.body) if isinstance(response.body, bytes) else len(
            response.body.encode("utf-8")
        )
        if body_size > self._max_response_bytes:
            raise ProviderBoundaryError("Massive response exceeded the validation size limit")
        payload = decode_json_object(response)
        status = payload.get("status")
        if status is not None and status not in allowed_statuses:
            raise ProviderBoundaryError("Massive response status was not an allowed success state")
        return payload, _aware_utc(self._now(), "retrieved_at")

    def _paged_items(
        self,
        initial_url: str,
        *,
        path_prefix: str,
        stable_key: Callable[[dict[str, Any]], object] | None = None,
        context: Mapping[str, Any] | None = None,
        allowed_statuses: frozenset[str] = frozenset({"OK"}),
    ) -> tuple[PaginatedResult, datetime]:
        retrieved: list[datetime] = []

        def fetch(next_url: str | None) -> ProviderPage:
            payload, retrieved_at = self._request_json(
                next_url or initial_url, allowed_statuses=allowed_statuses
            )
            retrieved.append(retrieved_at)
            results = payload.get("results")
            if not isinstance(results, list):
                raise ProviderBoundaryError("Massive response results must be an array")
            if any(not isinstance(item, dict) for item in results):
                raise ProviderBoundaryError("Massive response contained a non-object record")
            return ProviderPage(
                items=results,
                next_cursor=self._validated_next_url(payload.get("next_url"), path_prefix),
                provenance={
                    "request_id": payload.get("request_id"),
                    "response_status": payload.get("status"),
                    **dict(context or {}),
                },
            )

        result = paginate(fetch, max_pages=self._max_pages, stable_key=stable_key)
        if not result.complete:
            raise PaginationError("Massive pagination did not prove exhaustion")
        return result, max(retrieved)

    def metadata(
        self, ticker: str, input_cutoff: datetime
    ) -> list[CompanyMetadataRecord]:
        cutoff = _aware_utc(input_cutoff, "input_cutoff")
        symbol = ticker.strip().upper()
        payload, retrieved_at = self._request_json(
            self._url(f"/v3/reference/tickers/{symbol}")
        )
        raw = payload.get("results")
        if not isinstance(raw, dict):
            raise ProviderBoundaryError("Massive ticker results must be an object")
        actual_ticker = _required_text(raw, "ticker").upper()
        if actual_ticker != symbol:
            raise ProviderBoundaryError("Massive ticker response did not match the request")
        company_name = _required_text(raw, "name")
        active = raw.get("active")
        if active is not None and not isinstance(active, bool):
            raise ProviderBoundaryError("Massive ticker active status is malformed")
        currency = str(raw.get("currency_name") or "").strip().upper() or None
        if currency is not None and len(currency) != 3:
            raise ProviderBoundaryError("Massive ticker currency is malformed")
        provider_identity = (
            str(raw.get("composite_figi") or "").strip()
            or str(raw.get("share_class_figi") or "").strip()
            or actual_ticker
        )
        type_map = {
            "CS": "common_stock",
            "PFD": "preferred_stock",
            "ADRC": "adr",
            "ADRP": "adr",
            "ETF": "etf",
            "FUND": "fund",
            "REIT": "reit",
            "WARRANT": "warrant",
        }
        record = CompanyMetadataRecord(
            ticker=actual_ticker,
            company_name=company_name,
            cik=raw.get("cik"),
            exchange=raw.get("primary_exchange"),
            industry=raw.get("sic_description"),
            sic=raw.get("sic_code"),
            listing_date=_optional_date(raw.get("list_date"), "list_date"),
            active=active,
            listing_type=type_map.get(str(raw.get("type") or "").upper()),
            source="Massive ticker overview",
            source_record_id=f"massive:ticker:{provider_identity}",
            effective_from=retrieved_at,
            observed_at=retrieved_at,
            available_at=retrieved_at,
            retrieved_at=retrieved_at,
            provenance={
                "provider": "Massive",
                "endpoint": "/v3/reference/tickers/{ticker}",
                "request_id": payload.get("request_id"),
                "provider_last_updated_utc": raw.get("last_updated_utc"),
                "currency": currency,
                "composite_figi": raw.get("composite_figi"),
                "share_class_figi": raw.get("share_class_figi"),
                "pit_availability": "retrieval_only_provider_first_known_time_unavailable",
            },
        )
        self._metadata_cache[symbol] = record
        return [record] if record.available_at <= cutoff else []

    def _price_basis(
        self, ticker: str, start: date, end: date, *, adjusted: bool
    ) -> tuple[PaginatedResult, datetime]:
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        return self._paged_items(
            self._url(path, {"adjusted": adjusted, "sort": "asc", "limit": 5000}),
            path_prefix=f"/v2/aggs/ticker/{ticker}/range/1/day/",
            stable_key=lambda item: item.get("t", -1),
            context={"adjusted": adjusted},
            # Aggregate responses on delayed-data plans can validly carry
            # status=DELAYED while returning a complete results array.
            allowed_statuses=frozenset({"OK", "DELAYED"}),
        )

    def prices(
        self, ticker: str, start: date, end: date, input_cutoff: datetime, *,
        admit_current_retrieval: bool = False,
        known_currency: str | None = None,
        known_exchange: str | None = None,
    ) -> list[PriceObservation]:
        """Retrieve daily prices while preserving retrieval-only availability.

        ``admit_current_retrieval`` is reserved for the bounded due-evaluation
        wrapper.  It returns only records produced by this invocation even when
        their truthful retrieval timestamp is just after ``input_cutoff``; the
        wrapper then applies its stricter authorized-attempt boundary.
        """
        if end < start:
            raise ValueError("price range end precedes start")
        cutoff = _aware_utc(input_cutoff, "input_cutoff")
        symbol = ticker.strip().upper()
        if known_currency is not None and known_exchange is not None:
            metadata = None
            currency = known_currency
            exchange = known_exchange
        else:
            metadata = self._metadata_cache.get(symbol)
            if metadata is None:
                metadata_rows = self.metadata(
                    symbol, datetime.max.replace(tzinfo=timezone.utc)
                )
                metadata = metadata_rows[0]
            currency = metadata.provenance.get("currency")
            exchange = metadata.exchange
        if not isinstance(currency, str) or not currency.strip():
            raise ProviderBoundaryError("Massive ticker currency is unavailable for price mapping")
        currency = currency.strip().upper()
        if not isinstance(exchange, str) or not exchange.strip():
            raise ProviderBoundaryError("Massive ticker exchange is unavailable for price mapping")
        exchange = exchange.strip().upper()
        raw_page, raw_retrieved = self._price_basis(symbol, start, end, adjusted=False)
        adjusted_page, adjusted_retrieved = self._price_basis(
            symbol, start, end, adjusted=True
        )
        retrieved_at = max(raw_retrieved, adjusted_retrieved)

        def by_timestamp(items: list[dict[str, Any]], basis: str) -> dict[int, dict[str, Any]]:
            output: dict[int, dict[str, Any]] = {}
            for item in items:
                timestamp = item.get("t")
                if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                    raise ProviderBoundaryError(f"Massive {basis} bar timestamp is malformed")
                if timestamp in output:
                    raise ProviderBoundaryError(f"Massive {basis} bars contain duplicate timestamps")
                output[timestamp] = item
            return output

        raw = by_timestamp(raw_page.items, "raw")
        adjusted = by_timestamp(adjusted_page.items, "adjusted")
        if set(raw) != set(adjusted):
            raise ProviderBoundaryError("Massive raw and adjusted bars cover different timestamps")
        eastern = ZoneInfo("America/New_York")
        output: list[PriceObservation] = []
        seen_dates: set[date] = set()
        for timestamp in sorted(raw):
            item = raw[timestamp]
            adjusted_item = adjusted[timestamp]
            trading_date = datetime.fromtimestamp(
                timestamp / 1000, tz=timezone.utc
            ).astimezone(eastern).date()
            if trading_date in seen_dates:
                raise ProviderBoundaryError("Massive daily bars contain duplicate trading dates")
            seen_dates.add(trading_date)
            raw_close = _required_number(item, "c")
            adjusted_close = _required_number(adjusted_item, "c")
            record = PriceObservation(
                ticker=symbol,
                trading_date=trading_date,
                open=_required_number(item, "o"),
                high=_required_number(item, "h"),
                low=_required_number(item, "l"),
                close=raw_close,
                split_adjusted_close=adjusted_close,
                adjustment_origin="provider_supplied",
                volume=_required_number(item, "v"),
                currency=currency,
                exchange=exchange,
                source="Massive custom bars",
                source_record_id=f"massive:daily:{symbol}:{trading_date.isoformat()}:raw+split",
                observed_at=retrieved_at,
                available_at=retrieved_at,
                retrieved_at=retrieved_at,
                provenance={
                    "provider": "Massive",
                    "endpoint": "/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}",
                    "raw_adjusted_parameter": False,
                    "split_adjusted_parameter": True,
                    "dividends_in_adjusted_series": False,
                    "raw_timestamp_ms": timestamp,
                    "raw_volume": str(item.get("v")),
                    "adjusted_volume": str(adjusted_item.get("v")),
                    "volume_adjustment_semantics": "provider_not_explicitly_verified",
                    "volume_capability_reason_code": "adjusted_volume_unverified",
                    "pit_availability": "retrieval_only_provider_correction_history_unavailable",
                    "raw_pages": [page.model_dump(mode="json") for page in raw_page.pages],
                    "adjusted_pages": [
                        page.model_dump(mode="json") for page in adjusted_page.pages
                    ],
                },
            )
            if record.available_at <= cutoff or admit_current_retrieval:
                output.append(record)
        return output

    def split_actions(
        self, ticker: str, start: date, end: date, input_cutoff: datetime, *,
        admit_current_retrieval: bool = False,
    ) -> SplitActionBatch:
        """Retrieve only split-like actions and retain proof of empty coverage."""

        if end < start:
            raise ValueError("split range end precedes start")
        cutoff = _aware_utc(input_cutoff, "input_cutoff")
        symbol = ticker.strip().upper()
        splits, retrieved_at = self._action_items(
            "/stocks/v1/splits",
            {
                "ticker": symbol,
                "execution_date.gte": start.isoformat(),
                "execution_date.lte": end.isoformat(),
                "sort": "execution_date.asc",
                "limit": 100,
            },
            stable_field="execution_date",
        )
        split_type = {
            "forward_split": "stock_split",
            "reverse_split": "reverse_split",
            "stock_dividend": "stock_dividend",
        }
        actions: list[CorporateActionRecord] = []
        for item in splits.items:
            actual_ticker = _required_text(item, "ticker").upper()
            if actual_ticker != symbol:
                raise ProviderBoundaryError("Massive split ticker did not match the request")
            adjustment_type = _required_text(item, "adjustment_type")
            if adjustment_type not in split_type:
                raise ProviderBoundaryError("Massive split adjustment type is unsupported")
            split_from = _required_number(item, "split_from")
            split_to = _required_number(item, "split_to")
            if split_from <= 0 or split_to <= 0:
                raise ProviderBoundaryError("Massive split ratio is invalid")
            execution_date = _optional_date(item.get("execution_date"), "execution_date")
            if execution_date is None:
                raise ProviderBoundaryError("Massive split execution_date is required")
            identity = str(item.get("id") or "").strip() or hashlib.sha256(
                json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            action_classification = classify_corporate_action(adjustment_type)
            record = CorporateActionRecord(
                ticker=symbol,
                event_type=split_type[adjustment_type],
                effective_date=execution_date,
                ratio=split_to / split_from,
                source="Massive splits",
                source_record_id=f"massive:split:{identity}",
                observed_at=retrieved_at,
                available_at=retrieved_at,
                retrieved_at=retrieved_at,
                provenance={
                    "provider": "Massive",
                    "endpoint": "/stocks/v1/splits",
                    "split_from": str(split_from),
                    "split_to": str(split_to),
                    "adjustment_type": adjustment_type,
                    "action_classification": action_classification.model_dump(
                        mode="json", exclude_none=False
                    ),
                    "historical_adjustment_factor": item.get("historical_adjustment_factor"),
                    "pit_availability": "retrieval_only_provider_first_known_time_unavailable",
                },
            )
            if record.available_at <= cutoff or admit_current_retrieval:
                actions.append(record)
        source_ids = tuple(sorted(
            f"massive:split-page:{page.content_hash}" for page in splits.pages
        ))
        return SplitActionBatch(
            requested_start=start,
            requested_end=end,
            retrieved_at=retrieved_at,
            actions=tuple(sorted(actions, key=lambda item: (
                item.effective_date or start, item.source_record_id
            ))),
            page_source_ids=source_ids,
            pagination_complete=splits.complete,
        )

    def _action_items(
        self, path: str, parameters: Mapping[str, object], *, stable_field: str
    ) -> tuple[PaginatedResult, datetime]:
        return self._paged_items(
            self._url(path, parameters),
            path_prefix=path,
            stable_key=lambda item: item.get(stable_field, ""),
        )

    def corporate_actions(
        self, ticker: str, start: date, end: date, input_cutoff: datetime
    ) -> list[CorporateActionRecord]:
        if end < start:
            raise ValueError("corporate-action range end precedes start")
        cutoff = _aware_utc(input_cutoff, "input_cutoff")
        symbol = ticker.strip().upper()
        splits, split_retrieved = self._action_items(
            "/stocks/v1/splits",
            {
                "ticker": symbol,
                "execution_date.gte": start.isoformat(),
                "execution_date.lte": end.isoformat(),
                "sort": "execution_date.asc",
                "limit": 100,
            },
            stable_field="execution_date",
        )
        dividends, dividend_retrieved = self._action_items(
            "/stocks/v1/dividends",
            {
                "ticker": symbol,
                "ex_dividend_date.gte": start.isoformat(),
                "ex_dividend_date.lte": end.isoformat(),
                "sort": "ex_dividend_date.asc",
                "limit": 100,
            },
            stable_field="ex_dividend_date",
        )
        output: list[CorporateActionRecord] = []
        split_type = {
            "forward_split": "stock_split",
            "reverse_split": "reverse_split",
            "stock_dividend": "stock_dividend",
        }
        for item in splits.items:
            actual_ticker = _required_text(item, "ticker").upper()
            if actual_ticker != symbol:
                raise ProviderBoundaryError("Massive split ticker did not match the request")
            adjustment_type = _required_text(item, "adjustment_type")
            if adjustment_type not in split_type:
                raise ProviderBoundaryError("Massive split adjustment type is unsupported")
            split_from = _required_number(item, "split_from")
            split_to = _required_number(item, "split_to")
            if split_from <= 0 or split_to <= 0:
                raise ProviderBoundaryError("Massive split ratio is invalid")
            execution_date = _optional_date(item.get("execution_date"), "execution_date")
            if execution_date is None:
                raise ProviderBoundaryError("Massive split execution_date is required")
            identity = str(item.get("id") or "").strip() or hashlib.sha256(
                json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            action_classification = classify_corporate_action(adjustment_type)
            output.append(CorporateActionRecord(
                ticker=symbol,
                event_type=split_type[adjustment_type],
                effective_date=execution_date,
                ratio=split_to / split_from,
                source="Massive splits",
                source_record_id=f"massive:split:{identity}",
                observed_at=split_retrieved,
                available_at=split_retrieved,
                retrieved_at=split_retrieved,
                provenance={
                    "provider": "Massive",
                    "endpoint": "/stocks/v1/splits",
                    "split_from": str(split_from),
                    "split_to": str(split_to),
                    "adjustment_type": adjustment_type,
                    "action_classification": action_classification.model_dump(
                        mode="json", exclude_none=False
                    ),
                    "historical_adjustment_factor": item.get("historical_adjustment_factor"),
                    "pit_availability": "retrieval_only_provider_first_known_time_unavailable",
                },
            ))
        for item in dividends.items:
            actual_ticker = _required_text(item, "ticker").upper()
            if actual_ticker != symbol:
                raise ProviderBoundaryError("Massive dividend ticker did not match the request")
            identity = str(item.get("id") or "").strip() or hashlib.sha256(
                json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            output.append(CorporateActionRecord(
                ticker=symbol,
                event_type="cash_dividend",
                declaration_date=_optional_date(item.get("declaration_date"), "declaration_date"),
                record_date=_optional_date(item.get("record_date"), "record_date"),
                ex_date=_optional_date(item.get("ex_dividend_date"), "ex_dividend_date"),
                payment_date=_optional_date(item.get("pay_date"), "pay_date"),
                amount=_required_number(item, "cash_amount"),
                currency=_required_text(item, "currency"),
                source="Massive dividends",
                source_record_id=f"massive:dividend:{identity}",
                observed_at=dividend_retrieved,
                available_at=dividend_retrieved,
                retrieved_at=dividend_retrieved,
                provenance={
                    "provider": "Massive",
                    "endpoint": "/stocks/v1/dividends",
                    "distribution_type": item.get("distribution_type"),
                    "frequency": item.get("frequency"),
                    "split_adjusted_cash_amount": item.get("split_adjusted_cash_amount"),
                    "historical_adjustment_factor": item.get("historical_adjustment_factor"),
                    "pit_availability": "retrieval_only_provider_first_known_time_unavailable",
                },
            ))
        return sorted(
            [item for item in output if item.available_at <= cutoff],
            key=lambda item: (
                item.effective_date or item.ex_date or item.record_date
                or item.declaration_date or item.payment_date or start,
                item.source_record_id,
            ),
        )

    def ticker_events(
        self, ticker: str, input_cutoff: datetime
    ) -> list[CorporateActionRecord]:
        cutoff = _aware_utc(input_cutoff, "input_cutoff")
        symbol = ticker.strip().upper()
        payload, retrieved_at = self._request_json(
            self._url(f"/vX/reference/tickers/{symbol}/events", {"types": "ticker_change"})
        )
        results = payload.get("results")
        if not isinstance(results, dict):
            raise ProviderBoundaryError("Massive ticker-event results must be an object")
        events = results.get("events", [])
        if not isinstance(events, list) or any(not isinstance(item, dict) for item in events):
            raise ProviderBoundaryError("Massive ticker events must be an array of objects")
        changes: list[tuple[date, str]] = []
        for item in events:
            if item.get("type") != "ticker_change":
                continue
            detail = item.get("ticker_change")
            if not isinstance(detail, dict):
                raise ProviderBoundaryError("Massive ticker-change detail is malformed")
            event_date = _optional_date(item.get("date"), "ticker event date")
            if event_date is None:
                raise ProviderBoundaryError("Massive ticker-change date is required")
            changes.append((event_date, _required_text(detail, "ticker").upper()))
        changes.sort()
        output: list[CorporateActionRecord] = []
        for (_, old_ticker), (event_date, new_ticker) in zip(changes, changes[1:]):
            output.append(CorporateActionRecord(
                ticker=new_ticker,
                event_type="ticker_change",
                effective_date=event_date,
                old_ticker=old_ticker,
                new_ticker=new_ticker,
                source="Massive ticker events (experimental)",
                source_record_id=f"massive:ticker-change:{old_ticker}:{new_ticker}:{event_date}",
                observed_at=retrieved_at,
                available_at=retrieved_at,
                retrieved_at=retrieved_at,
                provenance={
                    "provider": "Massive",
                    "endpoint": "/vX/reference/tickers/{id}/events",
                    "experimental": True,
                    "request_id": payload.get("request_id"),
                    "pit_availability": "retrieval_only_provider_first_known_time_unavailable",
                },
            ))
        return [item for item in output if item.available_at <= cutoff]
