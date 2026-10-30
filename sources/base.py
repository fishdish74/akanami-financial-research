"""Shared request behavior with source-specific limits and testable retries."""

from __future__ import annotations

import gzip
import io
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Callable, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from config import SourceRequestConfig
from entities import Entity
from schemas import Candidate


class CandidateSource(Protocol):
    name: str

    def discover(self, entity: Entity, limit: int) -> Sequence[Candidate]:
        ...


class SourceRequestError(RuntimeError):
    """Sanitized network failure safe for summaries and logs."""


# Company Facts payloads can be materially larger than RSS/submissions responses. These
# ceilings allow normal SEC documents while bounding both inbound bytes and gzip expansion.
DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024


def _safe_endpoint(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or "invalid-host"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}{parsed.path}"


class HttpClient:
    def __init__(
        self,
        config: SourceRequestConfig,
        *,
        opener: Callable[..., object] = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_decompressed_bytes: int = DEFAULT_MAX_DECOMPRESSED_BYTES,
    ) -> None:
        if max_response_bytes <= 0 or max_decompressed_bytes <= 0:
            raise ValueError("HTTP response size limits must be positive")
        self.config = config
        self._opener = opener
        self._sleep = sleeper
        self._monotonic = monotonic
        self._last_request_at: float | None = None
        self.max_response_bytes = max_response_bytes
        self.max_decompressed_bytes = max_decompressed_bytes

    def get(self, url: str) -> bytes:
        error: Exception | None = None
        safe_endpoint = _safe_endpoint(url)
        for attempt in range(self.config.retries + 1):
            if self._last_request_at is not None:
                elapsed = self._monotonic() - self._last_request_at
                if elapsed < self.config.delay_seconds:
                    self._sleep(self.config.delay_seconds - elapsed)
            request = Request(url, headers=self.config.headers)
            self._last_request_at = self._monotonic()
            try:
                with self._opener(request, timeout=self.config.timeout_seconds) as response:  # type: ignore[attr-defined]
                    payload = response.read(self.max_response_bytes + 1)
                    if len(payload) > self.max_response_bytes:
                        raise SourceRequestError(
                            f"response size exceeded {self.max_response_bytes} bytes for {safe_endpoint}"
                        )
                    if response.headers.get("Content-Encoding", "").casefold() == "gzip":
                        try:
                            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
                                payload = stream.read(self.max_decompressed_bytes + 1)
                        except (EOFError, OSError) as exc:
                            raise SourceRequestError(
                                f"malformed gzip response for {safe_endpoint}"
                            ) from exc
                        if len(payload) > self.max_decompressed_bytes:
                            raise SourceRequestError(
                                "decompressed response size exceeded "
                                f"{self.max_decompressed_bytes} bytes for {safe_endpoint}"
                            )
                    return payload
            except HTTPError as exc:
                error = exc
                retryable = exc.code in {408, 425, 429} or 500 <= exc.code <= 599
                if not retryable or attempt >= self.config.retries:
                    exc.close()
                    break
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = 0.5 * (attempt + 1)
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        try:
                            parsed = parsedate_to_datetime(retry_after)
                            delay = max(0.0, parsed.timestamp() - time.time())
                        except (TypeError, ValueError, OverflowError):
                            pass
                self._sleep(min(max(delay, 0.0), 30.0))
                exc.close()
            except (URLError, TimeoutError, OSError) as exc:
                error = exc
                if attempt < self.config.retries:
                    self._sleep(0.5 * (attempt + 1))
        error_name = type(error).__name__ if error else "unknown error"
        raise SourceRequestError(
            f"request failed after {self.config.retries + 1} attempt(s) "
            f"for {safe_endpoint} ({error_name})"
        ) from error


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
