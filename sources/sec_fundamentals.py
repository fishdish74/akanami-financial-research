"""Rate-limited SEC Company Facts retrieval with immutable raw preservation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from config import SourceRequestConfig, require_sec_user_agent
from entities import Entity
from sources.base import HttpClient


def _require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class RawSecDocument:
    payload: bytes
    source_url: str
    request_identifier: str
    retrieved_at: datetime
    content_hash: str
    path: Path
    from_cache: bool


class RawSecCache:
    """Content-addressed raw bytes plus metadata; source responses are never replaced."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def _write_new(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            if temporary.exists():
                temporary.unlink()

    def put(
        self,
        *,
        payload: bytes,
        source_url: str,
        request_identifier: str,
        retrieved_at: datetime,
    ) -> RawSecDocument:
        retrieved_at = _require_aware(retrieved_at, "retrieved_at")
        digest = hashlib.sha256(payload).hexdigest()
        raw_path = self.root / "raw" / f"{digest}.json"
        metadata_path = self.root / "metadata" / f"{digest}.json"
        self._write_new(raw_path, payload)
        metadata = {
            "provider": "sec",
            "request_identifier": request_identifier,
            "source_url": source_url,
            "retrieved_at": retrieved_at.astimezone(timezone.utc).isoformat(),
            "content_hash": digest,
            "byte_count": len(payload),
            "raw_relative_path": str(raw_path.relative_to(self.root)),
        }
        self._write_new(
            metadata_path,
            json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        # The immutable content blob is deduplicated, while every retrieval is
        # retained as its own immutable audit event.
        event_identity = hashlib.sha256(
            f"{request_identifier}\0{retrieved_at.isoformat()}\0{digest}".encode("utf-8")
        ).hexdigest()
        event_path = self.root / "retrieval_events" / f"{event_identity}.json"
        self._write_new(
            event_path,
            json.dumps(
                {
                    "provider": "sec",
                    "request_identifier": request_identifier,
                    "source_url": source_url,
                    "retrieved_at": retrieved_at.isoformat(),
                    "content_hash": digest,
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8") + b"\n",
        )
        return RawSecDocument(
            payload=payload,
            source_url=source_url,
            request_identifier=request_identifier,
            retrieved_at=retrieved_at,
            content_hash=digest,
            path=raw_path,
            from_cache=False,
        )

    def latest(
        self,
        request_identifier: str,
        *,
        now: datetime,
        max_age: timedelta,
    ) -> RawSecDocument | None:
        now = _require_aware(now, "now")
        metadata_roots = [self.root / "metadata", self.root / "retrieval_events"]
        if not any(path.is_dir() for path in metadata_roots):
            return None
        matches: list[tuple[datetime, dict[str, object]]] = []
        for metadata_path in (
            path for root in metadata_roots if root.is_dir() for path in root.glob("*.json")
        ):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("request_identifier") != request_identifier:
                    continue
                retrieved_at = _require_aware(
                    datetime.fromisoformat(str(metadata["retrieved_at"])), "cached retrieved_at"
                )
                metadata.setdefault(
                    "raw_relative_path", f"raw/{metadata['content_hash']}.json"
                )
                matches.append((retrieved_at, metadata))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        if not matches:
            return None
        retrieved_at, metadata = max(matches, key=lambda item: item[0])
        if retrieved_at > now:
            raise ValueError("cached SEC retrieved_at is in the future")
        if now.astimezone(timezone.utc) - retrieved_at.astimezone(timezone.utc) > max_age:
            return None
        raw_path = self.root / str(metadata["raw_relative_path"])
        root = self.root.resolve()
        if not raw_path.resolve().is_relative_to(root):
            raise ValueError("cached SEC metadata points outside the cache root")
        try:
            payload = raw_path.read_bytes()
        except OSError:
            return None
        digest = hashlib.sha256(payload).hexdigest()
        if digest != metadata.get("content_hash"):
            raise ValueError("cached SEC raw document failed its content-hash check")
        return RawSecDocument(
            payload=payload,
            source_url=str(metadata["source_url"]),
            request_identifier=request_identifier,
            retrieved_at=retrieved_at,
            content_hash=digest,
            path=raw_path,
            from_cache=True,
        )


class SecCompanyFactsClient:
    """Fetch one company at a time; bulk SEC archives remain intentionally deferred."""

    def __init__(
        self,
        config: SourceRequestConfig,
        cache: RawSecCache,
        *,
        http: HttpClient | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        max_age: timedelta = timedelta(days=1),
    ) -> None:
        require_sec_user_agent(config.headers.get("User-Agent", ""))
        self.http = http or HttpClient(config)
        self.cache = cache
        self.now = now
        self.max_age = max_age

    def company_facts(self, entity: Entity) -> RawSecDocument:
        if not entity.cik:
            raise ValueError(f"SEC fundamentals require a CIK for {entity.ticker}")
        cik = entity.cik.zfill(10)
        request_identifier = f"companyfacts:CIK{cik}"
        current = self.now()
        cached = self.cache.latest(
            request_identifier, now=current, max_age=self.max_age
        )
        if cached is not None:
            return cached
        source_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        payload = self.http.get(source_url)
        # Validate top-level JSON before preserving a response as a usable artifact.
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("SEC returned malformed Company Facts JSON") from exc
        if not isinstance(parsed, dict) or "facts" not in parsed:
            raise ValueError("SEC Company Facts response is missing facts")
        return self.cache.put(
            payload=payload,
            source_url=source_url,
            request_identifier=request_identifier,
            retrieved_at=current,
        )
