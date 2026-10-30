"""Small content-addressed cache abstraction for future external providers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict
from pydantic import AwareDatetime, field_validator


class CacheMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    retrieval_timestamp: AwareDatetime
    expiry_timestamp: AwareDatetime
    request_identifier: str
    content_hash: str

    @field_validator("retrieval_timestamp", "expiry_timestamp", mode="after")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)


class CacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metadata: CacheMetadata
    payload: object


class CacheLookup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    entry: CacheEntry | None = None
    stale_by_seconds: float | None = None


class FileCache:
    """Optional JSON cache. Keys are hashed, payloads must never contain secrets."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def _identity(request_identifier: str) -> str:
        return hashlib.sha256(request_identifier.encode("utf-8")).hexdigest()

    def path_for(self, request_identifier: str) -> Path:
        return self.root / f"{self._identity(request_identifier)}.json"

    def _load_verified(self, request_identifier: str, current: datetime) -> Optional[CacheEntry]:
        path = self.path_for(request_identifier)
        if not path.is_file():
            return None
        entry = CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
        if entry.metadata.retrieval_timestamp > current:
            raise ValueError("cache retrieval timestamp is in the future")
        payload = json.dumps(entry.payload, sort_keys=True, separators=(",", ":"), default=str)
        actual_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if actual_hash != entry.metadata.content_hash:
            raise ValueError("cached payload failed its content-hash check")
        return entry

    def lookup(self, request_identifier: str, now: datetime | None = None) -> CacheLookup:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("cache lookup time must be timezone-aware")
        current = current.astimezone(timezone.utc)
        entry = self._load_verified(request_identifier, current)
        if entry is None:
            return CacheLookup(status="MISS")
        if entry.metadata.expiry_timestamp >= current:
            return CacheLookup(status="FRESH", entry=entry, stale_by_seconds=0)
        return CacheLookup(
            status="STALE", entry=entry,
            stale_by_seconds=(current - entry.metadata.expiry_timestamp).total_seconds(),
        )

    def get(self, request_identifier: str, now: datetime | None = None) -> Optional[CacheEntry]:
        lookup = self.lookup(request_identifier, now)
        return lookup.entry if lookup.status == "FRESH" else None

    def put(self, entry: CacheEntry) -> Path:
        if entry.metadata.expiry_timestamp < entry.metadata.retrieval_timestamp:
            raise ValueError("cache expiry cannot precede retrieval")
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(entry.payload, sort_keys=True, separators=(",", ":"), default=str)
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != entry.metadata.content_hash:
            raise ValueError("cache content hash does not match payload")
        path = self.path_for(entry.metadata.request_identifier)
        if path.exists():
            existing = self.get(
                entry.metadata.request_identifier,
                now=entry.metadata.retrieval_timestamp,
            )
            if existing is not None:
                raise FileExistsError("unexpired cache entry already exists")
        handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(entry.model_dump_json(indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path
