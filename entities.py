"""Small watchlist-scoped company/entity registry with boundary-safe matching."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from identifiers import normalize_ticker


# Symbols which are ordinary English words or single letters require explicit
# market notation (or a company-name/alias mention). Keep this policy in one
# place so every source uses the same conservative rule.
COMMON_WORD_TICKERS = frozenset({"COST", "NOW", "ON", "ALL", "IT", "AI", "A", "T", "F"})


def normalize_entity_name(value: str) -> str:
    """Normalize case, Unicode, and punctuation without collapsing word boundaries."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        identity = normalize_entity_name(cleaned)
        if cleaned and identity not in seen:
            seen.add(identity)
            result.append(cleaned)
    return result


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    exchange: Optional[str] = None
    alternate_listings: list[str] = Field(default_factory=list)
    cik: Optional[str] = None

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> str:
        return normalize_ticker(value)

    @field_validator("canonical_name", mode="before")
    @classmethod
    def clean_name(cls, value: object) -> str:
        name = str(value).strip()
        if not name:
            raise ValueError("canonical_name cannot be empty")
        return name

    @field_validator("aliases", "alternate_listings", mode="before")
    @classmethod
    def clean_lists(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return _unique([value])
        if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
            raise ValueError("entity aliases must be a string or sequence")
        return _unique(value)

    @field_validator("exchange", mode="before")
    @classmethod
    def clean_exchange(cls, value: object) -> Optional[str]:
        exchange = str(value or "").strip().upper()
        return exchange or None

    @field_validator("cik", mode="before")
    @classmethod
    def normalize_cik(cls, value: object) -> Optional[str]:
        cik = str(value or "").strip()
        if not cik:
            return None
        if not cik.isdigit() or len(cik) > 10:
            raise ValueError("CIK must contain at most 10 digits")
        return cik.zfill(10)

    def searchable_names(self) -> list[str]:
        values = [self.ticker, self.canonical_name, *self.aliases, *self.alternate_listings]
        return _unique(normalize_entity_name(value) for value in values)

    def appears_in(self, text: str) -> bool:
        """Match normalized aliases as complete token sequences, not substrings."""

        haystack = f" {normalize_entity_name(text)} "
        names = self.searchable_names()
        normalized_ticker = normalize_entity_name(self.ticker)
        # Short/common-word tickers are only treated as mentions when the source
        # uses explicit ticker notation. Company names and aliases still match.
        if self.ticker in COMMON_WORD_TICKERS:
            names = [name for name in names if name != normalized_ticker]
            explicit = re.search(
                rf"(?<![A-Za-z0-9])(?:\$|[A-Z][A-Z0-9.-]*\s*:\s*){re.escape(self.ticker)}(?![A-Za-z0-9])|\({re.escape(self.ticker)}\)",
                text,
                flags=re.IGNORECASE,
            )
            if explicit:
                return True
        return any(f" {name} " in haystack for name in names)


class EntityRegistry:
    def __init__(self, entities: Iterable[Entity] = ()) -> None:
        self.entities = list(entities)
        identifiers: dict[str, str] = {}
        ciks: dict[str, str] = {}
        tickers: set[str] = set()
        for entity in self.entities:
            if entity.ticker in tickers:
                raise ValueError(f"duplicate entity ticker: {entity.ticker}")
            tickers.add(entity.ticker)
            for identifier in entity.searchable_names():
                owner = identifiers.get(identifier)
                if owner is not None and owner != entity.ticker:
                    raise ValueError(
                        f"ambiguous entity identifier {identifier!r}: {owner} and {entity.ticker}"
                    )
                identifiers[identifier] = entity.ticker
            if entity.cik:
                owner = ciks.get(entity.cik)
                if owner is not None and owner != entity.ticker:
                    raise ValueError(
                        f"CIK {entity.cik} is assigned to both {owner} and {entity.ticker}"
                    )
                ciks[entity.cik] = entity.ticker

    @classmethod
    def load(cls, path: Path) -> "EntityRegistry":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("entities", []), list):
            raise ValueError("entity registry must contain an entities list")
        return cls(Entity.model_validate(item) for item in raw.get("entities", []))

    def resolve(self, value: str) -> Optional[Entity]:
        needle = normalize_entity_name(value)
        for entity in self.entities:
            if needle in entity.searchable_names():
                return entity
        return None

    def normalize_mentions(self, text: str) -> list[Entity]:
        return [entity for entity in self.entities if entity.appears_in(text)]
