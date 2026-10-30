"""Deterministic URL, identity, syndicated-copy, and lightweight event matching."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import yaml

from schemas import Candidate, DuplicateStatus


TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "ref",
    "referrer",
    "source",
    "campaign",
    "campaignid",
}
UPDATE_TERMS = {
    "updated",
    "revised",
    "raises",
    "raised",
    "cuts",
    "cut",
    "increase",
    "decrease",
    "approval",
    "approved",
    "rejected",
    "closes",
    "closed",
    "completed",
    "cancelled",
    "canceled",
    "settles",
    "settled",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
}
EVENT_KEYWORDS: dict[str, set[str]] = {
    "earnings": {"earnings", "results", "quarter", "revenue", "eps"},
    "guidance": {"guidance", "outlook", "forecast"},
    "acquisition": {"acquisition", "acquire", "merger", "deal"},
    "financing": {"financing", "debt", "offering", "issuance", "loan"},
    "management": {"ceo", "cfo", "executive", "resigns", "appointed"},
    "regulatory": {"regulator", "investigation", "antitrust", "approval", "sec"},
    "litigation": {"lawsuit", "litigation", "court", "settlement"},
    "product": {"launch", "product", "chip", "platform", "delay"},
    "contract": {"contract", "customer", "supplier", "partnership"},
    "restructuring": {"bankruptcy", "restructuring", "layoffs", "liquidity"},
    "filing": {"10-k", "10-q", "8-k", "20-f", "6-k", "s-1", "filing"},
}
DIRECTION_UP = {"rises", "rise", "rose", "gains", "gain", "up", "raises", "raised", "increase", "increases"}
DIRECTION_DOWN = {"falls", "fall", "fell", "drops", "drop", "down", "cuts", "cut", "decrease", "decreases"}


def normalize_url(url: str) -> str:
    """Canonicalize an HTTP(S) URL and remove known tracking parameters."""

    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower() or "https"
    if scheme not in {"http", "https"}:
        raise ValueError("only http and https source URLs are supported")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("source URL must include a hostname")
    try:
        host = host.encode("idna").decode("ascii")
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError("source URL contains an invalid hostname or port") from exc
    if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        host = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query: list[tuple[str, str]] = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.casefold()
        if lowered.startswith("utm_") or lowered in TRACKING_PARAMS:
            continue
        query.append((key, value))
    query.sort(key=lambda pair: (pair[0].casefold(), pair[1]))
    return urlunsplit((scheme, host, path, urlencode(query, doseq=True), ""))


def normalized_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def title_tokens(title: str) -> set[str]:
    return {token for token in normalized_title(title).split() if token not in STOPWORDS}


def token_similarity(first: str, second: str) -> float:
    first_tokens = title_tokens(first)
    second_tokens = title_tokens(second)
    union = first_tokens | second_tokens
    return len(first_tokens & second_tokens) / len(union) if union else 0.0


def candidate_id(candidate: Candidate) -> str:
    identity = normalize_url(str(candidate.source_url))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def content_hash(candidate: Candidate) -> Optional[str]:
    text = (candidate.content or candidate.snippet).strip()
    if not text:
        return None
    compact = " ".join(unicodedata.normalize("NFKC", text).split()).casefold()
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def event_types(title: str) -> set[str]:
    tokens = title_tokens(title)
    return {name for name, words in EVENT_KEYWORDS.items() if tokens & words}


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)(?:\$|€|£)?\d+(?:[.,]\d+)*(?:%|[mbkt])?", text.casefold()))


def _frontmatter(text: str) -> dict[str, object]:
    if not text.startswith("---\n"):
        return {}
    parts = text.split("---", 2)
    if len(parts) != 3:
        return {}
    try:
        value = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, list) else [value]
    return tuple(str(item).strip() for item in values if str(item).strip())


def _as_date(value: object) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class ResearchRecord:
    title: str
    source: str = ""
    published: Optional[date] = None
    companies: tuple[str, ...] = ()
    tickers: tuple[str, ...] = ()
    normalized_url: Optional[str] = None
    candidate_id: Optional[str] = None
    content_hash: Optional[str] = None

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> "ResearchRecord":
        return cls(
            title=candidate.title,
            source=candidate.source,
            published=candidate.published.date() if candidate.published else None,
            companies=tuple(value.casefold() for value in candidate.companies),
            tickers=tuple(value.upper() for value in candidate.tickers),
            normalized_url=normalize_url(str(candidate.source_url)),
            candidate_id=candidate_id(candidate),
            content_hash=content_hash(candidate),
        )


@dataclass
class ExistingResearchIndex:
    urls: set[str] = field(default_factory=set)
    candidate_ids: set[str] = field(default_factory=set)
    content_hashes: set[str] = field(default_factory=set)
    titles: set[str] = field(default_factory=set)
    records: list[ResearchRecord] = field(default_factory=list)

    @classmethod
    def scan(cls, vault_dir: Path) -> "ExistingResearchIndex":
        index = cls()
        for path in vault_dir.rglob("*.md"):
            relative = path.relative_to(vault_dir)
            if relative.parts and relative.parts[0] in {"Templates", "_tools"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            metadata = _frontmatter(text)
            raw_url = metadata.get("source_url")
            # Web Clipper commonly stores the original URL in source.
            if not raw_url and str(metadata.get("source", "")).startswith(("http://", "https://")):
                raw_url = metadata.get("source")
            normalized_url: Optional[str] = None
            if raw_url:
                try:
                    normalized_url = normalize_url(str(raw_url))
                    index.urls.add(normalized_url)
                except ValueError:
                    pass
            stored_id = str(metadata.get("candidate_id") or "").strip() or None
            stored_hash = str(metadata.get("content_hash") or "").strip() or None
            if stored_id:
                index.candidate_ids.add(stored_id)
            if stored_hash:
                index.content_hashes.add(stored_hash)
            title = str(metadata.get("title") or "").strip()
            if not title:
                heading = re.search(r"(?m)^#\s+(.+)$", text)
                title = heading.group(1).strip() if heading else path.stem
            index.titles.add(normalized_title(title))
            index.records.append(
                ResearchRecord(
                    title=title,
                    source=str(metadata.get("source") or "").strip(),
                    published=_as_date(metadata.get("published")),
                    companies=tuple(value.casefold() for value in _as_list(metadata.get("companies"))),
                    tickers=tuple(value.upper() for value in _as_list(metadata.get("tickers"))),
                    normalized_url=normalized_url,
                    candidate_id=stored_id,
                    content_hash=stored_hash,
                )
            )
        return index

    def exact_duplicate(self, candidate: Candidate) -> bool:
        return (
            normalize_url(str(candidate.source_url)) in self.urls
            or candidate_id(candidate) in self.candidate_ids
        )


def _same_entities(candidate: ResearchRecord, known: ResearchRecord) -> bool:
    if candidate.tickers and known.tickers:
        return bool(set(candidate.tickers) & set(known.tickers))
    if candidate.companies and known.companies:
        return bool(set(candidate.companies) & set(known.companies))
    return True  # Legacy notes may not contain normalized entity metadata.


def _within_event_window(candidate: ResearchRecord, known: ResearchRecord, days: int = 21) -> bool:
    if not candidate.published or not known.published:
        return True
    return abs((candidate.published - known.published).days) <= days


def _compatible_event_context(candidate: ResearchRecord, known: ResearchRecord) -> bool:
    if not _same_entities(candidate, known) or not _within_event_window(candidate, known):
        return False
    candidate_events = event_types(candidate.title)
    known_events = event_types(known.title)
    return not candidate_events or not known_events or bool(candidate_events & known_events)


def _meaningful_update(candidate: ResearchRecord, known: ResearchRecord) -> bool:
    tokens = title_tokens(candidate.title)
    if tokens & UPDATE_TERMS:
        return True
    candidate_numbers = _numbers(candidate.title)
    known_numbers = _numbers(known.title)
    return bool(candidate_numbers and known_numbers and candidate_numbers != known_numbers)


def _opposite_direction(candidate: ResearchRecord, known: ResearchRecord) -> bool:
    current = title_tokens(candidate.title)
    previous = title_tokens(known.title)
    return bool((current & DIRECTION_UP and previous & DIRECTION_DOWN) or
                (current & DIRECTION_DOWN and previous & DIRECTION_UP))


def _match_classification(current: ResearchRecord, known: ResearchRecord) -> DuplicateStatus:
    similarity = token_similarity(current.title, known.title)
    if normalized_title(current.title) == normalized_title(known.title):
        return "SYNDICATED_DUPLICATE"
    event_overlap = bool(event_types(current.title) & event_types(known.title))
    if event_overlap and (_opposite_direction(current, known) or _meaningful_update(current, known)) and similarity >= 0.35:
        return "SAME_EVENT_MEANINGFUL_UPDATE"
    if similarity >= 0.62 and event_overlap:
        if current.source and known.source and current.source.casefold() != known.source.casefold():
            return "SAME_EVENT_CONFIRMATION"
        return "SAME_EVENT_NO_NEW_INFO"
    if similarity >= 0.78:
        return "SAME_EVENT_CONFIRMATION"
    return "NEW_EVENT"


def classify_duplicate(
    candidate: Candidate,
    index: ExistingResearchIndex,
    run_candidates: Iterable[Candidate | str] = (),
) -> DuplicateStatus:
    """Classify using conservative deterministic signals; uncertainty stays new/reviewable."""

    if index.exact_duplicate(candidate):
        return "EXACT_DUPLICATE"
    current = ResearchRecord.from_candidate(candidate)

    known_records = list(index.records)
    for item in run_candidates:
        if isinstance(item, Candidate):
            known_records.append(ResearchRecord.from_candidate(item))
        else:
            known_records.append(ResearchRecord(title=str(item)))

    if any(
        (current.normalized_url and current.normalized_url == known.normalized_url)
        or (current.candidate_id and current.candidate_id == known.candidate_id)
        for known in known_records
    ):
        return "EXACT_DUPLICATE"
    if current.content_hash and any(
        current.content_hash == known.content_hash
        and _compatible_event_context(current, known)
        and not _opposite_direction(current, known)
        for known in known_records
    ):
        return "SYNDICATED_DUPLICATE"

    plausible = [
        known for known in known_records
        if _compatible_event_context(current, known)
    ]
    if not plausible:
        return "NEW_EVENT"
    # Evaluate every compatible record. The hierarchy favors preserving new
    # economic information over a superficially closer no-change headline.
    priority = {
        "SYNDICATED_DUPLICATE": 5,
        "SAME_EVENT_MEANINGFUL_UPDATE": 4,
        "SAME_EVENT_CONFIRMATION": 3,
        "SAME_EVENT_NO_NEW_INFO": 2,
        "NEW_EVENT": 1,
    }
    classified = [
        (
            priority[_match_classification(current, known)],
            token_similarity(current.title, known.title),
            normalized_title(known.title),
            known.source.casefold(),
            _match_classification(current, known),
        )
        for known in plausible
    ]
    return max(classified, key=lambda item: (item[0], item[1], item[2], item[3]))[4]
