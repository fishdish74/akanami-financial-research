"""Google News RSS discovery; publisher access controls are never bypassed."""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from xml.etree import ElementTree

from pydantic import ValidationError

from config import SourceRequestConfig
from dedup import candidate_id
from entities import Entity
from schemas import Candidate
from sources.base import HttpClient, utc_now


HIGH_QUALITY = {"Reuters", "Bloomberg", "The Wall Street Journal", "Financial Times"}
SECONDARY = {"CNBC", "Barron's"}


class SourceParseError(ValueError):
    """A source payload could not be parsed at all."""


def _plain_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(html.unescape(without_tags).split())


def _publication_tier(publisher: str) -> int:
    if publisher in HIGH_QUALITY:
        return 2
    if publisher in SECONDARY:
        return 3
    return 4


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def parse_google_news_feed(
    payload: bytes,
    entity: Entity,
    limit: int,
    *,
    discovered_at: datetime | None = None,
) -> tuple[list[Candidate], list[str]]:
    """Parse valid items while returning deterministic warnings for malformed ones."""

    if limit <= 0:
        return [], []
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise SourceParseError("Google News returned malformed XML") from exc
    timestamp = discovered_at or utc_now()
    candidates: list[Candidate] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(root.findall("./channel/item"), 1):
        raw_title = item.findtext("title", default="").strip()
        headline, separator, publisher = raw_title.rpartition(" - ")
        if not separator:
            headline, publisher = raw_title, "Google News"
        link = item.findtext("link", default="").strip()
        description = _plain_text(item.findtext("description", default=""))
        try:
            searchable = f"{headline} {description}"
            mentioned = entity.appears_in(searchable)
            candidate = Candidate(
                title=headline,
                source=publisher,
                source_url=link,
                published=_parse_date(item.findtext("pubDate", default="").strip()),
                discovered=timestamp,
                author=None,
                content="",
                snippet=description,
                source_tier=_publication_tier(publisher),
                companies=[entity.canonical_name] if mentioned else [],
                tickers=[entity.ticker] if mentioned else [],
                source_type="google-news",
                content_scope="excerpt",
            )
        except ValidationError:
            warnings.append(f"skipped malformed Google News item {position}")
            continue
        identity = candidate_id(candidate)
        if identity in seen_ids:
            warnings.append(f"skipped duplicate Google News item {position}")
            continue
        seen_ids.add(identity)
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates, warnings


class GoogleNewsRssSource:
    name = "google-news"

    def __init__(self, config: SourceRequestConfig, *, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient(config)
        self.last_warnings: list[str] = []

    def discover(self, entity: Entity, limit: int) -> list[Candidate]:
        query = quote_plus(f'"{entity.canonical_name}" OR {entity.ticker} when:30d')
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        candidates, warnings = parse_google_news_feed(self.http.get(url), entity, limit)
        self.last_warnings = warnings
        return candidates
