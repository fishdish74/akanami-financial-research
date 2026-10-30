"""Primary-source discovery from the SEC submissions JSON endpoint."""

from __future__ import annotations

import json
from datetime import datetime, time, timezone

from pydantic import ValidationError

from config import SourceRequestConfig, require_sec_user_agent
from entities import Entity
from schemas import Candidate
from sources.base import HttpClient, utc_now


MATERIAL_FORMS = {"10-K", "10-Q", "8-K", "20-F", "6-K", "S-1", "DEF 14A"}


def parse_sec_submissions(
    payload: bytes,
    entity: Entity,
    limit: int,
    *,
    discovered_at: datetime | None = None,
) -> tuple[list[Candidate], list[str]]:
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SEC returned malformed submissions JSON") from exc
    recent = raw.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict):
        raise ValueError("SEC submissions JSON is missing filings.recent")
    keys = ("accessionNumber", "filingDate", "reportDate", "form", "primaryDocument")
    columns = {key: recent.get(key, []) for key in keys}
    if any(not isinstance(value, list) for value in columns.values()):
        raise ValueError("SEC submissions JSON contains malformed recent-filing columns")

    cik = (entity.cik or "").zfill(10)
    timestamp = discovered_at or utc_now()
    candidates: list[Candidate] = []
    warnings: list[str] = []
    row_count = max((len(value) for value in columns.values()), default=0)
    for position in range(row_count):
        try:
            accession = columns["accessionNumber"][position]
            filing_date = columns["filingDate"][position]
            report_date = columns["reportDate"][position]
            form = columns["form"][position]
            document = columns["primaryDocument"][position]
        except IndexError:
            warnings.append(f"skipped incomplete SEC filing row {position + 1}")
            continue
        if form not in MATERIAL_FORMS:
            continue
        try:
            published = datetime.combine(
                datetime.strptime(str(filing_date), "%Y-%m-%d").date(),
                time.min,
                tzinfo=timezone.utc,
            )
            accession_path = str(accession).replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession_path}/{document}"
            )
            description = (
                f"Official SEC {form} filing for {entity.canonical_name} "
                f"filed {filing_date}; reporting period {report_date or 'not specified'}."
            )
            candidate = Candidate(
                title=f"{entity.canonical_name} {form} filing",
                source="SEC",
                source_url=filing_url,
                published=published,
                discovered=timestamp,
                author=entity.canonical_name,
                content="",
                snippet=description,
                source_tier=1,
                companies=[entity.canonical_name],
                tickers=[entity.ticker],
                source_type="sec",
                content_scope="metadata_only",
            )
        except (ValueError, ValidationError):
            warnings.append(f"skipped malformed SEC filing row {position + 1}")
            continue
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates, warnings


class SecSubmissionsSource:
    name = "sec"

    def __init__(self, config: SourceRequestConfig, *, http: HttpClient | None = None) -> None:
        require_sec_user_agent(config.headers.get("User-Agent", ""))
        self.http = http or HttpClient(config)
        self.last_warnings: list[str] = []

    def discover(self, entity: Entity, limit: int) -> list[Candidate]:
        if not entity.cik:
            self.last_warnings = [f"SEC source skipped: no CIK configured for {entity.ticker}"]
            return []
        cik = entity.cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        candidates, warnings = parse_sec_submissions(self.http.get(url), entity, limit)
        self.last_warnings = warnings
        return candidates
