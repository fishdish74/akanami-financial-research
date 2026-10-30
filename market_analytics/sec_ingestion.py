"""Orchestrate SEC raw-source preservation, normalization, and idempotent database inserts."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from entities import Entity
from sources.sec_fundamentals import RawSecDocument, SecCompanyFactsClient

from .database import MarketDatabase
from .sec_xbrl import assess_company_facts_capability, parse_company_facts


def ingest_company_facts_payload(
    database: MarketDatabase,
    entity: Entity,
    payload: bytes,
    *,
    retrieved_at: datetime,
    source_url: str,
    request_identifier: str,
    raw_relative_path: str | None = None,
    processed_at: datetime | None = None,
) -> dict[str, Any]:
    if database.read_only:
        raise PermissionError("SEC ingestion requires a writable database")
    database.require_sec_schema()
    capability = assess_company_facts_capability(payload)
    cik, sec_name, facts, warnings = parse_company_facts(
        payload, retrieved_at=retrieved_at, capability=capability
    )
    if entity.cik and entity.cik.zfill(10) != cik:
        raise ValueError("SEC response CIK does not match the requested company")
    content_hash = hashlib.sha256(payload).hexdigest()
    with database.transaction() as connection:
        recorded_at = database._processing_timestamp(processed_at)
        company_id = database._upsert_company(
            connection,
            ticker=entity.ticker,
            canonical_name=entity.canonical_name or sec_name,
            exchange=entity.exchange,
            cik=cik,
            aliases=entity.aliases,
        )
        artifact_id = database._insert_source_artifact(
            connection,
            provider="sec",
            request_identifier=request_identifier,
            source_url=source_url,
            retrieved_at=retrieved_at.astimezone(timezone.utc).isoformat(),
            content_hash=content_hash,
            relative_path=raw_relative_path,
            byte_count=len(payload),
        )
        inserted, retired = database._reconcile_financial_facts(
            connection, company_id, facts, artifact_id, "SEC Company Facts",
            recorded_at=recorded_at,
        )
    return {
        "ticker": entity.ticker,
        "cik": cik,
        "sec_entity_name": sec_name,
        "facts_normalized": len(facts),
        "facts_inserted": inserted,
        "facts_retired": retired,
        "duplicates_ignored": len(facts) - inserted,
        "warnings": warnings,
        "content_hash": content_hash,
        "source_url": source_url,
    }


def sync_sec_company(
    database: MarketDatabase,
    entity: Entity,
    client: SecCompanyFactsClient,
    *,
    cache_root: Path,
) -> dict[str, Any]:
    if database.read_only:
        raise PermissionError("SEC ingestion requires a writable database")
    database.require_sec_schema()
    document: RawSecDocument = client.company_facts(entity)
    try:
        relative_path = str(document.path.relative_to(cache_root))
    except ValueError:
        relative_path = None
    result = ingest_company_facts_payload(
        database,
        entity,
        document.payload,
        retrieved_at=document.retrieved_at,
        source_url=document.source_url,
        request_identifier=document.request_identifier,
        raw_relative_path=relative_path,
    )
    result["from_cache"] = document.from_cache
    return result
