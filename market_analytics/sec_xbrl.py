"""SEC Company Facts parsing into canonical, provenance-preserving financial facts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from .capabilities import (
    SecurityCapability,
    evaluate_security_capability,
    require_fundamental_normalization,
)


MAPPING_PATH = Path(__file__).with_name("sec_xbrl_mapping.yaml")
STANDARD_SEC_TAXONOMIES = frozenset(
    {"us-gaap", "dei", "srt", "invest", "ecd", "ffd"}
)
CORE_NORMALIZATION_METRICS = frozenset(
    {"revenue", "net_income", "operating_cash_flow", "total_assets"}
)


@dataclass(frozen=True)
class ConceptMapping:
    canonical_metric: str
    priority: int
    units: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class NormalizedFinancialFact:
    canonical_metric: str
    normalized_value: Decimal
    unit: str
    currency: str | None
    taxonomy: str
    source_concept: str
    source_unit: str
    source_value: Decimal
    source_fact_id: str
    mapping_version: str
    mapping_priority: int
    period_type: str
    start_date: str | None
    end_date: str
    fiscal_year: int | None
    fiscal_quarter: int | None
    filed_at: str
    available_at: str
    accession_number: str
    form: str
    is_amendment: bool
    quality_status: str
    warning: str | None
    retrieved_at: str
    source: str = "SEC Company Facts"
    formula_version: str | None = None
    derivation: dict[str, Any] | None = None


def load_concept_mapping(
    path: Path = MAPPING_PATH,
) -> tuple[str, dict[tuple[str, str], ConceptMapping]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("concepts"), dict):
        raise ValueError("SEC XBRL mapping must contain a concepts mapping")
    version = str(raw.get("mapping_version", "")).strip()
    if not version:
        raise ValueError("SEC XBRL mapping version is required")
    output: dict[tuple[str, str], ConceptMapping] = {}
    for taxonomy, concepts in raw["concepts"].items():
        if not isinstance(concepts, dict):
            raise ValueError(f"invalid mapping taxonomy {taxonomy}")
        taxonomy_name = str(taxonomy).strip()
        if not taxonomy_name:
            raise ValueError("mapping taxonomy cannot be empty")
        for concept, values in concepts.items():
            if not isinstance(values, dict):
                raise ValueError(f"invalid mapping for {taxonomy_name}:{concept}")
            mapping = ConceptMapping(
                canonical_metric=str(values["metric"]),
                priority=int(values["priority"]),
                units=tuple(str(item) for item in values["units"]),
                kind=str(values["kind"]),
            )
            if mapping.kind not in {"flow", "instant"} or mapping.priority < 1:
                raise ValueError(f"invalid mapping properties for {taxonomy_name}:{concept}")
            key = (taxonomy_name, str(concept))
            if key in output:
                raise ValueError(f"duplicate mapping for {taxonomy_name}:{concept}")
            output[key] = mapping
    return version, output


def _safe_decimal(value: object) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("XBRL fact value is not numeric") from exc
    if not math.isfinite(float(result)):
        raise ValueError("XBRL fact value must be finite")
    return result


def _period_type(raw: dict[str, Any], mapping: ConceptMapping) -> tuple[str, int | None] | None:
    form = str(raw.get("form", ""))
    fp = str(raw.get("fp", ""))
    if mapping.kind == "instant":
        return "POINT_IN_TIME", None
    start_text = raw.get("start")
    end_text = raw.get("end")
    if not start_text or not end_text:
        return None
    try:
        duration = (date.fromisoformat(str(end_text)) - date.fromisoformat(str(start_text))).days
    except ValueError:
        return None
    if form in {"10-K", "10-K/A"} and fp == "FY" and 300 <= duration <= 400:
        return "FY", None
    if form in {"10-Q", "10-Q/A"} and fp in {"Q1", "Q2", "Q3"} and 65 <= duration <= 120:
        return "Q", int(fp[1])
    return None


def _fact_identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def assess_company_facts_capability(
    payload: bytes,
    *,
    current_metadata_available: bool = True,
    active: bool | None = True,
    listing_type: str | None = "common_stock",
    foreign_issuer: bool = False,
    custom_taxonomy_replay_validated: bool = False,
) -> SecurityCapability:
    """Classify accounting/taxonomy support without normalizing any facts."""

    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SEC returned malformed Company Facts JSON") from exc
    facts = raw.get("facts") if isinstance(raw, dict) else None
    if not isinstance(facts, dict):
        raise ValueError("SEC Company Facts JSON is missing facts")
    taxonomies = {str(item) for item in facts}
    if "ifrs-full" in taxonomies:
        basis = "ifrs"
    elif "us-gaap" in taxonomies:
        basis = "us_gaap"
    else:
        basis = "unknown"
    custom = "custom" if taxonomies - STANDARD_SEC_TAXONOMIES else "standard"
    _, mappings = load_concept_mapping()
    standard_coverage = {
        mapping.canonical_metric
        for (taxonomy, concept), mapping in mappings.items()
        if taxonomy in facts
        and isinstance(facts[taxonomy], dict)
        and concept in facts[taxonomy]
    }
    materially_relies_on_custom = (
        custom == "custom"
        and len(standard_coverage & CORE_NORMALIZATION_METRICS) < 3
    )
    return evaluate_security_capability(
        current_metadata_available=current_metadata_available,
        active=active,
        listing_type=listing_type,
        foreign_issuer=foreign_issuer,
        fundamental_basis=basis,
        custom_taxonomy=custom,
        custom_taxonomy_replay_validated=(
            custom_taxonomy_replay_validated or not materially_relies_on_custom
        ),
        adjusted_volume_verified=False,
    )


def parse_company_facts(
    payload: bytes,
    *,
    retrieved_at: datetime,
    mapping_path: Path = MAPPING_PATH,
    capability: SecurityCapability | None = None,
) -> tuple[str, str, list[NormalizedFinancialFact], list[str]]:
    """Parse a Company Facts response without silently resolving conflicts."""

    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware")
    if capability is not None:
        require_fundamental_normalization(capability)

    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SEC returned malformed Company Facts JSON") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("facts"), dict):
        raise ValueError("SEC Company Facts JSON is missing facts")
    cik_raw = str(raw.get("cik", ""))
    if not cik_raw.isdigit():
        raise ValueError("SEC Company Facts JSON has an invalid CIK")
    cik = cik_raw.zfill(10)
    entity = str(raw.get("entityName", "")).strip()
    if not entity:
        raise ValueError("SEC Company Facts JSON is missing entityName")
    mapping_version, mappings = load_concept_mapping(mapping_path)
    retrieval = retrieved_at.astimezone(timezone.utc).isoformat()
    facts: list[NormalizedFinancialFact] = []
    warnings: list[str] = []
    unsupported_concepts: set[str] = set()
    seen: set[str] = set()

    for taxonomy, taxonomy_facts in sorted(raw["facts"].items()):
        if not isinstance(taxonomy_facts, dict):
            continue
        for concept, concept_payload in sorted(taxonomy_facts.items()):
            mapping = mappings.get((taxonomy, concept))
            if mapping is None:
                unsupported_concepts.add(f"{taxonomy}:{concept}")
                continue
            if not isinstance(concept_payload, dict):
                continue
            units = concept_payload.get("units", {})
            if not isinstance(units, dict):
                continue
            for source_unit in mapping.units:
                rows = units.get(source_unit, [])
                if not isinstance(rows, list):
                    continue
                for position, row in enumerate(rows):
                    if not isinstance(row, dict):
                        warnings.append(f"{concept}: skipped malformed fact row {position + 1}")
                        continue
                    form = str(row.get("form", ""))
                    if form not in {"10-K", "10-K/A", "10-Q", "10-Q/A"}:
                        continue
                    period = _period_type(row, mapping)
                    if period is None:
                        warnings.append(f"{concept}: skipped ambiguous cumulative or malformed period")
                        continue
                    try:
                        value = _safe_decimal(row.get("val"))
                        end_date = date.fromisoformat(str(row["end"])).isoformat()
                        filed_date = date.fromisoformat(str(row["filed"])).isoformat()
                    except (KeyError, ValueError) as exc:
                        warnings.append(f"{concept}: skipped malformed fact row {position + 1}: {exc}")
                        continue
                    accession = str(row.get("accn", "")).strip()
                    if not accession:
                        warnings.append(f"{concept}: skipped fact without accession number")
                        continue
                    period_type, fiscal_quarter = period
                    start_date = str(row.get("start")) if row.get("start") else None
                    fiscal_year = int(row["fy"]) if str(row.get("fy", "")).isdigit() else None
                    filed_at = f"{filed_date}T00:00:00+00:00"
                    # Company Facts provides a filing date, not an acceptance timestamp.
                    # End-of-day availability is conservative and prevents same-day leakage.
                    available_at = f"{filed_date}T23:59:59+00:00"
                    identity_payload = {
                        "cik": cik, "taxonomy": taxonomy, "concept": concept,
                        "unit": source_unit, "value": str(value), "start": start_date,
                        "end": end_date, "filed": filed_date, "accn": accession,
                        "form": form,
                    }
                    source_fact_id = _fact_identity(identity_payload)
                    if source_fact_id in seen:
                        continue
                    seen.add(source_fact_id)
                    unit = "raw" if source_unit in {"shares", "USD/shares"} else "raw"
                    currency = "USD" if source_unit in {"USD", "USD/shares"} else None
                    facts.append(
                        NormalizedFinancialFact(
                            canonical_metric=mapping.canonical_metric,
                            normalized_value=value,
                            unit=unit,
                            currency=currency,
                            taxonomy=taxonomy,
                            source_concept=concept,
                            source_unit=source_unit,
                            source_value=value,
                            source_fact_id=source_fact_id,
                            mapping_version=mapping_version,
                            mapping_priority=mapping.priority,
                            period_type=period_type,
                            start_date=start_date,
                            end_date=end_date,
                            fiscal_year=fiscal_year,
                            fiscal_quarter=fiscal_quarter,
                            filed_at=filed_at,
                            available_at=available_at,
                            accession_number=accession,
                            form=form,
                            is_amendment=form.endswith("/A"),
                            quality_status="valid",
                            warning=None,
                            retrieved_at=retrieval,
                        )
                    )

    groups: dict[tuple[str, str, str, str | None, str, str, str], list[int]] = {}
    for index, fact in enumerate(facts):
        key = (
            fact.taxonomy, fact.source_concept, fact.source_unit,
            fact.start_date, fact.end_date,
            fact.accession_number, fact.filed_at,
        )
        groups.setdefault(key, []).append(index)
    conflicting_indexes: set[int] = set()
    for indexes in groups.values():
        values = {facts[index].source_value for index in indexes}
        if len(values) > 1:
            conflicting_indexes.update(indexes)
    if conflicting_indexes:
        warnings.append(f"preserved {len(conflicting_indexes)} conflicting XBRL facts for review")
        facts = [
            replace(
                fact,
                quality_status="conflict",
                warning="conflicting values share the same filing and period context",
            ) if index in conflicting_indexes else fact
            for index, fact in enumerate(facts)
        ]

    if unsupported_concepts:
        warnings.append(
            f"unsupported taxonomy coverage: {len(unsupported_concepts)} concept(s) omitted"
        )
    facts.extend(_derive_facts(facts, mapping_version))
    return cik, entity, sorted(facts, key=lambda item: (item.end_date, item.canonical_metric, item.available_at, item.source_fact_id)), sorted(set(warnings))


def _derive_facts(
    facts: list[NormalizedFinancialFact], mapping_version: str
) -> list[NormalizedFinancialFact]:
    """Derive only unambiguous FCF and debt totals from same-filing inputs."""

    by_context: dict[tuple[str, str | None, str, str], dict[str, list[NormalizedFinancialFact]]] = {}
    for fact in facts:
        if fact.quality_status != "valid":
            continue
        context = (fact.accession_number, fact.start_date, fact.end_date, fact.period_type)
        by_context.setdefault(context, {}).setdefault(fact.canonical_metric, []).append(fact)
    output: list[NormalizedFinancialFact] = []
    for metrics in by_context.values():
        for target, left_name, right_name, operation, formula_version in (
            ("free_cash_flow", "operating_cash_flow", "capex", "subtract", "fcf-v1"),
            ("total_debt", "debt_current", "debt_noncurrent", "add", "debt-total-v1"),
        ):
            left = metrics.get(left_name, [])
            right = metrics.get(right_name, [])
            if len(left) != 1 or len(right) != 1:
                continue
            first, second = left[0], right[0]
            value = first.normalized_value - second.normalized_value if operation == "subtract" else first.normalized_value + second.normalized_value
            derivation = {
                "operation": operation,
                "inputs": [first.source_fact_id, second.source_fact_id],
                "input_metrics": [left_name, right_name],
            }
            identity = _fact_identity(
                {"target": target, "formula_version": formula_version, "derivation": derivation}
            )
            output.append(
                replace(
                    first,
                    canonical_metric=target,
                    normalized_value=value,
                    taxonomy="derived",
                    source_concept=f"{left_name}:{operation}:{right_name}",
                    source_unit="USD",
                    source_value=value,
                    source_fact_id=identity,
                    mapping_version=mapping_version,
                    mapping_priority=50,
                    formula_version=formula_version,
                    derivation=derivation,
                    warning=None,
                )
            )
    return output
