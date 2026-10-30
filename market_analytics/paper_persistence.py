"""Append-only persistence for immutable forward-paper decisions and outcomes.

Production decision persistence is available only through the explicitly reviewed
authorization boundary and its dedicated write gate. Production evaluation
persistence is separately gated and must be independently authorized.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .database import MarketDatabase
from .exchange_calendar import us_equity_calendar
from .forward_paper import (
    PaperDecision, PaperEvaluationOutcome, AdmittedEvaluationEvidence,
    ADJUSTMENT_BASIS_POLICY_VERSION, DECISION_PRICE_BASIS, FUTURE_PRICE_BASIS,
    RETURN_DEFINITION_VERSION, SplitBasisCoverage,
)


PAPER_SCHEMA_VERSION = 5
PAPER_SUPPORTED_SCHEMA_VERSIONS = frozenset({5, 6})
PAPER_WRITE_ENV = "FORWARD_PAPER_ENABLE_WRITES"
EVALUATION_WRITE_ENV = "FORWARD_PAPER_EVALUATION_ENABLE_WRITES"
PRODUCTION_DATABASE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "market_data.sqlite3"
).resolve()
SUCCESSFUL_OUTCOMES = frozenset({"evaluated", "terminal_value_verified"})
EVALUATION_AVAILABILITY_BUFFER = timedelta(minutes=15)
EVENT_IDENTITY_VERSION = "forward-paper-evaluation-v2"
PERSISTENCE_COVERAGE_VERSION = "split-coverage-admitted-v3"
SINGLE_EVENT_AUTHORIZATION_VERSION = "single-event-production-evaluation-v1"


@dataclass(frozen=True)
class SingleEventProductionAuthorization:
    """Explicit run-local permission, never loaded from environment or a boolean.

    Consumption is enforced by immutable rows for decision/horizon under the
    insert transaction, not this object's lifetime. No secrets belong here.
    """

    decision_id: str
    horizon_name: str
    target_session: date
    ticker: str
    authorization_version: str = SINGLE_EVENT_AUTHORIZATION_VERSION
    max_new_events: int = 1
    purpose: str = "controlled-single-evaluation"

    def __post_init__(self) -> None:
        if (self.authorization_version != SINGLE_EVENT_AUTHORIZATION_VERSION
            or self.purpose != "controlled-single-evaluation"
            or type(self.max_new_events) is not int or self.max_new_events != 1):
            raise ValueError("unsupported single-event authorization policy")
        if type(self.target_session) is not date:
            raise ValueError("authorization requires an exact target date")
        for value in (self.decision_id, self.horizon_name, self.ticker):
            if (not isinstance(value, str) or not value or value != value.strip()
                or any(c in value for c in "*?[]")):
                raise ValueError("authorization scope must be explicit without wildcards")
        if self.ticker != self.ticker.upper():
            raise ValueError("authorization ticker must be canonical uppercase")


PAPER_SCHEMA_V5 = """
CREATE TABLE forward_paper_decisions (
    decision_id TEXT PRIMARY KEY,
    rehearsal_id TEXT NOT NULL,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE RESTRICT,
    ticker_at_decision TEXT NOT NULL,
    exchange TEXT NOT NULL,
    decision_timestamp TEXT NOT NULL,
    decision_price TEXT NOT NULL,
    price_date TEXT NOT NULL,
    split_adjusted_close TEXT NOT NULL,
    ranking_status TEXT NOT NULL,
    opportunity_score TEXT,
    confidence TEXT NOT NULL,
    available_weight TEXT,
    capability_state_json TEXT NOT NULL,
    peer_identity_json TEXT NOT NULL,
    peer_classification_version TEXT,
    evidence_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    ranking_version TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    calculation_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    calendar_policy_version TEXT NOT NULL,
    horizon_policy_version TEXT NOT NULL,
    delisting_policy_version TEXT NOT NULL,
    return_policy_version TEXT NOT NULL,
    frozen_payload_hash TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    decision_payload_hash TEXT NOT NULL,
    decision_payload_json TEXT NOT NULL,
    persisted_at TEXT NOT NULL,
    CHECK(length(frozen_payload_hash)=64),
    CHECK(length(snapshot_hash)=64),
    CHECK(length(decision_payload_hash)=64)
);

CREATE TABLE forward_paper_horizons (
    decision_id TEXT NOT NULL REFERENCES forward_paper_decisions(decision_id) ON DELETE RESTRICT,
    horizon_name TEXT NOT NULL,
    target_date TEXT NOT NULL,
    resolved_session TEXT NOT NULL,
    exchange TEXT NOT NULL,
    calendar_policy_version TEXT NOT NULL,
    horizon_policy_version TEXT NOT NULL,
    delisting_policy_version TEXT NOT NULL,
    return_policy_version TEXT NOT NULL,
    horizon_payload_hash TEXT NOT NULL,
    horizon_payload_json TEXT NOT NULL,
    persisted_at TEXT NOT NULL,
    PRIMARY KEY(decision_id, horizon_name),
    CHECK(length(horizon_payload_hash)=64)
);
CREATE INDEX idx_forward_horizons_due
ON forward_paper_horizons(resolved_session, decision_id, horizon_name);

CREATE TABLE forward_paper_evaluation_events (
    evaluation_event_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    horizon_name TEXT NOT NULL,
    evaluation_timestamp TEXT NOT NULL,
    outcome_code TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL,
    observed_value TEXT,
    observed_value_type TEXT,
    price_return TEXT,
    successor_ticker TEXT,
    calendar_policy_version TEXT NOT NULL,
    horizon_policy_version TEXT NOT NULL,
    delisting_policy_version TEXT NOT NULL,
    return_policy_version TEXT NOT NULL,
    correction_of_event_id TEXT REFERENCES forward_paper_evaluation_events(evaluation_event_id) ON DELETE RESTRICT,
    event_payload_hash TEXT NOT NULL,
    event_payload_json TEXT NOT NULL,
    persisted_at TEXT NOT NULL,
    FOREIGN KEY(decision_id, horizon_name)
      REFERENCES forward_paper_horizons(decision_id, horizon_name) ON DELETE RESTRICT,
    CHECK(length(event_payload_hash)=64)
);
CREATE INDEX idx_forward_events_horizon
ON forward_paper_evaluation_events(decision_id, horizon_name, evaluation_timestamp, evaluation_event_id);

CREATE TABLE forward_paper_event_provenance (
    evaluation_event_id TEXT NOT NULL REFERENCES forward_paper_evaluation_events(evaluation_event_id) ON DELETE RESTRICT,
    provenance_kind TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    PRIMARY KEY(evaluation_event_id, provenance_kind, source_identifier)
);

CREATE TRIGGER forward_paper_decisions_no_update
BEFORE UPDATE ON forward_paper_decisions BEGIN
  SELECT RAISE(ABORT, 'forward-paper decisions are append-only');
END;
CREATE TRIGGER forward_paper_decisions_no_delete
BEFORE DELETE ON forward_paper_decisions BEGIN
  SELECT RAISE(ABORT, 'forward-paper decisions are append-only');
END;
CREATE TRIGGER forward_paper_horizons_no_update
BEFORE UPDATE ON forward_paper_horizons BEGIN
  SELECT RAISE(ABORT, 'forward-paper horizons are append-only');
END;
CREATE TRIGGER forward_paper_horizons_no_delete
BEFORE DELETE ON forward_paper_horizons BEGIN
  SELECT RAISE(ABORT, 'forward-paper horizons are append-only');
END;
CREATE TRIGGER forward_paper_events_no_update
BEFORE UPDATE ON forward_paper_evaluation_events BEGIN
  SELECT RAISE(ABORT, 'forward-paper events are append-only');
END;
CREATE TRIGGER forward_paper_events_no_delete
BEFORE DELETE ON forward_paper_evaluation_events BEGIN
  SELECT RAISE(ABORT, 'forward-paper events are append-only');
END;
CREATE TRIGGER forward_paper_provenance_no_update
BEFORE UPDATE ON forward_paper_event_provenance BEGIN
  SELECT RAISE(ABORT, 'forward-paper provenance is append-only');
END;
CREATE TRIGGER forward_paper_provenance_no_delete
BEFORE DELETE ON forward_paper_event_provenance BEGIN
  SELECT RAISE(ABORT, 'forward-paper provenance is append-only');
END;
"""


class PaperPersistenceError(RuntimeError):
    """Base error for fail-closed paper persistence."""


class PaperWriteDisabled(PermissionError, PaperPersistenceError):
    """The dedicated paper-persistence gate is not explicitly enabled."""


class PaperIntegrityConflict(PaperPersistenceError):
    """A deterministic identity already exists with different canonical content."""


class PaperPolicyMismatch(PaperPersistenceError):
    """An event does not use the exact policy versions frozen with its horizon."""


@dataclass(frozen=True)
class ReviewedProductionAuthorization:
    """One explicitly reviewed decision/hash pair authorized for production."""

    decision_id: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise ValueError("production paper authorization requires a decision ID")
        if not _is_sha256(self.artifact_sha256):
            raise ValueError("production paper authorization requires a SHA-256")


def paper_writes_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Only the exact value ``1`` enables the dedicated disposable write path."""

    values = os.environ if environment is None else environment
    return values.get(PAPER_WRITE_ENV) == "1"


def evaluation_writes_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Evaluation events require a second, exact opt-in gate."""

    values = os.environ if environment is None else environment
    return values.get(EVALUATION_WRITE_ENV) == "1"


def _normalize(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical timestamps must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, float):
        raise TypeError("floats are not allowed in canonical paper payloads")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical payload keys must be strings")
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical payload type: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Serialize paper data deterministically without paths, secrets, or floats."""

    return json.dumps(
        _normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sql_statements(script: str) -> list[str]:
    """Split SQLite DDL without breaking trigger bodies at inner semicolons."""

    statements: list[str] = []
    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            statements.append(pending.strip())
            pending = ""
    if pending.strip():
        raise ValueError("incomplete paper schema statement")
    return statements


def _utc_iso(value: datetime) -> str:
    return _normalize(value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _expected_decision_id(decision: PaperDecision) -> str:
    payload = {
        "rehearsal_id": decision.rehearsal_id,
        "ticker": decision.ticker,
        "snapshot_hash": decision.snapshot_hash,
    }
    # Match the already-frozen forward-paper identity contract.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _decision_payload(decision: PaperDecision, frozen_payload_hash: str) -> dict[str, Any]:
    return {
        "decision": decision.model_dump(mode="python"),
        "frozen_payload_hash": frozen_payload_hash,
    }


def _event_payload(
    outcome: PaperEvaluationOutcome,
    evaluation_timestamp: datetime,
    correction_of_event_id: str | None,
    provenance: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    return _event_normalize({
        "identity_version": EVENT_IDENTITY_VERSION,
        "outcome": outcome.model_dump(mode="python"),
        "evaluation_timestamp": evaluation_timestamp,
        "correction_of_event_id": correction_of_event_id,
        "provenance": {
            key: tuple(sorted(set(value))) for key, value in sorted(provenance.items())
        },
    })


def _event_normalize(value: Any, field: str = "") -> Any:
    """Event-only normalization; frozen decision/snapshot hashes are unchanged.

    Provenance IDs are sets. Reason codes retain their primary-reason order.
    Raw coverage page IDs are diagnostics, never semantic identity inputs.
    Decimal formatting uses fixed notation without context-dependent rounding.
    """
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite event decimal")
        if value == 0:
            return "0"
        text = format(value, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    if isinstance(value, Mapping):
        return {k: _event_normalize(v, k) for k, v in value.items()
                if k != "raw_page_source_ids"}
    if isinstance(value, (tuple, list)):
        values = [_event_normalize(v) for v in value]
        if field in {"source_ids", "source_page_ids", "split_coverage_source_ids",
                     "terminal_event_source_ids", "intervening_split_source_ids"}:
            return sorted(set(values))
        return values
    return value


def semantic_coverage_id(ticker: str, coverage: SplitBasisCoverage) -> str:
    """Derive coverage identity; caller IDs/raw pages have no authority.

    v3 adds completeness and exact admitted availability to the interval/content
    contract. Out-of-range records remain diagnostic; append separately rejects
    any split-bearing evaluated bundle, including inadmissible records.
    """
    records = {
        canonical_bytes(_event_normalize(item.model_dump(mode="python"))).decode()
        for item in coverage.evidence
        if coverage.requested_start <= item.effective_date <= coverage.requested_end
    }
    digest = canonical_hash(_event_normalize({
        "version": PERSISTENCE_COVERAGE_VERSION,
        "adjustment_policy": ADJUSTMENT_BASIS_POLICY_VERSION,
        "ticker": ticker,
        "requested_start": coverage.requested_start,
        "requested_end": coverage.requested_end,
        "complete": coverage.complete,
        "available_at": coverage.available_at,
        "evidence": [json.loads(record) for record in sorted(records)],
    }))
    return f"{PERSISTENCE_COVERAGE_VERSION}:{digest}"


def _admitted_provenance(outcome: PaperEvaluationOutcome) -> dict[str, tuple[str, ...]]:
    evidence = outcome.admitted_evidence
    if evidence is None:
        return {}
    sources = {"provider_request": tuple(sorted(set(evidence.bar.source_page_ids)))}
    if evidence.retrieval_policy_version is not None:
        sources["authorized_retrieval"] = (
            f"policy:{evidence.retrieval_policy_version}",
            f"started_at:{_utc_iso(evidence.retrieval_started_at)}",
            f"completed_at:{_utc_iso(evidence.retrieval_completed_at)}",
            *tuple(f"evidence_available_at:{_utc_iso(t)}" for t in sorted({
                evidence.bar.available_at, evidence.coverage.available_at,
            })),
        )
    return sources


def proposed_evaluation_identity(
    outcome: PaperEvaluationOutcome,
    *,
    evaluation_timestamp: datetime,
    correction_of_event_id: str | None = None,
    provenance: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, str, str]:
    """Return deterministic event ID, hash, and JSON without persistence."""

    sources = (_admitted_provenance(outcome) if outcome.status == "evaluated" else
               {key: tuple(value) for key, value in (provenance or {}).items()})
    if outcome.provider_snapshot_id:
        sources["provider_snapshot"] = tuple(sorted(set(
            (*sources.get("provider_snapshot", ()), outcome.provider_snapshot_id)
        )))
    if outcome.terminal_event_source_ids:
        sources["terminal_event"] = tuple(sorted(set(
            (*sources.get("terminal_event", ()), *outcome.terminal_event_source_ids)
        )))
    payload = _event_payload(
        outcome, evaluation_timestamp, correction_of_event_id, sources
    )
    payload_hash = canonical_hash(payload)
    event_id = canonical_hash({
        "kind": EVENT_IDENTITY_VERSION, "payload": payload_hash
    })[:24]
    return event_id, payload_hash, canonical_bytes(payload).decode()


class PaperPersistence:
    """Repository for schema-v5 append-only paper decisions and outcomes."""

    def __init__(
        self,
        database: MarketDatabase,
        *,
        environment: Mapping[str, str] | None = None,
        failure_injector: Callable[[str], None] | None = None,
        production_authorization: ReviewedProductionAuthorization | None = None,
        evaluation_authorization: SingleEventProductionAuthorization | None = None,
    ) -> None:
        self.database = database
        self.environment = environment
        self.failure_injector = failure_injector
        self.production_authorization = production_authorization
        self.evaluation_authorization = evaluation_authorization

    def _authorize(self) -> None:
        if self.database.read_only:
            raise PaperWriteDisabled("paper persistence requires a writable disposable database")
        if not paper_writes_enabled(self.environment):
            raise PaperWriteDisabled("forward-paper persistence gate is disabled")
        if (
            self.database.path.resolve() == PRODUCTION_DATABASE_PATH
            and self.production_authorization is None
        ):
            raise PaperWriteDisabled(
                "production paper persistence requires an exact reviewed authorization"
            )

    def _inject(self, stage: str) -> None:
        if self.failure_injector is not None:
            self.failure_injector(stage)

    def _authorize_evaluation(self) -> SingleEventProductionAuthorization | None:
        if self.database.path.resolve() == PRODUCTION_DATABASE_PATH:
            return self.authorize_production_evaluation()
        self._authorize()
        if not evaluation_writes_enabled(self.environment):
            raise PaperWriteDisabled(
                "forward-paper evaluation persistence gate is disabled"
            )
        return None

    def authorize_production_evaluation(self) -> SingleEventProductionAuthorization:
        """Default deny. Both existing gates plus exact typed scope are required.

        This is separate from decision-write authorization: it never grants
        permission to insert a decision, research note or ranking view.
        """
        if (self.database.read_only or not paper_writes_enabled(self.environment)
            or not evaluation_writes_enabled(self.environment)
            or type(self.evaluation_authorization) is not SingleEventProductionAuthorization):
            raise PaperWriteDisabled("production evaluation persistence is not authorized")
        authorization = self.evaluation_authorization
        # Recheck policy even if an object was constructed through unsafe tooling.
        authorization.__post_init__()
        decision = self.paper_decision(authorization.decision_id)
        if decision is None or decision.ticker != authorization.ticker:
            raise PaperWriteDisabled("production evaluation authorization decision/ticker mismatch")
        horizon = next((h for h in decision.evaluation_horizons
                        if h.name == authorization.horizon_name), None)
        if horizon is None or horizon.resolved_session_date != authorization.target_session:
            raise PaperWriteDisabled("production evaluation authorization horizon/target mismatch")
        return authorization

    def migrate_v4_to_v5(self) -> int:
        """Transactionally add paper tables to an explicitly authorized disposable v4 DB."""

        self._authorize()
        version = self.database.schema_version()
        if version in PAPER_SUPPORTED_SCHEMA_VERSIONS:
            return version
        if version != 4:
            raise PaperPersistenceError("paper migration requires schema v4")
        with self.database.transaction() as connection:
            for index, statement in enumerate(_sql_statements(PAPER_SCHEMA_V5)):
                connection.execute(statement)
                self._inject(f"migration_statement_{index}")
            connection.execute(
                "UPDATE schema_metadata SET value=? WHERE key='schema_version'",
                (str(PAPER_SCHEMA_VERSION),),
            )
            self._inject("migration_before_commit")
        return self.database.schema_version()

    def _require_schema(self) -> None:
        if self.database.schema_version() not in PAPER_SUPPORTED_SCHEMA_VERSIONS:
            raise PaperPersistenceError("forward-paper schema v5 or v6 is required")

    def persist_decision(
        self,
        decision: PaperDecision,
        *,
        frozen_payload_hash: str,
        persisted_at: datetime,
    ) -> bool:
        """Atomically append one decision and every frozen horizon.

        Returns ``True`` for a new insert and ``False`` for an identical no-op.
        """

        self._authorize()
        self._require_schema()
        if not _is_sha256(frozen_payload_hash) or not _is_sha256(decision.snapshot_hash):
            raise ValueError("paper hashes must be lowercase SHA-256 values")
        if self.database.path.resolve() == PRODUCTION_DATABASE_PATH:
            authorization = self.production_authorization
            if authorization is None or (
                decision.decision_id != authorization.decision_id
                or frozen_payload_hash != authorization.artifact_sha256
            ):
                raise PaperWriteDisabled(
                    "production decision/hash differs from the reviewed authorization"
                )
        if decision.decision_id != _expected_decision_id(decision):
            raise ValueError("decision identity does not match frozen inputs")
        if len(decision.evaluation_horizons) != 3:
            raise ValueError("paper decision requires exactly three frozen horizons")
        if len({item.name for item in decision.evaluation_horizons}) != 3:
            raise ValueError("paper decision horizons must be unique")
        if any(item.resolved_session_date is None for item in decision.evaluation_horizons):
            raise ValueError("paper decision horizons must have resolved sessions")
        payload = _decision_payload(decision, frozen_payload_hash)
        payload_json = canonical_bytes(payload).decode("utf-8")
        payload_hash = canonical_hash(payload)
        persisted = _utc_iso(persisted_at)
        existing = self.database.connection.execute(
            "SELECT decision_payload_hash FROM forward_paper_decisions WHERE decision_id=?",
            (decision.decision_id,),
        ).fetchone()
        if existing is not None:
            if existing["decision_payload_hash"] != payload_hash:
                raise PaperIntegrityConflict("decision ID conflicts with stored canonical payload")
            stored_horizons = self.horizons(decision.decision_id)
            expected = [
                (item.name, canonical_hash(item.model_dump(mode="python")))
                for item in sorted(decision.evaluation_horizons, key=lambda value: value.name)
            ]
            actual = [(row["horizon_name"], row["horizon_payload_hash"]) for row in stored_horizons]
            if actual != expected:
                raise PaperIntegrityConflict("stored horizon set conflicts with decision payload")
            return False

        policies = {
            (
                item.calendar_policy_version,
                item.horizon_policy_version,
                item.delisting_policy_version,
                item.return_definition_version,
            )
            for item in decision.evaluation_horizons
        }
        if len(policies) != 1:
            raise ValueError("all frozen horizons must use one policy-version set")
        calendar_version, horizon_version, delisting_version, return_version = next(iter(policies))
        capability = {
            "eligibility_status": decision.eligibility_status,
            "selection_status": decision.selection_status,
            "reason_codes": decision.capability_reason_codes,
        }
        evidence = {
            "summary": decision.evidence_summary,
            "metrics": [item.model_dump(mode="python") for item in decision.key_metrics],
            "components": [item.model_dump(mode="python") for item in decision.metric_contributions],
            "unresolved_risks": decision.unresolved_risks,
        }
        provenance = {
            "source_artifact_ids": decision.source_artifact_ids,
            "mapping_versions": decision.mapping_versions,
            "provider_snapshot_ids": decision.provider_snapshot_ids,
            "analytics_run_id": decision.analytics_run_id,
        }
        with self.database.transaction() as connection:
            # Repeat the identity check while holding BEGIN IMMEDIATE so two
            # concurrent writers cannot turn an identical replay into a UNIQUE
            # failure or a conflicting replay into an overwrite.
            locked_existing = connection.execute(
                "SELECT decision_payload_hash FROM forward_paper_decisions WHERE decision_id=?",
                (decision.decision_id,),
            ).fetchone()
            if locked_existing is not None:
                if locked_existing["decision_payload_hash"] != payload_hash:
                    raise PaperIntegrityConflict(
                        "decision ID conflicts with stored canonical payload"
                    )
                locked_horizons = list(connection.execute(
                    """SELECT horizon_name,horizon_payload_hash
                       FROM forward_paper_horizons WHERE decision_id=?
                       ORDER BY horizon_name""",
                    (decision.decision_id,),
                ))
                expected = [
                    (item.name, canonical_hash(item.model_dump(mode="python")))
                    for item in sorted(
                        decision.evaluation_horizons, key=lambda value: value.name
                    )
                ]
                actual = [
                    (row["horizon_name"], row["horizon_payload_hash"])
                    for row in locked_horizons
                ]
                if actual != expected:
                    raise PaperIntegrityConflict(
                        "stored horizon set conflicts with decision payload"
                    )
                return False
            connection.execute(
                """
                INSERT INTO forward_paper_decisions(
                    decision_id,rehearsal_id,company_id,ticker_at_decision,exchange,
                    decision_timestamp,decision_price,price_date,split_adjusted_close,
                    ranking_status,opportunity_score,confidence,available_weight,
                    capability_state_json,peer_identity_json,peer_classification_version,
                    evidence_json,provenance_json,ranking_version,pipeline_version,
                    calculation_version,config_hash,calendar_policy_version,
                    horizon_policy_version,delisting_policy_version,return_policy_version,
                    frozen_payload_hash,snapshot_hash,decision_payload_hash,
                    decision_payload_json,persisted_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    decision.decision_id, decision.rehearsal_id, decision.company_id,
                    decision.ticker, decision.exchange, _utc_iso(decision.decision_timestamp),
                    str(decision.decision_raw_close), decision.decision_price_date.isoformat(),
                    str(decision.decision_split_adjusted_close), decision.ranking_status,
                    str(decision.opportunity_score) if decision.opportunity_score is not None else None,
                    decision.confidence,
                    str(decision.available_weight) if decision.available_weight is not None else None,
                    canonical_bytes(capability).decode(),
                    canonical_bytes([item.model_dump(mode="python") for item in decision.peer_identities]).decode(),
                    decision.peer_classification_rule,
                    canonical_bytes(evidence).decode(), canonical_bytes(provenance).decode(),
                    decision.ranking_version, decision.pipeline_version,
                    decision.calculation_version, decision.config_hash,
                    calendar_version, horizon_version, delisting_version, return_version,
                    frozen_payload_hash, decision.snapshot_hash, payload_hash, payload_json, persisted,
                ),
            )
            self._inject("after_decision")
            for position, item in enumerate(decision.evaluation_horizons, 1):
                horizon_payload = item.model_dump(mode="python")
                connection.execute(
                    """
                    INSERT INTO forward_paper_horizons(
                        decision_id,horizon_name,target_date,resolved_session,exchange,
                        calendar_policy_version,horizon_policy_version,
                        delisting_policy_version,return_policy_version,
                        horizon_payload_hash,horizon_payload_json,persisted_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        decision.decision_id, item.name, item.target_not_before.isoformat(),
                        item.resolved_session_date.isoformat(), decision.exchange,
                        item.calendar_policy_version, item.horizon_policy_version,
                        item.delisting_policy_version, item.return_definition_version,
                        canonical_hash(horizon_payload),
                        canonical_bytes(horizon_payload).decode(), persisted,
                    ),
                )
                self._inject(f"after_horizon_{position}")
            self._inject("before_decision_commit")
        return True

    def validate_evaluation(
        self,
        outcome: PaperEvaluationOutcome,
        *,
        evaluation_timestamp: datetime,
        as_of: datetime,
    ) -> None:
        """Bind an outcome to its authoritative decision and frozen horizon."""

        self._require_schema()
        if evaluation_timestamp.tzinfo is None or evaluation_timestamp.utcoffset() is None:
            raise PaperPersistenceError("evaluation timestamp must be timezone-aware")
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise PaperPersistenceError("evaluation as_of must be timezone-aware")
        event_time = evaluation_timestamp.astimezone(timezone.utc)
        cutoff = as_of.astimezone(timezone.utc)
        if event_time > cutoff:
            raise PaperPersistenceError("evaluation timestamp cannot be later than as_of")

        decision = self.paper_decision(outcome.decision_id)
        if decision is None:
            raise PaperPersistenceError("evaluation references an unknown frozen decision")
        try:
            frozen = next(
                item for item in decision.evaluation_horizons
                if item.name == outcome.horizon
            )
        except StopIteration as exc:
            raise PaperPersistenceError(
                "evaluation references an unknown frozen horizon"
            ) from exc
        horizon = self.database.connection.execute(
            """SELECT * FROM forward_paper_horizons
               WHERE decision_id=? AND horizon_name=?""",
            (outcome.decision_id, outcome.horizon),
        ).fetchone()
        if horizon is None:
            raise PaperPersistenceError("evaluation references an unknown frozen horizon")
        if outcome.decision_id != decision.decision_id:
            raise PaperPersistenceError("evaluation decision ID differs from frozen decision")
        if outcome.ticker != decision.ticker:
            raise PaperPersistenceError("evaluation ticker differs from frozen decision")
        if horizon["exchange"] != decision.exchange:
            raise PaperPolicyMismatch("frozen horizon exchange differs from frozen decision")
        if outcome.target_not_before != frozen.target_not_before:
            raise PaperPolicyMismatch("evaluation target differs from frozen horizon")
        if (
            frozen.resolved_session_date is None
            or outcome.resolved_session_date != frozen.resolved_session_date
            or horizon["resolved_session"] != frozen.resolved_session_date.isoformat()
        ):
            raise PaperPolicyMismatch("evaluation session differs from frozen horizon")
        if horizon["target_date"] != frozen.target_not_before.isoformat():
            raise PaperPolicyMismatch("stored horizon target differs from frozen decision")
        if outcome.return_type != frozen.return_type or outcome.total_return_included:
            raise PaperPolicyMismatch("evaluation return semantics differ from frozen horizon")
        expected_policies = (
            frozen.calendar_policy_version,
            frozen.horizon_policy_version,
            frozen.delisting_policy_version,
            frozen.return_definition_version,
        )
        stored_policies = (
            horizon["calendar_policy_version"], horizon["horizon_policy_version"],
            horizon["delisting_policy_version"], horizon["return_policy_version"],
        )
        event_policies = (
            outcome.calendar_policy_version, outcome.horizon_policy_version,
            outcome.delisting_policy_version, outcome.return_definition_version,
        )
        if stored_policies != expected_policies or event_policies != expected_policies:
            raise PaperPolicyMismatch(
                "evaluation policy versions differ from frozen horizon"
            )

        session = us_equity_calendar(decision.exchange).session(
            frozen.resolved_session_date
        )
        if session.closes_at is None:
            raise PaperPolicyMismatch("frozen evaluation session is not an open session")
        eligible_at = session.closes_at + EVALUATION_AVAILABILITY_BUFFER
        before_eligible_pending = (
            outcome.status == "pending"
            and outcome.reason_codes == ("target_session_not_closed",)
            and event_time < eligible_at
        )
        if event_time < eligible_at and not before_eligible_pending:
            raise PaperPersistenceError(
                "evaluation timestamp precedes close availability eligibility"
            )

        value = outcome.evaluation_value
        value_type = outcome.evaluation_value_type
        supplied_return = outcome.price_return
        snapshot = outcome.provider_snapshot_id
        terminal_sources = outcome.terminal_event_source_ids
        primary_reason = {
            "evaluated": "regular_session_close",
            "pending": None,
            "missing_price": "split_adjusted_close_missing",
            "provider_gap": "provider_bar_missing",
            "inactive_unresolved": "terminal_event_evidence_missing",
            "terminal_value_verified": "verified_cash_acquisition",
            "unsupported_identity_transition": "identity_continuity_unverified",
            "unsupported_adjustment_basis": None,
        }.get(outcome.status)
        if outcome.status == "pending":
            if outcome.reason_codes not in {
                ("target_session_not_closed",), ("temporary_halt",),
            }:
                raise PaperPersistenceError("pending outcome has incompatible reasons")
            if (
                outcome.reason_codes == ("temporary_halt",)
                and not terminal_sources
            ):
                raise PaperPersistenceError(
                    "temporary-halt outcome requires terminal event evidence"
                )
        elif outcome.status == "unsupported_adjustment_basis":
            if (
                not outcome.reason_codes
                or outcome.reason_codes[0] not in {
                    "split_coverage_unavailable", "intervening_split_detected"
                }
                or any(
                    reason != "verified_ticker_continuity"
                    for reason in outcome.reason_codes[1:]
                )
            ):
                raise PaperPersistenceError(
                    "unsupported adjustment basis has incompatible reasons"
                )
        elif (
            primary_reason is None
            or not outcome.reason_codes
            or outcome.reason_codes[0] != primary_reason
            or any(
                reason != "verified_ticker_continuity"
                for reason in outcome.reason_codes[1:]
            )
        ):
            raise PaperPersistenceError(
                f"{outcome.status} outcome has incompatible reasons"
            )
        continuity_fields_present = (
            outcome.successor_ticker is not None
            and bool(terminal_sources)
            and "verified_ticker_continuity" in outcome.reason_codes
        )
        continuity_fields_absent = (
            outcome.successor_ticker is None
            and not terminal_sources
            and "verified_ticker_continuity" not in outcome.reason_codes
        )
        if outcome.status in {
            "evaluated", "missing_price", "provider_gap",
        } and not (
            continuity_fields_present or continuity_fields_absent
        ):
            raise PaperPersistenceError(
                "ticker continuity fields must describe one admitted evidence set"
            )
        if outcome.status == "evaluated":
            if value is None or supplied_return is None or snapshot is None:
                raise PaperPersistenceError(
                    "evaluated outcome requires price, return, and provider snapshot"
                )
            if value_type != frozen.price_field:
                raise PaperPolicyMismatch(
                    "evaluation price field differs from frozen horizon"
                )
            expected_return = (
                value / decision.decision_split_adjusted_close - Decimal(1)
            )
            if supplied_return != expected_return:
                raise PaperPersistenceError(
                    "evaluation return does not match the frozen decision price"
                )
            if (
                not outcome.future_price_basis
                or not outcome.split_coverage_source_ids
                or outcome.intervening_split_source_ids
                or outcome.rebasing_applied
                or outcome.cumulative_split_factor is not None
            ):
                raise PaperPersistenceError(
                    "evaluated outcome lacks compatible split-basis provenance"
                )
        elif outcome.status == "terminal_value_verified":
            if (
                value is None
                or value_type != "terminal_cash_consideration"
                or supplied_return is None
                or not terminal_sources
                or snapshot is not None
            ):
                raise PaperPersistenceError(
                    "verified terminal outcome requires terminal value evidence only"
                )
            expected_return = (
                value / decision.decision_split_adjusted_close - Decimal(1)
            )
            if supplied_return != expected_return:
                raise PaperPersistenceError(
                    "terminal return does not match the frozen decision price"
                )
        elif outcome.status == "missing_price":
            if value is not None or value_type is not None or supplied_return is not None:
                raise PaperPersistenceError("missing-price outcome cannot carry a return")
            if snapshot is None:
                raise PaperPersistenceError(
                    "missing-price outcome requires its admitted provider snapshot"
                )
        elif outcome.status in {"provider_gap", "pending"}:
            if any(item is not None for item in (value, value_type, supplied_return, snapshot)):
                raise PaperPersistenceError(
                    f"{outcome.status} outcome cannot carry a final value or return"
                )
        elif outcome.status in {
            "inactive_unresolved", "unsupported_identity_transition",
        }:
            if any(item is not None for item in (value, value_type, supplied_return, snapshot)):
                raise PaperPersistenceError(
                    f"{outcome.status} outcome cannot carry a final value or return"
                )
            if not terminal_sources:
                raise PaperPersistenceError(
                    f"{outcome.status} outcome requires terminal event evidence"
                )
        elif outcome.status == "unsupported_adjustment_basis":
            if any(item is not None for item in (value, value_type, supplied_return)):
                raise PaperPersistenceError(
                    "unsupported adjustment basis cannot carry a value or return"
                )
            if outcome.reason_codes[0] == "intervening_split_detected":
                if not outcome.intervening_split_source_ids:
                    raise PaperPersistenceError(
                        "intervening split outcome requires split evidence"
                    )
            elif outcome.intervening_split_source_ids:
                raise PaperPersistenceError(
                    "unavailable split coverage cannot claim intervening splits"
                )
        else:
            raise PaperPersistenceError("unsupported evaluation status")

    def append_evaluation(
        self,
        outcome: PaperEvaluationOutcome,
        *,
        evaluation_timestamp: datetime,
        correction_of_event_id: str | None = None,
        provenance: Mapping[str, Sequence[str]] | None = None,
        persisted_at: datetime,
        as_of: datetime,
    ) -> tuple[str, bool]:
        """Append an immutable evaluation and its provenance in one transaction."""

        authorization = self._authorize_evaluation()
        if authorization is not None:
            if (outcome.decision_id != authorization.decision_id
                or outcome.horizon != authorization.horizon_name
                or outcome.resolved_session_date != authorization.target_session
                or outcome.ticker != authorization.ticker
                or outcome.status != "evaluated"
                or correction_of_event_id is not None):
                raise PaperWriteDisabled("event is outside single-event production authorization")
        self.validate_evaluation(
            outcome,
            evaluation_timestamp=evaluation_timestamp,
            as_of=as_of,
        )
        if outcome.status == "evaluated":
            self._validate_admitted_evidence(outcome, evaluation_timestamp, as_of)
            # Semantic provenance is already in the typed evidence bundle. Do
            # not accept a parallel, unvalidated channel into event identity.
            if provenance:
                raise PaperPersistenceError("evaluated provenance must come from admitted evidence")
        sources = (_admitted_provenance(outcome) if outcome.status == "evaluated" else
                   {key: tuple(value) for key, value in (provenance or {}).items()})
        if outcome.provider_snapshot_id:
            sources["provider_snapshot"] = tuple(sorted(set(
                (*sources.get("provider_snapshot", ()), outcome.provider_snapshot_id)
            )))
        if outcome.terminal_event_source_ids:
            sources["terminal_event"] = tuple(sorted(set(
                (*sources.get("terminal_event", ()), *outcome.terminal_event_source_ids)
            )))
        event_id, payload_hash, event_json = proposed_evaluation_identity(
            outcome,
            evaluation_timestamp=evaluation_timestamp,
            correction_of_event_id=correction_of_event_id,
            provenance=sources,
        )
        evaluated_at = _utc_iso(evaluation_timestamp)
        persisted = _utc_iso(persisted_at)
        event_policies = (
            outcome.calendar_policy_version, outcome.horizon_policy_version,
            outcome.delisting_policy_version, outcome.return_definition_version,
        )
        if correction_of_event_id is not None:
            prior = self.database.connection.execute(
                """SELECT decision_id,horizon_name,correction_of_event_id
                   FROM forward_paper_evaluation_events
                   WHERE evaluation_event_id=?""",
                (correction_of_event_id,),
            ).fetchone()
            if prior is None or (prior["decision_id"], prior["horizon_name"]) != (
                outcome.decision_id, outcome.horizon
            ):
                raise PaperIntegrityConflict("correction must reference an event on the same horizon")
            seen = {event_id}
            cursor = correction_of_event_id
            while cursor is not None:
                if cursor in seen:
                    raise PaperIntegrityConflict("evaluation correction cycle detected")
                seen.add(cursor)
                link = self.database.connection.execute(
                    """SELECT correction_of_event_id
                       FROM forward_paper_evaluation_events
                       WHERE evaluation_event_id=?""",
                    (cursor,),
                ).fetchone()
                if link is None:
                    raise PaperIntegrityConflict("correction chain contains an unknown event")
                cursor = link["correction_of_event_id"]
        existing = self.database.connection.execute(
            "SELECT event_payload_hash FROM forward_paper_evaluation_events WHERE evaluation_event_id=?",
            (event_id,),
        ).fetchone()
        if existing is not None:
            if existing["event_payload_hash"] != payload_hash:
                raise PaperIntegrityConflict("event ID conflicts with stored canonical payload")
            return event_id, False
        with self.database.transaction() as connection:
            locked_existing = connection.execute(
                """SELECT event_payload_hash FROM forward_paper_evaluation_events
                   WHERE evaluation_event_id=?""",
                (event_id,),
            ).fetchone()
            if locked_existing is not None:
                if locked_existing["event_payload_hash"] != payload_hash:
                    raise PaperIntegrityConflict(
                        "event ID conflicts with stored canonical payload"
                    )
                return event_id, False
            if authorization is not None:
                # BEGIN IMMEDIATE serializes contenders. Count persisted state,
                # including unsuccessful/older events, so restart or a fresh
                # authorization object cannot replenish this scope's allowance.
                consumed = connection.execute(
                    """SELECT COUNT(*) FROM forward_paper_evaluation_events
                       WHERE decision_id=? AND horizon_name=?""",
                    (authorization.decision_id, authorization.horizon_name),
                ).fetchone()[0]
                if consumed >= authorization.max_new_events:
                    raise PaperWriteDisabled("single-event production authorization is consumed")
            connection.execute(
                """
                INSERT INTO forward_paper_evaluation_events(
                    evaluation_event_id,decision_id,horizon_name,evaluation_timestamp,
                    outcome_code,reason_codes_json,observed_value,observed_value_type,
                    price_return,successor_ticker,calendar_policy_version,
                    horizon_policy_version,delisting_policy_version,return_policy_version,
                    correction_of_event_id,event_payload_hash,event_payload_json,persisted_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event_id, outcome.decision_id, outcome.horizon, evaluated_at,
                    outcome.status, canonical_bytes(outcome.reason_codes).decode(),
                    str(outcome.evaluation_value) if outcome.evaluation_value is not None else None,
                    outcome.evaluation_value_type,
                    str(outcome.price_return) if outcome.price_return is not None else None,
                    outcome.successor_ticker, *event_policies, correction_of_event_id,
                    payload_hash, event_json, persisted,
                ),
            )
            self._inject("after_evaluation")
            for kind in sorted(sources):
                for source_id in sorted(set(sources[kind])):
                    connection.execute(
                        """INSERT INTO forward_paper_event_provenance(
                               evaluation_event_id,provenance_kind,source_identifier
                           ) VALUES (?,?,?)""",
                        (event_id, kind, source_id),
                    )
            self._inject("after_evaluation_provenance")
            self._inject("before_evaluation_commit")
        return event_id, True

    def _validate_admitted_evidence(
        self, outcome: PaperEvaluationOutcome, event_time: datetime, as_of: datetime,
    ) -> None:
        from .paper_evaluation import (
            AUTHORIZED_RETRIEVAL_POLICY_VERSION, AUTHORIZED_RETRIEVAL_MAX_ELAPSED,
        )
        if (
            outcome.adjustment_basis_policy_version != ADJUSTMENT_BASIS_POLICY_VERSION
            or outcome.decision_price_basis != DECISION_PRICE_BASIS
            or outcome.future_price_basis != FUTURE_PRICE_BASIS
            or outcome.return_definition_version != RETURN_DEFINITION_VERSION
        ):
            raise PaperPolicyMismatch("unsupported evaluated adjustment policy/basis")
        if outcome.admitted_evidence is None:
            raise PaperPersistenceError("evaluated event requires structured admitted evidence")
        # Revalidate even models produced through unchecked model_copy/update.
        evidence_data = outcome.admitted_evidence.model_dump(mode="python")
        evidence_data["coverage"]["source_ids"] = tuple(sorted(set(
            evidence_data["coverage"]["source_ids"]
        )))
        evidence = AdmittedEvaluationEvidence.model_validate(evidence_data)
        start, end = evidence.retrieval_started_at, evidence.retrieval_completed_at
        boundary, cutoff = evidence.authorized_boundary, evidence.operator_cutoff
        if not start <= end <= boundary <= as_of or cutoff > boundary:
            raise PaperPersistenceError("invalid evidence retrieval boundary")
        if evidence.retrieval_policy_version is not None:
            if (evidence.retrieval_policy_version != AUTHORIZED_RETRIEVAL_POLICY_VERSION
                or start < cutoff or boundary != end
                or end > cutoff + AUTHORIZED_RETRIEVAL_MAX_ELAPSED):
                raise PaperPersistenceError("invalid authorized retrieval window")
        elif boundary != cutoff:
            raise PaperPersistenceError("historical evidence boundary must equal cutoff")
        decision = self.paper_decision(outcome.decision_id)
        assert decision is not None
        coverage, bar = evidence.coverage, evidence.bar
        # No structured continuity proof is supported by this persistence
        # boundary yet. Labels/source IDs must never authorize another security.
        if (bar.ticker != decision.ticker or outcome.successor_ticker is not None
            or outcome.terminal_event_source_ids
            or "verified_ticker_continuity" in outcome.reason_codes):
            raise PaperPersistenceError("cross-ticker evaluated persistence is unsupported")
        if (
            not coverage.complete
            or coverage.requested_start != decision.decision_price_date + timedelta(days=1)
            or coverage.requested_end != boundary.astimezone(
                ZoneInfo(us_equity_calendar(decision.exchange).timezone_name)
            ).date()
            or set(coverage.source_ids) != set(outcome.split_coverage_source_ids)
            or coverage.evidence
        ):
            raise PaperPersistenceError("incomplete, mismatched, or split-bearing coverage")
        expected_coverage_id = semantic_coverage_id(decision.ticker, coverage)
        if set(coverage.source_ids) != {expected_coverage_id}:
            raise PaperPersistenceError("coverage identity does not match normalized contents")
        if (bar.provider_snapshot_id != outcome.provider_snapshot_id
            or bar.split_adjusted_close != outcome.evaluation_value
            or bar.trading_date != outcome.resolved_session_date
            or bar.ticker != (outcome.successor_ticker or outcome.ticker)
            or us_equity_calendar(bar.exchange).exchange != us_equity_calendar(decision.exchange).exchange
            or not bar.regular_session):
            raise PaperPersistenceError("admitted price does not match evaluated outcome")
        times = (bar.available_at, coverage.available_at)
        if any(t > boundary or t < start or t > end for t in times):
            raise PaperPersistenceError("admitted evidence is outside retrieval window")
        eligible = us_equity_calendar(decision.exchange).session(
            outcome.resolved_session_date
        ).closes_at + EVALUATION_AVAILABILITY_BUFFER
        if event_time != max(eligible, *times):
            raise PaperPersistenceError("event timestamp must derive from admitted evidence")

    def decision(self, decision_id: str) -> dict[str, Any] | None:
        row = self.database.connection.execute(
            "SELECT * FROM forward_paper_decisions WHERE decision_id=?", (decision_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def paper_decision(self, decision_id: str) -> PaperDecision | None:
        """Load the immutable decision model through the normal persistence read API."""

        row = self.decision(decision_id)
        if row is None:
            return None
        payload = json.loads(row["decision_payload_json"])
        return PaperDecision.model_validate(payload["decision"])

    def all_decisions(self) -> list[dict[str, Any]]:
        return [
            dict(row) for row in self.database.connection.execute(
                "SELECT * FROM forward_paper_decisions ORDER BY decision_timestamp,decision_id"
            )
        ]

    def horizons(self, decision_id: str) -> list[dict[str, Any]]:
        return [
            dict(row) for row in self.database.connection.execute(
                """SELECT * FROM forward_paper_horizons WHERE decision_id=?
                   ORDER BY horizon_name""",
                (decision_id,),
            )
        ]

    def evaluation_events(
        self, decision_id: str, horizon: str | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM forward_paper_evaluation_events WHERE decision_id=?"
        values: list[str] = [decision_id]
        if horizon is not None:
            sql += " AND horizon_name=?"
            values.append(horizon)
        sql += " ORDER BY evaluation_timestamp,evaluation_event_id"
        return [dict(row) for row in self.database.connection.execute(sql, values)]

    def current_outcome(self, decision_id: str, horizon: str) -> dict[str, Any] | None:
        events = self.evaluation_events(decision_id, horizon)
        if not events:
            return None
        corrected = {row["correction_of_event_id"] for row in events if row["correction_of_event_id"]}
        effective = [row for row in events if row["evaluation_event_id"] not in corrected]
        successful = [row for row in effective if row["outcome_code"] in SUCCESSFUL_OUTCOMES]
        return successful[-1] if successful else effective[-1] if effective else None

    def unresolved_horizons(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        rows = self.database.connection.execute(
            """SELECT h.*,d.ticker_at_decision,d.decision_timestamp
               FROM forward_paper_horizons h
               JOIN forward_paper_decisions d ON d.decision_id=h.decision_id
               ORDER BY h.resolved_session,h.decision_id,h.horizon_name"""
        )
        for row in rows:
            state = self.current_outcome(row["decision_id"], row["horizon_name"])
            if state is None or state["outcome_code"] not in SUCCESSFUL_OUTCOMES:
                output.append(dict(row))
        return output

    def evaluations_due(self, as_of: datetime) -> list[dict[str, Any]]:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("due-evaluation cutoff must be timezone-aware")
        cutoff_date = as_of.astimezone(timezone.utc).date().isoformat()
        return [
            row for row in self.unresolved_horizons()
            if row["resolved_session"] <= cutoff_date
        ]
