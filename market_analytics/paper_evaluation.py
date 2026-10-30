"""Deterministic due-horizon evaluation orchestration without scheduling."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator,
)

from .exchange_calendar import us_equity_calendar
from .forward_paper import (
    AdmittedEvaluationEvidence,
    EvaluationHorizon,
    EvaluationPriceBar,
    PaperDecision,
    PaperEvaluationOutcome,
    SplitBasisCoverage,
    TerminalEventEvidence,
    evaluate_horizon_outcome_with_evidence,
)
from .paper_persistence import (
    PRODUCTION_DATABASE_PATH,
    PaperPersistence,
    PaperWriteDisabled,
    canonical_hash,
    proposed_evaluation_identity,
    semantic_coverage_id,
)


EVALUATION_WORKFLOW_VERSION = "due-evaluation-v1"
CLOSE_AVAILABILITY_POLICY_VERSION = "regular-close-plus-15m-v1"
CLOSE_AVAILABILITY_BUFFER = timedelta(minutes=15)
PROVIDER_GAP_RETRY_POLICY_VERSION = "provider-gap-initial-plus-2-v1"
PROVIDER_GAP_MAX_ATTEMPTS = 3
PROVIDER_GAP_MIN_RETRY_INTERVAL = timedelta(minutes=60)
AUTHORIZED_RETRIEVAL_POLICY_VERSION = "authorized-evaluation-retrieval-v1"
AUTHORIZED_RETRIEVAL_MAX_ELAPSED = timedelta(minutes=5)
EvaluationMode = Literal["dry_run", "authorized_write"]
ProviderFailureCode = Literal[
    "network_timeout", "http_429", "provider_5xx", "malformed_response",
    "provider_auth_failure", "provider_unavailable",
]


class EvaluationFetchResult(BaseModel):
    """Narrow provider response for exactly one frozen trading session.

    Batch-level provenance_ids are diagnostic. Canonical page provenance is
    attached to the selected bar or admitted split coverage instead.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_start: date
    requested_end: date
    bars: tuple[EvaluationPriceBar, ...] = ()
    terminal_event: TerminalEventEvidence | None = None
    provenance_ids: tuple[str, ...] = ()
    provider_sends: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    failure_code: ProviderFailureCode | None = None
    failure_detail: str | None = None
    retrieval_policy_version: str | None = None
    retrieval_started_at: AwareDatetime | None = None
    retrieval_completed_at: AwareDatetime | None = None
    authorized_bar_ids: tuple[str, ...] = ()
    split_basis_coverage: SplitBasisCoverage | None = None
    authorized_split_ids: tuple[str, ...] = ()

    @field_validator("failure_detail")
    @classmethod
    def bounded_failure_detail(cls, value: str | None) -> str | None:
        if value is None:
            return None
        clean = " ".join(value.split())
        return clean[:240]

    @field_validator("retrieval_started_at", "retrieval_completed_at", mode="after")
    @classmethod
    def normalize_retrieval_time(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(timezone.utc) if value is not None else None

    @field_validator("authorized_bar_ids", "authorized_split_ids", mode="after")
    @classmethod
    def canonical_bar_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("authorized bar IDs must be non-empty")
        return tuple(sorted(set(value)))

    @model_validator(mode="after")
    def complete_retrieval_authorization(self) -> "EvaluationFetchResult":
        values = (
            self.retrieval_policy_version,
            self.retrieval_started_at,
            self.retrieval_completed_at,
        )
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError("authorized retrieval metadata must be complete")
        if self.authorized_bar_ids and self.retrieval_policy_version is None:
            raise ValueError("authorized bar IDs require retrieval metadata")
        if self.authorized_split_ids and self.retrieval_policy_version is None:
            raise ValueError("authorized split IDs require retrieval metadata")
        return self


class DueEvaluationProvider(Protocol):
    def fetch_resolved_session(
        self,
        *,
        decision: PaperDecision,
        horizon: EvaluationHorizon,
        start_date: date,
        end_date: date,
        as_of: datetime,
    ) -> EvaluationFetchResult: ...


class EvaluationRunItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str
    ticker: str
    horizon: str
    resolved_session: date
    close_eligible_at: AwareDatetime
    outcome_code: str
    reason_codes: tuple[str, ...] = ()
    observed_value: Decimal | None = None
    price_return: Decimal | None = None
    event_id: str | None = None
    event_payload_hash: str | None = None
    correction_of_event_id: str | None = None
    write_attempted: bool = False
    write_committed: bool = False
    failure_code: str | None = None
    retrieval_policy_version: str | None = None
    retrieval_started_at: AwareDatetime | None = None
    retrieval_completed_at: AwareDatetime | None = None
    admitted_evidence_available_at: tuple[AwareDatetime, ...] = ()


class EvaluationRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    workflow_version: str = EVALUATION_WORKFLOW_VERSION
    close_availability_policy_version: str = CLOSE_AVAILABILITY_POLICY_VERSION
    provider_gap_retry_policy_version: str = PROVIDER_GAP_RETRY_POLICY_VERSION
    authorized_retrieval_policy_version: str = AUTHORIZED_RETRIEVAL_POLICY_VERSION
    as_of: AwareDatetime
    mode: EvaluationMode
    horizons_inspected: int
    due_count: int
    evaluated_count: int
    pending_count: int
    provider_gap_count: int
    missing_price_count: int
    inactive_unresolved_count: int
    terminal_count: int
    unsupported_identity_count: int
    unsupported_adjustment_basis_count: int
    failed_count: int
    provider_sends: int
    retries: int
    writes_attempted: int
    writes_committed: int
    event_ids: tuple[str, ...]
    items: tuple[EvaluationRunItem, ...]


def _frozen_horizon(decision: PaperDecision, name: str) -> EvaluationHorizon:
    try:
        horizon = next(item for item in decision.evaluation_horizons if item.name == name)
    except StopIteration as exc:
        raise ValueError("persisted horizon is absent from its immutable decision") from exc
    if horizon.resolved_session_date is None:
        raise ValueError("persisted horizon has no resolved session")
    return horizon


def _pending_before_available(
    decision: PaperDecision, horizon: EvaluationHorizon
) -> PaperEvaluationOutcome:
    return PaperEvaluationOutcome(
        decision_id=decision.decision_id,
        ticker=decision.ticker,
        horizon=horizon.name,
        target_not_before=horizon.target_not_before,
        resolved_session_date=horizon.resolved_session_date,
        status="pending",
        reason_codes=("target_session_not_closed",),
        calendar_policy_version=horizon.calendar_policy_version,
        horizon_policy_version=horizon.horizon_policy_version,
        delisting_policy_version=horizon.delisting_policy_version,
        return_definition_version=horizon.return_definition_version,
    )


def _event_timestamp(
    outcome: PaperEvaluationOutcome,
    *,
    as_of: datetime,
    eligible_at: datetime,
    admitted_evidence_available_at: tuple[datetime, ...],
) -> datetime:
    """Derive identity time only from policy time and admitted evidence.

    Before close eligibility, the existing fallback is the run cutoff.  At or after
    eligibility, unresolved outcomes use the run cutoff; resolved outcomes use the
    latest of close eligibility and their admitted evidence availability.
    """

    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("event timestamp as_of must be timezone-aware")
    if eligible_at.tzinfo is None or eligible_at.utcoffset() is None:
        raise ValueError("event timestamp eligibility must be timezone-aware")
    cutoff = as_of.astimezone(timezone.utc)
    if outcome.status == "pending" and as_of < eligible_at:
        return cutoff
    candidates = [eligible_at.astimezone(timezone.utc)]
    for evidence_time in admitted_evidence_available_at:
        if evidence_time.tzinfo is None or evidence_time.utcoffset() is None:
            raise ValueError("admitted evidence availability must be timezone-aware")
        admitted = evidence_time.astimezone(timezone.utc)
        if admitted > cutoff:
            raise ValueError("admitted evidence cannot be available after as_of")
        candidates.append(admitted)
    if outcome.status in {
        "provider_gap", "pending", "inactive_unresolved",
        "unsupported_identity_transition",
    }:
        candidates.append(cutoff)
    result = max(candidates)
    if result > cutoff:
        raise ValueError("evaluation event timestamp cannot be later than as_of")
    return result


def _authorized_retrieval_boundary(
    fetch: EvaluationFetchResult, *, operator_as_of: datetime
) -> tuple[datetime, frozenset[str], frozenset[str]]:
    """Validate one bounded retrieval initiated by this evaluation attempt."""

    if fetch.retrieval_policy_version is None:
        return operator_as_of, frozenset(), frozenset()
    if fetch.retrieval_policy_version != AUTHORIZED_RETRIEVAL_POLICY_VERSION:
        raise ValueError("unsupported evaluation retrieval policy version")
    assert fetch.retrieval_started_at is not None
    assert fetch.retrieval_completed_at is not None
    cutoff = operator_as_of.astimezone(timezone.utc)
    started = fetch.retrieval_started_at.astimezone(timezone.utc)
    completed = fetch.retrieval_completed_at.astimezone(timezone.utc)
    if started < cutoff:
        raise ValueError("authorized retrieval started before the evaluation cutoff")
    if completed < started:
        raise ValueError("authorized retrieval completion precedes its start")
    if completed > cutoff + AUTHORIZED_RETRIEVAL_MAX_ELAPSED:
        raise ValueError("authorized retrieval exceeded its bounded evaluation window")
    return (
        completed,
        frozenset(fetch.authorized_bar_ids),
        frozenset(fetch.authorized_split_ids),
    )


def _proposed_correction(
    repository: PaperPersistence,
    outcome: PaperEvaluationOutcome,
    *,
    evaluation_timestamp: datetime,
    provenance: dict[str, tuple[str, ...]],
) -> str | None:
    current = repository.current_outcome(outcome.decision_id, outcome.horizon)
    if current is None:
        return None
    existing_parent = current["correction_of_event_id"]
    same_id, _, _ = proposed_evaluation_identity(
        outcome,
        evaluation_timestamp=evaluation_timestamp,
        correction_of_event_id=existing_parent,
        provenance=provenance,
    )
    if same_id == current["evaluation_event_id"]:
        return existing_parent
    return current["evaluation_event_id"]


def _provider_gap_retry_state(
    repository: PaperPersistence,
    *,
    decision_id: str,
    horizon: str,
    as_of: datetime,
) -> Literal["eligible", "backoff", "exhausted"]:
    gaps = [
        row for row in repository.evaluation_events(decision_id, horizon)
        if row["outcome_code"] == "provider_gap"
    ]
    if not gaps:
        return "eligible"
    if len(gaps) >= PROVIDER_GAP_MAX_ATTEMPTS:
        return "exhausted"
    last = datetime.fromisoformat(gaps[-1]["evaluation_timestamp"])
    if as_of < last + PROVIDER_GAP_MIN_RETRY_INTERVAL:
        return "backoff"
    return "eligible"


def evaluate_due_paper_horizons(
    repository: PaperPersistence,
    *,
    as_of: datetime,
    mode: EvaluationMode = "dry_run",
    provider: DueEvaluationProvider | None = None,
) -> EvaluationRunReport:
    """Inspect, propose, and optionally persist due outcomes through one owner.

    Production writes require both gates and an exact single-event authorization.
    Scope is filtered before any provider access; append rechecks it independently.
    """

    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("due-evaluation as_of must be timezone-aware")
    cutoff = as_of.astimezone(timezone.utc)
    production_scope = None
    if (
        mode == "authorized_write"
        and repository.database.path.resolve() == PRODUCTION_DATABASE_PATH
    ):
        try:
            production_scope = repository.authorize_production_evaluation()
        except (PaperWriteDisabled, ValueError) as exc:
            raise PaperWriteDisabled("due-evaluation workflow cannot write production without exact authorization") from exc

    unresolved = repository.unresolved_horizons()
    due = repository.evaluations_due(cutoff)
    if production_scope is not None:
        due = [row for row in due if (
            row["decision_id"] == production_scope.decision_id
            and row["horizon_name"] == production_scope.horizon_name
        )]
        if any(row["resolved_session"] != production_scope.target_session.isoformat()
               or row["ticker_at_decision"] != production_scope.ticker for row in due):
            raise PaperWriteDisabled("production due row differs from authorized scope")
    items: list[EvaluationRunItem] = []
    sends = retries = writes_attempted = writes_committed = 0
    for row in due:
        decision = repository.paper_decision(row["decision_id"])
        if decision is None:
            raise ValueError("due horizon references a missing decision")
        horizon = _frozen_horizon(decision, row["horizon_name"])
        calendar_owner = us_equity_calendar(decision.exchange)
        session = calendar_owner.session(horizon.resolved_session_date)
        if session.closes_at is None:
            raise ValueError("frozen resolved session is not open in its calendar")
        eligible_at = session.closes_at + CLOSE_AVAILABILITY_BUFFER
        fetch: EvaluationFetchResult | None = None
        admitted_evidence_available_at: tuple[datetime, ...] = ()
        evaluation_boundary = cutoff
        retrieval_policy_version: str | None = None
        retrieval_started_at: datetime | None = None
        retrieval_completed_at: datetime | None = None
        if cutoff < eligible_at:
            outcome = _pending_before_available(decision, horizon)
        elif provider is None:
            items.append(EvaluationRunItem(
                decision_id=decision.decision_id, ticker=decision.ticker,
                horizon=horizon.name, resolved_session=horizon.resolved_session_date,
                close_eligible_at=eligible_at, outcome_code="failed",
                failure_code="provider_unavailable",
            ))
            continue
        else:
            retry_state = _provider_gap_retry_state(
                repository, decision_id=decision.decision_id,
                horizon=horizon.name, as_of=cutoff,
            )
            if retry_state != "eligible":
                items.append(EvaluationRunItem(
                    decision_id=decision.decision_id, ticker=decision.ticker,
                    horizon=horizon.name,
                    resolved_session=horizon.resolved_session_date,
                    close_eligible_at=eligible_at, outcome_code="pending",
                    reason_codes=(f"provider_gap_retry_{retry_state}",),
                ))
                continue
            fetch = provider.fetch_resolved_session(
                decision=decision,
                horizon=horizon,
                start_date=horizon.resolved_session_date,
                end_date=horizon.resolved_session_date,
                as_of=cutoff,
            )
            sends += fetch.provider_sends
            retries += fetch.retries
            if (
                fetch.requested_start != horizon.resolved_session_date
                or fetch.requested_end != horizon.resolved_session_date
            ):
                raise ValueError("provider fetch exceeded the frozen one-session scope")
            if fetch.failure_code is not None:
                items.append(EvaluationRunItem(
                    decision_id=decision.decision_id, ticker=decision.ticker,
                    horizon=horizon.name,
                    resolved_session=horizon.resolved_session_date,
                    close_eligible_at=eligible_at, outcome_code="failed",
                    failure_code=fetch.failure_code,
                ))
                continue
            (
                evaluation_boundary,
                authorized_bar_ids,
                authorized_split_ids,
            ) = _authorized_retrieval_boundary(
                fetch, operator_as_of=cutoff
            )
            retrieval_policy_version = fetch.retrieval_policy_version
            retrieval_started_at = fetch.retrieval_started_at
            retrieval_completed_at = fetch.retrieval_completed_at
            admitted_bars = tuple(
                bar for bar in fetch.bars
                if bar.available_at <= cutoff
                or (
                    bar.provider_snapshot_id in authorized_bar_ids
                    and retrieval_started_at is not None
                    and retrieval_completed_at is not None
                    and retrieval_started_at <= bar.available_at <= retrieval_completed_at
                )
            )
            # R1 authorizes only prices returned by this exact retrieval attempt.
            # Terminal evidence retains the original operator-cutoff rule.
            admitted_terminal = (
                fetch.terminal_event
                if fetch.terminal_event is None
                or fetch.terminal_event.available_at <= cutoff
                else None
            )
            admitted_split_coverage = fetch.split_basis_coverage
            if admitted_split_coverage is not None:
                relevant_evidence = tuple(
                    item for item in admitted_split_coverage.evidence
                    if admitted_split_coverage.requested_start
                    <= item.effective_date
                    <= admitted_split_coverage.requested_end
                )
                if retrieval_started_at is None or retrieval_completed_at is None:
                    coverage_in_window = admitted_split_coverage.available_at <= cutoff
                    evidence_authorized = all(
                        item.available_at <= cutoff
                        for item in relevant_evidence
                    )
                else:
                    coverage_in_window = (
                        retrieval_started_at
                        <= admitted_split_coverage.available_at
                        <= retrieval_completed_at
                    )
                    evidence_authorized = all(
                        item.source_id in authorized_split_ids
                        and retrieval_started_at <= item.available_at <= retrieval_completed_at
                        for item in relevant_evidence
                    )
                if not coverage_in_window or not evidence_authorized:
                    # Rejected coverage is diagnostic only; even its page ID
                    # must not enter the canonical outcome or event identity.
                    admitted_split_coverage = None
                elif len(relevant_evidence) != len(admitted_split_coverage.evidence):
                    admitted_split_coverage = admitted_split_coverage.model_copy(
                        update={"evidence": relevant_evidence}
                    )
            evaluated = evaluate_horizon_outcome_with_evidence(
                decision,
                horizon=horizon.name,
                calendar_owner=calendar_owner,
                evaluation_timestamp=evaluation_boundary,
                bars=admitted_bars,
                terminal_event=admitted_terminal,
                split_basis_coverage=admitted_split_coverage,
            )
            outcome = evaluated.outcome
            admitted_evidence_available_at = tuple(
                evaluated.admitted_evidence_available_at
            )

        event_timestamp = _event_timestamp(
            outcome,
            as_of=evaluation_boundary,
            eligible_at=eligible_at,
            admitted_evidence_available_at=admitted_evidence_available_at,
        )
        provenance = {
            # Bind page provenance to the bar actually used in the outcome.
            # A provider-wide batch may also describe rejected future evidence.
            "provider_request": tuple(sorted({
                page_id for bar in admitted_bars
                if bar.provider_snapshot_id == outcome.provider_snapshot_id
                for page_id in bar.source_page_ids
            })) if fetch is not None else (),
        }
        if retrieval_policy_version is not None:
            provenance["authorized_retrieval"] = (
                f"policy:{retrieval_policy_version}",
                f"started_at:{retrieval_started_at.isoformat()}",
                f"completed_at:{retrieval_completed_at.isoformat()}",
                *tuple(
                    f"evidence_available_at:{value.isoformat()}"
                    for value in admitted_evidence_available_at
                ),
            )
        if outcome.status == "evaluated":
            selected = next(bar for bar in admitted_bars
                            if bar.provider_snapshot_id == outcome.provider_snapshot_id)
            assert admitted_split_coverage is not None
            coverage_ids = (semantic_coverage_id(decision.ticker, admitted_split_coverage),)
            outcome = outcome.model_copy(update={
                "split_coverage_source_ids": coverage_ids,
                "admitted_evidence": AdmittedEvaluationEvidence(
                retrieval_started_at=retrieval_started_at or min(admitted_evidence_available_at),
                retrieval_completed_at=retrieval_completed_at or evaluation_boundary,
                authorized_boundary=evaluation_boundary,
                operator_cutoff=cutoff,
                retrieval_policy_version=retrieval_policy_version,
                bar=selected,
                coverage=admitted_split_coverage.model_copy(update={
                    "raw_page_source_ids": (), "source_ids": coverage_ids,
                }),
            )})
            provenance = {}
        repository.validate_evaluation(
            outcome,
            evaluation_timestamp=event_timestamp,
            as_of=evaluation_boundary,
        )
        correction = _proposed_correction(
            repository, outcome,
            evaluation_timestamp=event_timestamp,
            provenance=provenance,
        )
        event_id, event_hash, _ = proposed_evaluation_identity(
            outcome,
            evaluation_timestamp=event_timestamp,
            correction_of_event_id=correction,
            provenance=provenance,
        )
        attempted = committed = False
        if mode == "authorized_write":
            attempted = True
            writes_attempted += 1
            stored_id, inserted = repository.append_evaluation(
                outcome,
                evaluation_timestamp=event_timestamp,
                correction_of_event_id=correction,
                provenance=provenance,
                persisted_at=evaluation_boundary,
                as_of=evaluation_boundary,
            )
            if stored_id != event_id:
                raise AssertionError("persisted event identity differs from proposal")
            committed = inserted
            writes_committed += int(inserted)
        items.append(EvaluationRunItem(
            decision_id=decision.decision_id, ticker=decision.ticker,
            horizon=horizon.name, resolved_session=horizon.resolved_session_date,
            close_eligible_at=eligible_at, outcome_code=outcome.status,
            reason_codes=tuple(outcome.reason_codes), event_id=event_id,
            event_payload_hash=event_hash, correction_of_event_id=correction,
            observed_value=outcome.evaluation_value,
            price_return=outcome.price_return,
            write_attempted=attempted, write_committed=committed,
            retrieval_policy_version=retrieval_policy_version,
            retrieval_started_at=retrieval_started_at,
            retrieval_completed_at=retrieval_completed_at,
            admitted_evidence_available_at=admitted_evidence_available_at,
        ))

    counts = {name: sum(item.outcome_code == name for item in items) for name in (
        "evaluated", "pending", "provider_gap", "missing_price",
        "inactive_unresolved", "terminal_value_verified",
        "unsupported_identity_transition", "failed",
        "unsupported_adjustment_basis",
    )}
    run_id = canonical_hash({
        "workflow_version": EVALUATION_WORKFLOW_VERSION,
        "as_of": cutoff,
        "mode": mode,
        "due": tuple((row["decision_id"], row["horizon_name"]) for row in due),
        # The run identity predates the additive operator-display fields below;
        # exclude them so release operations do not redefine existing identity.
        "items": tuple(item.model_dump(
            mode="python", exclude={"observed_value", "price_return"}
        ) for item in items),
    })[:24]
    return EvaluationRunReport(
        run_id=run_id, as_of=cutoff, mode=mode,
        horizons_inspected=len(unresolved), due_count=len(due),
        evaluated_count=counts["evaluated"], pending_count=counts["pending"],
        provider_gap_count=counts["provider_gap"],
        missing_price_count=counts["missing_price"],
        inactive_unresolved_count=counts["inactive_unresolved"],
        terminal_count=counts["terminal_value_verified"],
        unsupported_identity_count=counts["unsupported_identity_transition"],
        unsupported_adjustment_basis_count=counts["unsupported_adjustment_basis"],
        failed_count=counts["failed"], provider_sends=sends, retries=retries,
        writes_attempted=writes_attempted, writes_committed=writes_committed,
        event_ids=tuple(item.event_id for item in items if item.event_id is not None),
        items=tuple(items),
    )
