"""Deterministic confidence derivation shared by production and V6 validation."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dedup import candidate_id
from schemas import Candidate

if TYPE_CHECKING:
    from shadow_live import LiveShadowClaim, LiveShadowPayload
    from schemas import ModelAnalysisPayload, ProductionModelAnalysisPayload


RULE_TABLE_VERSION = "confidence-rules-v2"
LEGACY_RULE_TABLE_VERSION = "confidence-rules-v1"
SupportedRuleTableVersion = Literal["confidence-rules-v1", "confidence-rules-v2"]

FinalConfidence = Literal["insufficient", "low", "medium", "high"]
ConfidenceRuleId = Literal["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]
ConfidenceCaveat = Literal[
    "inference",
    "limited_history",
    "missing_info",
    "conflict",
    "opinion_source",
    "single_source",
    "attributed_only",
]
ConfidenceScope = Literal["overall", "claim", "statement_made", "proposition_truth"]
SourceCategory = Literal["primary", "company", "reporting", "opinion", "forum_opinion"]
SupportStatus = Literal["supported", "inference", "unsupported", "unresolved"]
CausalStance = Literal[
    "none", "source_asserted", "attributed", "model_inference", "unresolved"
]


_CONFIDENCE_ORDER: dict[FinalConfidence, int] = {
    "insufficient": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}
_SOURCE_CEILING: dict[int, FinalConfidence] = {
    1: "high",
    2: "medium",
    3: "low",
    4: "low",
}

RULE_TABLE: tuple[dict[str, str], ...] = (
    {"rule_id": "R1", "meaning": "unresolved, unsupported, or uncovered evidence"},
    {"rule_id": "R2", "meaning": "forum or community opinion"},
    {"rule_id": "R3", "meaning": "contradiction across independent validated artifacts"},
    {"rule_id": "R4", "meaning": "model inference"},
    {"rule_id": "R5", "meaning": "attributed statement versus proposition"},
    {"rule_id": "R6", "meaning": "independently corroborated trusted evidence"},
    {"rule_id": "R7", "meaning": "fully covered Tier-1 factual evidence"},
    {"rule_id": "R8", "meaning": "material missing information or limited history"},
    {"rule_id": "R9", "meaning": "conservative default"},
)

_RULE_EXPLANATIONS: dict[str, str] = {
    "R1": "unresolved, unsupported, or incompletely covered evidence is insufficient",
    "R2": "forum or community opinion is limited to low confidence",
    "R3": "a contradiction across independent validated artifacts limits confidence to low",
    "R4": "model inference is capped at medium and cannot become sourced fact",
    "R5": "attribution verifies the statement separately from the proposition",
    "R6": "multiple independent validated sources support the source-tier ceiling",
    "R7": "fully covered, uncontested Tier-1 factual evidence reaches its ceiling",
    "R8": "material missing information or limited history caps the base result at medium",
    "R9": "no higher-confidence categorical rule applies, so the default is low",
}


def source_confidence_ceiling(source_tier: int) -> FinalConfidence:
    try:
        return _SOURCE_CEILING[source_tier]
    except KeyError as exc:
        raise ValueError("source tier must be between 1 and 4") from exc


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


class ConfidenceInputVector(BaseModel):
    """Closed, code-owned observations allowed to affect final confidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_tier: int = Field(ge=1, le=4)
    source_category: SourceCategory
    support_status: SupportStatus
    full_evidence_coverage: bool
    validated_evidence_ids: list[str]
    independent_source_ids: list[str]
    contradiction_evidence_ids: list[str]
    materially_relevant_missing_info: bool
    materially_relevant_limited_history: bool
    stance: CausalStance
    factual_claim: bool

    @field_validator(
        "validated_evidence_ids", "independent_source_ids", "contradiction_evidence_ids"
    )
    @classmethod
    def normalize_ids(cls, value: list[str]) -> list[str]:
        return _unique(value)

    @model_validator(mode="after")
    def pointers_are_validated(self) -> "ConfidenceInputVector":
        validated = set(self.validated_evidence_ids)
        if not set(self.independent_source_ids).issubset(validated):
            raise ValueError("independent sources must be validated evidence IDs")
        if not set(self.contradiction_evidence_ids).issubset(validated):
            raise ValueError("contradictions must cite validated evidence IDs")
        if self.full_evidence_coverage and not validated:
            raise ValueError("full evidence coverage requires validated evidence")
        return self


class ConfidenceRecord(BaseModel):
    """Persistable result with its exact rule version and input vector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: ConfidenceScope
    confidence: FinalConfidence
    rule_id: ConfidenceRuleId
    source_tier: int = Field(ge=1, le=4)
    confidence_ceiling: FinalConfidence
    caveats: list[ConfidenceCaveat]
    independent_source_count: int = Field(ge=0)
    validated_inputs: list[str]
    explanation: str
    rule_table_version: SupportedRuleTableVersion
    input_vector: ConfidenceInputVector


class ClaimConfidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_index: int = Field(ge=1)
    claim: ConfidenceRecord | None
    statement_made: ConfidenceRecord | None
    proposition_truth: ConfidenceRecord | None

    @model_validator(mode="after")
    def has_result(self) -> "ClaimConfidenceResult":
        if self.claim is None and self.statement_made is None and self.proposition_truth is None:
            raise ValueError("claim confidence result cannot be empty")
        return self


class PayloadConfidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_table_version: SupportedRuleTableVersion
    legacy_model_confidence_ignored: bool
    overall: ConfidenceRecord
    claims: list[ClaimConfidenceResult]


class StructuredMetricObservation(BaseModel):
    """Optional cheap deterministic input for same-metric/same-period conflicts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    period: str
    value: Decimal
    evidence_id: str

    @field_validator("metric", "period", "evidence_id")
    @classmethod
    def nonempty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("structured observation fields cannot be empty")
        return cleaned


def detect_structured_metric_contradictions(
    observations: Sequence[StructuredMetricObservation],
    *,
    relative_tolerance: Decimal = Decimal("0.01"),
) -> list[str]:
    """Return evidence IDs for materially different same-metric/same-period values."""

    if relative_tolerance < 0:
        raise ValueError("relative tolerance cannot be negative")
    groups: dict[tuple[str, str], list[StructuredMetricObservation]] = defaultdict(list)
    for item in observations:
        groups[(item.metric.casefold(), item.period)].append(item)
    conflicts: list[str] = []
    for items in groups.values():
        if len({item.evidence_id for item in items}) < 2:
            continue
        values = [item.value for item in items]
        scale = max((abs(value) for value in values), default=Decimal(0))
        materially_different = (
            max(values) != min(values)
            if scale == 0
            else (max(values) - min(values)) / scale > relative_tolerance
        )
        if materially_different:
            conflicts.extend(item.evidence_id for item in items)
    return _unique(conflicts)


def _cap(value: FinalConfidence, ceiling: FinalConfidence) -> FinalConfidence:
    return min(value, ceiling, key=_CONFIDENCE_ORDER.__getitem__)


def _caveats(vector: ConfidenceInputVector) -> list[ConfidenceCaveat]:
    result: list[ConfidenceCaveat] = []
    if vector.stance == "model_inference" or vector.support_status == "inference":
        result.append("inference")
    if vector.materially_relevant_limited_history:
        result.append("limited_history")
    if vector.materially_relevant_missing_info:
        result.append("missing_info")
    if vector.contradiction_evidence_ids:
        result.append("conflict")
    if vector.source_category in {"opinion", "forum_opinion"}:
        result.append("opinion_source")
    if len(vector.independent_source_ids) == 1:
        result.append("single_source")
    return result


def _validated_inputs(vector: ConfidenceInputVector) -> list[str]:
    return [
        f"source_tier:{vector.source_tier}",
        f"source_category:{vector.source_category}",
        f"support_status:{vector.support_status}",
        f"evidence_coverage:{'full' if vector.full_evidence_coverage else 'incomplete'}",
        f"validated_evidence_count:{len(vector.validated_evidence_ids)}",
        f"independent_source_count:{len(vector.independent_source_ids)}",
        f"stance:{vector.stance}",
        f"factual_claim:{str(vector.factual_claim).lower()}",
        f"verified_contradiction:{str(bool(vector.contradiction_evidence_ids)).lower()}",
        f"missing_info_relevant:{str(vector.materially_relevant_missing_info).lower()}",
        f"limited_history_relevant:{str(vector.materially_relevant_limited_history).lower()}",
    ]


def _record(
    vector: ConfidenceInputVector,
    *,
    scope: ConfidenceScope,
    confidence: FinalConfidence,
    rule_id: ConfidenceRuleId,
    extra_caveats: Sequence[ConfidenceCaveat] = (),
) -> ConfidenceRecord:
    ceiling = source_confidence_ceiling(vector.source_tier)
    bounded = _cap(confidence, ceiling)
    caveats = _unique([*_caveats(vector), *extra_caveats])
    explanation = (
        f"{rule_id}: {_RULE_EXPLANATIONS[rule_id]}; "
        f"final={bounded}, trusted_source_ceiling={ceiling}."
    )
    return ConfidenceRecord(
        scope=scope,
        confidence=bounded,
        rule_id=rule_id,
        source_tier=vector.source_tier,
        confidence_ceiling=ceiling,
        caveats=caveats,
        independent_source_count=len(vector.independent_source_ids),
        validated_inputs=_validated_inputs(vector),
        explanation=explanation,
        rule_table_version=RULE_TABLE_VERSION,
        input_vector=vector,
    )


def derive_confidence(
    vector: ConfidenceInputVector,
    *,
    scope: ConfidenceScope = "claim",
) -> ConfidenceRecord:
    """Apply the small categorical table without reading source prose or model confidence."""

    ceiling = source_confidence_ceiling(vector.source_tier)
    if (
        vector.support_status in {"unresolved", "unsupported"}
        or not vector.full_evidence_coverage
    ):
        return _record(vector, scope=scope, confidence="insufficient", rule_id="R1")
    if vector.source_category in {"opinion", "forum_opinion"}:
        return _record(vector, scope=scope, confidence="low", rule_id="R2")
    if vector.contradiction_evidence_ids:
        return _record(vector, scope=scope, confidence="low", rule_id="R3")
    if vector.stance == "model_inference" or vector.support_status == "inference":
        inference_base: FinalConfidence = (
            ceiling if len(vector.independent_source_ids) >= 2 else "low"
        )
        return _record(
            vector,
            scope=scope,
            confidence=_cap(inference_base, "medium"),
            rule_id="R4",
        )
    if vector.stance == "attributed":
        attributed_confidence: FinalConfidence = (
            _cap(source_confidence_ceiling(vector.source_tier), "medium")
            if len(vector.independent_source_ids) >= 2
            else "low"
        )
        return _record(
            vector,
            scope=scope,
            confidence=attributed_confidence,
            rule_id="R5",
            extra_caveats=("attributed_only",),
        )

    if len(vector.independent_source_ids) >= 2 and vector.source_tier <= 2:
        base: FinalConfidence = ceiling
        base_rule: ConfidenceRuleId = "R6"
    elif vector.source_tier == 1 and vector.factual_claim:
        base = ceiling
        base_rule = "R7"
    else:
        base = "low"
        base_rule = "R9"

    if vector.materially_relevant_missing_info or vector.materially_relevant_limited_history:
        return _record(
            vector,
            scope=scope,
            confidence=_cap(base, "medium"),
            rule_id="R8",
        )
    return _record(vector, scope=scope, confidence=base, rule_id=base_rule)


def derive_attributed_confidence(
    vector: ConfidenceInputVector,
) -> tuple[ConfidenceRecord, ConfidenceRecord]:
    """Separate proof that a statement was made from truth of its proposition."""

    if vector.stance != "attributed":
        raise ValueError("attributed confidence requires attributed stance")
    if vector.support_status in {"unsupported", "unresolved"} or not vector.full_evidence_coverage:
        statement = _record(
            vector, scope="statement_made", confidence="insufficient", rule_id="R1"
        )
        proposition = _record(
            vector,
            scope="proposition_truth",
            confidence="insufficient",
            rule_id="R1",
            extra_caveats=("attributed_only",),
        )
        return statement, proposition
    statement = _record(
        vector,
        scope="statement_made",
        confidence=source_confidence_ceiling(vector.source_tier),
        rule_id="R5",
    )
    if vector.contradiction_evidence_ids:
        proposition_confidence: FinalConfidence = "low"
        proposition_rule: ConfidenceRuleId = "R3"
    elif len(vector.independent_source_ids) >= 2:
        proposition_confidence = _cap(source_confidence_ceiling(vector.source_tier), "medium")
        proposition_rule = "R5"
    else:
        proposition_confidence = "low"
        proposition_rule = "R5"
    proposition = _record(
        vector,
        scope="proposition_truth",
        confidence=proposition_confidence,
        rule_id=proposition_rule,
        extra_caveats=("attributed_only",),
    )
    return statement, proposition


def _candidate_evidence_class(candidate: Candidate) -> str:
    source_type = candidate.source_type.casefold()
    return (
        "forum_community"
        if "forum" in source_type or "community" in source_type
        else "article_event"
    )


def _source_category(candidate: Candidate) -> SourceCategory:
    """Classify only trusted candidate metadata; model claim labels cannot change it."""

    if _candidate_evidence_class(candidate) == "forum_community":
        return "forum_opinion"
    source_type = candidate.source_type.casefold()
    if "filing" in source_type or "regulatory" in source_type:
        return "primary"
    if "company" in source_type or "transcript" in source_type:
        return "company"
    if "opinion" in source_type or "commentary" in source_type:
        return "opinion"
    return "reporting"


def _claim_vector(
    candidate: Candidate,
    claim: "LiveShadowClaim",
    *,
    evidence_artifacts: Mapping[str, str],
    contradiction_evidence_ids: Sequence[str] = (),
) -> ConfidenceInputVector:
    unknown = set(claim.evidence_ids) - set(evidence_artifacts)
    if unknown:
        raise ValueError("claim references an unvalidated evidence pointer")
    stance: CausalStance = claim.causality.stance if claim.causality else "none"
    support_status: SupportStatus = (
        "unresolved" if stance == "unresolved" else claim.support
    )
    validated_ids = list(claim.evidence_ids)
    independent_ids = _independent_evidence_ids(validated_ids, evidence_artifacts)
    relevant_history = (
        "limited_history" in claim.facets
        and bool(validated_ids)
        and claim.category in {"source_opinion", "ai_inference", "forum_community_opinion"}
    )
    return ConfidenceInputVector(
        source_tier=candidate.source_tier,
        source_category=_source_category(candidate),
        support_status=support_status,
        full_evidence_coverage=bool(validated_ids),
        validated_evidence_ids=validated_ids,
        independent_source_ids=independent_ids,
        contradiction_evidence_ids=list(contradiction_evidence_ids),
        materially_relevant_missing_info=False,
        materially_relevant_limited_history=relevant_history,
        stance=stance,
        factual_claim=claim.category == "reported_fact",
    )


def derive_payload_confidence(
    payload: "LiveShadowPayload",
    candidate: Candidate,
    *,
    validated_evidence_artifacts: Mapping[str, str] | None = None,
) -> PayloadConfidenceResult:
    """Derive V6 confidence; conflict requires distinct validated source artifacts."""

    allowed_id = f"article_event:{candidate_id(candidate)}"
    evidence_artifacts = {allowed_id: allowed_id}
    for evidence_id, artifact_id in (validated_evidence_artifacts or {}).items():
        cleaned_evidence = str(evidence_id).strip()
        cleaned_artifact = str(artifact_id).strip()
        if not cleaned_evidence or not cleaned_artifact:
            raise ValueError("validated evidence and artifact identities cannot be blank")
        if cleaned_evidence == allowed_id and cleaned_artifact != allowed_id:
            raise ValueError("candidate evidence artifact identity cannot be overridden")
        evidence_artifacts[cleaned_evidence] = cleaned_artifact
    for claim in payload.claims:
        unknown = set(claim.evidence_ids) - set(evidence_artifacts)
        if unknown:
            raise ValueError("claim references an unvalidated evidence pointer")
        if (
            _candidate_evidence_class(candidate) == "forum_community"
            and claim.evidence_ids
            and claim.category != "forum_community_opinion"
        ):
            raise ValueError("forum evidence cannot be promoted into a core category")

    conflict_claims = [claim for claim in payload.claims if "contradiction" in claim.facets]
    conflict_evidence = _unique([
        evidence_id for claim in conflict_claims for evidence_id in claim.evidence_ids
    ])
    contradiction_ids = (
        _independent_evidence_ids(conflict_evidence, evidence_artifacts)
        if len(conflict_claims) >= 2
        else []
    )
    if len(contradiction_ids) < 2:
        contradiction_ids = []
    claim_results: list[ClaimConfidenceResult] = []
    vectors: list[ConfidenceInputVector] = []
    for index, claim in enumerate(payload.claims, 1):
        claim_conflict_ids = (
            contradiction_ids
            if "contradiction" in claim.facets
            and set(contradiction_ids).issubset(set(claim.evidence_ids))
            else []
        )
        vector = _claim_vector(
            candidate,
            claim,
            evidence_artifacts=evidence_artifacts,
            contradiction_evidence_ids=claim_conflict_ids,
        )
        vectors.append(vector)
        if vector.stance == "attributed":
            statement, proposition = derive_attributed_confidence(vector)
            claim_results.append(ClaimConfidenceResult(
                claim_index=index,
                claim=None,
                statement_made=statement,
                proposition_truth=proposition,
            ))
        else:
            claim_results.append(ClaimConfidenceResult(
                claim_index=index,
                claim=derive_confidence(vector),
                statement_made=None,
                proposition_truth=None,
            ))

    overall_support: SupportStatus
    if any(vector.support_status == "unresolved" for vector in vectors):
        overall_support = "unresolved"
    elif any(vector.support_status == "unsupported" for vector in vectors):
        overall_support = "unsupported"
    elif any(vector.support_status == "inference" for vector in vectors):
        overall_support = "inference"
    else:
        overall_support = "supported"
    if any(vector.stance == "model_inference" for vector in vectors):
        overall_stance: CausalStance = "model_inference"
    elif any(vector.stance == "unresolved" for vector in vectors):
        overall_stance = "unresolved"
    elif any(vector.stance == "attributed" for vector in vectors):
        overall_stance = "attributed"
    elif any(vector.stance == "source_asserted" for vector in vectors):
        overall_stance = "source_asserted"
    else:
        overall_stance = "none"
    all_evidence = _unique([
        evidence_id for vector in vectors for evidence_id in vector.validated_evidence_ids
    ])
    overall_vector = ConfidenceInputVector(
        source_tier=candidate.source_tier,
        source_category=_source_category(candidate),
        support_status=overall_support,
        full_evidence_coverage=all(vector.full_evidence_coverage for vector in vectors),
        validated_evidence_ids=all_evidence,
        independent_source_ids=_independent_evidence_ids(all_evidence, evidence_artifacts),
        contradiction_evidence_ids=contradiction_ids,
        materially_relevant_missing_info=False,
        materially_relevant_limited_history=any(
            vector.materially_relevant_limited_history for vector in vectors
        ),
        stance=overall_stance,
        factual_claim=all(vector.factual_claim for vector in vectors),
    )
    overall = derive_confidence(overall_vector, scope="overall")
    return PayloadConfidenceResult(
        rule_table_version=RULE_TABLE_VERSION,
        legacy_model_confidence_ignored=True,
        overall=overall,
        claims=claim_results,
    )


def _independent_evidence_ids(
    evidence_ids: Sequence[str], evidence_artifacts: Mapping[str, str]
) -> list[str]:
    """Choose one deterministic evidence ID per distinct underlying artifact."""

    representatives: list[str] = []
    seen_artifacts: set[str] = set()
    for evidence_id in _unique(evidence_ids):
        artifact_id = evidence_artifacts.get(evidence_id)
        if artifact_id is None:
            raise ValueError("evidence identity is not mapped to a validated artifact")
        if artifact_id not in seen_artifacts:
            seen_artifacts.add(artifact_id)
            representatives.append(evidence_id)
    return representatives


def derive_production_confidence(
    payload: "ProductionModelAnalysisPayload | ModelAnalysisPayload",
    candidate: Candidate,
) -> PayloadConfidenceResult:
    """Derive new production confidence without reading legacy model confidence."""

    allowed_id = f"article_event:{candidate_id(candidate)}"
    evidence_artifacts = {allowed_id: allowed_id}
    vectors: list[ConfidenceInputVector] = []
    claim_results: list[ClaimConfidenceResult] = []
    missing_info = bool(payload.information_needed_next)
    for index, claim in enumerate(payload.claims, 1):
        unknown = set(claim.evidence_ids) - {allowed_id}
        if unknown:
            raise ValueError("claim references an unvalidated evidence pointer")
        if (
            _candidate_evidence_class(candidate) == "forum_community"
            and claim.evidence_ids
            and claim.category != "forum_community_opinion"
        ):
            raise ValueError("forum evidence cannot be promoted into a core category")
        stance: CausalStance = {
            "none": "none",
            "attributed": "attributed",
            "inference": "model_inference",
        }[claim.causal_role]
        validated_ids = list(claim.evidence_ids)
        vector = ConfidenceInputVector(
            source_tier=candidate.source_tier,
            source_category=_source_category(candidate),
            support_status=claim.support,
            full_evidence_coverage=bool(validated_ids),
            validated_evidence_ids=validated_ids,
            independent_source_ids=_independent_evidence_ids(
                validated_ids, evidence_artifacts
            ),
            contradiction_evidence_ids=[],
            materially_relevant_missing_info=missing_info,
            materially_relevant_limited_history="limited_history" in claim.facets,
            stance=stance,
            factual_claim=claim.category == "reported_fact",
        )
        vectors.append(vector)
        if stance == "attributed":
            statement, proposition = derive_attributed_confidence(vector)
            claim_results.append(ClaimConfidenceResult(
                claim_index=index,
                claim=None,
                statement_made=statement,
                proposition_truth=proposition,
            ))
        else:
            claim_results.append(ClaimConfidenceResult(
                claim_index=index,
                claim=derive_confidence(vector),
                statement_made=None,
                proposition_truth=None,
            ))

    if any(vector.support_status == "unsupported" for vector in vectors):
        overall_support: SupportStatus = "unsupported"
    elif any(vector.support_status == "inference" for vector in vectors):
        overall_support = "inference"
    else:
        overall_support = "supported"
    if any(vector.stance == "model_inference" for vector in vectors):
        overall_stance: CausalStance = "model_inference"
    elif any(vector.stance == "attributed" for vector in vectors):
        overall_stance = "attributed"
    else:
        overall_stance = "none"
    all_evidence = _unique([
        evidence_id for vector in vectors for evidence_id in vector.validated_evidence_ids
    ])
    overall_vector = ConfidenceInputVector(
        source_tier=candidate.source_tier,
        source_category=_source_category(candidate),
        support_status=overall_support,
        full_evidence_coverage=all(vector.full_evidence_coverage for vector in vectors),
        validated_evidence_ids=all_evidence,
        independent_source_ids=_independent_evidence_ids(all_evidence, evidence_artifacts),
        contradiction_evidence_ids=[],
        materially_relevant_missing_info=missing_info,
        materially_relevant_limited_history=any(
            vector.materially_relevant_limited_history for vector in vectors
        ),
        stance=overall_stance,
        factual_claim=all(vector.factual_claim for vector in vectors),
    )
    return PayloadConfidenceResult(
        rule_table_version=RULE_TABLE_VERSION,
        legacy_model_confidence_ignored=True,
        overall=derive_confidence(overall_vector, scope="overall"),
        claims=claim_results,
    )
