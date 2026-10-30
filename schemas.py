"""Strict data models used between discovery, analysis, rendering, and reporting."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timezone
import re
from typing import Annotated, Literal, Optional

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from identifiers import normalize_ticker


Importance = Literal["critical", "high", "medium", "low"]
LegacyModelConfidence = Literal["high", "medium", "low"]
CodeOwnedConfidence = Literal["insufficient", "low", "medium", "high"]
# Compatibility alias for callers that deserialize historical V5 payloads.
Confidence = LegacyModelConfidence
Sentiment = Literal["bullish", "bearish", "mixed", "neutral"]
EvidenceClass = Literal[
    "source_artifact", "article_event", "filing", "metric",
    "structured_finding", "provenance_run", "forum_community",
]
ClaimCategory = Literal[
    "reported_fact", "management_statement", "source_opinion",
    "ai_inference", "forum_community_opinion",
]
ClaimSupport = Literal["supported", "inference", "unsupported"]
CausalRole = Literal["none", "attributed", "inference"]
ClaimFacet = Literal[
    "catalyst", "risk", "bullish", "bearish", "novelty", "contradiction",
    "limited_history",
]
Decision = Literal["REJECT", "LINK ONLY", "SAVE", "SAVE + REVIEW", "DEFER"]
RejectionReason = Literal[
    "irrelevant",
    "incidental mention",
    "duplicate",
    "no new information",
    "low-quality source",
    "promotional",
    "unsupported rumor",
    "insufficient financial relevance",
    "inaccessible / insufficient content",
    "low-value prediction or clickbait",
]
DuplicateStatus = Literal[
    "EXACT_DUPLICATE",
    "SYNDICATED_DUPLICATE",
    "SAME_EVENT_NO_NEW_INFO",
    "SAME_EVENT_CONFIRMATION",
    "SAME_EVENT_MEANINGFUL_UPDATE",
    "NEW_EVENT",
]


def _clean_nonempty(value: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("value cannot be empty")
    return cleaned


NonEmptyText = Annotated[str, AfterValidator(_clean_nonempty)]


_CAUSAL_ASSERTION = re.compile(
    r"\b(?:caus(?:e|es|ed|ing)|dr(?:ive|ives|ove|iven|iving)|led to|leads? to|"
    r"result(?:s|ed|ing)? (?:in|from)|due to|because(?: of)?|stemming from|responsible for|"
    r"contribut(?:e|es|ed|ing) to|affect(?:s|ed|ing)?|impact(?:s|ed|ing)?|therefore)\b"
    r"|\battribut(?:able|e|es|ed|ing)\b(?:\W+\w+){0,8}\W+\bto\b"
    r"|\bconsistent with (?:an? )?(?:possible|potential)\b",
    re.IGNORECASE,
)
_CAUSAL_UNCERTAINTY = re.compile(
    r"\b(?:may|might|could|appears?|suggests?|potentially|possibly|possible|perhaps)\b",
    re.IGNORECASE,
)
_EVIDENTIAL_SUBJECT = r"(?:evidence|data|filing|source|record|disclosure|analysis)"
_EVIDENTIAL_DENIAL = re.compile(
    rf"\b(?:"
    rf"(?:the\s+)?(?:available\s+)?{_EVIDENTIAL_SUBJECT}\s+(?:does|do)\s+not\s+"
    r"(?:establish|show|demonstrate|prove|substantiate)\s+(?:that|whether)"
    r"|no(?:\s+\w+){0,4}\s+evidence\s+(?:establishing\s+)?(?:that|whether)"
    r"|insufficient\s+evidence\s+to\s+conclude\s+(?:that|whether)"
    r"|cannot\s+(?:determine|conclude)\s+(?:that|whether)"
    r"|(?:it\s+is\s+)?(?:unclear|uncertain)\s+whether"
    r"|not\s+enough\s+evidence\s+to\s+say\s+that"
    rf"|(?:the\s+)?(?:available\s+)?{_EVIDENTIAL_SUBJECT}\s+(?:does|do)\s+not\s+"
    r"support\s+the\s+claim\s+that"
    rf"|(?:the\s+)?(?:available\s+)?{_EVIDENTIAL_SUBJECT}\s+fails?\s+to\s+"
    r"establish\s+(?:that|whether)"
    r")\b",
    re.IGNORECASE,
)
_DENIAL_SCOPE_END = re.compile(r"[.!?;]|,?\s+\b(?:but|yet|however)\b", re.IGNORECASE)
_ATTRIBUTOR = (
    r"(?:management|the company|company management|chief executive|CEO|CFO|executive|"
    r"analyst|source|article|report|regulator|agency|forum user|forum participant|"
    r"forum poster|participant|poster|author)"
)
_EXPLICIT_ATTRIBUTION = re.compile(
    rf"(?:\baccording to\s+{_ATTRIBUTOR}\b|\b{_ATTRIBUTOR}\b[^.!?]{{0,80}}\b"
    r"(?:said|says|stated|states|reported|reports|attributed|attributes|claimed|claims|"
    r"argued|argues|suggested|suggests|wrote|asserted|asserts)\b)",
    re.IGNORECASE,
)


def _asserted_causal_matches(text: str) -> list[re.Match[str]]:
    """Return causal matches not scoped under a narrow evidential denial."""

    denied_spans: list[tuple[int, int]] = []
    for denial in _EVIDENTIAL_DENIAL.finditer(text):
        boundary = _DENIAL_SCOPE_END.search(text, denial.end())
        denied_spans.append((denial.end(), boundary.start() if boundary else len(text)))
    return [
        match for match in _CAUSAL_ASSERTION.finditer(text)
        if not any(start <= match.start() < end for start, end in denied_spans)
    ]


def has_causal_assertion(text: str) -> bool:
    """Return whether wording asserts causality outside evidential-denial scope."""

    return bool(_asserted_causal_matches(text))


def causal_assertions_are_hedged(text: str) -> bool:
    """Require uncertainty in the same sentence before each causal construction."""

    matches = _asserted_causal_matches(text)
    if not matches:
        return False
    for match in matches:
        sentence_start = max(text.rfind(mark, 0, match.start()) for mark in ".!?;") + 1
        if not _CAUSAL_UNCERTAINTY.search(text[sentence_start : match.end()]):
            return False
    return True


def has_explicit_causal_attribution(text: str) -> bool:
    """Require attribution before each causal assertion in the same sentence/scope."""

    matches = _asserted_causal_matches(text)
    if not matches:
        return False
    for causal_match in matches:
        scope_start = max(
            (text.rfind(mark, 0, causal_match.start()) for mark in ".!?;"),
            default=-1,
        ) + 1
        scope = text[scope_start : causal_match.start()]
        if _EXPLICIT_ATTRIBUTION.search(scope) is None:
            return False
    return True


def _sequence_input(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    raise ValueError("value must be a string or sequence")


def _unique_clean(values: list[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        identity = cleaned.casefold()
        if identity not in seen:
            seen.add(identity)
            result.append(cleaned)
    return result


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


class Candidate(BaseModel):
    """Normalized source item before filtering or analysis."""

    model_config = ConfigDict(extra="forbid")

    title: NonEmptyText
    source: NonEmptyText
    source_url: HttpUrl
    published: Optional[AwareDatetime] = None
    discovered: AwareDatetime
    author: Optional[str] = None
    content: str = ""
    snippet: str = ""
    source_tier: int = Field(ge=1, le=4)
    companies: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    source_type: NonEmptyText
    content_scope: Literal["full_text", "excerpt", "metadata_only"] = "excerpt"

    @field_validator("published", "discovered", mode="after")
    @classmethod
    def normalize_datetimes(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None

    @field_validator("content", "snippet", mode="before")
    @classmethod
    def clean_optional_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("author", mode="before")
    @classmethod
    def clean_author(cls, value: object) -> Optional[str]:
        cleaned = str(value or "").strip()
        return cleaned or None

    @field_validator("companies", mode="before")
    @classmethod
    def clean_companies(cls, value: object) -> list[str]:
        return _unique_clean(_sequence_input(value))

    @field_validator("tickers", mode="before")
    @classmethod
    def normalize_tickers(cls, value: object) -> list[str]:
        return [normalize_ticker(ticker) for ticker in _unique_clean(_sequence_input(value))]


class EvidenceItem(BaseModel):
    """Trusted evidence identity available to generated claims."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9:._/-]{2,127}$")
    evidence_class: EvidenceClass
    label: NonEmptyText
    source_url: Optional[HttpUrl] = None
    available_at: Optional[AwareDatetime] = None
    provenance_id: Optional[str] = None

    @field_validator("available_at", mode="after")
    @classmethod
    def normalize_available_at(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None


class EvidenceLocation(BaseModel):
    """Optional stable location inside one evidence item; never inferred."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9:._/-]{2,127}$")
    location_type: Literal["line_range", "structured_field", "fact_id", "paragraph", "span"]
    identifier: NonEmptyText
    start: Optional[int] = Field(default=None, ge=0)
    end: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def ordered_range(self) -> "EvidenceLocation":
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("evidence location end cannot precede start")
        return self


class ClaimContent(BaseModel):
    """Model-owned claim meaning, without a model-owned identifier."""

    model_config = ConfigDict(extra="forbid")

    text: NonEmptyText
    category: ClaimCategory
    support: ClaimSupport
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_locations: list[EvidenceLocation] = Field(default_factory=list)
    causal_role: CausalRole = Field(
        default="none",
        description=(
            "none: no causal assertion; attributed: an explicitly named source makes the "
            "causal assertion; inference: the model makes an explicitly uncertain causal inference"
        ),
    )
    facets: list[ClaimFacet] = Field(default_factory=list)

    @field_validator("evidence_ids", "facets", mode="before")
    @classmethod
    def clean_claim_lists(cls, value: object) -> list[str]:
        return _unique_clean(_sequence_input(value))

    @model_validator(mode="after")
    def validate_support_contract(self) -> ClaimContent:
        if self.support in {"supported", "inference"} and not self.evidence_ids:
            raise ValueError("supported claims and inferences require evidence references")
        if self.support == "unsupported" and self.evidence_ids:
            raise ValueError("unsupported claims cannot cite evidence")
        if self.category == "ai_inference" and self.support == "supported":
            raise ValueError("AI inference must use support='inference', not 'supported'")
        if self.category != "ai_inference" and self.support == "inference":
            raise ValueError("support='inference' is reserved for AI inference")
        if self.support == "unsupported" and self.category != "ai_inference":
            raise ValueError("only explicitly labeled AI inference may be unsupported")
        causal_language = has_causal_assertion(self.text)
        if causal_language and self.causal_role == "none":
            raise ValueError("causal language requires a causal-capable role")
        if self.causal_role != "none" and not causal_language:
            raise ValueError("causal role requires explicit causal language")
        if self.causal_role == "attributed":
            if self.category not in {
                "management_statement", "source_opinion", "forum_community_opinion",
            } or self.support != "supported":
                raise ValueError(
                    "attributed causality requires a supported management/source/forum claim"
                )
            if not has_explicit_causal_attribution(self.text):
                raise ValueError("attributed causality must remain explicit in claim text")
        if self.causal_role == "inference":
            if self.category != "ai_inference" or self.support not in {"inference", "unsupported"}:
                raise ValueError("causal inference must remain an AI inference")
            if not causal_assertions_are_hedged(self.text):
                raise ValueError("causal inference must use explicit uncertainty language")
        location_ids = {location.evidence_id for location in self.evidence_locations}
        if not location_ids.issubset(set(self.evidence_ids)):
            raise ValueError("evidence locations must reference the claim's evidence IDs")
        return self


class ModelEvidenceClaim(ClaimContent):
    """A claim exactly as returned by the model; Python assigns its ID."""


class EvidenceClaim(ClaimContent):
    """A validated claim with a deterministic code-owned identifier."""

    claim_id: str = Field(pattern=r"^claim:[A-Za-z0-9._-]{1,100}$")


class ModelAnalysisPayload(BaseModel):
    """Historical V5 model response, retained only for exact legacy deserialization."""

    model_config = ConfigDict(extra="forbid")

    themes: list[str]
    sector: Optional[str]
    importance: Importance
    confidence: Confidence
    sentiment: Sentiment
    claims: list[ModelEvidenceClaim]
    open_questions: list[str]
    information_needed_next: list[str]

    @model_validator(mode="after")
    def require_claims(self) -> "ModelAnalysisPayload":
        if not self.claims:
            raise ValueError("analysis requires at least one claim")
        normalized = [" ".join(claim.text.split()).casefold() for claim in self.claims]
        if len(set(normalized)) != len(normalized):
            raise ValueError("model claims contain duplicate text")
        return self


class ProductionModelAnalysisPayload(BaseModel):
    """Current model response; final confidence is intentionally absent and code-owned."""

    model_config = ConfigDict(extra="forbid")

    themes: list[str]
    sector: Optional[str]
    importance: Importance
    sentiment: Sentiment
    claims: list[ModelEvidenceClaim]
    open_questions: list[str]
    information_needed_next: list[str]

    @model_validator(mode="after")
    def require_claims(self) -> "ProductionModelAnalysisPayload":
        if not self.claims:
            raise ValueError("analysis requires at least one claim")
        normalized = [" ".join(claim.text.split()).casefold() for claim in self.claims]
        if len(set(normalized)) != len(normalized):
            raise ValueError("model claims contain duplicate text")
        return self


def assign_claim_ids(claims: list[ModelEvidenceClaim]) -> list[EvidenceClaim]:
    """Assign stable IDs from validated model order; the model never supplies IDs."""

    return [
        EvidenceClaim(claim_id=f"claim:{index:03d}", **claim.model_dump())
        for index, claim in enumerate(claims, 1)
    ]


def select_summary_claims(claims: list[EvidenceClaim], limit: int = 4) -> list[EvidenceClaim]:
    """Select a small deterministic summary while retaining key limitations."""

    selected: list[EvidenceClaim] = []

    def add(claim: EvidenceClaim) -> None:
        if claim.claim_id not in {item.claim_id for item in selected} and len(selected) < limit:
            selected.append(claim)

    for required_facet in ("contradiction", "limited_history"):
        for claim in claims:
            if required_facet in claim.facets:
                add(claim)
                break
    for claim in claims:
        if claim.support != "unsupported":
            add(claim)
    if not selected and claims:
        add(claims[0])
    return selected


def summary_text_for_claims(claims: list[EvidenceClaim]) -> str:
    """Render only claim text plus deterministic attribution labels."""

    labels = {
        "reported_fact": "Reported fact",
        "management_statement": "Management statement",
        "source_opinion": "Source opinion",
        "ai_inference": "AI inference",
        "forum_community_opinion": "Forum/community opinion",
    }
    return " ".join(f"{labels[claim.category]}: {claim.text}" for claim in claims)


def derived_claim_views(claims: list[EvidenceClaim]) -> dict[str, list[str]]:
    """Derive all repeated human-readable views from the authoritative claims."""

    categories = {
        "reported_facts": "reported_fact",
        "company_management_statements": "management_statement",
        "analyst_source_opinion": "source_opinion",
        "ai_inference": "ai_inference",
    }
    facets = {
        "potential_catalysts": "catalyst",
        "potential_risks": "risk",
        "bullish_evidence": "bullish",
        "bearish_evidence": "bearish",
        "what_is_new": "novelty",
        "contradictions": "contradiction",
    }
    result = {
        field: [claim.text for claim in claims if claim.category == category]
        for field, category in categories.items()
    }
    result.update({
        field: [claim.text for claim in claims if facet in claim.facets]
        for field, facet in facets.items()
    })
    return result


class AnalysisPayload(BaseModel):
    """Fields the model may produce before deterministic rendering."""

    model_config = ConfigDict(extra="forbid")

    title: NonEmptyText
    source: NonEmptyText
    source_url: HttpUrl
    author: Optional[str] = None
    published: Optional[date] = None
    accessed: AwareDatetime
    companies: list[str]
    tickers: list[str]
    themes: list[str]
    sector: Optional[str] = None
    importance: Importance
    confidence: CodeOwnedConfidence
    analysis_version: Literal["market-analysis-v5", "market-analysis-v7"] = (
        "market-analysis-v5"
    )
    confidence_owner: Literal["legacy_model", "python"] = "legacy_model"
    confidence_rule_version: Optional[Literal["confidence-rules-v2"]] = None
    sentiment: Sentiment
    needs_review: bool
    executive_summary: NonEmptyText
    executive_summary_claim_ids: list[str]
    evidence_catalog: list[EvidenceItem]
    claims: list[EvidenceClaim]
    reported_facts: list[str]
    company_management_statements: list[str]
    analyst_source_opinion: list[str]
    ai_inference: list[str]
    potential_catalysts: list[str]
    potential_risks: list[str]
    bullish_evidence: list[str]
    bearish_evidence: list[str]
    what_is_new: list[str]
    contradictions: list[str]
    open_questions: list[str]
    information_needed_next: list[str]

    @field_validator("accessed", mode="after")
    @classmethod
    def normalize_accessed(cls, value: datetime) -> datetime:
        return _utc(value)

    @field_validator("author", "sector", mode="before")
    @classmethod
    def clean_optional_fields(cls, value: object) -> Optional[str]:
        cleaned = str(value or "").strip()
        return cleaned or None

    @field_validator(
        "companies",
        "themes",
        "reported_facts",
        "company_management_statements",
        "analyst_source_opinion",
        "ai_inference",
        "potential_catalysts",
        "potential_risks",
        "bullish_evidence",
        "bearish_evidence",
        "what_is_new",
        "contradictions",
        "open_questions",
        "information_needed_next",
        mode="before",
    )
    @classmethod
    def clean_lists(cls, value: object) -> list[str]:
        return _unique_clean(_sequence_input(value))

    @field_validator("tickers", mode="before")
    @classmethod
    def normalize_analysis_tickers(cls, value: object) -> list[str]:
        return [normalize_ticker(ticker) for ticker in _unique_clean(_sequence_input(value))]

    @field_validator("executive_summary_claim_ids", mode="before")
    @classmethod
    def clean_summary_claim_ids(cls, value: object) -> list[str]:
        return _unique_clean(_sequence_input(value))

    @model_validator(mode="after")
    def validate_claim_level_evidence(self) -> "AnalysisPayload":
        if self.confidence_owner == "python":
            if self.analysis_version != "market-analysis-v7":
                raise ValueError("Python-owned confidence requires market-analysis-v7")
            if self.confidence_rule_version != "confidence-rules-v2":
                raise ValueError("Python-owned confidence requires confidence-rules-v2")
        else:
            if self.analysis_version != "market-analysis-v5":
                raise ValueError("legacy model confidence requires market-analysis-v5")
            if self.confidence_rule_version is not None:
                raise ValueError("legacy model confidence cannot claim a Python rule version")
            if self.confidence == "insufficient":
                raise ValueError("legacy V5 confidence does not support insufficient")
        evidence = {item.evidence_id: item for item in self.evidence_catalog}
        if len(evidence) != len(self.evidence_catalog):
            raise ValueError("evidence_catalog contains duplicate evidence IDs")
        claims = {claim.claim_id: claim for claim in self.claims}
        if len(claims) != len(self.claims):
            raise ValueError("claims contain duplicate claim IDs")
        unknown_summary = set(self.executive_summary_claim_ids) - set(claims)
        if unknown_summary:
            raise ValueError("executive summary references unknown claim IDs")
        if not self.executive_summary_claim_ids:
            raise ValueError("executive summary requires at least one claim reference")
        expected_ids = [claim.claim_id for claim in select_summary_claims(self.claims)]
        if self.executive_summary_claim_ids != expected_ids:
            raise ValueError("executive summary claim IDs must be selected by code")
        expected_summary = summary_text_for_claims(
            [claims[claim_id] for claim_id in self.executive_summary_claim_ids]
        )
        normalized_summary = " ".join(self.executive_summary.split())
        normalized_expected = " ".join(expected_summary.split())
        if normalized_summary != normalized_expected:
            raise ValueError(
                "executive summary must exactly match its referenced claims in order"
            )
        for claim in self.claims:
            unknown = set(claim.evidence_ids) - set(evidence)
            if unknown:
                raise ValueError(
                    f"claim {claim.claim_id} references unknown evidence IDs: {sorted(unknown)}"
                )
            classes = {evidence[item].evidence_class for item in claim.evidence_ids}
            if claim.category == "forum_community_opinion":
                if classes and classes != {"forum_community"}:
                    raise ValueError("forum claims may cite only forum/community evidence")
            elif "forum_community" in classes:
                raise ValueError("forum/community evidence cannot support core claims")

        for field_name, expected in derived_claim_views(self.claims).items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} must be derived exactly from claims")
        return self


class ProcessingMetadata(BaseModel):
    """Trusted pipeline metadata, never delegated to source content or a model."""

    model_config = ConfigDict(extra="forbid")

    model_id: Optional[NonEmptyText] = None
    prompt_version: NonEmptyText
    pipeline_version: NonEmptyText
    analysis_mode: Literal["live_ai", "offline_smoke"] = "live_ai"
    analysis_completed: bool = True


class ProcessingDecision(BaseModel):
    """Validated decision metadata recorded by deterministic pipeline code."""

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    event_classification: DuplicateStatus
    candidate_id: str = Field(pattern=r"^[0-9a-f]{20}$")
    rejection_reason: Optional[RejectionReason] = None
    metadata: ProcessingMetadata
