"""Experimental market-analysis-v6-shadow causal contract; not a production route."""

from __future__ import annotations

import re
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from renderer import escape_external_markdown


SHADOW_PROMPT_VERSION = "market-analysis-v6-shadow"

_CAUSAL_LINK_TRIPWIRE = re.compile(
    r"\b(?:caus(?:e|es|ed|ing)|dr(?:ive|ives|ove|iven|iving)|led to|leads? to|"
    r"result(?:s|ed|ing)? (?:in|from)|due to|because(?: of)?|responsible for|"
    r"contribut(?:e|es|ed|ing) to|affect(?:s|ed|ing)?|impact(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)
_EVIDENTIAL_CAUSAL_DENIAL_TRIPWIRE = re.compile(
    r"\b(?:"
    r"(?:the\s+)?(?:available\s+)?(?:evidence|data|record|source|filing|disclosure|analysis)"
    r"\s+(?:does|do|did)\s+not\s+(?:establish|show|demonstrate|prove|substantiate|support)"
    r"\s+(?:that|whether)"
    r"|no\b[^.!?;]{0,180}\b(?:evidence|data)\b\s+"
    r"(?:establish(?:es|ed|ing)?|show(?:s|ed|ing)?|demonstrat(?:es|ed|ing)|"
    r"prov(?:es|ed|ing)|support(?:s|ed|ing)?)\s+(?:that|whether)"
    r"|insufficient\s+(?:evidence|data)\s+to\s+"
    r"(?:establish|show|demonstrate|prove|conclude)\s+(?:that|whether)"
    r"|(?:it\s+is\s+)?(?:unclear|uncertain)\s+whether"
    r"|(?:we\s+)?cannot\s+(?:determine|conclude)\s+(?:that|whether)"
    r")\b",
    re.IGNORECASE,
)
_ANAPHORIC_EVIDENCE_DENIAL = re.compile(
    r"\b(?:no\b[^.!?;]{0,120}\bevidence\b\s+"
    r"(?:establish(?:es|ed|ing)?|show(?:s|ed|ing)?|demonstrat(?:es|ed|ing)|"
    r"prov(?:es|ed|ing)|support(?:s|ed|ing)?)|"
    r"(?:the\s+)?(?:evidence|data|record|source)\s+(?:does|do|did)\s+not\s+"
    r"(?:establish|show|demonstrate|prove|support))\s+that\s+(?:claim|assertion|attribution)\b",
    re.IGNORECASE,
)

ShadowStance = Literal[
    "source_asserted", "attributed", "model_inference", "unresolved"
]
ShadowPolarity = Literal["affirmed", "negated"]
ShadowHedge = Literal["none", "possible", "likely"]


def evidential_denial_causal_propositions(text: str) -> tuple[str, ...]:
    """Return local source spans for each explicit evidentially denied proposition.

    The spans are used only to bind a denial to model-supplied, validated cause/link/effect
    selectors. They never infer the proposition or its stance themselves.
    """

    propositions: list[str] = []
    for match in _EVIDENTIAL_CAUSAL_DENIAL_TRIPWIRE.finditer(text):
        boundary = re.search(r"[.!?;]", text[match.end():])
        end = match.end() + boundary.start() + 1 if boundary else len(text)
        segment = text[match.start():end].strip()
        if _CAUSAL_LINK_TRIPWIRE.search(segment):
            propositions.append(segment)
    for match in _ANAPHORIC_EVIDENCE_DENIAL.finditer(text):
        sentence_start = max(text.rfind(mark, 0, match.start()) for mark in ".!?;") + 1
        boundary = re.search(r"[.!?;]", text[match.end():])
        sentence_end = match.end() + boundary.start() + 1 if boundary else len(text)
        segment = text[sentence_start:sentence_end].strip()
        if _CAUSAL_LINK_TRIPWIRE.search(segment):
            propositions.append(segment)
    # Preserve source order while preventing content/snippet repetition from multiplying
    # one denial requirement.
    unique: list[str] = []
    seen: set[str] = set()
    for proposition in sorted(propositions, key=text.find):
        normalized = " ".join(proposition.split()).casefold()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(proposition)
    return tuple(unique)


def has_evidential_denial_causal_proposition(text: str) -> bool:
    """Tripwire explicit non-establishment of a causal proposition."""

    return bool(evidential_denial_causal_propositions(text))


class SpanSelector(BaseModel):
    """Model-owned exact phrase and explicit one-based occurrence selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    occurrence: int = Field(ge=1)

    @field_validator("text")
    @classmethod
    def exact_inert_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("span selector text cannot have surrounding whitespace")
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value):
            raise ValueError("span selector text cannot contain unsafe control characters")
        return value


class ResolvedSpan(BaseModel):
    """Python-owned position derived from an exact selector; never model output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)
    occurrence: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_range(self) -> "ResolvedSpan":
        if self.end <= self.start:
            raise ValueError("span end must follow span start")
        return self


class ShadowCausality(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stance: ShadowStance
    polarity: ShadowPolarity | None
    cause: str = Field(min_length=1, max_length=200)
    effect: str = Field(min_length=1, max_length=200)
    attributed_to: str | None = Field(default=None, max_length=120)
    hedge: ShadowHedge
    cause_span: SpanSelector
    effect_span: SpanSelector
    link_span: SpanSelector | None = None

    @field_validator("cause", "effect", "attributed_to")
    @classmethod
    def short_inert_phrase(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("causal phrase cannot be blank")
        if re.search(r"[\x00-\x1f\x7f]", cleaned):
            raise ValueError("causal phrase cannot contain control characters")
        if re.search(r"[.!?;]", cleaned):
            raise ValueError("causal phrase must be a short phrase, not a sentence")
        return cleaned

    @model_validator(mode="after")
    def stance_constraints(self) -> "ShadowCausality":
        if self.stance == "attributed" and not (self.attributed_to or "").strip():
            raise ValueError("attributed causality requires attributed_to")
        if self.stance == "source_asserted" and self.attributed_to is not None:
            raise ValueError("source_asserted causality cannot name a third-party attribution")
        if self.stance == "model_inference" and self.attributed_to is not None:
            raise ValueError("model inference cannot name an external attribution")
        if self.stance == "model_inference" and self.hedge == "none":
            raise ValueError("model inference must remain hedged")
        if self.stance == "unresolved" and self.polarity is not None:
            raise ValueError("unresolved causality must not assert proposition polarity")
        if self.stance != "unresolved" and self.polarity is None:
            raise ValueError("resolved causal stance requires proposition polarity")
        if self.stance != "model_inference" and self.link_span is None:
            raise ValueError("source and unresolved causality require a bound link span")
        return self


class ShadowClaim(BaseModel):
    """A claim with explicit causal structure and no lexical role inference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    evidence_ids: list[str]
    causality: ShadowCausality | None

    @model_validator(mode="after")
    def spans_bind_exactly(self) -> "ShadowClaim":
        if self.causality is None:
            return self
        resolved: dict[str, ResolvedSpan] = {}
        pairs: tuple[tuple[str, str | None, SpanSelector | None], ...] = (
            ("cause", self.causality.cause, self.causality.cause_span),
            ("effect", self.causality.effect, self.causality.effect_span),
            ("link", None, self.causality.link_span),
        )
        for label, declared, selector in pairs:
            if selector is None:
                continue
            resolved[label] = resolve_span(self.text, selector)
            if declared is not None and declared != selector.text:
                raise ValueError(f"{label} value must exactly match its span selector")
        if self.causality.stance == "attributed":
            proposition_start = min(span.start for span in resolved.values())
            proposition_end = max(span.end for span in resolved.values())
            sentence_start = max(
                (self.text.rfind(mark, 0, proposition_start) for mark in ".!?;"),
                default=-1,
            ) + 1
            following = [
                position for mark in ".!?;"
                if (position := self.text.find(mark, proposition_end)) >= 0
            ]
            sentence_end = min(following, default=len(self.text))
            if not all(
                sentence_start <= span.start and span.end <= sentence_end
                for span in resolved.values()
            ):
                raise ValueError("causal selectors must bind within one proposition sentence")
            attribution = re.escape(self.causality.attributed_to or "")
            reporting = (
                r"(?:said|says|stated|states|reported|reports|claimed|claims|argued|argues|"
                r"suggested|suggests|wrote|asserted|asserts|speculated|concluded|found|"
                r"quotes?|saying)"
            )
            sentence = self.text[sentence_start:sentence_end]
            bound_attribution = re.search(
                rf"(?:according\s+to\s+{attribution}\b|"
                rf"\b{attribution}\b[^.!?;]{{0,80}}\b{reporting}\b|"
                rf"\b{reporting}\b[^.!?;]{{0,40}}\b{attribution}\b)",
                sentence,
                re.IGNORECASE,
            )
            if bound_attribution is None:
                raise ValueError(
                    "attributed causality must bind speaker, cause, link, and effect "
                    "within the same proposition sentence"
                )
        return self


def exact_occurrences(text: str, phrase: str) -> tuple[int, ...]:
    """Return every raw, case-sensitive code-point start, including overlapping matches."""

    starts: list[int] = []
    offset = 0
    while True:
        start = text.find(phrase, offset)
        if start < 0:
            return tuple(starts)
        starts.append(start)
        offset = start + 1


def resolve_span(text: str, selector: SpanSelector) -> ResolvedSpan:
    """Resolve an explicit occurrence without normalization, guessing, or repair."""

    starts = exact_occurrences(text, selector.text)
    if not starts:
        raise ValueError(f"exact span text is absent: {selector.text!r}")
    if selector.occurrence > len(starts):
        raise ValueError(
            f"span occurrence {selector.occurrence} is unavailable for "
            f"{selector.text!r}; matches={len(starts)}"
        )
    start = starts[selector.occurrence - 1]
    return ResolvedSpan(
        start=start,
        end=start + len(selector.text),
        text=selector.text,
        occurrence=selector.occurrence,
    )


def _unique_selector(text: str, expected: str | None) -> SpanSelector | None:
    if expected is None:
        return None
    starts = exact_occurrences(text, expected)
    if not starts:
        raise ValueError(f"expected span is absent: {expected!r}")
    if len(starts) != 1:
        raise ValueError(f"expected span is ambiguous: {expected!r}")
    return SpanSelector(text=expected, occurrence=1)


def resolved_causal_spans(claim: ShadowClaim) -> dict[str, ResolvedSpan | None]:
    """Expose auditable code-owned offsets for validation, rendering, and evaluation."""

    if claim.causality is None:
        return {"cause": None, "effect": None, "link": None}
    return {
        "cause": resolve_span(claim.text, claim.causality.cause_span),
        "effect": resolve_span(claim.text, claim.causality.effect_span),
        "link": (
            resolve_span(claim.text, claim.causality.link_span)
            if claim.causality.link_span is not None else None
        ),
    }


def shadow_claim_from_draft_label(label: Mapping[str, object]) -> ShadowClaim:
    """Convert a reviewed draft label into the executable shadow schema."""

    text = str(label["text"])
    if not bool(label["causal_proposition"]):
        return ShadowClaim(
            text=text,
            evidence_ids=[str(item) for item in label.get("evidence_ids", ())],
            causality=None,
        )
    spans = label.get("span_expectations")
    if not isinstance(spans, Mapping):
        raise ValueError("causal draft label requires span expectations")
    causality = ShadowCausality(
        stance=label["stance"],
        polarity=label["polarity"],
        cause=str(label["cause"]),
        effect=str(label["effect"]),
        attributed_to=(
            str(label["attributed_to"]) if label.get("attributed_to") is not None else None
        ),
        hedge=label["hedge"],
        cause_span=_unique_selector(text, str(spans["cause"])),
        effect_span=_unique_selector(text, str(spans["effect"])),
        link_span=_unique_selector(
            text, str(spans["link"]) if spans.get("link") is not None else None
        ),
    )
    return ShadowClaim(
        text=text,
        evidence_ids=[str(item) for item in label.get("evidence_ids", ())],
        causality=causality,
    )


def render_shadow_claim(claim: ShadowClaim) -> str:
    """Render a deterministic evaluation-only view; this is not Obsidian output."""

    safe = escape_external_markdown
    lines = [
        "## Causal Shadow (Experimental)",
        "",
        f"- Text: {safe(claim.text)}",
        "- Evidence IDs: " + (
            ", ".join(f"`{safe(item)}`" for item in claim.evidence_ids) or "none"
        ),
    ]
    if claim.causality is None:
        lines.append("- Causality: none")
        return "\n".join(lines) + "\n"
    causal = claim.causality
    resolved = resolved_causal_spans(claim)
    cause_span = resolved["cause"]
    effect_span = resolved["effect"]
    link_span = resolved["link"]
    assert cause_span is not None and effect_span is not None
    if causal.stance == "model_inference":
        connector = "may have caused" if causal.hedge == "possible" else "likely caused"
        rendered_statement = (
            f"AI inference ({causal.hedge}): {safe(causal.cause)} "
            f"{connector} {safe(causal.effect)}."
        )
    elif causal.stance == "unresolved":
        rendered_statement = (
            f"Causality unresolved: whether {safe(causal.cause)} caused "
            f"{safe(causal.effect)} is not established."
        )
    else:
        connector = "caused" if causal.polarity == "affirmed" else "did not cause"
        owner = (
            f"{safe(causal.attributed_to)} asserts"
            if causal.stance == "attributed" else "Source asserts"
        )
        rendered_statement = (
            f"{owner}: {safe(causal.cause)} {connector} {safe(causal.effect)}."
        )
    lines.extend((
        f"- Stance: `{causal.stance}`",
        f"- Polarity: `{causal.polarity}`",
        f"- Cause: {safe(causal.cause)} [{cause_span.start}:{cause_span.end}]",
        f"- Effect: {safe(causal.effect)} [{effect_span.start}:{effect_span.end}]",
        f"- Attributed to: {safe(causal.attributed_to) if causal.attributed_to else 'none'}",
        f"- Hedge: `{causal.hedge}`",
        f"- Rendered causal statement: {rendered_statement}",
        "- Link span: " + (
            f"{safe(link_span.text)} [{link_span.start}:{link_span.end}]"
            if link_span else "none"
        ),
    ))
    return "\n".join(lines) + "\n"


def content_smuggling_flags(causality: ShadowCausality) -> list[str]:
    """Flag suspicious phrase content for review without deriving causal semantics."""

    flags: list[str] = []
    causal_terms = re.compile(
        r"\b(?:cause[ds]?|causing|dr[io]ve[ns]?|led|lead(?:s|ing)?|result(?:ed|s)?|"
        r"contribut(?:e[ds]?|ing)|affect(?:ed|s|ing)?)\b",
        re.IGNORECASE,
    )
    clause_terms = re.compile(r"\b(?:which|because|although|while|whereas)\b", re.IGNORECASE)
    for field_name in ("cause", "effect", "attributed_to"):
        value = getattr(causality, field_name)
        if not value:
            continue
        if causal_terms.search(value):
            flags.append(f"{field_name}:contains_causal_verb")
        if clause_terms.search(value):
            flags.append(f"{field_name}:contains_clause_marker")
    return flags
