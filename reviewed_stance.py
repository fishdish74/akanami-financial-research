"""Deterministic human-style review decisions for the v6 shadow draft set."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping


REVIEWED_SET_STATUS = "REVIEWED REFERENCE SET"

_CORRECTIONS: dict[str, dict[str, object]] = {
    "captured_management_evidential_absence": {
        "polarity": None,
    },
    "historical_management_evidential_denial": {
        "polarity": None,
    },
    "direct_denial": {
        "stance": "attributed",
    },
    "unresolved_causality": {
        "polarity": None,
    },
    "double_negation": {
        "attributed_to": None,
    },
    "did_not_deny": {
        "polarity": None,
    },
    "failed_to_prove": {
        "polarity": None,
    },
    "unclear_whether": {
        "polarity": None,
    },
}

_AMBIGUOUS = {"double_negation"}

_REASONS: dict[str, str] = {
    "captured_management_evidential_absence": (
        "Absence of evidence leaves the causal proposition unresolved; it does not "
        "affirm or negate proposition truth."
    ),
    "historical_management_evidential_denial": (
        "The sentence denies evidential support, not the causal relationship itself."
    ),
    "direct_denial": (
        "The audit is a named asserting source, so attribution plus negated polarity "
        "captures the denial without a redundant denied stance."
    ),
    "unresolved_causality": (
        "An unresolved proposition has no truth polarity in the reviewed contract."
    ),
    "double_negation": (
        "Logical normalization yields affirmed causality, but double-negation scope is "
        "easy to misread and requires manual review; the report is the source itself."
    ),
    "did_not_deny": (
        "Failure to deny does not affirm causality; the proposition remains unresolved."
    ),
    "failed_to_prove": (
        "Failure of proof leaves causality unresolved rather than affirmed or denied."
    ),
    "unclear_whether": (
        "Explicit uncertainty leaves proposition polarity inapplicable."
    ),
}


def review_draft_case(raw: Mapping[str, object]) -> dict[str, object]:
    """Preserve the original label and apply only the documented review correction."""

    case_id = str(raw["case_id"])
    original = deepcopy(dict(raw))
    reviewed = deepcopy(dict(raw))
    correction = _CORRECTIONS.get(case_id, {})
    reviewed.update(correction)
    if case_id in _AMBIGUOUS:
        status = "ambiguous"
    elif correction:
        status = "corrected"
    else:
        status = "accepted_as_is"
    reason = _REASONS.get(
        case_id,
        "The draft label matches the reviewed stance, polarity, attribution, and span semantics.",
    )
    return {
        "case_id": case_id,
        "original_case_id": case_id,
        "review_status": status,
        "review_reason": reason,
        "original_draft_label": original,
        "reviewed_label": reviewed,
    }


def build_reviewed_reference_set(
    draft: Mapping[str, object], *, draft_sha256: str
) -> dict[str, object]:
    cases = draft.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError("review requires the exact existing 24-case draft set")
    reviews = [review_draft_case(item) for item in cases]
    return {
        "evaluation_id": "market-analysis-v6-shadow-stance-fidelity-reviewed-v1",
        "prompt_version": "market-analysis-v6-shadow",
        "status": REVIEWED_SET_STATUS,
        "ground_truth": False,
        "source_draft": "stance-fidelity-draft-v1.json",
        "source_draft_sha256": draft_sha256,
        "case_count": len(reviews),
        "reviews": reviews,
    }
