from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from schemas import (
    AnalysisPayload,
    Candidate,
    ModelEvidenceClaim,
    assign_claim_ids,
    derived_claim_views,
    select_summary_claims,
    summary_text_for_claims,
)


ANALYTICS_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "analytics-demo.json"


def candidate(**overrides: object) -> Candidate:
    values: dict[str, object] = {
        "title": "NVIDIA reports quarterly earnings and guidance",
        "source": "Example Wire",
        "source_url": "https://example.com/nvidia-results?utm_source=test",
        "published": datetime(2026, 9, 28, 20, 0, tzinfo=timezone.utc),
        "discovered": datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc),
        "author": "Example Author",
        "content": "NVIDIA Corporation reported quarterly earnings and updated guidance with enough fixture-only text for deterministic offline testing.",
        "snippet": "NVIDIA reported results and guidance.",
        "source_tier": 2,
        "companies": ["NVIDIA Corporation"],
        "tickers": ["NVDA"],
        "source_type": "manual",
    }
    values.update(overrides)
    return Candidate.model_validate(values)


def analysis(**overrides: object) -> AnalysisPayload:
    values: dict[str, object] = {
        "title": "NVIDIA reports quarterly earnings and guidance",
        "source": "Example Wire",
        "source_url": "https://example.com/nvidia-results",
        "author": "Example Author",
        "published": "2026-09-28",
        "accessed": "2026-09-29T12:00:00Z",
        "companies": ["NVIDIA Corporation"],
        "tickers": ["NVDA"],
        "themes": ["AI infrastructure"],
        "sector": "Semiconductors",
        "importance": "high",
        "confidence": "medium",
        "sentiment": "mixed",
        "needs_review": True,
        "executive_summary": "Fixture placeholder replaced when claims are generated.",
        "executive_summary_claim_ids": [],
        "reported_facts": ["A fixture fact."],
        "company_management_statements": ["Management made a fixture statement."],
        "analyst_source_opinion": ["The source used optimistic framing."],
        "ai_inference": ["A deterministic fixture-only inference."],
        "potential_catalysts": ["A fixture catalyst."],
        "potential_risks": ["A fixture risk."],
        "bullish_evidence": ["A fixture bullish point."],
        "bearish_evidence": ["A fixture bearish point."],
        "what_is_new": ["A fixture update."],
        "contradictions": [],
        "open_questions": ["A fixture question?"],
        "information_needed_next": ["A fixture follow-up."],
    }
    values.update(overrides)
    if "evidence_catalog" not in overrides and "claims" not in overrides:
        values.pop("executive_summary", None)
        values.pop("executive_summary_claim_ids", None)
        evidence_id = "article_event:fixture-source"
        values["evidence_catalog"] = [{
            "evidence_id": evidence_id,
            "evidence_class": "article_event",
            "label": "Fixture source",
            "source_url": values["source_url"],
            "available_at": values["accessed"],
        }]
        model_claims: list[ModelEvidenceClaim] = []
        by_text: dict[str, int] = {}

        def add(text: str, category: str, support: str, facet: str | None = None) -> None:
            identity = " ".join(text.split()).casefold()
            if identity in by_text:
                if facet:
                    existing = model_claims[by_text[identity]]
                    model_claims[by_text[identity]] = existing.model_copy(
                        update={"facets": [*existing.facets, facet]}
                    )
                return
            by_text[identity] = len(model_claims)
            model_claims.append(ModelEvidenceClaim(
                text=text,
                category=category,
                support=support,
                evidence_ids=[evidence_id],
                evidence_locations=[],
                causal_role="none",
                facets=[facet] if facet else [],
            ))

        categories = {
            "reported_facts": ("reported_fact", "supported"),
            "company_management_statements": ("management_statement", "supported"),
            "analyst_source_opinion": ("source_opinion", "supported"),
            "ai_inference": ("ai_inference", "inference"),
        }
        for field_name, (category, support) in categories.items():
            seen: set[str] = set()
            for raw_text in values[field_name]:
                text = str(raw_text).strip()
                identity = text.casefold()
                if not text or identity in seen:
                    continue
                seen.add(identity)
                add(text, category, support)
        facets = {
            "potential_catalysts": "catalyst",
            "potential_risks": "risk",
            "bullish_evidence": "bullish",
            "bearish_evidence": "bearish",
            "what_is_new": "novelty",
            "contradictions": "contradiction",
        }
        for field_name, facet in facets.items():
            for raw_text in values[field_name]:
                text = str(raw_text).strip()
                if text:
                    add(text, "ai_inference", "inference", facet)
        claims = assign_claim_ids(model_claims)
        values["claims"] = claims
        summary_claims = select_summary_claims(claims)
        values["executive_summary_claim_ids"] = [claim.claim_id for claim in summary_claims]
        values["executive_summary"] = summary_text_for_claims(summary_claims)
        values.update(derived_claim_views(claims))
    return AnalysisPayload.model_validate(values)
