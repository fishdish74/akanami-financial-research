"""Manual JSON input for deterministic offline demos and tests."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from entities import Entity
from schemas import Candidate


class ManualJsonSource:
    name = "manual"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.last_warnings: list[str] = []

    def discover(self, entity: Entity, limit: int) -> list[Candidate]:
        if not self.path.is_file():
            raise ValueError(f"manual candidate file does not exist: {self.path}")
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("manual candidate file contains malformed JSON") from exc
        items = raw.get("candidates") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            raise ValueError("manual candidate file must be a list or contain a candidates list")
        candidates: list[Candidate] = []
        self.last_warnings = []
        for position, item in enumerate(items, 1):
            try:
                candidate = Candidate.model_validate(item)
            except ValidationError:
                self.last_warnings.append(
                    f"skipped manual candidate {position}: schema validation failed"
                )
                continue
            updates: dict[str, object] = {
                "source_type": "manual",
                "source_tier": 4,
            }
            # Missing attribution is inferred only when the record itself names
            # the target. Explicit manual attribution is otherwise preserved,
            # and unrelated records remain unattributed for downstream rejection.
            text = " ".join(filter(None, [candidate.title, candidate.snippet, candidate.content or ""]))
            if entity.appears_in(text):
                if not candidate.companies:
                    updates["companies"] = [entity.canonical_name]
                if not candidate.tickers:
                    updates["tickers"] = [entity.ticker]
            candidate = Candidate.model_validate(
                {**candidate.model_dump(mode="python"), **updates}
            )
            candidates.append(candidate)
            if len(candidates) >= limit:
                break
        return candidates
