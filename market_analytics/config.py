"""Validated analytics configuration loaded from readable YAML."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import Horizon


CONFIG_PATH = Path(__file__).with_name("analytics_config.yaml")
COMPONENTS = {
    "valuation", "quality", "growth", "financial_strength",
    "catalysts", "news", "risk", "data_quality",
}


class Versions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ranking: str
    calculation: str
    pipeline: str


class FreshnessThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    price: int = Field(gt=0)
    filing: int = Field(gt=0)
    fundamentals: int = Field(gt=0)
    annual_fundamentals: int = Field(gt=0)
    estimate: int = Field(gt=0)
    news: int = Field(gt=0)
    thesis: int = Field(gt=0)


class PeerSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primary_scope: Literal["industry", "sector"] = "industry"
    allow_sector_fallback: bool = False
    minimum_companies: int = Field(default=2, ge=2)


class AnalyticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    versions: Versions
    freshness_days: FreshnessThresholds
    peer_selection: PeerSelectionConfig
    ranking_weights: dict[Horizon, dict[str, Decimal]]
    metric_directions: dict[str, Literal["higher", "lower", "context"]]
    sector_relevance: dict[str, dict[str, Literal["relevant", "weak", "not_applicable"]]]

    @model_validator(mode="after")
    def validate_weights(self) -> "AnalyticsConfig":
        required = {"one_day", "one_week", "one_month", "one_year"}
        if set(self.ranking_weights) != required:
            raise ValueError("ranking weights must define all four horizons")
        for horizon, weights in self.ranking_weights.items():
            if set(weights) != COMPONENTS:
                raise ValueError(f"{horizon} weights must define exactly {sorted(COMPONENTS)}")
            if any(weight < 0 for weight in weights.values()):
                raise ValueError(f"{horizon} weights cannot be negative")
            if abs(sum(weights.values()) - Decimal("1")) > Decimal("0.000001"):
                raise ValueError(f"{horizon} weights must sum to 1")
        return self


def load_config(path: Path = CONFIG_PATH) -> AnalyticsConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AnalyticsConfig.model_validate(raw)


def config_hash(config: AnalyticsConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def metric_relevance(config: AnalyticsConfig, sector: str, metric: str) -> str:
    sector_map = config.sector_relevance.get(sector, {})
    default_map = config.sector_relevance.get("default", {})
    return sector_map.get(metric, default_map.get(metric, "relevant"))
