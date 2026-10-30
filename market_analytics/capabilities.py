"""Small fail-closed capability model for provider/security boundaries."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SecurityType = Literal[
    "active_us_common_stock", "foreign_issuer_adr", "inactive_delisted", "unknown"
]
FundamentalBasis = Literal["us_gaap", "ifrs", "unknown"]
NormalizationCapability = Literal["supported", "unsupported", "requires_replay"]
IdentityContinuity = Literal["supported", "incomplete", "unsupported"]
DatasetSupport = Literal["supported", "partial", "unavailable", "provider_limitation"]
AdjustedVolumeSemantics = Literal["verified", "unverified"]
CustomTaxonomy = Literal["standard", "custom", "unknown"]
CapabilityReason = Literal[
    "unsupported_ifrs",
    "unsupported_accounting_basis",
    "incomplete_identity_continuity",
    "provider_identity_unavailable",
    "custom_taxonomy_requires_replay",
    "inactive_security",
    "adjusted_volume_unverified",
    "ambiguous_corporate_action_type",
]


class SecurityCapability(BaseModel):
    """One deterministic decision about what may safely flow downstream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    security_type: SecurityType
    fundamental_basis: FundamentalBasis
    fundamental_normalization: NormalizationCapability
    identity_continuity: IdentityContinuity
    corporate_actions: DatasetSupport
    price_history: DatasetSupport
    adjusted_volume_semantics: AdjustedVolumeSemantics
    custom_taxonomy: CustomTaxonomy
    reason_codes: list[CapabilityReason] = Field(default_factory=list)
    explanations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_reasons(self) -> "SecurityCapability":
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("capability reason codes must be unique")
        if self.fundamental_basis == "ifrs" and "unsupported_ifrs" not in self.reason_codes:
            raise ValueError("IFRS capability must carry unsupported_ifrs")
        if (
            self.custom_taxonomy == "custom"
            and self.fundamental_normalization == "requires_replay"
            and "custom_taxonomy_requires_replay" not in self.reason_codes
        ):
            raise ValueError("unvalidated custom taxonomy must carry its replay reason")
        return self

    @property
    def ranking_eligible(self) -> bool:
        return (
            self.security_type == "active_us_common_stock"
            and self.identity_continuity == "supported"
            and self.fundamental_basis == "us_gaap"
            and self.fundamental_normalization == "supported"
        )


def active_us_stock_capability() -> SecurityCapability:
    """Explicit v1 supported universe; adjusted-volume claims remain excluded."""

    return SecurityCapability(
        security_type="active_us_common_stock",
        fundamental_basis="us_gaap",
        fundamental_normalization="supported",
        identity_continuity="supported",
        corporate_actions="supported",
        price_history="supported",
        adjusted_volume_semantics="unverified",
        custom_taxonomy="standard",
        reason_codes=["adjusted_volume_unverified"],
        explanations=[
            "Adjusted-volume semantics are not verified and cannot be used as adjusted-volume evidence."
        ],
    )


def evaluate_security_capability(
    *,
    current_metadata_available: bool,
    active: bool | None,
    listing_type: str | None,
    foreign_issuer: bool,
    fundamental_basis: FundamentalBasis,
    custom_taxonomy: CustomTaxonomy = "standard",
    custom_taxonomy_replay_validated: bool = False,
    historical_prices_available: bool = False,
    historical_actions_available: bool = False,
    ticker_events_available: bool | None = None,
    adjusted_volume_verified: bool = False,
) -> SecurityCapability:
    """Evaluate observed facts without inferring identity or accounting support."""

    reasons: list[CapabilityReason] = []
    explanations: list[str] = []

    if not current_metadata_available:
        security_type: SecurityType = "unknown"
        identity: IdentityContinuity = "incomplete"
        reasons.extend(["provider_identity_unavailable", "incomplete_identity_continuity"])
        explanations.append(
            "Current provider identity is unavailable; historical observations do not prove legal-entity continuity."
        )
    elif active is False:
        security_type = "inactive_delisted"
        identity = "incomplete" if ticker_events_available is not True else "supported"
        reasons.append("inactive_security")
        explanations.append("The provider identifies the security as inactive.")
        if identity != "supported":
            reasons.append("incomplete_identity_continuity")
            explanations.append("Ticker/delisting continuity is incomplete.")
    elif foreign_issuer or listing_type == "adr":
        security_type = "foreign_issuer_adr"
        identity = "supported"
    elif active is True and listing_type == "common_stock":
        security_type = "active_us_common_stock"
        identity = "supported"
    else:
        security_type = "unknown"
        identity = "unsupported"
        reasons.append("incomplete_identity_continuity")
        explanations.append("Security type or active identity cannot be established safely.")

    if fundamental_basis == "ifrs":
        normalization: NormalizationCapability = "unsupported"
        reasons.append("unsupported_ifrs")
        explanations.append("The v1 normalizer supports US GAAP only; IFRS fundamentals are excluded.")
    elif fundamental_basis != "us_gaap":
        normalization = "unsupported"
        reasons.append("unsupported_accounting_basis")
        explanations.append("The accounting basis is unknown or unsupported.")
    elif custom_taxonomy == "custom" and not custom_taxonomy_replay_validated:
        normalization = "requires_replay"
        reasons.append("custom_taxonomy_requires_replay")
        explanations.append("Material custom-taxonomy reliance requires an explicit replay before normalization.")
    else:
        normalization = "supported"

    price_history: DatasetSupport = (
        "supported" if current_metadata_available else
        "partial" if historical_prices_available else "unavailable"
    )
    if current_metadata_available and active is False and historical_prices_available:
        price_history = "partial"
    corporate_actions: DatasetSupport = (
        "supported" if current_metadata_available and ticker_events_available is not False else
        "partial" if historical_actions_available else "provider_limitation"
    )

    volume_semantics: AdjustedVolumeSemantics = (
        "verified" if adjusted_volume_verified else "unverified"
    )
    if not adjusted_volume_verified:
        reasons.append("adjusted_volume_unverified")
        explanations.append("Raw volume may be used, but adjusted-volume behavior is unverified.")

    return SecurityCapability(
        security_type=security_type,
        fundamental_basis=fundamental_basis,
        fundamental_normalization=normalization,
        identity_continuity=identity,
        corporate_actions=corporate_actions,
        price_history=price_history,
        adjusted_volume_semantics=volume_semantics,
        custom_taxonomy=custom_taxonomy,
        reason_codes=list(dict.fromkeys(reasons)),
        explanations=list(dict.fromkeys(explanations)),
    )


def require_fundamental_normalization(capability: SecurityCapability) -> None:
    """Fail before US-GAAP normalization when the capability is not supported."""

    if capability.fundamental_normalization != "supported":
        code = next(
            (
                item for item in capability.reason_codes
                if item in {
                    "unsupported_ifrs", "unsupported_accounting_basis",
                    "custom_taxonomy_requires_replay",
                }
            ),
            "unsupported_accounting_basis",
        )
        raise ValueError(f"fundamental normalization blocked: {code}")


class CorporateActionClassification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_provider_type: str
    normalized_action_type: str | None = None
    normalization_status: Literal["provider_as_reported", "derived", "ambiguous"]
    reason_code: CapabilityReason | None = None
    evidence_reference: str | None = None


def classify_corporate_action(
    raw_provider_type: str,
    *,
    verified_economic_type: str | None = None,
    evidence_reference: str | None = None,
) -> CorporateActionClassification:
    """Preserve provider semantics; derive only from explicit verified evidence."""

    direct = {
        "forward_split": "stock_split",
        "reverse_split": "reverse_split",
        "stock_split": "stock_split",
    }
    if raw_provider_type in direct:
        return CorporateActionClassification(
            raw_provider_type=raw_provider_type,
            normalized_action_type=direct[raw_provider_type],
            normalization_status="provider_as_reported",
        )
    if verified_economic_type is not None:
        if not evidence_reference:
            raise ValueError("derived corporate-action classification requires evidence_reference")
        if verified_economic_type not in {"stock_split", "reverse_split", "stock_dividend"}:
            raise ValueError("verified economic action type is unsupported")
        return CorporateActionClassification(
            raw_provider_type=raw_provider_type,
            normalized_action_type=verified_economic_type,
            normalization_status="derived",
            evidence_reference=evidence_reference,
        )
    return CorporateActionClassification(
        raw_provider_type=raw_provider_type,
        normalized_action_type=None,
        normalization_status="ambiguous",
        reason_code="ambiguous_corporate_action_type",
    )


class VolumeComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["exact", "equivalent_after_rounding", "different"]
    adjusted_volume_semantics: AdjustedVolumeSemantics = "unverified"
    reason_code: CapabilityReason = "adjusted_volume_unverified"


def compare_reported_volume(
    provider_volume: Decimal,
    secondary_volume: Decimal,
    *,
    secondary_quantum: Decimal = Decimal("1"),
) -> VolumeComparison:
    """Compare display values without inventing adjusted-volume semantics."""

    if provider_volume == secondary_volume:
        status = "exact"
    elif provider_volume.quantize(secondary_quantum, rounding=ROUND_HALF_UP) == secondary_volume:
        status = "equivalent_after_rounding"
    else:
        status = "different"
    return VolumeComparison(status=status)
