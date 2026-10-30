from __future__ import annotations

import unittest

from pydantic import ValidationError

from discovery import DiscoveryConfig, run_discovery
from tests.test_discovery import NOW, item


EXPECTED_WEIGHTS = {
    "novelty": 20,
    "event_materiality": 25,
    "evidence_quality": 20,
    "catalyst_strength": 10,
    "fundamental_interest": 10,
    "growth_acceleration": 5,
    "urgency": 5,
    "data_completeness": 5,
}


class DiscoveryConfigValidationRegressionTests(unittest.TestCase):
    def test_missing_required_weight_is_rejected(self) -> None:
        weights = dict(EXPECTED_WEIGHTS)
        weights.pop("urgency")
        with self.assertRaisesRegex(ValidationError, "urgency"):
            DiscoveryConfig(weights=weights)

    def test_unexpected_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unexpected"):
            DiscoveryConfig(weights={**EXPECTED_WEIGHTS, "hype": 0})

    def test_negative_weight_is_rejected(self) -> None:
        weights = dict(EXPECTED_WEIGHTS)
        weights["novelty"] = -1
        weights["event_materiality"] = 46
        with self.assertRaisesRegex(ValidationError, "nonnegative"):
            DiscoveryConfig(weights=weights)

    def test_nonfinite_weights_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                DiscoveryConfig(weights={**EXPECTED_WEIGHTS, "novelty": value})

    def test_invalid_weight_total_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "total 100"):
            DiscoveryConfig(weights={**EXPECTED_WEIGHTS, "novelty": 19})

    def test_thresholds_must_be_finite_integers_from_zero_to_one_hundred(self) -> None:
        for field, value in (
            ("minimum_score", -1),
            ("minimum_score", 101),
            ("minimum_materiality", -1),
            ("minimum_materiality", 101),
            ("minimum_evidence_quality", -1),
            ("minimum_evidence_quality", 101),
            ("minimum_score", float("nan")),
            ("minimum_score", float("inf")),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                DiscoveryConfig(**{field: value})

    def test_known_good_default_config_passes(self) -> None:
        config = DiscoveryConfig()
        self.assertEqual(config.weights, EXPECTED_WEIGHTS)
        self.assertEqual(sum(config.weights.values()), 100)

    def test_valid_custom_config_preserves_existing_scoring_formula(self) -> None:
        weights = {
            "novelty": 10,
            "event_materiality": 30,
            "evidence_quality": 20,
            "catalyst_strength": 10,
            "fundamental_interest": 10,
            "growth_acceleration": 5,
            "urgency": 10,
            "data_completeness": 5,
        }
        run = run_discovery(
            [item()], config=DiscoveryConfig(weights=weights), discovery_timestamp=NOW
        )
        self.assertEqual(run.candidates[0].research_priority_score, 83)
        self.assertEqual(run.selected.event_id, run.candidates[0].event_id)


def _source(
    *,
    source: str,
    url: str,
    snapshot: str,
    value: str,
    discovered_offset: int = 0,
):
    return item(
        source=source,
        source_url=url,
        source_snapshot_id=snapshot,
        source_independence_key=source,
        primary_source=source == "Issuer",
        material_numbers={"contract_value": value},
        discovered_at=NOW.replace(minute=discovered_offset),
    )


class MaterialNumberConflictRegressionTests(unittest.TestCase):
    def test_two_independent_sources_with_equivalent_values_confirm(self) -> None:
        run = run_discovery(
            [
                _source(source="Issuer", url="https://issuer.example/event", snapshot="s1", value="$100 million"),
                _source(source="Wire", url="https://wire.example/event", snapshot="s2", value="100M", discovered_offset=1),
            ],
            discovery_timestamp=NOW,
        )
        candidate = run.candidates[0]
        self.assertEqual(candidate.material_number_conflicts, [])
        self.assertIn("contract_value", candidate.material_numbers)
        self.assertEqual(len(candidate.material_number_evidence["contract_value"]), 2)
        self.assertEqual(candidate.dimensions.evidence_quality, 100)

    def test_two_independent_sources_with_different_values_conflict(self) -> None:
        run = run_discovery(
            [
                _source(source="Issuer", url="https://issuer.example/event", snapshot="s1", value="$100 million"),
                _source(source="Wire", url="https://wire.example/event", snapshot="s2", value="$250 million", discovered_offset=1),
            ],
            discovery_timestamp=NOW,
        )
        candidate = run.candidates[0]
        self.assertEqual(candidate.material_number_conflicts, ["contract_value"])
        self.assertNotIn("contract_value", candidate.material_numbers)
        self.assertEqual(
            {entry.raw_value for entry in candidate.material_number_evidence["contract_value"]},
            {"$100 million", "$250 million"},
        )
        self.assertEqual(candidate.dimensions.evidence_quality, 90)
        self.assertIn("Conflicting material number: contract_value", candidate.risk_flags)

    def test_three_sources_two_vs_one_preserve_conflict_without_majority_truth(self) -> None:
        run = run_discovery(
            [
                _source(source="Issuer", url="https://issuer.example/event", snapshot="s1", value="$100 million"),
                _source(source="Wire", url="https://wire.example/event", snapshot="s2", value="100M", discovered_offset=1),
                _source(source="Trade", url="https://trade.example/event", snapshot="s3", value="$250 million", discovered_offset=2),
            ],
            discovery_timestamp=NOW,
        )
        candidate = run.candidates[0]
        self.assertEqual(candidate.material_number_conflicts, ["contract_value"])
        self.assertNotIn("contract_value", candidate.material_numbers)
        self.assertEqual(len(candidate.material_number_evidence["contract_value"]), 3)
        self.assertEqual(candidate.dimensions.evidence_quality, 90)

    def test_same_value_with_different_numeric_formatting_is_not_a_conflict(self) -> None:
        run = run_discovery(
            [
                _source(source="Issuer", url="https://issuer.example/event", snapshot="s1", value="100000000"),
                _source(source="Wire", url="https://wire.example/event", snapshot="s2", value="$100 million", discovered_offset=1),
            ],
            discovery_timestamp=NOW,
        )
        self.assertEqual(run.candidates[0].material_number_conflicts, [])
        self.assertEqual(run.candidates[0].dimensions.evidence_quality, 100)

    def test_different_units_or_currencies_are_not_treated_as_confirmation(self) -> None:
        for first, second in (
            ("USD 100 million", "EUR 100 million"),
            ("100 percent", "100 shares"),
        ):
            with self.subTest(first=first, second=second):
                run = run_discovery(
                    [
                        _source(source="Issuer", url="https://issuer.example/event", snapshot="s1", value=first),
                        _source(source="Wire", url="https://wire.example/event", snapshot="s2", value=second, discovered_offset=1),
                    ],
                    discovery_timestamp=NOW,
                )
                self.assertEqual(
                    run.candidates[0].material_number_conflicts, ["contract_value"]
                )
                self.assertEqual(run.candidates[0].dimensions.evidence_quality, 90)

    def test_source_order_does_not_change_material_number_semantics(self) -> None:
        sources = [
            _source(source="Issuer", url="https://issuer.example/event", snapshot="s1", value="$100 million"),
            _source(source="Wire", url="https://wire.example/event", snapshot="s2", value="$250 million", discovered_offset=1),
        ]
        first = run_discovery(sources, discovery_timestamp=NOW).candidates[0]
        second = run_discovery(list(reversed(sources)), discovery_timestamp=NOW).candidates[0]
        self.assertEqual(first.material_numbers, second.material_numbers)
        self.assertEqual(first.material_number_conflicts, second.material_number_conflicts)
        self.assertEqual(first.material_number_evidence, second.material_number_evidence)
        self.assertEqual(first.dimensions.evidence_quality, second.dimensions.evidence_quality)


if __name__ == "__main__":
    unittest.main()
