"""F5 fixed applicable denominator: no imputed observations or redistributed weight."""
import unittest
from decimal import Decimal as D
from unittest.mock import patch

from market_analytics.config import load_config
from market_analytics.models import PeerComparison, QualityIssue
from market_analytics.providers import FixtureProvider
from market_analytics.ranking import COMPONENT_METRICS, _peer_component, rank_company, rank_universe
from market_analytics.peers import compare_metric
from market_analytics.pipeline import run_provider_pipeline
from tests.helpers import ANALYTICS_FIXTURE


def comparison(name, score=100, **updates):
    return PeerComparison(ticker="QLTY", metric=name, value=D(1), peer_median=D(1),
        percentile=D(score), z_score=D(0), relative_spread=D(0), direction="higher",
        status="valid").model_copy(update=updates)


class RankingCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.names = COMPONENT_METRICS["quality"]
        self.comparisons = {n: comparison(n) for names in COMPONENT_METRICS.values() for n in names}
        self.company = FixtureProvider(ANALYTICS_FIXTURE).company("QLTY").model_copy(update={"news_score": D(100)})
        self.config = load_config()

    def rank(self, comparisons, issues=()):
        with patch("market_analytics.ranking.catalyst_score", return_value=D(25)), patch(
                "market_analytics.ranking.risk_resilience_score", return_value=D(100)):
            return rank_company(self.company, comparisons, list(issues), self.config, "one_year", self.company.as_of)

    def test_exact_audit_inflation_blocked(self):
        self.comparisons["ROE"] = comparison("ROE", 0)
        complete = self.rank(self.comparisons)
        missing = {k:v for k,v in self.comparisons.items() if k != "ROE"}
        def legacy(comparisons, names):
            _, absent = _peer_component(comparisons, names)
            values = [comparisons[n].percentile for n in names if n in comparisons
                      and comparisons[n].status == "valid" and comparisons[n].percentile is not None]
            return (sum(values)/len(values)).quantize(D(1)), absent
        with patch("market_analytics.ranking._peer_component", side_effect=legacy):
            old = self.rank(missing)
        self.assertEqual((complete.components.quality, complete.opportunity_score), (D(83), D(90)))
        self.assertEqual((old.components.quality, old.opportunity_score), (D(100), D(94)))
        fixed = self.rank(missing)
        self.assertEqual((fixed.components.quality, fixed.opportunity_score), (D(83), D(90)))
        self.assertIn("ROE", fixed.unavailable_inputs)

    def test_complete_constituents_unchanged(self):
        self.assertEqual(_peer_component(self.comparisons, self.names), (D(100), {}))
        self.assertEqual(self.rank(self.comparisons).opportunity_score, D(94))
        self.assertEqual(self.rank(self.comparisons).ranking_version, "ranking-v2")

    def test_removing_weak_or_strong_never_improves(self):
        for score in (0, 20, 100):
            values = {n: comparison(n, 50) for n in self.names}
            values["ROE"] = comparison("ROE", score)
            before = _peer_component(values, self.names)[0]
            del values["ROE"]
            self.assertLessEqual(_peer_component(values, self.names)[0], before)

    def test_multiple_and_all_missing(self):
        self.assertEqual(_peer_component({"ROE":comparison("ROE")}, self.names)[0], D(17))
        self.assertIsNone(_peer_component({}, self.names)[0])

    def test_verified_structural_na_only(self):
        self.comparisons["ROE"] = comparison("ROE", percentile=None, status="not_applicable",
            warning="ROE is configured as not applicable for Synthetic")
        score, absent = _peer_component(self.comparisons, self.names)
        self.assertEqual(score,D(100)); self.assertEqual(absent["ROE"],"structural not-applicable")
        for warning in (None, "denominator is zero", "negative denominator makes this metric misleading"):
            self.comparisons["ROE"] = comparison("ROE", percentile=None,status="not_applicable",warning=warning)
            self.assertEqual(_peer_component(self.comparisons,self.names)[0],D(83))
        values={n:comparison(n,percentile=None,status="not_applicable",warning=f"{n} is configured as not applicable for Synthetic") for n in self.names}
        self.assertIsNone(_peer_component(values,self.names)[0])

    def test_insufficient_peers_stale_and_invalid_keep_weight(self):
        for status,warning,reason in [("missing_data","insufficient_peer_set","insufficient peer set"),
                ("missing_data","stale data removed","stale data"),
                ("invalid_input","incompatible_financial_periods","adverse or economically invalid input")]:
            self.comparisons["ROE"]=comparison("ROE",percentile=None,status=status,warning=warning)
            score,absent=_peer_component(self.comparisons,self.names)
            self.assertEqual(score,D(83));self.assertEqual(absent["ROE"],reason)
        peer=compare_metric("QLTY","ROE",D(1),{"QLTY":D(1)},"higher")
        self.assertIsNone(peer.percentile)

    def test_blocking_and_evidence_gates_unchanged(self):
        blocked=self.rank(self.comparisons,[QualityIssue(code="test",severity="BLOCKING",message="synthetic")])
        self.assertEqual(blocked.ranking_status,"blocked");self.assertIsNone(blocked.opportunity_score)
        empty=self.rank({});self.assertEqual(empty.ranking_status,"insufficient_evidence")
        self.assertIsNone(empty.opportunity_score)

    def test_f6_invalid_roa_and_quality_evidence_not_promoted(self):
        from datetime import date
        provider=FixtureProvider(ANALYTICS_FIXTURE); company=provider.company("QLTY")
        inputs=dict(company.financials)
        inputs["net_income"]=inputs["net_income"].model_copy(update={"period":inputs["net_income"].period.model_copy(update={"start_date":date(2023,1,1),"end_date":date(2023,12,31)})})
        provider._companies["QLTY"]=company.model_copy(update={"financials":inputs})
        run=run_provider_pipeline(provider)
        self.assertIsNone(run.metrics["QLTY"]["ROA"].value)
        self.assertIsNone(run.comparisons["QLTY"]["ROA"].value)
        self.assertEqual(run.comparisons["QLTY"]["Net Margin"].status,"invalid_input")
        comps=run.comparisons["QLTY"]
        valid=[comps[n].percentile for n in self.names if comps[n].status=="valid" and comps[n].percentile is not None]
        score,_=_peer_component(comps,self.names)
        if valid:
            from decimal import ROUND_HALF_UP
            self.assertEqual(score,(sum(valid)/D(len(self.names))).quantize(D(1),rounding=ROUND_HALF_UP))
        else:self.assertIsNone(score)

    def test_stable_ties_and_determinism(self):
        first=self.rank(self.comparisons)
        second=self.rank(dict(reversed(list(self.comparisons.items()))))
        self.assertEqual(first,second)
        a=first.model_copy(update={"ticker":"AAA"});z=first.model_copy(update={"ticker":"ZZZ"})
        self.assertEqual([r.ticker for r in rank_universe([z,a])],["AAA","ZZZ"])
