import json
import unittest
from pathlib import Path

from rerank import Candidate, evaluate, rerank

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "cycling_routes.json"


class RerankerTests(unittest.TestCase):
    def test_demo_quantifies_coverage_tradeoff(self):
        scenario = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        docs = [Candidate(**row) for row in scenario["candidates"]]
        weights = scenario["aspect_weights"]
        baseline, diverse = [rerank(docs, weights, k=3, diversity=d) for d in (0, 0.75)]
        self.assertEqual([p.id for p in baseline], ["city-loop", "station-loop", "market-loop"])
        self.assertEqual([p.id for p in diverse], ["city-loop", "forest-route", "ramp-route"])
        a, b = [evaluate(order, scenario["judgments"], weights) for order in (baseline, diverse)]
        self.assertEqual(a["observed_weighted_coverage"], 0.5)
        self.assertEqual(b["observed_weighted_coverage"], 1)
        self.assertGreater(a["mean_input_relevance"], b["mean_input_relevance"])

    def test_soft_aspect_scores_leave_remaining_coverage(self):
        docs = [Candidate("a", 1, {"x": 0.75}), Candidate("b", 0.8, {"x": 0.4})]
        picks = rerank(docs, {"x": 1}, k=2, diversity=1)
        self.assertAlmostEqual(picks[1].novelty, 0.1)

    def test_weight_normalization_and_ties(self):
        docs = [Candidate(k, 1, {"x": 0.5}) for k in ["b", "a"]]
        picks = rerank(docs, {"x": 0.1}, k=1, diversity=1)
        self.assertEqual(picks[0].id, "a")
        self.assertEqual(picks, rerank(docs[::-1], {"x": 1}, k=1, diversity=1))

    def test_empty_and_oversized_k(self):
        docs = [Candidate("a", 1, {})]
        self.assertEqual(rerank(docs, {"x": 1}, k=0, diversity=0), [])
        self.assertEqual(len(rerank(docs, {"x": 1}, k=10, diversity=0)), 1)
        self.assertIsNone(evaluate([], {}, {"x": 1})["judgment_rate"])

    def test_missing_and_judged_empty_are_distinct(self):
        docs = [Candidate("a", 1, {"x": 1}), Candidate("b", 1, {"x": 1})]
        picks = rerank(docs, {"x": 1}, k=2, diversity=1)
        report = evaluate(picks, {"a": []}, {"x": 1})
        self.assertEqual(report["judgment_rate"], 0.5)
        self.assertEqual(report["observed_weighted_coverage"], 0)

    def test_invalid_configuration(self):
        docs = [Candidate("a", 1, {"x": 1})]
        for weights, k, diversity in [
            ({"y": 1}, 1, 0),
            ({"x": 0}, 1, 0),
            ({"x": 1}, -1, 0),
            ({"x": 1}, 1, float("nan")),
        ]:
            with self.subTest(weights=weights, k=k, diversity=diversity):
                with self.assertRaises(ValueError):
                    rerank(docs, weights, k=k, diversity=diversity)

    def test_duplicate_ids_and_nonfinite_scores_are_rejected(self):
        docs = [Candidate("a", 1, {}), Candidate("a", 0.8, {})]
        with self.assertRaises(ValueError):
            rerank(docs, {"x": 1}, k=2, diversity=0.5)
        with self.assertRaises(ValueError):
            rerank([Candidate("a", float("inf"), {})], {"x": 1}, k=1, diversity=0.5)
