# Diversity-Aware Retrieval Lab

An offline Python reranker that trades off individual relevance against coverage of different search intents. Keep the retriever and candidate pool fixed; compare relevance-only ranking with a diversity-aware policy.

```bash
python rerank.py examples/cycling_routes.json --k 3 --diversity 0.75
python -m unittest discover -s tests -v
```

The fictional cycling example has three competing intents: city, scenic and accessible routes. The relevance-only ranking chooses three city routes. The diversified ranking takes one route per intent. Observed weighted coverage moves from 0.50 to 1.00 while mean **input** relevance drops from 0.93 to about 0.87. These are fixture results, not a real-world gain.

## Ranking and evaluation

Normalize the intent weights to sum to one. Each intent starts with uncovered mass of 1. At each step, score candidates by `(1 - diversity) * relevance + diversity * weighted_novelty`. Novelty is the weighted sum of each intent's remaining mass times the candidate's aspect score. After selecting a candidate, multiply each intent's remaining mass by `1 - aspect_score`. This is an xQuAD-inspired heuristic; missing aspect scores are zero. Ties prefer higher relevance and then the lexicographically smaller ID. Runtime is O(k × n × a) with k selections, n candidates and a intents.

The report evaluates against separate binary judgments, not the aspect scores used for ranking. Missing judgments do not contribute to observed coverage, so reported coverage is a lower bound; judgment rate shows how many ranked items had labels. An empty judgment list is a judged item with no relevant aspects. The mean input score describes the ranking's score tradeoff, not measured user satisfaction.

The stdlib-only demo assumes upstream relevance and aspect scores already exist. It does not calibrate them, infer intent weights or retrieve candidates excluded by the first stage. The sample is synthetic; deploy only after evaluating on representative queries with blinded relevance and aspect judgments.
