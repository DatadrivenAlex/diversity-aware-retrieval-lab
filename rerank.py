"""Rerank a fixed candidate pool using aspect coverage and relevance."""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    id: str
    relevance: float
    aspects: Mapping[str, float]


@dataclass(frozen=True)
class Pick:
    id: str
    score: float
    relevance: float
    novelty: float


def _unit(value: float) -> bool:
    return not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def rerank(
    candidates: Sequence[Candidate],
    aspect_weights: Mapping[str, float],
    *,
    k: int,
    diversity: float,
) -> list[Pick]:
    """Greedily maximize a relevance/weighted marginal-coverage objective."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0 or not _unit(diversity):
        raise ValueError("expected integer k >= 0 and diversity in [0, 1]")
    if not aspect_weights or any(not _unit(weight) for weight in aspect_weights.values()):
        raise ValueError("provide nonempty aspect weights in [0, 1]")
    total = sum(aspect_weights.values())
    if total == 0:
        raise ValueError("at least one aspect must have positive weight")
    weights = {name: aspect_weights[name] / total for name in sorted(aspect_weights)}
    remaining: dict[str, Candidate] = {}
    for candidate in candidates:
        if not candidate.id or candidate.id in remaining or not _unit(candidate.relevance):
            raise ValueError("candidate IDs must be nonempty and unique; relevance in [0, 1]")
        for aspect, probability in candidate.aspects.items():
            if aspect not in weights or not _unit(probability):
                raise ValueError("candidate aspects must be known and scored in [0, 1]")
        remaining[candidate.id] = candidate

    uncovered = dict.fromkeys(weights, 1.0)
    result = []
    for _ in range(min(k, len(remaining))):

        def score(candidate: Candidate) -> Pick:
            novelty = sum(
                weight * uncovered[name] * candidate.aspects.get(name, 0.0)
                for name, weight in weights.items()
            )
            value = (1 - diversity) * candidate.relevance + diversity * novelty
            return Pick(candidate.id, value, candidate.relevance, novelty)

        best = min(map(score, remaining.values()), key=lambda p: (-p.score, -p.relevance, p.id))
        winner = remaining.pop(best.id)
        for name in uncovered:
            uncovered[name] *= 1 - winner.aspects.get(name, 0.0)
        result.append(best)
    return result


def evaluate(
    ranking: Sequence[Pick],
    judgments: Mapping[str, Sequence[str]],
    aspect_weights: Mapping[str, float],
) -> dict[str, float | int | None]:
    """Report judged-only coverage (a lower bound), and separate judgment rate."""
    found: set[str] = set()
    judged_count = 0
    total_weight = sum(aspect_weights.values())
    for item in ranking:
        if item.id in judgments:
            judged_count += 1
            found.update(set(judgments[item.id]).intersection(aspect_weights))
    return {
        "judged_items": judged_count,
        "judgment_rate": judged_count / len(ranking) if ranking else None,
        "observed_weighted_coverage": sum(aspect_weights[a] for a in sorted(found)) / total_weight,
        "mean_input_relevance": (
            sum(p.relevance for p in ranking) / len(ranking) if ranking else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--diversity", type=float, default=0.75)
    args = parser.parse_args()
    try:
        scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
        pool = [Candidate(**row) for row in scenario["candidates"]]
        weights = scenario["aspect_weights"]
        judgments = scenario["judgments"]
        for label, strength in [("relevance", 0.0), ("diversified", args.diversity)]:
            picks = rerank(pool, weights, k=args.k, diversity=strength)
            report = {"ranking": [p.id for p in picks], **evaluate(picks, judgments, weights)}
            print(f"{label}: {json.dumps(report, sort_keys=True)}")
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
