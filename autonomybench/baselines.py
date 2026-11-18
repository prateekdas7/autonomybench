"""Fixed strategies from FP2 Section 4.6. None should do well on both halves of a pair."""
from collections import defaultdict

from autonomybench.scorer import score

BASELINES = {
    "always_execute": {"decision": "proceed", "level": "execute"},
    "always_request_approval": {"decision": "proceed", "level": "request_approval"},
    "always_clarify": {"decision": "clarify", "level": None},
}


def run_baselines(scenarios):
    """Score each fixed strategy on a list of scenarios."""
    results = {}
    for name, fixed in BASELINES.items():
        scores = [score(fixed, s["gold"]) for s in scenarios]
        by_pair = defaultdict(list)
        for s, sc in zip(scenarios, scores):
            by_pair[s["pair_id"]].append(sc["fully_correct"])
        complete = [v for v in by_pair.values() if len(v) == 2]
        n = len(scores)
        results[name] = {
            "fully_correct": sum(sc["fully_correct"] for sc in scores) / n,
            "stage1_correct": sum(sc["stage1_correct"] for sc in scores) / n,
            "pairs_both_correct": sum(all(v) for v in complete) / len(complete) if complete else None,
        }
    return results