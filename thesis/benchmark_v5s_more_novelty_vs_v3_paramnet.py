import numpy as np
import sys
import os
from typing import Dict, Tuple

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'HPOBench'))

# --- PATCHES FOR PARAMNET (copied from other benchmarks) ---
import types
class MockLockUtils:
    def synchronized(self, *args, **kwargs):
        def decorator(f):
            return f
        return decorator
sys.modules['oslo_concurrency'] = types.ModuleType('oslo_concurrency')
sys.modules['oslo_concurrency'].lockutils = MockLockUtils()

import sklearn.ensemble
import sklearn.tree
if not hasattr(sklearn.ensemble, 'forest'):
    try:
        from sklearn.ensemble import _forest
        sys.modules['sklearn.ensemble.forest'] = _forest
    except ImportError:
        pass
if not hasattr(sklearn.tree, 'tree'):
    try:
        sys.modules['sklearn.tree.tree'] = sklearn.tree
    except ImportError:
        pass

from thesis.hpo_lgs_v3 import HPOptimizer as HPO_v3
from thesis.hpo_lgs_v5_simple import HPOptimizer as HPO_v5s

try:
    from hpobench.benchmarks.surrogates.paramnet_benchmark import (
        ParamNetAdultOnStepsBenchmark,
        ParamNetHiggsOnStepsBenchmark,
        ParamNetLetterOnStepsBenchmark,
        ParamNetMnistOnStepsBenchmark,
        ParamNetOptdigitsOnStepsBenchmark,
        ParamNetPokerOnStepsBenchmark,
    )
except ImportError:
    print("HPOBench not found. Please install it.")
    sys.exit(1)


def _get_bounds(benchmark) -> Tuple[list, object]:
    cs = benchmark.get_configuration_space()
    bounds = []
    for hp in cs.get_hyperparameters():
        if hasattr(hp, "lower"):
            bounds.append((hp.lower, hp.upper))
        else:
            bounds.append((0.0, 1.0))
    return bounds, cs


def run_single(benchmark_cls, seed: int, budget: int, variant: str) -> Tuple[float, float]:
    b = benchmark_cls(rng=seed)
    bounds, cs = _get_bounds(b)

    if variant == "v3":
        opt = HPO_v3(bounds=bounds, maximize=False, seed=seed, n_candidates=30)
    elif variant == "v5s_more_novelty":
        opt = HPO_v5s(
            bounds=bounds,
            maximize=False,
            seed=seed,
            total_budget=budget,
            n_candidates=25,
            novelty_weight=0.4,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    best_y = np.inf

    for _ in range(budget):
        x = opt.ask()
        config = {}
        for k, hp in enumerate(cs.get_hyperparameters()):
            val = x[k]
            if hasattr(hp, "lower") and isinstance(hp.lower, int):
                val = int(round(val))
            config[hp.name] = val
        y = b.objective_function(config)["function_value"]
        opt.tell(x, y)
        if y < best_y:
            best_y = y

    return best_y, opt.best_y


def run_comparison(benchmark_cls, name: str, budget: int, n_seeds: int, start_seed: int = 200) -> Dict[str, float]:
    wins_v5s = 0
    wins_v3 = 0
    ties = 0
    bests_v5s = []
    bests_v3 = []
    rows = []  # per-seed detailed rows for this dataset

    for i in range(n_seeds):
        seed = start_seed + i
        best_v5s, _ = run_single(benchmark_cls, seed, budget, variant="v5s_more_novelty")
        best_v3, _ = run_single(benchmark_cls, seed, budget, variant="v3")

        bests_v5s.append(best_v5s)
        bests_v3.append(best_v3)

        if best_v5s + 1e-6 < best_v3:
            wins_v5s += 1
            winner = "v5s"
        elif best_v3 + 1e-6 < best_v5s:
            wins_v3 += 1
            winner = "v3"
        else:
            ties += 1
            winner = "tie"

        rows.append((name, seed, best_v5s, best_v3, winner))
        print(f"{name:10s} | seed={seed} | v5s_more_novelty={best_v5s:.6f} | v3={best_v3:.6f} | winner={winner}")

    mean_v5s = float(np.mean(bests_v5s))
    mean_v3 = float(np.mean(bests_v3))

    print(
        f"SUMMARY {name:10s} | wins_v5s={wins_v5s} | wins_v3={wins_v3} | ties={ties} | "
        f"mean_v5s={mean_v5s:.6f} | mean_v3={mean_v3:.6f}"
    )

    return {
        "wins_v5s": wins_v5s,
        "wins_v3": wins_v3,
        "ties": ties,
        "mean_v5s": mean_v5s,
        "mean_v3": mean_v3,
        "rows": rows,
    }


if __name__ == "__main__":
    benchmarks = [
        ("Adult", ParamNetAdultOnStepsBenchmark),
        ("Higgs", ParamNetHiggsOnStepsBenchmark),
        ("Letter", ParamNetLetterOnStepsBenchmark),
        ("Mnist", ParamNetMnistOnStepsBenchmark),
        ("Optdigits", ParamNetOptdigitsOnStepsBenchmark),
        ("Poker", ParamNetPokerOnStepsBenchmark),
    ]

    budget = 200
    n_seeds = 20
    start_seed = 600

    overall_wins_v5s = 0
    overall_wins_v3 = 0
    overall_ties = 0
    all_rows = []

    print("Running v5s_more_novelty vs v3 on ParamNet benchmarks...")

    for name, cls in benchmarks:
        stats = run_comparison(cls, name, budget, n_seeds, start_seed)
        overall_wins_v5s += stats["wins_v5s"]
        overall_wins_v3 += stats["wins_v3"]
        overall_ties += stats["ties"]
        all_rows.extend(stats["rows"])

    total = overall_wins_v5s + overall_wins_v3 + overall_ties
    winrate_v5s = overall_wins_v5s / total if total > 0 else 0.0

    print("\nFINAL SUMMARY:")
    print(
        f"v5s_more_novelty wins={overall_wins_v5s}, v3 wins={overall_wins_v3}, ties={overall_ties}, "
        f"winrate_v5s={winrate_v5s:.3f}"
    )

    # Print compact comparative table seed x dataset
    print("\nSEED-BY-DATASET TABLE (v5s_more_novelty vs v3):")
    print("dataset   | seed |   v5s_more_novelty   |        v3        | winner")
    for dataset, seed, best_v5s, best_v3, winner in all_rows:
        print(
            f"{dataset:9s} | {seed:4d} | {best_v5s:18.6f} | {best_v3:12.6f} | {winner}"
        )
