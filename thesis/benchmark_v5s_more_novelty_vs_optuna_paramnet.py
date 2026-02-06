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

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

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


def _get_bounds_and_cs(benchmark) -> Tuple[list, object]:
    cs = benchmark.get_configuration_space()
    bounds = []
    for hp in cs.get_hyperparameters():
        if hasattr(hp, "lower"):
            bounds.append((hp.lower, hp.upper))
        else:
            bounds.append((0.0, 1.0))
    return bounds, cs


def _vector_to_config(x_vec, cs) -> dict:
    config = {}
    for val, hp in zip(x_vec, cs.get_hyperparameters()):
        v = float(val)
        if hasattr(hp, "lower") and hasattr(hp, "upper"):
            lo, hi = hp.lower, hp.upper
            if lo is not None and hi is not None:
                v = min(max(v, lo), hi)
            if isinstance(lo, int) and isinstance(hi, int):
                v = int(round(v))
        config[hp.name] = v
    return config


def optuna_search(objective, dim: int, budget: int, seed: int) -> float:
    if not HAS_OPTUNA:
        print("Optuna not available; returning NaN")
        return float("nan")

    def _objective(trial: "optuna.trial.Trial") -> float:
        x_norm = np.array(
            [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)],
            dtype=float,
        )
        return float(objective(x_norm))

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=int(budget), show_progress_bar=False)
    return float(study.best_value)


def run_single(benchmark_cls, seed: int, budget: int, variant: str) -> Tuple[float, float]:
    bench = benchmark_cls(rng=seed)
    bounds, cs = _get_bounds_and_cs(bench)
    dim = len(bounds)

    if variant == "v5s_more_novelty":
        opt = HPO_v5s(
            bounds=bounds,
            maximize=False,
            seed=seed,
            total_budget=budget,
            n_candidates=25,
            novelty_weight=0.4,
        )
        best_y = np.inf
        for _ in range(budget):
            x = opt.ask()
            config = _vector_to_config(x, cs)
            y = bench.objective_function(config)["function_value"]
            opt.tell(x, y)
            if y < best_y:
                best_y = y
        return best_y, opt.best_y

    if variant == "optuna":
        lo_hi = [(float(lo), float(hi)) for lo, hi in bounds]

        def _objective(x_norm: np.ndarray) -> float:
            x_vec = [lo + xn * (hi - lo) for xn, (lo, hi) in zip(x_norm, lo_hi)]
            cfg = _vector_to_config(x_vec, cs)
            return float(bench.objective_function(cfg)["function_value"])

        best_val = optuna_search(_objective, dim=dim, budget=budget, seed=seed)
        return best_val, best_val

    raise ValueError(f"Unknown variant: {variant}")


def run_comparison(benchmark_cls, name: str, budget: int, n_seeds: int, start_seed: int = 200) -> Dict[str, float]:
    wins_v5s = 0
    wins_optuna = 0
    ties = 0
    bests_v5s = []
    bests_optuna = []
    rows = []

    for i in range(n_seeds):
        seed = start_seed + i
        best_v5s, _ = run_single(benchmark_cls, seed, budget, variant="v5s_more_novelty")
        best_optuna, _ = run_single(benchmark_cls, seed, budget, variant="optuna")

        bests_v5s.append(best_v5s)
        bests_optuna.append(best_optuna)

        if best_v5s + 1e-6 < best_optuna:
            wins_v5s += 1
            winner = "v5s"
        elif best_optuna + 1e-6 < best_v5s:
            wins_optuna += 1
            winner = "optuna"
        else:
            ties += 1
            winner = "tie"

        rows.append((name, seed, best_v5s, best_optuna, winner))
        print(f"{name:10s} | seed={seed} | v5s_more_novelty={best_v5s:.6f} | optuna={best_optuna:.6f} | winner={winner}")

    mean_v5s = float(np.mean(bests_v5s))
    mean_optuna = float(np.mean(bests_optuna))

    print(
        f"SUMMARY {name:10s} | wins_v5s={wins_v5s} | wins_optuna={wins_optuna} | ties={ties} | "
        f"mean_v5s={mean_v5s:.6f} | mean_optuna={mean_optuna:.6f}"
    )

    return {
        "wins_v5s": wins_v5s,
        "wins_optuna": wins_optuna,
        "ties": ties,
        "mean_v5s": mean_v5s,
        "mean_optuna": mean_optuna,
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

    if not HAS_OPTUNA:
        print("Optuna is not installed. Please install optuna to run this benchmark.")
        sys.exit(1)

    overall_wins_v5s = 0
    overall_wins_optuna = 0
    overall_ties = 0
    all_rows = []

    print("Running v5s_more_novelty vs Optuna on ParamNet benchmarks...")

    for name, cls in benchmarks:
        stats = run_comparison(cls, name, budget, n_seeds, start_seed)
        overall_wins_v5s += stats["wins_v5s"]
        overall_wins_optuna += stats["wins_optuna"]
        overall_ties += stats["ties"]
        all_rows.extend(stats["rows"])

    total = overall_wins_v5s + overall_wins_optuna + overall_ties
    winrate_v5s = overall_wins_v5s / total if total > 0 else 0.0

    print("\nFINAL SUMMARY:")
    print(
        f"v5s_more_novelty wins={overall_wins_v5s}, optuna wins={overall_wins_optuna}, ties={overall_ties}, "
        f"winrate_v5s={winrate_v5s:.3f}"
    )

    print("\nSEED-BY-DATASET TABLE (v5s_more_novelty vs optuna):")
    print("dataset   | seed |   v5s_more_novelty   |      optuna      | winner")
    for dataset, seed, best_v5s, best_optuna, winner in all_rows:
        print(
            f"{dataset:9s} | {seed:4d} | {best_v5s:18.6f} | {best_optuna:14.6f} | {winner}"
        )
