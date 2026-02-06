#!/usr/bin/env python3
"""
Benchmark: Optuna TPE vs ALBA_V1 vs Random Search
on WFG test suite (WFG1 - WFG7).

WFG è una suite standard di benchmark per ottimizzazione (originariamente
multi-obiettivo, qui usiamo la scalarizzazione sul primo obiettivo per
renderlo single-objective, come spesso fatto per HPO benchmarking).

Environment: conda py39 (con pymoo installato)

Usage:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/benchmark_wfg_optuna_alba_random.py [--budget 200] [--seeds 42 43 44] [--n_var 10]
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# WFG from pymoo
from pymoo.problems import get_problem

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_alba_v1():
    """Load ALBA_V1 from thesis folder."""
    sys.path.insert(0, "/mnt/workspace/thesis")
    from ALBA_V1 import ALBA
    return ALBA


class WFGWrapper:
    """
    Wrapper per problemi WFG (pymoo).
    Converte il problema in single-objective usando il primo obiettivo.
    Le variabili sono normalizzate in [0, 1].
    """

    def __init__(
        self,
        problem_name: str,
        n_var: int = 10,
        n_obj: int = 2,
        scalarization: str = "f0",
        weights: Optional[np.ndarray] = None,
        eps: float = 1e-12,
    ):
        self.problem_name = problem_name
        self.n_var = n_var
        self.n_obj = n_obj
        self.problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)
        self.dim = n_var
        self.bounds = [(0.0, 1.0)] * n_var
        self.scalarization = str(scalarization)
        self.weights = weights
        self.eps = float(eps)
        self.n_evals = 0

    def reset(self):
        self.n_evals = 0

    def evaluate_vector(self, x: np.ndarray) -> np.ndarray:
        """Evaluate WFG and return the full objective vector F (minimize)."""
        # WFG expects x in [0, 2*i] for variable i (standard bounds).
        # pymoo's WFG already handles internal scaling, but let's be safe:
        # The problem.xu gives upper bounds. Scale from [0,1] to [xl, xu].
        xl = self.problem.xl
        xu = self.problem.xu
        x_scaled = xl + x * (xu - xl)

        # Evaluate
        out = {}
        self.problem._evaluate(x_scaled.reshape(1, -1), out)
        f = np.array(out["F"][0], dtype=float)  # shape (n_obj,)
        self.n_evals += 1

        return f

    def scalarize(self, f: np.ndarray) -> float:
        """Convert multiobjective vector f into a scalar to minimize."""
        if f.ndim != 1 or f.shape[0] != self.n_obj:
            raise ValueError(f"Expected f shape ({self.n_obj},), got {f.shape}")

        method = self.scalarization
        if method == "f0":
            return float(f[0])

        w = self.weights
        if w is None:
            w = np.ones(self.n_obj, dtype=float) / float(self.n_obj)
        w = np.asarray(w, dtype=float)
        if w.shape != (self.n_obj,):
            raise ValueError(f"weights must have shape ({self.n_obj},), got {w.shape}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if float(np.sum(w)) <= 0:
            raise ValueError("weights must sum to a positive number")
        w = w / float(np.sum(w))

        if method == "wsum":
            return float(np.dot(w, f))

        if method == "tcheby":
            # Classical weighted Tchebycheff around reference point z=0.
            # WFG objectives are non-negative; using z=0 keeps it stationary.
            return float(np.max(w * np.abs(f)))

        if method == "wgm":
            # Weighted geometric mean (requires positivity). Use eps for safety.
            f_pos = np.maximum(f, self.eps)
            return float(np.exp(np.dot(w, np.log(f_pos))))

        raise ValueError(f"Unknown scalarization: {method}")

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate and return the scalarized objective (minimize)."""
        f = self.evaluate_vector(x)
        return self.scalarize(f)

        # Return first objective (single-objective proxy)
        return float(f[0])

    def evaluate_array(self, x: np.ndarray) -> float:
        return self.evaluate(x)


def run_random_search(wrapper: WFGWrapper, budget: int, seed: int) -> Tuple[float, List[float]]:
    """Random search baseline."""
    rng = np.random.default_rng(seed)
    wrapper.reset()

    best_y = float("inf")
    curve = []

    for _ in range(budget):
        x = rng.uniform(0, 1, wrapper.dim)
        y = wrapper.evaluate(x)
        best_y = min(best_y, y)
        curve.append(best_y)

    return best_y, curve


def run_optuna_tpe(wrapper: WFGWrapper, budget: int, seed: int) -> Tuple[float, List[float]]:
    """Optuna TPE sampler."""
    wrapper.reset()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    curve = []
    best_y = float("inf")

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_y
        x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(wrapper.dim)])
        y = wrapper.evaluate(x)
        best_y = min(best_y, y)
        curve.append(best_y)
        return y

    study.optimize(objective, n_trials=budget, show_progress_bar=False)

    return study.best_value, curve


def run_alba_v1(wrapper: WFGWrapper, budget: int, seed: int, ALBA) -> Tuple[float, List[float]]:
    """ALBA_V1 optimizer."""
    wrapper.reset()

    opt = ALBA(
        bounds=wrapper.bounds,
        maximize=False,
        seed=seed,
        split_depth_max=8,
        total_budget=budget,
        global_random_prob=0.05,
        stagnation_threshold=50,
    )

    best_y = float("inf")
    curve = []

    for _ in range(budget):
        x = opt.ask()
        y = wrapper.evaluate(x)
        opt.tell(x, y)
        best_y = min(best_y, y)
        curve.append(best_y)

    return best_y, curve


def _seeded_weights(problem_name: str, n_obj: int, seed: int) -> np.ndarray:
    """Deterministic weights per (problem, seed) so all algorithms see same scalarization."""
    # Stable-ish hash without relying on Python's randomized hash.
    key = f"{problem_name}|{seed}|{n_obj}"
    h = 2166136261
    for ch in key.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    rng = np.random.default_rng(h)
    # Dirichlet weights (strictly positive and sum to 1)
    alpha = np.ones(n_obj, dtype=float)
    return rng.dirichlet(alpha)


def main():
    parser = argparse.ArgumentParser(description="WFG benchmark: Optuna vs ALBA vs Random")
    parser.add_argument("--budget", type=int, default=200, help="Number of evaluations per run")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--n_var", type=int, default=10, help="Number of variables (dimensionality)")
    parser.add_argument("--n_obj", type=int, default=2, help="Number of objectives (only first used)")
    parser.add_argument(
        "--scalarization",
        type=str,
        default="f0",
        choices=["f0", "wsum", "tcheby", "wgm"],
        help="How to turn F (multiobjective) into a scalar objective to minimize",
    )
    parser.add_argument("--problems", type=str, nargs="*", default=None, help="WFG problems to test (default: wfg1-wfg7)")
    parser.add_argument("--results_file", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    budget = args.budget
    seeds = args.seeds
    n_var = args.n_var
    n_obj = args.n_obj
    scalarization = args.scalarization
    problems = args.problems or [f"wfg{i}" for i in range(1, 8)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = args.results_file or os.path.join(
        RESULTS_DIR,
        f"wfg_optuna_alba_random_b{budget}_d{n_var}_{timestamp}.json",
    )

    print("=" * 70)
    print("WFG BENCHMARK: Optuna TPE vs ALBA_V1 vs Random")
    print("=" * 70)
    print(f"Budget: {budget}")
    print(f"Seeds: {seeds}")
    print(f"Dimensions: {n_var}")
    print(f"Objectives: {n_obj}")
    print(f"Scalarization: {scalarization}")
    print(f"Problems: {problems}")
    print(f"Results: {results_file}")
    print("=" * 70)

    ALBA = load_alba_v1()

    results: Dict[str, Any] = {
        "config": {
            "budget": budget,
            "seeds": seeds,
            "n_var": n_var,
            "n_obj": n_obj,
            "scalarization": scalarization,
            "problems": problems,
            "timestamp": timestamp,
        },
        "results": {},
    }

    for prob_name in problems:
        print(f"\n>>> {prob_name.upper()}")
        # Weights are chosen per-seed; wrapper will be re-instantiated inside the seed loop.

        prob_results = {
            "random": {"finals": [], "curves": []},
            "optuna_tpe": {"finals": [], "curves": []},
            "alba_v1": {"finals": [], "curves": []},
        }

        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            t0 = time.time()

            weights = _seeded_weights(prob_name, n_obj=n_obj, seed=seed)
            wrapper = WFGWrapper(
                prob_name,
                n_var=n_var,
                n_obj=n_obj,
                scalarization=scalarization,
                weights=weights,
            )

            # Random
            y_rand, curve_rand = run_random_search(wrapper, budget, seed)
            prob_results["random"]["finals"].append(y_rand)
            prob_results["random"]["curves"].append(curve_rand)

            # Optuna TPE
            y_tpe, curve_tpe = run_optuna_tpe(wrapper, budget, seed)
            prob_results["optuna_tpe"]["finals"].append(y_tpe)
            prob_results["optuna_tpe"]["curves"].append(curve_tpe)

            # ALBA_V1
            y_alba, curve_alba = run_alba_v1(wrapper, budget, seed, ALBA)
            prob_results["alba_v1"]["finals"].append(y_alba)
            prob_results["alba_v1"]["curves"].append(curve_alba)

            # Save weights used for this run (so results are reproducible)
            prob_results.setdefault("weights", {})
            prob_results["weights"][str(seed)] = [float(x) for x in weights]

            elapsed = time.time() - t0
            print(f"Random={y_rand:.4f}  TPE={y_tpe:.4f}  ALBA={y_alba:.4f}  ({elapsed:.1f}s)")

        # Summary per problem
        for algo in ["random", "optuna_tpe", "alba_v1"]:
            finals = prob_results[algo]["finals"]
            prob_results[algo]["mean"] = float(np.mean(finals))
            prob_results[algo]["std"] = float(np.std(finals))
            prob_results[algo]["min"] = float(np.min(finals))

        results["results"][prob_name] = prob_results

        # Save incrementally
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std of final objective)")
    print("=" * 70)
    print(f"{'Problem':<10} {'Random':<20} {'Optuna TPE':<20} {'ALBA_V1':<20} {'Winner':<10}")
    print("-" * 70)

    wins = {"random": 0, "optuna_tpe": 0, "alba_v1": 0}

    for prob_name in problems:
        pr = results["results"][prob_name]
        rand_str = f"{pr['random']['mean']:.4f} ± {pr['random']['std']:.4f}"
        tpe_str = f"{pr['optuna_tpe']['mean']:.4f} ± {pr['optuna_tpe']['std']:.4f}"
        alba_str = f"{pr['alba_v1']['mean']:.4f} ± {pr['alba_v1']['std']:.4f}"

        means = {
            "random": pr["random"]["mean"],
            "optuna_tpe": pr["optuna_tpe"]["mean"],
            "alba_v1": pr["alba_v1"]["mean"],
        }
        winner = min(means, key=means.get)
        wins[winner] += 1

        print(f"{prob_name:<10} {rand_str:<20} {tpe_str:<20} {alba_str:<20} {winner:<10}")

    print("-" * 70)
    print(f"Wins: Random={wins['random']}  Optuna={wins['optuna_tpe']}  ALBA={wins['alba_v1']}")
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
