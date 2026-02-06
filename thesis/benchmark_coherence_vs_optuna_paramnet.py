#!/usr/bin/env python3
"""Benchmark: ALBA Coherence vs Optuna TPE on ParamNet surrogates.

Runs on all 6 ParamNet datasets (Adult, Higgs, Letter, Mnist, Optdigits, Poker).
Budget=400, multiple seeds.

Example:
  python thesis/benchmark_coherence_vs_optuna_paramnet.py --budget 400 --seeds 0-9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

# --- HPOBench patches ---
sys.path.insert(0, "/mnt/workspace/HPOBench")

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

from hpobench.benchmarks.surrogates.paramnet_benchmark import (
    ParamNetAdultOnStepsBenchmark,
    ParamNetHiggsOnStepsBenchmark,
    ParamNetLetterOnStepsBenchmark,
    ParamNetMnistOnStepsBenchmark,
    ParamNetOptdigitsOnStepsBenchmark,
    ParamNetPokerOnStepsBenchmark,
)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, "/mnt/workspace/thesis")
from alba_framework_coherence import ALBA as ALBA_COH

RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"

BENCHMARKS = [
    ("Adult", ParamNetAdultOnStepsBenchmark),
    ("Higgs", ParamNetHiggsOnStepsBenchmark),
    ("Letter", ParamNetLetterOnStepsBenchmark),
    ("Mnist", ParamNetMnistOnStepsBenchmark),
    ("Optdigits", ParamNetOptdigitsOnStepsBenchmark),
    ("Poker", ParamNetPokerOnStepsBenchmark),
]


def _parse_seeds(arg: str) -> List[int]:
    arg = arg.strip()
    if "-" in arg and "," not in arg:
        a, b = arg.split("-", 1)
        return list(range(int(a), int(b) + 1))
    out: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _parse_checkpoints(arg: str, budget: int) -> List[int]:
    cps = [int(x.strip()) for x in arg.split(",") if x.strip()]
    cps = sorted(set(c for c in cps if 1 <= c <= budget))
    if budget not in cps:
        cps.append(budget)
    return cps


def _get_bounds_and_cs(benchmark):
    cs = benchmark.get_configuration_space()
    bounds = []
    for hp in cs.get_hyperparameters():
        if hasattr(hp, "lower"):
            bounds.append((hp.lower, hp.upper))
        else:
            bounds.append((0.0, 1.0))
    return bounds, cs


def save_results(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run_coherence(benchmark_cls, seed: int, budget: int, checkpoints: List[int]) -> Dict[str, float]:
    b = benchmark_cls(rng=seed)
    bounds, cs = _get_bounds_and_cs(b)
    hp_list = list(cs.get_hyperparameters())

    opt = ALBA_COH(
        bounds=bounds,
        maximize=False,
        seed=seed,
        total_budget=budget,
        use_coherence_gating=True,
    )

    best = float("inf")
    cp_results: Dict[str, float] = {}

    for it in range(budget):
        x = opt.ask()
        config = {}
        for k, hp in enumerate(hp_list):
            val = x[k]
            if hasattr(hp, "lower") and isinstance(hp.lower, int):
                val = int(round(val))
            config[hp.name] = val
        y = b.objective_function(config)["function_value"]
        opt.tell(x, y)
        best = min(best, y)
        if (it + 1) in checkpoints:
            cp_results[str(it + 1)] = float(best)

    return cp_results


def run_optuna(benchmark_cls, seed: int, budget: int, checkpoints: List[int]) -> Dict[str, float]:
    b = benchmark_cls(rng=seed)
    cs = b.get_configuration_space()
    hp_list = list(cs.get_hyperparameters())

    best = float("inf")
    cp_results: Dict[str, float] = {}
    trial_count = [0]

    def objective(trial):
        nonlocal best
        config = {}
        for hp in hp_list:
            if hasattr(hp, "lower"):
                if isinstance(hp.lower, int):
                    config[hp.name] = trial.suggest_int(hp.name, int(hp.lower), int(hp.upper))
                else:
                    config[hp.name] = trial.suggest_float(hp.name, hp.lower, hp.upper)
            else:
                config[hp.name] = trial.suggest_float(hp.name, 0.0, 1.0)
        y = b.objective_function(config)["function_value"]
        best = min(best, y)
        trial_count[0] += 1
        if trial_count[0] in checkpoints:
            cp_results[str(trial_count[0])] = float(best)
        return y

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)

    if str(budget) not in cp_results:
        cp_results[str(budget)] = float(best)

    return cp_results


def main() -> int:
    p = argparse.ArgumentParser(description="Coherence vs Optuna on ParamNet")
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--checkpoints", default="100,200,400")
    p.add_argument("--seeds", default="0-9")
    args = p.parse_args()

    seeds = _parse_seeds(args.seeds)
    checkpoints = _parse_checkpoints(args.checkpoints, args.budget)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{RESULTS_DIR}/coherence_vs_optuna_paramnet_b{args.budget}_s{seeds[0]}-{seeds[-1]}_{timestamp}.json"

    results: Dict[str, Any] = {
        "config": {
            "budget": args.budget,
            "checkpoints": checkpoints,
            "seeds": seeds,
            "timestamp": timestamp,
        },
        "datasets": {},
    }
    save_results(out_path, results)

    print(f"PARAMNET BATTERY | budget={args.budget} | seeds={seeds}")
    print(f"checkpoints={checkpoints}")
    print(f"save={out_path}")
    print("=" * 80)

    total_wins_coh = 0
    total_wins_opt = 0
    total_ties = 0

    for name, benchmark_cls in BENCHMARKS:
        results["datasets"].setdefault(name, {"coherence": {}, "optuna": {}})
        wins_coh = 0
        wins_opt = 0
        ties = 0

        for seed in seeds:
            t0 = time.time()
            try:
                cp_coh = run_coherence(benchmark_cls, seed, args.budget, checkpoints)
                cp_opt = run_optuna(benchmark_cls, seed, args.budget, checkpoints)

                results["datasets"][name]["coherence"][str(seed)] = cp_coh
                results["datasets"][name]["optuna"][str(seed)] = cp_opt
                save_results(out_path, results)

                final_coh = cp_coh.get(str(args.budget), float("inf"))
                final_opt = cp_opt.get(str(args.budget), float("inf"))

                if final_coh + 1e-8 < final_opt:
                    winner = "COH"
                    wins_coh += 1
                elif final_opt + 1e-8 < final_coh:
                    winner = "OPT"
                    wins_opt += 1
                else:
                    winner = "TIE"
                    ties += 1

                elapsed = time.time() - t0
                print(f"{name:10s} seed={seed:2d} | COH={final_coh:.6f} OPT={final_opt:.6f} -> {winner} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"{name:10s} seed={seed:2d} | ERROR: {e}")

        total_wins_coh += wins_coh
        total_wins_opt += wins_opt
        total_ties += ties
        print(f"{name:10s} SUMMARY: COH wins={wins_coh}, OPT wins={wins_opt}, ties={ties}")
        print("-" * 80)

    total = total_wins_coh + total_wins_opt + total_ties
    winrate_coh = total_wins_coh / total if total > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"FINAL: Coherence wins={total_wins_coh}, Optuna wins={total_wins_opt}, ties={total_ties}")
    print(f"Coherence winrate: {winrate_coh:.1%}")
    print(f"Saved: {out_path}")

    results["summary"] = {
        "coherence_wins": total_wins_coh,
        "optuna_wins": total_wins_opt,
        "ties": total_ties,
        "coherence_winrate": winrate_coh,
    }
    save_results(out_path, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
