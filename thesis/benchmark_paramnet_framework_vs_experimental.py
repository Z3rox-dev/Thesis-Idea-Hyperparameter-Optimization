#!/usr/bin/env python3
"""HPOBench ParamNet battery: alba_framework vs alba_experimental.

- Minimizes function_value (loss) from HPOBench ParamNet surrogate.
- Runs over multiple ParamNet datasets.
- Stores incremental checkpoints to JSON.

Intended env: conda `py39`.

Example:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/benchmark_paramnet_framework_vs_experimental.py --budget 400 --seeds 70-79
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np

# Reduce noisy warning spam from old surrogate pickles / sklearn / numpy.
warnings.filterwarnings("ignore")

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)

# ParamNet surrogates shipped with HPOBench are pickled with older scikit-learn
# module paths (e.g. sklearn.ensemble.forest). Add a small import alias so
# unpickling works with modern scikit-learn.
try:
    import sklearn.ensemble._forest as _sk_forest  # type: ignore

    sys.modules.setdefault("sklearn.ensemble.forest", _sk_forest)
except Exception:
    pass

try:
    import sklearn.tree._classes as _sk_tree_classes  # type: ignore

    sys.modules.setdefault("sklearn.tree.tree", _sk_tree_classes)
except Exception:
    pass

# HPOBench (vendored)
sys.path.insert(0, "/mnt/workspace/HPOBench")

# Compat for older deps that use np.float/np.int/np.bool.
# Use setattr (no attribute access) and suppress FutureWarning that NumPy may emit
# when defining deprecated aliases.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*np\\.bool.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*np\\.int.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*np\\.float.*")

    _np_dict = np.__dict__
    if "float" not in _np_dict:
        setattr(np, "float", float)  # type: ignore[attr-defined]
    if "int" not in _np_dict:
        setattr(np, "int", int)  # type: ignore[attr-defined]
    if "bool" not in _np_dict:
        setattr(np, "bool", bool)  # type: ignore[attr-defined]

import ConfigSpace as CS

from hpobench.benchmarks.surrogates.paramnet_benchmark import (  # type: ignore[import]
    ParamNetAdultOnStepsBenchmark,
    ParamNetHiggsOnStepsBenchmark,
    ParamNetLetterOnStepsBenchmark,
    ParamNetMnistOnStepsBenchmark,
    ParamNetOptdigitsOnStepsBenchmark,
    ParamNetPokerOnStepsBenchmark,
)

# Add thesis to path for ALBA imports
sys.path.insert(0, "/mnt/workspace/thesis")
from alba_framework import ALBA as ALBA_FRAMEWORK  # noqa: E402
from ALBA_V1_experimental import ALBA as ALBA_EXP  # noqa: E402


RESULTS_DIR = "/mnt/workspace/thesis/benchmark_results"

_PARAMNET_STEPS_MAP = {
    "adult": ParamNetAdultOnStepsBenchmark,
    "higgs": ParamNetHiggsOnStepsBenchmark,
    "letter": ParamNetLetterOnStepsBenchmark,
    "mnist": ParamNetMnistOnStepsBenchmark,
    "optdigits": ParamNetOptdigitsOnStepsBenchmark,
    "poker": ParamNetPokerOnStepsBenchmark,
}


def _parse_seeds(arg: str) -> List[int]:
    arg = arg.strip()
    if not arg:
        return []
    if "-" in arg and "," not in arg:
        a, b = arg.split("-", 1)
        lo = int(a)
        hi = int(b)
        if hi < lo:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    out: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a)
            hi = int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def save_results(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _x01_to_cs_config(x01: np.ndarray, cs: CS.ConfigurationSpace) -> CS.Configuration:
    """Map x in [0,1]^d to ConfigSpace Configuration, respecting log-scale."""
    x01 = np.asarray(x01, dtype=float)
    hps = cs.get_hyperparameters()
    if len(x01) != len(hps):
        raise ValueError(f"Expected x of length {len(hps)}, got {len(x01)}")

    values: Dict[str, Any] = {}
    for i, hp in enumerate(hps):
        v01 = float(np.clip(x01[i], 0.0, 1.0))
        lo = float(getattr(hp, "lower"))
        hi = float(getattr(hp, "upper"))

        if getattr(hp, "log", False):
            val = float(np.exp(np.log(lo) + v01 * (np.log(hi) - np.log(lo))))
        else:
            val = float(lo + v01 * (hi - lo))

        if isinstance(hp, CS.UniformIntegerHyperparameter):
            val = int(round(val))
            val = max(int(hp.lower), min(int(hp.upper), int(val)))

        values[hp.name] = val

    return CS.Configuration(cs, values=values)


def main() -> int:
    p = argparse.ArgumentParser(description="ParamNet battery: alba_framework vs alba_experimental")
    p.add_argument("--datasets", default="adult,higgs,letter,mnist,optdigits,poker")
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--seeds", default="70-79", help="Comma list and/or ranges, e.g. '70-79' or '70,71,72'")
    p.add_argument("--step", type=int, default=50, help="Fidelity step for ParamNet (default 50)")
    args = p.parse_args()

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    seeds = _parse_seeds(args.seeds)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{RESULTS_DIR}/paramnet_framework_exp_b{args.budget}_s{seeds[0]}-{seeds[-1]}_{timestamp}.json"

    results: Dict[str, Any] = {
        "config": {
            "datasets": datasets,
            "budget": args.budget,
            "seeds": seeds,
            "step": args.step,
            "timestamp": timestamp,
        },
        "datasets": {},
    }
    save_results(out_path, results)

    print("=" * 78)
    print(f"PARAMNET BATTERY: alba_framework vs alba_experimental")
    print(f"budget={args.budget} | seeds={seeds} | step={args.step}")
    print(f"datasets={datasets}")
    print(f"save={out_path}")
    print("=" * 78)
    print()

    for ds in datasets:
        if ds not in _PARAMNET_STEPS_MAP:
            print(f"Skipping unsupported dataset: {ds}")
            continue

        print(f"== {ds.upper()} ==")
        bench = _PARAMNET_STEPS_MAP[ds]()
        cs = bench.get_configuration_space()
        dim = len(cs.get_hyperparameters())

        results["datasets"].setdefault(ds, {"framework": {}, "experimental": {}, "optuna": {}, "errors": {}})
        save_results(out_path, results)

        for seed in seeds:
            t0 = time.time()
            try:
                # ---------------- FRAMEWORK (bounds array) ----------------
                opt_framework = ALBA_FRAMEWORK(
                    bounds=[(0.0, 1.0)] * dim,
                    seed=seed,
                    maximize=False,
                    total_budget=args.budget,
                    split_depth_max=8,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                )

                best_framework = float("inf")
                for it in range(args.budget):
                    x = opt_framework.ask()
                    cfg = _x01_to_cs_config(x, cs)
                    res = bench.objective_function(configuration=cfg, fidelity={"step": int(args.step)})
                    y = float(res["function_value"])
                    opt_framework.tell(x, y)
                    best_framework = min(best_framework, y)

                results["datasets"][ds]["framework"][str(seed)] = float(best_framework)
                save_results(out_path, results)

                # -------------- EXPERIMENTAL (bounds array) --------------
                opt_exp = ALBA_EXP(
                    bounds=[(0.0, 1.0)] * dim,
                    seed=seed,
                    maximize=False,
                    total_budget=args.budget,
                    split_depth_max=8,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                )

                best_exp = float("inf")
                for it in range(args.budget):
                    x = opt_exp.ask()
                    cfg = _x01_to_cs_config(x, cs)
                    res = bench.objective_function(configuration=cfg, fidelity={"step": int(args.step)})
                    y = float(res["function_value"])
                    opt_exp.tell(x, y)
                    best_exp = min(best_exp, y)

                results["datasets"][ds]["experimental"][str(seed)] = float(best_exp)
                save_results(out_path, results)

                # ---------------- OPTUNA (TPE) ----------------
                optuna_best = float("inf")

                def optuna_objective(trial: optuna.Trial) -> float:
                    nonlocal optuna_best
                    x = np.array(
                        [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)],
                        dtype=float,
                    )
                    cfg = _x01_to_cs_config(x, cs)
                    res = bench.objective_function(configuration=cfg, fidelity={"step": int(args.step)})
                    y = float(res["function_value"])
                    if y < optuna_best:
                        optuna_best = y
                    return y

                sampler = optuna.samplers.TPESampler(seed=int(seed), multivariate=True)
                study = optuna.create_study(direction="minimize", sampler=sampler)
                study.optimize(optuna_objective, n_trials=int(args.budget), show_progress_bar=False)

                results["datasets"][ds]["optuna"][str(seed)] = float(optuna_best)
                save_results(out_path, results)

                # Minimal, per-seed output: only method name + score
                print(f"seed {seed}")
                pair = [
                    ("alba_framework", float(best_framework)),
                    ("alba_experimental", float(best_exp)),
                    ("optuna_tpe", float(optuna_best)),
                ]
                pair.sort(key=lambda t: t[1])
                for name, score in pair:
                    print(f"  {name} {score:.6f}")

            except Exception as e:
                results["datasets"][ds]["errors"][str(seed)] = str(e)
                save_results(out_path, results)
                print(f"seed {seed}")
                print(f"  alba_framework ERROR")
                print(f"  alba_experimental ERROR")
                print(f"  optuna_tpe ERROR")
                print(f"  error {e}")
        print()

    print(f"✓ Results saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
