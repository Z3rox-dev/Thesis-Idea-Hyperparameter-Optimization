#!/usr/bin/env python3
"""HPOBench ParamNet battery: ALBA_V1 (param_space) vs ALBA_V1_experimental.

- Minimizes function_value (loss) from HPOBench ParamNet surrogate.
- Runs over multiple ParamNet datasets.
- Stores incremental checkpoints to JSON.

Intended env: conda `py39`.

Example:
  source /mnt/workspace/miniconda3/bin/activate py39
  python thesis/benchmark_paramnet_v1_vs_experimental_battery.py --budget 400 --seeds 70-79
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

# Compat for older deps that use np.float/np.int/np.bool
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
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


def _parse_checkpoints(arg: str, budget: int) -> List[int]:
    cps: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        cps.append(int(part))
    cps = [c for c in sorted(set(cps)) if 1 <= c <= budget]
    if budget not in cps:
        cps.append(budget)
    return cps


def save_results(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _param_space_from_cs(cs: CS.ConfigurationSpace) -> Tuple[Dict[str, Any], int]:
    """Build ALBA param_space spec from ConfigSpace.

    Supports UniformFloat/UniformInteger; respects log-scale if present.
    """
    param_space: Dict[str, Any] = {}
    hps = cs.get_hyperparameters()
    for hp in hps:
        if isinstance(hp, CS.UniformFloatHyperparameter):
            if getattr(hp, "log", False):
                param_space[hp.name] = (float(hp.lower), float(hp.upper), "log")
            else:
                param_space[hp.name] = (float(hp.lower), float(hp.upper))
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            # Integer range; log-scale is rare but handle anyway
            if getattr(hp, "log", False):
                param_space[hp.name] = (float(hp.lower), float(hp.upper), "log")
            else:
                param_space[hp.name] = (float(hp.lower), float(hp.upper), "int")
        else:
            raise ValueError(f"Unsupported hyperparameter type: {hp} ({type(hp)})")
    return param_space, len(hps)


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
    p = argparse.ArgumentParser(description="ParamNet battery: ALBA_V1(param_space) vs ALBA_V1_experimental")
    p.add_argument("--datasets", default="adult,higgs,letter,mnist,optdigits,poker")
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--checkpoints", default="100,200,400")
    p.add_argument("--seeds", default="70-79", help="Comma list and/or ranges, e.g. '70-79' or '70,71,72'")
    p.add_argument("--step", type=int, default=50, help="Fidelity step for ParamNet (default 50)")
    args = p.parse_args()

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    seeds = _parse_seeds(args.seeds)
    checkpoints = _parse_checkpoints(args.checkpoints, args.budget)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{RESULTS_DIR}/paramnet_v1_vs_experimental_b{args.budget}_s{seeds[0]}-{seeds[-1]}_{timestamp}.json"

    sys.path.insert(0, "/mnt/workspace/thesis")
    from ALBA_V1 import ALBA as ALBA_V1
    from ALBA_V1_experimental import ALBA as ALBA_EXP

    results: Dict[str, Any] = {
        "config": {
            "datasets": datasets,
            "budget": args.budget,
            "checkpoints": checkpoints,
            "seeds": seeds,
            "step": args.step,
            "timestamp": timestamp,
        },
        "datasets": {},
    }
    save_results(out_path, results)

    print(f"PARAMNET BATTERY | budget={args.budget} | seeds={seeds}")
    print(f"datasets={datasets}")
    print(f"checkpoints={checkpoints}")
    print(f"save={out_path}")

    for ds in datasets:
        if ds not in _PARAMNET_STEPS_MAP:
            print(f"Skipping unsupported dataset: {ds}")
            continue

        bench = _PARAMNET_STEPS_MAP[ds]()
        cs = bench.get_configuration_space()
        param_space, dim = _param_space_from_cs(cs)

        results["datasets"].setdefault(ds, {"v1": {}, "experimental": {}, "errors": {}})
        save_results(out_path, results)

        for seed in seeds:
            t0 = time.time()
            try:
                # ---------------- V1 (param_space) ----------------
                opt_v1 = ALBA_V1(
                    param_space=param_space,
                    seed=seed,
                    maximize=False,
                    total_budget=args.budget,
                    split_depth_max=8,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                )

                best = float("inf")
                cp: Dict[str, float] = {}
                for it in range(args.budget):
                    cfg_dict = opt_v1.ask()  # dict
                    cfg = CS.Configuration(cs, values=cfg_dict)
                    res = bench.objective_function(configuration=cfg, fidelity={"step": int(args.step)})
                    y = float(res["function_value"])
                    opt_v1.tell(cfg_dict, y)
                    best = min(best, y)
                    if (it + 1) in checkpoints:
                        cp[str(it + 1)] = float(best)
                results["datasets"][ds]["v1"][str(seed)] = cp
                save_results(out_path, results)

                # -------------- Experimental (array) --------------
                opt_exp = ALBA_EXP(
                    bounds=[(0.0, 1.0)] * dim,
                    seed=seed,
                    maximize=False,
                    total_budget=args.budget,
                    split_depth_max=8,
                    global_random_prob=0.05,
                    stagnation_threshold=50,
                )

                best = float("inf")
                cp = {}
                for it in range(args.budget):
                    x = opt_exp.ask()
                    cfg = _x01_to_cs_config(x, cs)
                    res = bench.objective_function(configuration=cfg, fidelity={"step": int(args.step)})
                    y = float(res["function_value"])
                    opt_exp.tell(x, y)
                    best = min(best, y)
                    if (it + 1) in checkpoints:
                        cp[str(it + 1)] = float(best)
                results["datasets"][ds]["experimental"][str(seed)] = cp
                save_results(out_path, results)

                elapsed = time.time() - t0
                v1_final = results["datasets"][ds]["v1"][str(seed)].get(str(args.budget))
                exp_final = results["datasets"][ds]["experimental"][str(seed)].get(str(args.budget))
                winner = "V1" if (v1_final is not None and exp_final is not None and v1_final < exp_final) else "EXP"
                print(f"{ds} seed {seed}: V1={v1_final:.6f} EXP={exp_final:.6f} -> {winner} ({elapsed:.1f}s)")

            except Exception as e:
                results["datasets"][ds]["errors"][str(seed)] = str(e)
                save_results(out_path, results)
                print(f"{ds} seed {seed}: ERROR: {e}")

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
