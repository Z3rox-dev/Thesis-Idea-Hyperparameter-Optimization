from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Compatibilità con vecchie dipendenze che usano np.float / np.int / np.bool
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
import ConfigSpace as CS

from hpobench.benchmarks.surrogates.paramnet_benchmark import (  # type: ignore[import]
    ParamNetAdultOnStepsBenchmark,
    ParamNetHiggsOnStepsBenchmark,
    ParamNetLetterOnStepsBenchmark,
    ParamNetMnistOnStepsBenchmark,
    ParamNetOptdigitsOnStepsBenchmark,
    ParamNetPokerOnStepsBenchmark,
)
from hpo_lgs_v3 import HPOptimizer as LGSv3HPO
from hpo_lgs_v4 import HPOptimizer as LGSv4HPO
from hpo_minimal import HPOptimizer as MinimalHPO

try:
    import optuna

    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False


_PARAMNET_STEPS_MAP = {
    "adult": ParamNetAdultOnStepsBenchmark,
    "higgs": ParamNetHiggsOnStepsBenchmark,
    "letter": ParamNetLetterOnStepsBenchmark,
    "mnist": ParamNetMnistOnStepsBenchmark,
    "optdigits": ParamNetOptdigitsOnStepsBenchmark,
    "poker": ParamNetPokerOnStepsBenchmark,
}


def build_paramnet_adapter(
    dataset: str,
) -> Tuple[
    Any,
    CS.ConfigurationSpace,
    List[CS.Hyperparameter],
    List[Tuple[float, float]],
    List[str],
]:
    """Costruisce benchmark ParamNet(dataset) + mapping da [0,1]^d a ConfigSpace."""
    key = dataset.lower()
    if key not in _PARAMNET_STEPS_MAP:
        raise ValueError(f"Dataset ParamNet non supportato: {dataset}")
    bench_cls = _PARAMNET_STEPS_MAP[key]
    bench = bench_cls()
    cs = bench.get_configuration_space()
    hps = cs.get_hyperparameters()

    bounds: List[Tuple[float, float]] = []
    types: List[str] = []

    for hp in hps:
        if isinstance(hp, CS.UniformFloatHyperparameter):
            bounds.append((float(hp.lower), float(hp.upper)))
            types.append("float")
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            bounds.append((float(hp.lower), float(hp.upper)))
            types.append("int")
        else:
            raise ValueError(f"Unsupported hyperparameter type: {hp} ({type(hp)})")

    return bench, cs, hps, bounds, types


def xnorm_to_config(
    x_norm: np.ndarray,
    cs: CS.ConfigurationSpace,
    hps: List[CS.Hyperparameter],
    bounds: List[Tuple[float, float]],
    types: List[str],
) -> CS.Configuration:
    """Mappa un vettore x_norm in [0,1]^d a una Configuration di ConfigSpace."""
    x_norm = np.asarray(x_norm, dtype=float)
    values: Dict[str, Any] = {}

    for val, hp, (lo, hi), t in zip(x_norm, hps, bounds, types):
        v = lo + float(val) * (hi - lo)
        if t == "int":
            v = int(round(v))
            v = max(int(hp.lower), min(int(hp.upper), int(v)))  # type: ignore[arg-type]
        values[hp.name] = v

    return CS.Configuration(cs, values=values)


def random_search(
    objective: Any,
    dim: int,
    budget: int,
    seed: int,
) -> float:
    """Random search su [0,1]^dim per minimizzare l'obiettivo."""
    rng = np.random.default_rng(seed)
    best = float("inf")
    for _ in range(int(budget)):
        x_norm = rng.random(dim)
        val = float(objective(x_norm))
        if val < best:
            best = val
    return best


def optuna_search(
    objective: Any,
    dim: int,
    budget: int,
    seed: int,
) -> Tuple[float, List[float]]:
    """Optuna TPE su [0,1]^dim per minimizzare l'obiettivo.

    Ritorna (best_loss, history_loss_per_trial).
    """
    if not OPTUNA_AVAILABLE:
        return float("nan"), []

    # Silenzia i log di Optuna (niente INFO/WARNING su stdout)
    import logging

    optuna.logging.set_verbosity(optuna.logging.ERROR)  # type: ignore[attr-defined]

    history: List[float] = []

    def _objective(trial: "optuna.trial.Trial") -> float:  # type: ignore[name-defined]
        x_norm = np.array(
            [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)],
            dtype=float,
        )
        val = float(objective(x_norm))
        history.append(val)
        return val

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
    )  # type: ignore[attr-defined]
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="quad_hpo_paramnet_benchmark",
    )  # type: ignore[attr-defined]
    study.optimize(_objective, n_trials=int(budget), show_progress_bar=False)  # type: ignore[attr-defined]
    return float(study.best_value), history  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark QuadHPO vs Random/Optuna su HPOBench ParamNet (surrogate, multi-dataset)."
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=200,
        help="Numero di trial per metodo per seed.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="151",
        help="Lista di seed separati da virgola, es: '0,1,2'.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="adult,higgs,letter,mnist,optdigits,poker",
        help="Lista di dataset ParamNet (adult,higgs,letter,mnist,optdigits,poker).",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    print("=" * 78)
    print("Benchmark QuadHPO vs Random/Optuna su HPOBench ParamNet (surrogate)")
    print("=" * 78)
    print(f"Seeds: {seeds}")
    print(f"Budget per seed: {args.budget}")
    print(f"Datasets: {', '.join(datasets)}")
    print("=" * 78)

    # Log file nella cartella thesis/tests
    tests_dir = Path(__file__).resolve().parent / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = tests_dir / f"benchmark_paramnet_results_{timestamp}.txt"
    log_file = log_path.open("w", encoding="utf-8")
    trace_path = tests_dir / f"benchmark_paramnet_traces_{timestamp}.txt"
    trace_file = trace_path.open("w", encoding="utf-8")
    header_bar = "=" * 78
    log_file.write(header_bar + "\n")
    log_file.write("PARAMNET BENCHMARK SUMMARY: QuadHPO vs Random vs Optuna\n")
    log_file.write(header_bar + "\n")
    log_file.write(f"Timestamp       : {timestamp}\n")
    log_file.write(f"Seeds           : {args.seeds}\n")
    log_file.write(f"Budget per seed : {args.budget}\n")
    log_file.write(f"Datasets        : {', '.join(datasets)}\n")
    log_file.write(header_bar + "\n\n")

    # Per ParamNet, l'obiettivo è sempre un loss da MINIMIZZARE.
    maximize = False

    for dataset in datasets:
        print("\n" + "-" * 78)
        print(f"Dataset: {dataset}")
        print("-" * 78)

        try:
            bench, cs, hps, bounds, types = build_paramnet_adapter(dataset)
        except Exception as exc:
            print(f"Impossibile inizializzare ParamNet per dataset '{dataset}': {exc}")
            continue

        def objective(x_norm: np.ndarray, epochs: int = 1) -> float:  # noqa: ARG001
            """Wrapper che valuta HPOBench dato x_norm in [0,1]^d."""
            cfg = xnorm_to_config(x_norm, cs, hps, bounds, types)
            # Usiamo sempre il massimo budget di steps per avere una valutazione "finale".
            res = bench.objective_function(cfg, fidelity={"step": 50})
            return float(res["function_value"])

        dim = len(bounds)

        best_minimal: List[float] = []
        best_v3: List[float] = []
        best_v4: List[float] = []
        best_rand: List[float] = []
        best_opt: List[float] = []
        rows: List[Tuple[str, int, float]] = []
        minimal_histories: Dict[int, List[float]] = {}
        v3_histories: Dict[int, List[float]] = {}
        v4_histories: Dict[int, List[float]] = {}
        opt_histories: Dict[int, List[float]] = {}

        for seed in seeds:
            # Traccia per-seed per Minimal
            minimal_hist: List[float] = []

            def objective_minimal(x_norm: np.ndarray) -> float:
                cfg = xnorm_to_config(x_norm, cs, hps, bounds, types)
                res = bench.objective_function(cfg, fidelity={"step": 50})
                loss = float(res["function_value"])
                minimal_hist.append(loss)
                return -loss  # Minimal maximizes, so negate

            # Debug log file per minimal
            debug_log_path = tests_dir / f"debug_minimal_{dataset}_{seed}_{timestamp}.txt"
            debug_log_file = debug_log_path.open("w", encoding="utf-8")
            
            def debug_logger(msg: str) -> None:
                debug_log_file.write(msg + "\n")
                debug_log_file.flush()

            hpo_m = MinimalHPO(
                bounds=[(0.0, 1.0)] * dim,
                maximize=True,
                seed=seed,
                debug_log=debug_logger,
            )
            best_x, best_neg = hpo_m.optimize(objective_minimal, budget=args.budget)
            debug_log_file.close()
            minimal_best = -best_neg  # Convert back to loss
            best_minimal.append(minimal_best)
            minimal_histories[seed] = minimal_hist

            # Traccia per-seed per LGS v4
            v4_hist: List[float] = []

            def objective_v4(x_norm: np.ndarray) -> float:
                cfg = xnorm_to_config(x_norm, cs, hps, bounds, types)
                res = bench.objective_function(cfg, fidelity={"step": 50})
                loss = float(res["function_value"])
                v4_hist.append(loss)
                return loss  # LGS v4 minimizza

            hpo_v4 = LGSv4HPO(
                bounds=[(0.0, 1.0)] * dim,
                maximize=False,
                seed=seed,
            )
            _, v4_best = hpo_v4.optimize(objective_v4, budget=args.budget)
            best_v4.append(v4_best)
            v4_histories[seed] = v4_hist

            # Traccia per-seed per LGS v3
            v3_hist: List[float] = []

            def objective_v3(x_norm: np.ndarray) -> float:
                cfg = xnorm_to_config(x_norm, cs, hps, bounds, types)
                res = bench.objective_function(cfg, fidelity={"step": 50})
                loss = float(res["function_value"])
                v3_hist.append(loss)
                return loss  # LGS v3 minimizza

            hpo_v3 = LGSv3HPO(
                bounds=[(0.0, 1.0)] * dim,
                maximize=False,
                seed=seed,
            )
            _, v3_best = hpo_v3.optimize(objective_v3, budget=args.budget)
            best_v3.append(v3_best)
            v3_histories[seed] = v3_hist

            rand_best = random_search(objective, dim=dim, budget=args.budget, seed=seed)
            best_rand.append(rand_best)

            opt_best, opt_hist = optuna_search(objective, dim=dim, budget=args.budget, seed=seed)
            best_opt.append(opt_best)
            opt_histories[seed] = opt_hist

            print(
                f"[Seed {seed:3d}] "
                f"Minimal: {minimal_best:.6f} | "
                f"v3: {v3_best:.6f} | "
                f"v4: {v4_best:.6f} | "
                f"Random: {rand_best:.6f} | "
                f"Optuna: {opt_best:.6f}"
            )

            # Accumula righe per tabella ASCII nel log
            rows.append(("Minimal", seed, minimal_best))
            rows.append(("LGS_v3", seed, v3_best))
            rows.append(("LGS_v4", seed, v4_best))
            rows.append(("Random", seed, rand_best))
            rows.append(("Optuna", seed, opt_best))

        # Stampa statistiche per dataset su stdout
        if best_minimal:
            arr_m = np.array(best_minimal, dtype=float)
            print(f"Minimal mean best loss ({dataset}): {arr_m.mean():.6f} ± {arr_m.std():.6f}")
        if best_v3:
            arr_v3 = np.array(best_v3, dtype=float)
            print(f"LGS_v3  mean best loss ({dataset}): {arr_v3.mean():.6f} ± {arr_v3.std():.6f}")
        if best_v4:
            arr_v4 = np.array(best_v4, dtype=float)
            print(f"LGS_v4  mean best loss ({dataset}): {arr_v4.mean():.6f} ± {arr_v4.std():.6f}")
        if best_rand:
            arr_r = np.array(best_rand, dtype=float)
            print(f"Random mean best loss ({dataset}): {arr_r.mean():.6f} ± {arr_r.std():.6f}")
        if OPTUNA_AVAILABLE and any(np.isfinite(best_opt)):
            arr_o = np.array(best_opt, dtype=float)
            print(f"Optuna mean best loss ({dataset}): {arr_o.mean():.6f} ± {arr_o.std():.6f}")

        # Scrivi blocco riassuntivo per dataset nel log
        sep = "-" * 78
        log_file.write(sep + "\n")
        log_file.write(f"Dataset: {dataset} | Budget: {args.budget} | Seeds: {args.seeds}\n")
        log_file.write(sep + "\n")
        log_file.write("Method   | Seed | Best loss\n")
        log_file.write("---------+------+-----------\n")
        for method, seed, loss in rows:
            log_file.write(f"{method:<8} | {seed:4d} | {loss:11.8f}\n")

        log_file.write(sep + "\n")
        if best_minimal:
            log_file.write(f"Minimal mean best loss: {arr_m.mean():.8f} ± {arr_m.std():.8f}\n")
        if best_v3:
            log_file.write(f"LGS_v3  mean best loss: {arr_v3.mean():.8f} ± {arr_v3.std():.8f}\n")
        if best_v4:
            log_file.write(f"LGS_v4  mean best loss: {arr_v4.mean():.8f} ± {arr_v4.std():.8f}\n")
        if best_rand:
            log_file.write(f"Random mean best loss : {arr_r.mean():.8f} ± {arr_r.std():.8f}\n")
        if OPTUNA_AVAILABLE and any(np.isfinite(best_opt)):
            log_file.write(f"Optuna mean best loss : {arr_o.mean():.8f} ± {arr_o.std():.8f}\n")
        log_file.write("\n")

        # Scrivi tracce per questo dataset nel file dedicato
        trace_sep = "-" * 78
        for seed in seeds:
            # Minimal
            mh = minimal_histories.get(seed, [])
            if mh:
                trace_file.write(trace_sep + "\n")
                trace_file.write(f"Dataset: {dataset} | Method: Minimal | Seed: {seed} | Budget: {args.budget}\n")
                trace_file.write(trace_sep + "\n")
                trace_file.write("Trial | Loss\n")
                trace_file.write("------+-----------\n")
                for t_idx, loss in enumerate(mh, start=1):
                    trace_file.write(f"{t_idx:5d} | {loss:11.8f}\n")
                trace_file.write("\n")
            # LGS v3
            v3h = v3_histories.get(seed, [])
            if v3h:
                trace_file.write(trace_sep + "\n")
                trace_file.write(f"Dataset: {dataset} | Method: LGS_v3 | Seed: {seed} | Budget: {args.budget}\n")
                trace_file.write(trace_sep + "\n")
                trace_file.write("Trial | Loss\n")
                trace_file.write("------+-----------\n")
                for t_idx, loss in enumerate(v3h, start=1):
                    trace_file.write(f"{t_idx:5d} | {loss:11.8f}\n")
                trace_file.write("\n")
            # LGS v4
            vh = v4_histories.get(seed, [])
            if vh:
                trace_file.write(trace_sep + "\n")
                trace_file.write(f"Dataset: {dataset} | Method: LGS_v4 | Seed: {seed} | Budget: {args.budget}\n")
                trace_file.write(trace_sep + "\n")
                trace_file.write("Trial | Loss\n")
                trace_file.write("------+-----------\n")
                for t_idx, loss in enumerate(vh, start=1):
                    trace_file.write(f"{t_idx:5d} | {loss:11.8f}\n")
                trace_file.write("\n")
            # Optuna
            oh = opt_histories.get(seed, [])
            if oh:
                trace_file.write(trace_sep + "\n")
                trace_file.write(f"Dataset: {dataset} | Method: Optuna | Seed: {seed} | Budget: {args.budget}\n")
                trace_file.write(trace_sep + "\n")
                trace_file.write("Trial | Loss\n")
                trace_file.write("------+-----------\n")
                for t_idx, loss in enumerate(oh, start=1):
                    trace_file.write(f"{t_idx:5d} | {loss:11.8f}\n")
                trace_file.write("\n")

    log_file.close()
    trace_file.close()
    print(f"\nLog summary salvato in: {log_path}")
    print(f"Log tracce salvato in  : {trace_path}")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    # Patch per scikit-learn < 0.24 (necessario per caricare vecchi pickle di ParamNet)
    import sys
    import types

    try:
        import sklearn.ensemble
        import sklearn.tree

        if "sklearn.ensemble.forest" not in sys.modules:
            # Crea un modulo dummy che punta a sklearn.ensemble
            forest = types.ModuleType("sklearn.ensemble.forest")
            for attr in dir(sklearn.ensemble):
                setattr(forest, attr, getattr(sklearn.ensemble, attr))
            sys.modules["sklearn.ensemble.forest"] = forest

        if "sklearn.tree.tree" not in sys.modules:
            tree_mod = types.ModuleType("sklearn.tree.tree")
            for attr in dir(sklearn.tree):
                setattr(tree_mod, attr, getattr(sklearn.tree, attr))
            sys.modules["sklearn.tree.tree"] = tree_mod
    except ImportError:
        pass

    main()
