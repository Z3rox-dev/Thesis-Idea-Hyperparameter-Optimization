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

from hpobench.benchmarks.ml import TabularBenchmark  # type: ignore[import]
from hpo_main import HPOptimizer  # Current HPO algorithm

try:
    import optuna

    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False


def build_xgb_tabular_adapter(
    task_id: int,
) -> Tuple[
    TabularBenchmark,
    CS.ConfigurationSpace,
    List[CS.Hyperparameter],
    List[Tuple[float, float]],
    List[str],
    Dict[str, Any],
]:
    """Costruisce TabularBenchmark XGBoost + mapping da [0,1]^d a ConfigSpace."""
    bench = TabularBenchmark(model="xgb", task_id=task_id)
    cs = bench.get_configuration_space()
    hps = cs.get_hyperparameters()

    bounds: List[Tuple[float, float]] = []
    types: List[str] = []

    for hp in hps:
        # Tabular configspace è discreto (sequence) – mappiamo indice via [0,1] -> posizione
        if hasattr(hp, "sequence") and hp.sequence:
            seq = list(hp.sequence)
            bounds.append((0.0, float(len(seq) - 1)))
            types.append("index")
        elif isinstance(hp, CS.UniformFloatHyperparameter):
            bounds.append((float(hp.lower), float(hp.upper)))
            types.append("float")
        elif isinstance(hp, CS.UniformIntegerHyperparameter):
            bounds.append((float(hp.lower), float(hp.upper)))
            types.append("int")
        else:
            raise ValueError(f"Unsupported hyperparameter type: {hp} ({type(hp)})")

    # Usiamo la massima fidelity disponibile (massimo budget).
    max_fid = bench.get_max_fidelity()

    return bench, cs, hps, bounds, types, max_fid


def xnorm_to_config(
    x_norm: np.ndarray,
    cs: CS.ConfigurationSpace,
    hps: List[CS.Hyperparameter],
    bounds: List[Tuple[float, float]],
    types: List[str],
) -> CS.Configuration:
    """Mappa x_norm in [0,1]^d a una Configuration della TabularBenchmark."""
    x_norm = np.asarray(x_norm, dtype=float)
    values: Dict[str, Any] = {}

    for val, hp, (lo, hi), t in zip(x_norm, hps, bounds, types):
        if t == "index":
            # Map [0,1] a indice discreto della sequence
            seq = list(hp.sequence)  # type: ignore[attr-defined]
            idx = int(np.clip(np.floor(val * len(seq)), 0, len(seq) - 1))
            values[hp.name] = seq[idx]
        else:
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
) -> float:
    """Optuna TPE su [0,1]^dim per minimizzare l'obiettivo."""
    if not OPTUNA_AVAILABLE:
        return float("nan")

    def _objective(trial: "optuna.trial.Trial") -> float:  # type: ignore[name-defined]
        x_norm = np.array(
            [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)],
            dtype=float,
        )
        return float(objective(x_norm))

    sampler = optuna.samplers.TPESampler(seed=seed)  # type: ignore[attr-defined]
    study = optuna.create_study(direction="minimize", sampler=sampler)  # type: ignore[attr-defined]
    study.optimize(_objective, n_trials=int(budget), show_progress_bar=False)  # type: ignore[attr-defined]
    return float(study.best_value)  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark QuadHPO vs Random/Optuna su HPOBench Tabular XGBoost (surrogate)."
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Numero di trial per metodo per seed.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Lista di seed separati da virgola, es: '0,1,2'.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=31,
        help="OpenML task_id usato dal TabularBenchmark (es. 31).",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("=" * 78)
    print("Benchmark QuadHPO vs Random/Optuna su HPOBench Tabular XGBoost (surrogate)")
    print("=" * 78)
    print(f"Seeds: {seeds}")
    print(f"Budget per seed: {args.budget}")
    print(f"Task ID: {args.task_id}")
    print("=" * 78)

    # Log file nella cartella thesis/tests
    tests_dir = Path(__file__).resolve().parent / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = tests_dir / f"benchmark_xgb_tabular_results_{timestamp}.txt"
    log_file = log_path.open("w", encoding="utf-8")
    header_bar = "=" * 78
    log_file.write(header_bar + "\n")
    log_file.write("XGBOOST TABULAR BENCHMARK: QuadHPO vs Random vs Optuna\n")
    log_file.write(header_bar + "\n")
    log_file.write(f"Timestamp       : {timestamp}\n")
    log_file.write(f"Seeds           : {args.seeds}\n")
    log_file.write(f"Budget per seed : {args.budget}\n")
    log_file.write(f"Task ID         : {args.task_id}\n")
    log_file.write(header_bar + "\n\n")

    # Obiettivo della TabularBenchmark: loss (1 - acc), da minimizzare.
    maximize = False

    try:
        bench, cs, hps, bounds, types, max_fid = build_xgb_tabular_adapter(args.task_id)
    except Exception as exc:
        print(f"Impossibile inizializzare TabularBenchmark XGBoost (task_id={args.task_id}): {exc}")
        log_file.write(f"ERRORE: {exc}\n")
        log_file.close()
        print(f"\nLog salvato in: {log_path}")
        return

    def objective(x_norm: np.ndarray, epochs: int = 1) -> float:  # noqa: ARG001
        """Wrapper che valuta Tabular XGBoost dato x_norm in [0,1]^d."""
        cfg = xnorm_to_config(x_norm, cs, hps, bounds, types)
        res = bench.objective_function(cfg, fidelity=max_fid, metric="acc")
        return float(res["function_value"])

    dim = len(bounds)

    best_quad: List[float] = []
    best_rand: List[float] = []
    best_opt: List[float] = []
    rows: List[Tuple[str, int, float]] = []

    for seed in seeds:
        # Run QuadHPO (HPOptimizer from hpo_main)
        hpo = HPOptimizer(
            bounds=[(0.0, 1.0)] * dim,
            maximize=maximize,
            seed=seed,
        )
        _, quad_best = hpo.optimize(objective, budget=args.budget)
        best_quad.append(quad_best)

        rand_best = random_search(objective, dim=dim, budget=args.budget, seed=seed)
        best_rand.append(rand_best)

        opt_best = optuna_search(objective, dim=dim, budget=args.budget, seed=seed)
        best_opt.append(opt_best)

        print(
            f"[Seed {seed:3d}] "
            f"QuadHPO: {quad_best:.6f} | "
            f"Rand: {rand_best:.6f} | "
            f"Optuna: {opt_best:.6f}"
        )

        rows.append(("QuadHPO", seed, quad_best))
        rows.append(("Random", seed, rand_best))
        rows.append(("Optuna", seed, opt_best))

    # Stampa statistiche su stdout
    if best_quad:
        arr_quad = np.array(best_quad, dtype=float)
        print(f"QuadHPO mean best loss (xgb/tabular): {arr_quad.mean():.6f} ± {arr_quad.std():.6f}")
    if best_rand:
        arr_r = np.array(best_rand, dtype=float)
        print(f"Random  mean best loss (xgb/tabular): {arr_r.mean():.6f} ± {arr_r.std():.6f}")
    if OPTUNA_AVAILABLE and any(np.isfinite(best_opt)):
        arr_opt = np.array(best_opt, dtype=float)
        print(f"Optuna  mean best loss (xgb/tabular): {arr_opt.mean():.6f} ± {arr_opt.std():.6f}")

    # Determina vincitore
    means = {}
    if best_quad:
        means['QuadHPO'] = arr_quad.mean()
    if best_rand:
        means['Random'] = arr_r.mean()
    if OPTUNA_AVAILABLE and any(np.isfinite(best_opt)):
        means['Optuna'] = arr_opt.mean()
    
    if means:
        winner = min(means, key=means.get)
        print(f"\n>>> WINNER: {winner} with mean loss = {means[winner]:.6f}")

    # Scrivi tabella ASCII nel log
    sep = "-" * 78
    log_file.write(sep + "\n")
    log_file.write(f"Model: XGBoost Tabular | Task ID: {args.task_id} | Budget: {args.budget} | Seeds: {args.seeds}\n")
    log_file.write(sep + "\n")
    log_file.write("Method   | Seed | Best loss\n")
    log_file.write("---------+------+-----------\n")
    for method, seed, loss in rows:
        log_file.write(f"{method:<8} | {seed:4d} | {loss:11.8f}\n")

    log_file.write(sep + "\n")
    if best_quad:
        log_file.write(f"QuadHPO mean best loss : {arr_quad.mean():.8f} ± {arr_quad.std():.8f}\n")
    if best_rand:
        log_file.write(f"Random  mean best loss : {arr_r.mean():.8f} ± {arr_r.std():.8f}\n")
    if OPTUNA_AVAILABLE and any(np.isfinite(best_opt)):
        log_file.write(f"Optuna  mean best loss : {arr_opt.mean():.8f} ± {arr_opt.std():.8f}\n")
    if means:
        log_file.write(f"\nWINNER: {winner} with mean loss = {means[winner]:.8f}\n")
    log_file.write("\n")

    log_file.close()
    print(f"\nLog salvato in: {log_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()

