#!/usr/bin/env python3
"""
Benchmark QuadHPO (hpo_main V26) vs HPO Minimal vs Optuna on NN TabularBenchmark
"""

import sys
import os
sys.path.insert(0, '/mnt/workspace/HPOBench')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import types

# Import HPOBench Tabular benchmark
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark

# Import optimizers  
# from hpo_main import HPOptimizer as HPOptimizerMain
from hpo_v5s_more_novelty_standalone import HPOptimizerV5s as HPOptimizerMain
# from hpo_minimal import HPOptimizer as HPOptimizerMinimal
from thesis.hpo_lgs_v3 import HPOptimizer as HPOptimizerV3
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _enable_fast_hpobench_tabular_lookup(benchmark: TabularBenchmark) -> None:
    """Monkey-patch TabularBenchmark._search_dataframe to a MultiIndex-based lookup.

    This preserves exact results (same row selected) but avoids the per-call O(N) mask.
    """
    import pandas as pd  # local import to keep module import light

    df = benchmark.table
    if not hasattr(df, "columns") or "result" not in df.columns:
        return

    idx_cols = [c for c in df.columns if c != "result"]
    # Keep columns to allow robust access to 'result' after loc.
    indexed = df.set_index(idx_cols, drop=False)
    try:
        indexed = indexed.sort_index()
    except Exception:
        pass

    # Small LRU-ish cache (dict) keyed by tuple of values in idx_cols.
    cache: dict[tuple, object] = {}
    max_cache = 200_000

    def _search_dataframe_fast(self, row_dict, _df_unused):
        key = tuple(row_dict[c] for c in idx_cols)
        hit = cache.get(key)
        if hit is not None:
            return hit

        row = indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            if len(row) != 1:
                raise AssertionError(
                    f"The query has resulted into multiple matches. Query={row_dict} matches={len(row)}"
                )
            row = row.iloc[0]
        res = row["result"]

        if len(cache) < max_cache:
            cache[key] = res
        return res

    benchmark._search_dataframe = types.MethodType(_search_dataframe_fast, benchmark)


def _available_hpobench_tabular_task_ids(model: str = 'nn', data_dir: Optional[Path] = None) -> list[int]:
    """Return available task_ids for HPOBench TabularBenchmark datasets already present on disk."""
    try:
        import hpobench
    except Exception:
        return []

    base_dir = Path(data_dir) if data_dir is not None else (hpobench.config_file.data_dir / 'TabularData')
    model_dir = base_dir / model
    if not model_dir.exists():
        return []

    task_ids: list[int] = []
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue

        tid = int(child.name)
        parquet = child / f'{model}_{tid}_data.parquet.gzip'
        meta = child / f'{model}_{tid}_metadata.json'
        if parquet.exists() and meta.exists():
            task_ids.append(tid)

    task_ids.sort()
    return task_ids

def _openml_task_name(task_id: int, timeout_s: float = 15.0) -> Optional[str]:
    """Fetch a human-readable OpenML task name for a given task_id.

    Returns None if OpenML is unreachable or the response format is unexpected.
    """
    try:
        import requests
    except Exception:
        return None

    try:
        url = f'https://www.openml.org/api/v1/json/task/{int(task_id)}'
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        payload = r.json().get('task')
        if isinstance(payload, dict):
            return payload.get('task_name') or payload.get('name')
        return None
    except Exception:
        return None


def run_hpo_main(benchmark, cs, budget, seed, verbose=False):
    """Run QuadHPO V26 (hpo_main)"""
    np.random.seed(seed)
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
    dim = len(hp_names)
    
    optimizer = HPOptimizerMain(
        bounds=[(0, 1)] * dim,
        seed=seed,
        total_budget=budget,
        maximize=False
    )
    
    max_fidelity = benchmark.get_max_fidelity()
    
    trial_count = [0]
    best_so_far = [float('inf')]
    
    def objective(x):
        config = {}
        for i, name in enumerate(hp_names):
            seq = hp_seqs[name]
            idx = int(np.round(x[i] * (len(seq) - 1)))
            idx = max(0, min(len(seq) - 1, idx))
            config[name] = seq[idx]
        
        result = benchmark.objective_function(configuration=config, fidelity=max_fidelity)
        y = result['function_value']
        
        trial_count[0] += 1
        if y < best_so_far[0]:
            best_so_far[0] = y
        
        if verbose:
            print(f"    HPO_Main trial {trial_count[0]:3d}: loss={y:.6f} (best={best_so_far[0]:.6f})")
        
        return y
    
    best_x, best_y = optimizer.optimize(objective, budget=budget)
    return best_y


def run_random_search(benchmark, cs, budget, seed, verbose=False):
    """Run Random Search"""
    np.random.seed(seed)
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
    dim = len(hp_names)
    
    max_fidelity = benchmark.get_max_fidelity()
    
    trial_count = [0]
    best_so_far = [float('inf')]
    
    def objective(x):
        config = {}
        for i, name in enumerate(hp_names):
            seq = hp_seqs[name]
            idx = int(np.round(x[i] * (len(seq) - 1)))
            idx = max(0, min(len(seq) - 1, idx))
            config[name] = seq[idx]
        
        result = benchmark.objective_function(configuration=config, fidelity=max_fidelity)
        y = result['function_value']
        
        trial_count[0] += 1
        if y < best_so_far[0]:
            best_so_far[0] = y
        
        if verbose:
            print(f"    Random   trial {trial_count[0]:3d}: loss={y:.6f} (best={best_so_far[0]:.6f})")
        
        return y
    
    # Random Search Loop
    best_y = float('inf')
    for _ in range(budget):
        x = np.random.uniform(0, 1, dim)
        y = objective(x)
        if y < best_y:
            best_y = y
            
    return best_y


def run_hpo_v3(benchmark, cs, budget, seed, verbose=False):
    """Run HPO LGS v3"""
    np.random.seed(seed)
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
    dim = len(hp_names)
    
    optimizer = HPOptimizerV3(
        bounds=[(0, 1)] * dim,
        maximize=False,
        seed=seed
    )
    
    max_fidelity = benchmark.get_max_fidelity()
    
    trial_count = [0]
    best_so_far = [float('inf')]
    
    def objective(x):
        config = {}
        for i, name in enumerate(hp_names):
            seq = hp_seqs[name]
            idx = int(np.round(x[i] * (len(seq) - 1)))
            idx = max(0, min(len(seq) - 1, idx))
            config[name] = seq[idx]
        
        result = benchmark.objective_function(configuration=config, fidelity=max_fidelity)
        y = result['function_value']
        
        trial_count[0] += 1
        if y < best_so_far[0]:
            best_so_far[0] = y
        
        if verbose:
            print(f"    HPO_v3   trial {trial_count[0]:3d}: loss={y:.6f} (best={best_so_far[0]:.6f})")
        
        return y
    
    best_x, best_y = optimizer.optimize(objective, budget=budget)
    return best_y


def run_optuna(benchmark, cs, budget, seed, verbose=False):
    """Run Optuna TPE (multivariate)"""
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
    max_fidelity = benchmark.get_max_fidelity()
    
    trial_count = [0]
    best_so_far = [float('inf')]
    
    def objective(trial):
        config = {}
        for name in hp_names:
            seq = hp_seqs[name]
            config[name] = trial.suggest_categorical(name, seq)
        
        result = benchmark.objective_function(configuration=config, fidelity=max_fidelity)
        y = result['function_value']
        
        trial_count[0] += 1
        if y < best_so_far[0]:
            best_so_far[0] = y
        
        if verbose:
            print(f"    Optuna  trial {trial_count[0]:3d}: loss={y:.6f} (best={best_so_far[0]:.6f})")
        
        return y
    
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    
    return study.best_value


def run_turbo(benchmark, cs, budget, seed, verbose=False, n_trust_regions: int = 5):
    """Run TuRBO (TurboM) on the continuous [0,1]^d proxy space.

    Notes:
    - TuRBO minimizes f(x) by default, which matches our loss.
    - Mapping from continuous x to discrete hyperparameters is identical to other optimizers.
    """
    np.random.seed(seed)

    try:
        from turbo import TurboM, Turbo1
    except Exception as e:
        raise RuntimeError(
            "TuRBO is not installed. Install with: pip install gpytorch botorch && "
            "pip install git+https://github.com/uber-research/TuRBO.git"
        ) from e

    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    hp_seqs = {hp.name: list(hp.sequence) for hp in hps}
    dim = len(hp_names)

    max_fidelity = benchmark.get_max_fidelity()

    trial_count = [0]
    best_so_far = [float('inf')]

    def f(x):
        # x is in [lb, ub] as a 1D array
        config = {}
        for i, name in enumerate(hp_names):
            seq = hp_seqs[name]
            idx = int(np.round(float(x[i]) * (len(seq) - 1)))
            idx = max(0, min(len(seq) - 1, idx))
            config[name] = seq[idx]

        result = benchmark.objective_function(configuration=config, fidelity=max_fidelity)
        y = float(result['function_value'])

        trial_count[0] += 1
        if y < best_so_far[0]:
            best_so_far[0] = y

        if verbose:
            print(f"    TuRBO   trial {trial_count[0]:3d}: loss={y:.6f} (best={best_so_far[0]:.6f})")

        return y

    lb = np.zeros(dim)
    ub = np.ones(dim)
    if budget <= 2:
        raise ValueError(f"TuRBO requires budget > 2, got budget={budget}")

    n_init = max(10, 2 * dim)
    # TuRBO asserts max_evals > n_init
    n_init = min(n_init, budget - 1)

    n_tr = max(1, int(n_trust_regions))
    # TurboM asserts max_evals > n_trust_regions * n_init
    max_tr = max(1, (budget - 1) // max(1, n_init))
    n_tr = min(n_tr, max_tr)

    if n_tr >= 2:
        turbo = TurboM(
            f=f,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=budget,
            n_trust_regions=n_tr,
            batch_size=1,
            verbose=bool(verbose),
            use_ard=True,
            device='cpu',
            dtype='float64',
        )
    else:
        turbo = Turbo1(
            f=f,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=budget,
            batch_size=1,
            verbose=bool(verbose),
            use_ard=True,
            device='cpu',
            dtype='float64',
        )
    turbo.optimize()

    # TuRBO stores evaluations in turbo.fX as shape (n, 1)
    return float(np.min(turbo.fX))


def main():
    parser = argparse.ArgumentParser(description='Triple Benchmark: HPO_Main vs Random vs Optuna')
    parser.add_argument('--budget', type=int, default=150, help='Number of evaluations per seed')
    parser.add_argument('--seeds', type=str, default="20,21,22,23,24,25,26,27,28,29", help='Comma-separated seeds')
    parser.add_argument('--task_id', type=int, default=31, help='HPOBench TabularBenchmark task ID (must exist for the selected model)')
    parser.add_argument('--list_tasks', action='store_true', help='List locally available task IDs for model=nn and exit')
    parser.add_argument('--strict_task_id', action='store_true', help='Fail fast if the requested task_id is not available locally')
    parser.add_argument('--no_fast_lookup', action='store_true', help='Disable MultiIndex-based fast lookup (debug)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show per-trial output')
    parser.add_argument('--list_tasks_with_names', action='store_true', help='List locally available task IDs for model=nn with OpenML names and exit')
    parser.add_argument('--with_turbo', action='store_true', help='Also benchmark TuRBO (TurboM)')
    parser.add_argument('--turbo_trust_regions', type=int, default=5, help='TuRBO: number of trust regions for TurboM')
    parser.add_argument('--no_v3', action='store_true', help='Skip HPO_v3 (LGS v3)')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    budget = args.budget
    task_id = args.task_id

    available_task_ids = _available_hpobench_tabular_task_ids(model='nn')
    if args.list_tasks:
        print('Available HPOBench TabularBenchmark task IDs for model=nn:')
        if available_task_ids:
            print(', '.join(map(str, available_task_ids)))
        else:
            print('(none found locally)')
        return

    if args.list_tasks_with_names:
        print('Available HPOBench TabularBenchmark task IDs for model=nn (with OpenML names):')
        if not available_task_ids:
            print('(none found locally)')
            return

        for tid in available_task_ids:
            name = _openml_task_name(tid)
            if name is None:
                print(f'{tid}: (name unavailable / offline)')
            else:
                print(f'{tid}: {name}')
        return

    if available_task_ids and task_id not in available_task_ids:
        if args.strict_task_id:
            print(f"ERROR: task_id={task_id} is not available locally for model=nn.")
            print(f"Available: {', '.join(map(str, available_task_ids))}")
            print('Tip: run with --list_tasks to see options, or omit --strict_task_id to auto-fallback.')
            raise SystemExit(2)

        fallback = 31 if 31 in available_task_ids else available_task_ids[0]
        print(f"WARNING: task_id={task_id} is not available locally for model=nn.")
        print(f"Falling back to task_id={fallback}. Use --list_tasks or --strict_task_id to control this.")
        task_id = fallback
    
    print("=" * 80)
    print("Triple Benchmark: HPO_Main (V26) vs Random vs Optuna (multivariate)")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Budget per seed: {budget}")
    print(f"Task ID: {task_id}")
    
    # Create benchmark
    benchmark = TabularBenchmark(model='nn', task_id=task_id)
    if not args.no_fast_lookup:
        _enable_fast_hpobench_tabular_lookup(benchmark)
    cs = benchmark.get_configuration_space()
    
    hps = list(cs.values())
    hp_names = [hp.name for hp in hps]
    print(f"Dimensions: {len(hp_names)}")
    print(f"Hyperparameters: {hp_names}")
    print()
    
    results_main = []
    results_random = []
    results_v3 = []
    results_optuna = []
    results_turbo = []
    
    for seed in seeds:
        if args.verbose:
            print(f"\n--- Seed {seed} ---")
        
        # Run HPO Main (V26)
        if args.verbose:
            print("  HPO_Main (V26):")
        best_main = run_hpo_main(benchmark, cs, budget, seed, verbose=args.verbose)
        results_main.append(best_main)
        
        # Run Random Search
        if args.verbose:
            print("  Random Search:")
        best_random = run_random_search(benchmark, cs, budget, seed, verbose=args.verbose)
        results_random.append(best_random)

        best_v3 = None
        if not args.no_v3:
            # Run HPO v3
            if args.verbose:
                print("  HPO_v3:")
            best_v3 = run_hpo_v3(benchmark, cs, budget, seed, verbose=args.verbose)
            results_v3.append(best_v3)
        
        # Run Optuna
        if args.verbose:
            print("  Optuna:")
        best_optuna = run_optuna(benchmark, cs, budget, seed, verbose=args.verbose)
        results_optuna.append(best_optuna)

        if args.with_turbo:
            if args.verbose:
                print("  TuRBO (TurboM):")
            best_turbo = run_turbo(
                benchmark,
                cs,
                budget,
                seed,
                verbose=args.verbose,
                n_trust_regions=args.turbo_trust_regions,
            )
            results_turbo.append(best_turbo)
        
        line = f"[Seed {seed:3d}] HPO_v5s: {best_main:.6f} | Random: {best_random:.6f}"
        if not args.no_v3:
            line += f" | HPO_v3: {best_v3:.6f}"
        line += f" | Optuna: {best_optuna:.6f}"
        if args.with_turbo:
            line += f" | TuRBO: {best_turbo:.6f}"
        print(line)
    
    # Summary statistics
    mean_main = np.mean(results_main)
    std_main = np.std(results_main)
    mean_random = np.mean(results_random)
    std_random = np.std(results_random)
    if not args.no_v3:
        mean_v3 = np.mean(results_v3)
        std_v3 = np.std(results_v3)
    mean_optuna = np.mean(results_optuna)
    std_optuna = np.std(results_optuna)

    if args.with_turbo:
        mean_turbo = np.mean(results_turbo)
        std_turbo = np.std(results_turbo)
    
    print()
    print(f"HPO_v5s (standalone) mean best loss: {mean_main:.6f} ± {std_main:.6f}")
    print(f"Random Search    mean best loss: {mean_random:.6f} ± {std_random:.6f}")
    if not args.no_v3:
        print(f"HPO_v3           mean best loss: {mean_v3:.6f} ± {std_v3:.6f}")
    print(f"Optuna (multi)   mean best loss: {mean_optuna:.6f} ± {std_optuna:.6f}")
    if args.with_turbo:
        print(f"TuRBO (TurboM)   mean best loss: {mean_turbo:.6f} ± {std_turbo:.6f}")
    print()
    
    # Determine winner
    results = [
        ("HPO_v5s", mean_main),
        ("Random", mean_random),
        ("Optuna", mean_optuna)
    ]
    if not args.no_v3:
        results.insert(2, ("HPO_v3", mean_v3))
    if args.with_turbo:
        results.append(("TuRBO", mean_turbo))
    winner = min(results, key=lambda x: x[1])
    print(f">>> WINNER: {winner[0]} with mean loss = {winner[1]:.6f}")
    
    # Win counts
    if args.no_v3:
        if args.with_turbo:
            wins_main = sum(1 for i in range(len(seeds)) if results_main[i] <= min(results_random[i], results_optuna[i], results_turbo[i]))
            wins_random = sum(1 for i in range(len(seeds)) if results_random[i] <= min(results_main[i], results_optuna[i], results_turbo[i]))
            wins_optuna = sum(1 for i in range(len(seeds)) if results_optuna[i] <= min(results_main[i], results_random[i], results_turbo[i]))
            wins_turbo = sum(1 for i in range(len(seeds)) if results_turbo[i] <= min(results_main[i], results_random[i], results_optuna[i]))
            print(f"Win counts: HPO_v5s={wins_main}, Random={wins_random}, Optuna={wins_optuna}, TuRBO={wins_turbo}")
        else:
            wins_main = sum(1 for i in range(len(seeds)) if results_main[i] <= min(results_random[i], results_optuna[i]))
            wins_random = sum(1 for i in range(len(seeds)) if results_random[i] <= min(results_main[i], results_optuna[i]))
            wins_optuna = sum(1 for i in range(len(seeds)) if results_optuna[i] <= min(results_main[i], results_random[i]))
            print(f"Win counts: HPO_v5s={wins_main}, Random={wins_random}, Optuna={wins_optuna}")
    else:
        if args.with_turbo:
            wins_main = sum(1 for i in range(len(seeds)) if results_main[i] <= min(results_random[i], results_v3[i], results_optuna[i], results_turbo[i]))
            wins_random = sum(1 for i in range(len(seeds)) if results_random[i] <= min(results_main[i], results_v3[i], results_optuna[i], results_turbo[i]))
            wins_v3 = sum(1 for i in range(len(seeds)) if results_v3[i] <= min(results_main[i], results_random[i], results_optuna[i], results_turbo[i]))
            wins_optuna = sum(1 for i in range(len(seeds)) if results_optuna[i] <= min(results_main[i], results_random[i], results_v3[i], results_turbo[i]))
            wins_turbo = sum(1 for i in range(len(seeds)) if results_turbo[i] <= min(results_main[i], results_random[i], results_v3[i], results_optuna[i]))
            print(f"Win counts: HPO_v5s={wins_main}, Random={wins_random}, HPO_v3={wins_v3}, Optuna={wins_optuna}, TuRBO={wins_turbo}")
        else:
            wins_main = sum(1 for i in range(len(seeds)) if results_main[i] <= min(results_random[i], results_v3[i], results_optuna[i]))
            wins_random = sum(1 for i in range(len(seeds)) if results_random[i] <= min(results_main[i], results_v3[i], results_optuna[i]))
            wins_v3 = sum(1 for i in range(len(seeds)) if results_v3[i] <= min(results_main[i], results_random[i], results_optuna[i]))
            wins_optuna = sum(1 for i in range(len(seeds)) if results_optuna[i] <= min(results_main[i], results_random[i], results_v3[i]))
            print(f"Win counts: HPO_v5s={wins_main}, Random={wins_random}, HPO_v3={wins_v3}, Optuna={wins_optuna}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"/mnt/workspace/thesis/tests/benchmark_triple_nn_{timestamp}.txt"
    
    with open(log_path, 'w') as f:
        title = "Benchmark: HPO_v5s (standalone) vs Random vs Optuna"
        if not args.no_v3:
            title = "Benchmark: HPO_v5s (standalone) vs Random vs HPO_v3 vs Optuna"
        if args.with_turbo:
            title += " vs TuRBO"
        f.write(title + "\n")
        f.write("=" * 60 + "\n")
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Hyperparameters: {hp_names}\n\n")
        
        for i, seed in enumerate(seeds):
            f.write(f"[Seed {seed:3d}] HPO_v5s: {results_main[i]:.6f} | ")
            f.write(f"Random: {results_random[i]:.6f} | ")
            if not args.no_v3:
                f.write(f"HPO_v3: {results_v3[i]:.6f} | ")
            f.write(f"Optuna: {results_optuna[i]:.6f}")
            if args.with_turbo:
                f.write(f" | TuRBO: {results_turbo[i]:.6f}")
            f.write("\n")
        
        f.write(f"\nHPO_v5s (standalone): {mean_main:.6f} ± {std_main:.6f}\n")
        f.write(f"Random Search:      {mean_random:.6f} ± {std_random:.6f}\n")
        if not args.no_v3:
            f.write(f"HPO_v3:             {mean_v3:.6f} ± {std_v3:.6f}\n")
        f.write(f"Optuna (multi):     {mean_optuna:.6f} ± {std_optuna:.6f}\n")
        if args.with_turbo:
            f.write(f"TuRBO (TurboM):     {mean_turbo:.6f} ± {std_turbo:.6f}\n")
        f.write(f"\nWinner: {winner[0]}\n")
        if args.with_turbo:
            if args.no_v3:
                f.write(f"Win counts: HPO_v5s={wins_main}, Random={wins_random}, Optuna={wins_optuna}, TuRBO={wins_turbo}\n")
            else:
                f.write(f"Win counts: HPO_v5s={wins_main}, Random={wins_random}, HPO_v3={wins_v3}, Optuna={wins_optuna}, TuRBO={wins_turbo}\n")
        else:
            if args.no_v3:
                f.write(f"Win counts: HPO_v5s={wins_main}, Random={wins_random}, Optuna={wins_optuna}\n")
            else:
                f.write(f"Win counts: HPO_v5s={wins_main}, Random={wins_random}, HPO_v3={wins_v3}, Optuna={wins_optuna}\n")
    
    print(f"\nLog salvato in: {log_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
