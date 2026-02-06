#!/usr/bin/env python3
"""
Benchmark completo: ALBA vs Optuna vs TuRBO su tutti i benchmark nuovi.

1. HPOBench SVM (2D)
2. HPOBench LR (2D)  
3. HPOBench RF (Random Forest)
4. Nevergrad funzioni realistiche
"""

from __future__ import annotations
import sys
import warnings
import argparse
import numpy as np
from typing import Callable, Dict, Any

warnings.filterwarnings('ignore')


# ==== TuRBO utilities ====

def run_turbo_continuous(objective_fn, dim: int, budget: int, seed: int):
    """Run TuRBO-M on a continuous [0,1]^d objective."""
    try:
        from turbo import TurboM
        import torch
    except ImportError:
        return None
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    best_y = [float('inf')]
    
    class Objective:
        def __init__(self):
            self.dim = dim
            self.lb = np.zeros(dim)
            self.ub = np.ones(dim)
        
        def __call__(self, x):
            y = objective_fn(x)
            best_y[0] = min(best_y[0], y)
            return y
    
    f = Objective()
    
    n_trust_regions = min(5, max(2, budget // 50))
    n_init = min(2 * dim, budget // (n_trust_regions * 3))
    n_init = max(n_init, dim + 1)
    
    turbo_m = TurboM(
        f=f,
        lb=f.lb,
        ub=f.ub,
        n_init=n_init,
        max_evals=budget,
        n_trust_regions=n_trust_regions,
        batch_size=1,
        verbose=False,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        device="cpu",
        dtype="float64",
    )
    
    turbo_m.optimize()
    return best_y[0]


def run_optimizer_on_objective(
    objective: Callable,
    bounds: list,
    budget: int,
    seed: int,
    optimizer: str
) -> float:
    """Run a single optimizer on an objective."""
    
    dim = len(bounds)
    
    if optimizer == 'ALBA':
        sys.path.insert(0, '/mnt/workspace/thesis')
        from alba_framework_potential.optimizer import ALBA as ALBAPotentialOptimizer
        
        opt = ALBAPotentialOptimizer(bounds=bounds, seed=seed, maximize=False)
        for _ in range(budget):
            x = opt.ask()
            y = objective(x)
            opt.tell(x, y)
        return opt.best_y
    
    elif optimizer == 'Optuna':
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def optuna_obj(trial):
            x = np.array([trial.suggest_float(f"x{i}", lo, hi) for i, (lo, hi) in enumerate(bounds)])
            return objective(x)
        
        study = optuna.create_study(direction='minimize',
                                   sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(optuna_obj, n_trials=budget, show_progress_bar=False)
        return study.best_value
    
    elif optimizer == 'TuRBO':
        # Normalize to [0,1]^d
        def turbo_obj(x_01):
            x = np.array([lo + x_01[i] * (hi - lo) for i, (lo, hi) in enumerate(bounds)])
            return objective(x)
        
        result = run_turbo_continuous(turbo_obj, dim, budget, seed)
        return result if result is not None else float('inf')
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


# ==== HPOBench SVM Benchmark (SKIP - needs OpenML download) ====

def run_hpobench_svm(budget: int = 100, seeds: int = 3, task_ids: list = None):
    """Run HPOBench SVM benchmark - SKIP (requires OpenML data download)."""
    print("\n" + "=" * 70)
    print("HPOBench SVM Benchmark - SKIPPED (requires OpenML data download)")
    print("=" * 70)
    return None


# ==== HPOBench LR Benchmark (SKIP - needs OpenML download) ====

def run_hpobench_lr(budget: int = 100, seeds: int = 3, task_ids: list = None):
    """Run HPOBench LR benchmark - SKIP (requires OpenML data download)."""
    print("\n" + "=" * 70)
    print("HPOBench LR Benchmark - SKIPPED (requires OpenML data download)")
    print("=" * 70)
    return None


# ==== HPOBench RF Benchmark (SKIP - needs OpenML download) ====

def run_hpobench_rf(budget: int = 100, seeds: int = 3, task_ids: list = None):
    """Run HPOBench RF benchmark - SKIPPED (requires OpenML data download)."""
    print("\n" + "=" * 70)
    print("HPOBench RF Benchmark - SKIPPED (requires OpenML data download)")
    print("=" * 70)
    return None


# ==== High-Dimensional Synthetic Functions ====

def run_highdim_synthetic(budget: int = 200, seeds: int = 3):
    """Run high-dimensional synthetic benchmark (15D-20D)."""
    print("\n" + "=" * 70)
    print("High-Dimensional Synthetic Functions (15D-20D)")
    print("=" * 70)
    
    functions = [
        ("Sphere", 15),
        ("Rosenbrock", 15),
        ("Rastrigin", 15),
        ("Ackley", 20),
        ("Griewank", 20),
    ]
    
    results = {'ALBA': [], 'Optuna': [], 'TuRBO': []}
    
    for func_name, dim in functions:
        print(f"\n  {func_name} (dim={dim}):")
        
        if func_name == "Sphere":
            def objective(x):
                return float(np.sum(x**2))
            bounds = [(-5.0, 5.0)] * dim
        elif func_name == "Rosenbrock":
            def objective(x):
                return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))
            bounds = [(-5.0, 10.0)] * dim
        elif func_name == "Rastrigin":
            def objective(x):
                return float(10 * dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))
            bounds = [(-5.12, 5.12)] * dim
        elif func_name == "Ackley":
            def objective(x):
                a, b, c = 20, 0.2, 2 * np.pi
                sum1 = np.sum(x**2)
                sum2 = np.sum(np.cos(c * x))
                return float(-a * np.exp(-b * np.sqrt(sum1 / dim)) - np.exp(sum2 / dim) + a + np.e)
            bounds = [(-32.768, 32.768)] * dim
        elif func_name == "Griewank":
            def objective(x):
                sum_sq = np.sum(x**2) / 4000
                prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1))))
                return float(sum_sq - prod_cos + 1)
            bounds = [(-600.0, 600.0)] * dim
        
        for seed in range(seeds):
            print(f"    seed={seed}", end=" ", flush=True)
            
            alba_best = run_optimizer_on_objective(objective, bounds, budget, seed, 'ALBA')
            optuna_best = run_optimizer_on_objective(objective, bounds, budget, seed, 'Optuna')
            turbo_best = run_optimizer_on_objective(objective, bounds, budget, seed, 'TuRBO')
            
            results['ALBA'].append(alba_best)
            results['Optuna'].append(optuna_best)
            results['TuRBO'].append(turbo_best)
            
            bests = [('ALBA', alba_best), ('Optuna', optuna_best), ('TuRBO', turbo_best)]
            winner = min(bests, key=lambda x: x[1])[0]
            print(f"ALBA={alba_best:.2e} Optuna={optuna_best:.2e} TuRBO={turbo_best:.2e} -> {winner}")
    
    return _summarize_results("HighDim", results)


# ==== Nevergrad Realistic Functions ====

def run_nevergrad_realistic(budget: int = 200, seeds: int = 3):
    """Run Nevergrad realistic benchmark functions."""
    print("\n" + "=" * 70)
    print("Nevergrad-style Realistic Functions")
    print("=" * 70)
    
    # Funzioni realistiche da testare (implementate direttamente)
    functions = [
        ("Sphere", 10),
        ("Rastrigin", 10),
        ("Rosenbrock", 10),
        ("Cigar", 10),
        ("Ellipsoid", 10),
        ("Lunacek", 10),
    ]
    
    results = {'ALBA': [], 'Optuna': [], 'TuRBO': []}
    
    for func_name, dim in functions:
        print(f"\n  {func_name} (dim={dim}):")
        
        try:
            # Create nevergrad function
            if func_name == "Sphere":
                def objective(x):
                    return float(np.sum(x**2))
            elif func_name == "Rastrigin":
                def objective(x):
                    return float(10 * dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))
            elif func_name == "Rosenbrock":
                def objective(x):
                    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))
            elif func_name == "Cigar":
                def objective(x):
                    return float(x[0]**2 + 1e6 * np.sum(x[1:]**2))
            elif func_name == "Ellipsoid":
                def objective(x):
                    return float(np.sum((10**(6 * np.arange(dim) / (dim - 1))) * x**2))
            elif func_name == "Lunacek":
                def objective(x):
                    s = 1 - 1 / (2 * np.sqrt(dim + 20) - 8.2)
                    mu1 = 2.5
                    mu2 = -np.sqrt((mu1**2 - 1) / s)
                    sphere1 = np.sum((x - mu1)**2)
                    sphere2 = dim + np.sum((x - mu2)**2)
                    return float(min(sphere1, sphere2) + 10 * (dim - np.sum(np.cos(2 * np.pi * (x - mu1)))))
            
            bounds = [(-5.0, 5.0)] * dim
            
            for seed in range(seeds):
                print(f"    seed={seed}", end=" ", flush=True)
                
                alba_best = run_optimizer_on_objective(objective, bounds, budget, seed, 'ALBA')
                optuna_best = run_optimizer_on_objective(objective, bounds, budget, seed, 'Optuna')
                turbo_best = run_optimizer_on_objective(objective, bounds, budget, seed, 'TuRBO')
                
                results['ALBA'].append(alba_best)
                results['Optuna'].append(optuna_best)
                results['TuRBO'].append(turbo_best)
                
                bests = [('ALBA', alba_best), ('Optuna', optuna_best), ('TuRBO', turbo_best)]
                winner = min(bests, key=lambda x: x[1])[0]
                print(f"ALBA={alba_best:.2e} Optuna={optuna_best:.2e} TuRBO={turbo_best:.2e} -> {winner}")
                
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    return _summarize_results("Nevergrad", results)


# ==== Helper Functions ====

def _summarize_results(name: str, results: Dict[str, list]) -> Dict[str, list]:
    """Print summary and return results."""
    if not results['ALBA']:
        return None
    
    n = len(results['ALBA'])
    alba_wins = sum(1 for i in range(n) 
                    if results['ALBA'][i] < results['Optuna'][i] 
                    and results['ALBA'][i] < results['TuRBO'][i])
    optuna_wins = sum(1 for i in range(n) 
                      if results['Optuna'][i] < results['ALBA'][i] 
                      and results['Optuna'][i] < results['TuRBO'][i])
    turbo_wins = sum(1 for i in range(n) 
                     if results['TuRBO'][i] < results['ALBA'][i] 
                     and results['TuRBO'][i] < results['Optuna'][i])
    
    print(f"\n=== {name} Summary ===")
    print(f"ALBA wins: {alba_wins}/{n} ({100*alba_wins/n:.1f}%)")
    print(f"Optuna wins: {optuna_wins}/{n} ({100*optuna_wins/n:.1f}%)")
    print(f"TuRBO wins: {turbo_wins}/{n} ({100*turbo_wins/n:.1f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--only', type=str, default=None, 
                       help='svm, lr, rf, nevergrad, highdim, or all')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MEGA BENCHMARK: ALBA vs Optuna vs TuRBO")
    print("=" * 70)
    print(f"Budget: {args.budget} evaluations")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)
    
    all_results = {}
    
    if args.only is None or args.only == 'all' or args.only == 'svm':
        all_results['SVM'] = run_hpobench_svm(args.budget, args.seeds)
    
    if args.only is None or args.only == 'all' or args.only == 'lr':
        all_results['LR'] = run_hpobench_lr(args.budget, args.seeds)
    
    if args.only is None or args.only == 'all' or args.only == 'rf':
        all_results['RF'] = run_hpobench_rf(args.budget, args.seeds)
    
    if args.only is None or args.only == 'all' or args.only == 'nevergrad':
        all_results['Nevergrad'] = run_nevergrad_realistic(args.budget, args.seeds)
    
    if args.only is None or args.only == 'all' or args.only == 'highdim':
        all_results['HighDim'] = run_highdim_synthetic(args.budget, args.seeds)
    
    # Final summary
    print("\n" + "=" * 70)
    print("GRAND FINAL SUMMARY")
    print("=" * 70)
    
    total_alba = total_optuna = total_turbo = total_n = 0
    
    for name, results in all_results.items():
        if results is None:
            continue
        n = len(results['ALBA'])
        alba_w = sum(1 for i in range(n) 
                     if results['ALBA'][i] < results['Optuna'][i] 
                     and results['ALBA'][i] < results['TuRBO'][i])
        optuna_w = sum(1 for i in range(n) 
                       if results['Optuna'][i] < results['ALBA'][i] 
                       and results['Optuna'][i] < results['TuRBO'][i])
        turbo_w = sum(1 for i in range(n) 
                      if results['TuRBO'][i] < results['ALBA'][i] 
                      and results['TuRBO'][i] < results['Optuna'][i])
        
        total_alba += alba_w
        total_optuna += optuna_w
        total_turbo += turbo_w
        total_n += n
        
        print(f"{name}: ALBA {alba_w}/{n}, Optuna {optuna_w}/{n}, TuRBO {turbo_w}/{n}")
    
    if total_n > 0:
        print("-" * 70)
        print(f"TOTAL: ALBA {total_alba}/{total_n} ({100*total_alba/total_n:.1f}%), "
              f"Optuna {total_optuna}/{total_n} ({100*total_optuna/total_n:.1f}%), "
              f"TuRBO {total_turbo}/{total_n} ({100*total_turbo/total_n:.1f}%)")
    print("=" * 70)


if __name__ == '__main__':
    main()
