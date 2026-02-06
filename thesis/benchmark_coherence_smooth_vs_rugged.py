"""
Validate hypothesis: Coherence works better on SMOOTH landscapes vs RUGGED ones.

SMOOTH functions (informative gradients):
- Sphere: f(x) = sum(x^2) - unimodal, smooth
- Rosenbrock: f(x) = sum(100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2) - smooth valley

RUGGED functions (misleading gradients):
- Rastrigin: f(x) = 10d + sum(x^2 - 10*cos(2*pi*x)) - many local minima
- Ackley: exp-based with cosine oscillations - highly multimodal

If hypothesis is correct:
- Coherence >> Optuna on Sphere/Rosenbrock
- Coherence ~ Optuna on Rastrigin/Ackley
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np

sys.path.insert(0, "/mnt/workspace/thesis")
from alba_framework_coherence import ALBA as ALBA_COH

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic functions
# ─────────────────────────────────────────────────────────────────────────────

def sphere(x: np.ndarray) -> float:
    """Sphere: smooth, unimodal. Optimum at origin = 0."""
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock: smooth valley. Optimum at (1,1,...,1) = 0."""
    total = 0.0
    for i in range(len(x) - 1):
        total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return float(total)


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin: highly multimodal. Optimum at origin = 0."""
    d = len(x)
    return float(10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley: multimodal with many local minima. Optimum at origin = 0."""
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return float(-20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e)


def griewank(x: np.ndarray) -> float:
    """Griewank: moderately multimodal. Optimum at origin = 0."""
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(sum_sq - prod_cos + 1)


def levy(x: np.ndarray) -> float:
    """Levy: smooth with some oscillations. Optimum at (1,1,...,1) = 0."""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return float(term1 + term2 + term3)


FUNCTIONS = {
    # SMOOTH (informative gradients)
    "sphere": {"fn": sphere, "bounds": (-5.12, 5.12), "type": "SMOOTH"},
    "rosenbrock": {"fn": rosenbrock, "bounds": (-5.0, 10.0), "type": "SMOOTH"},
    "levy": {"fn": levy, "bounds": (-10.0, 10.0), "type": "SMOOTH"},
    # RUGGED (misleading gradients)
    "rastrigin": {"fn": rastrigin, "bounds": (-5.12, 5.12), "type": "RUGGED"},
    "ackley": {"fn": ackley, "bounds": (-32.768, 32.768), "type": "RUGGED"},
    "griewank": {"fn": griewank, "bounds": (-600.0, 600.0), "type": "RUGGED"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Optimizers
# ─────────────────────────────────────────────────────────────────────────────

def run_coherence(fn, bounds: List[Tuple[float, float]], budget: int, seed: int) -> float:
    opt = ALBA_COH(
        bounds=bounds,
        maximize=False,
        seed=seed,
    )
    best = float("inf")
    for _ in range(budget):
        x = opt.ask()
        y = fn(x)
        opt.tell(x, y)
        best = min(best, y)
    return best


def run_optuna(fn, bounds: List[Tuple[float, float]], budget: int, seed: int) -> float:
    dim = len(bounds)
    best = [float("inf")]
    
    def objective(trial):
        x = np.array([trial.suggest_float(f"x{i}", bounds[i][0], bounds[i][1]) for i in range(dim)])
        y = fn(x)
        best[0] = min(best[0], y)
        return y
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return best[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--dim", type=int, default=10)
    args = parser.parse_args()
    
    # Parse seeds
    if "-" in args.seeds:
        s, e = map(int, args.seeds.split("-"))
        seeds = list(range(s, e + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]
    
    budget = args.budget
    dim = args.dim
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "/mnt/workspace/thesis/benchmark_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/coherence_smooth_vs_rugged_d{dim}_b{budget}_{timestamp}.json"
    
    print(f"SMOOTH vs RUGGED | dim={dim} | budget={budget} | seeds={seeds}")
    print(f"save={results_file}")
    print("=" * 80)
    
    results = {
        "config": {"dim": dim, "budget": budget, "seeds": seeds},
        "smooth": {"coh_wins": 0, "opt_wins": 0, "ties": 0, "runs": []},
        "rugged": {"coh_wins": 0, "opt_wins": 0, "ties": 0, "runs": []},
    }
    
    for fn_name, fn_info in FUNCTIONS.items():
        fn = fn_info["fn"]
        lo, hi = fn_info["bounds"]
        fn_type = fn_info["type"].lower()
        bounds = [(lo, hi)] * dim
        
        print(f"\n{fn_name.upper()} ({fn_info['type']})")
        print("-" * 60)
        
        fn_coh_wins = 0
        fn_opt_wins = 0
        fn_ties = 0
        
        for seed in seeds:
            t0 = time.time()
            
            coh_best = run_coherence(fn, bounds, budget, seed)
            opt_best = run_optuna(fn, bounds, budget, seed)
            
            elapsed = time.time() - t0
            
            # Relative comparison (use ratio for scale-invariance)
            if coh_best < opt_best * 0.99:  # COH at least 1% better
                winner = "COH"
                fn_coh_wins += 1
                results[fn_type]["coh_wins"] += 1
            elif opt_best < coh_best * 0.99:  # OPT at least 1% better
                winner = "OPT"
                fn_opt_wins += 1
                results[fn_type]["opt_wins"] += 1
            else:
                winner = "TIE"
                fn_ties += 1
                results[fn_type]["ties"] += 1
            
            print(f"  seed={seed:2d} | COH={coh_best:.2e} OPT={opt_best:.2e} -> {winner} ({elapsed:.1f}s)")
            
            results[fn_type]["runs"].append({
                "function": fn_name,
                "seed": seed,
                "coh_best": coh_best,
                "opt_best": opt_best,
                "winner": winner,
            })
        
        print(f"  {fn_name} SUMMARY: COH={fn_coh_wins}, OPT={fn_opt_wins}, TIE={fn_ties}")
        
        # Save incrementally
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 80)
    print("HYPOTHESIS VALIDATION")
    print("=" * 80)
    
    smooth = results["smooth"]
    rugged = results["rugged"]
    
    smooth_total = smooth["coh_wins"] + smooth["opt_wins"]
    rugged_total = rugged["coh_wins"] + rugged["opt_wins"]
    
    smooth_wr = smooth["coh_wins"] / smooth_total * 100 if smooth_total > 0 else 0
    rugged_wr = rugged["coh_wins"] / rugged_total * 100 if rugged_total > 0 else 0
    
    print(f"SMOOTH functions: COH wins={smooth['coh_wins']}, OPT wins={smooth['opt_wins']}, ties={smooth['ties']} -> {smooth_wr:.1f}% winrate")
    print(f"RUGGED functions: COH wins={rugged['coh_wins']}, OPT wins={rugged['opt_wins']}, ties={rugged['ties']} -> {rugged_wr:.1f}% winrate")
    print()
    
    if smooth_wr > rugged_wr + 10:
        print("✓ HYPOTHESIS SUPPORTED: Coherence works better on SMOOTH landscapes")
    elif rugged_wr > smooth_wr + 10:
        print("✗ HYPOTHESIS REJECTED: Coherence works better on RUGGED landscapes (unexpected!)")
    else:
        print("~ INCONCLUSIVE: No significant difference between smooth and rugged")
    
    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
