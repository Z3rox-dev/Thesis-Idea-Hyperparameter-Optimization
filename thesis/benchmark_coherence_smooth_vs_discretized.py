#!/usr/bin/env python3
"""
Benchmark: Coherence vs Optuna su funzioni SMOOTH vs DISCRETIZED (a gradini)

OBIETTIVO: Dimostrare che NON è la funzione vera che conta, ma come la "vediamo"
           attraverso il surrogate/discretizzazione.

Stessa funzione sottostante (Sphere, Rosenbrock), ma:
- SMOOTH: valore esatto
- DISCRETIZED: valore arrotondato a N bins (simula XGBoost/RF)

Se l'ipotesi è corretta:
- Coherence vince su SMOOTH
- Coherence perde su DISCRETIZED (stessa funzione!)
"""

import sys
import os
import json
import argparse
import time
import warnings
from datetime import datetime

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alba_framework_coherence.optimizer import ALBA

# ============================================================================
# FUNZIONI: versione smooth e discretizzata
# ============================================================================

def sphere(x):
    """Sphere - unimodale, smooth perfetto"""
    return np.sum(x ** 2)

def rosenbrock(x):
    """Rosenbrock - smooth con valle curva"""
    total = 0
    for i in range(len(x) - 1):
        total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return total

def discretize(value, n_bins=50, vmin=0, vmax=100):
    """Discretizza un valore in N bins (simula output di albero decisionale)"""
    # Clamp al range
    value = np.clip(value, vmin, vmax)
    # Discretizza
    bin_size = (vmax - vmin) / n_bins
    bin_idx = int((value - vmin) / bin_size)
    bin_idx = min(bin_idx, n_bins - 1)
    # Ritorna centro del bin
    return vmin + (bin_idx + 0.5) * bin_size

# Wrapper per creare versioni discretizzate
def make_discretized(func, n_bins, bounds):
    """Crea versione discretizzata di una funzione"""
    # Stima range approssimativo per la funzione
    # Per Sphere in [-5,5]^d: max ~ 25*d
    # Per Rosenbrock in [-5,5]^d: può essere molto alto
    dim = len(bounds)
    if func.__name__ == 'sphere':
        vmax = 25 * dim * 1.5
    else:
        vmax = 1000 * dim
    
    def discretized_func(x):
        raw = func(x)
        return discretize(raw, n_bins=n_bins, vmin=0, vmax=vmax)
    
    discretized_func.__name__ = f"{func.__name__}_bins{n_bins}"
    return discretized_func

# ============================================================================
# OPTIMIZER RUNNERS
# ============================================================================

def run_coherence(func, bounds, budget, seed):
    """Esegue ALBA con Coherence"""
    dim = len(bounds)
    np.random.seed(seed)
    
    opt = ALBA(
        bounds=bounds,
        seed=seed,
        total_budget=budget
    )
    
    for _ in range(budget):
        x = opt.ask()
        y = func(np.array(x))
        opt.tell(x, y)
    
    return opt.best_y

def run_optuna(func, bounds, budget, seed):
    """Esegue Optuna TPE"""
    dim = len(bounds)
    
    def objective(trial):
        x = np.array([trial.suggest_float(f"x{i}", bounds[i][0], bounds[i][1]) 
                      for i in range(dim)])
        return func(x)
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    
    return study.best_value

# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(dim, budget, seeds, n_bins_list):
    """
    Confronta Coherence vs Optuna su:
    - Funzioni smooth
    - Stesse funzioni discretizzate con diversi numeri di bins
    """
    bounds = [(-5.0, 5.0)] * dim
    base_functions = [sphere, rosenbrock]
    
    results = {
        "dim": dim,
        "budget": budget,
        "seeds": seeds,
        "n_bins_list": n_bins_list,
        "data": {}
    }
    
    print(f"SMOOTH vs DISCRETIZED | dim={dim} | budget={budget} | seeds={seeds}")
    print(f"Discretization bins: {n_bins_list}")
    print("=" * 70)
    
    for base_func in base_functions:
        # Lista di varianti: smooth + varie discretizzazioni
        variants = [("SMOOTH", base_func)]
        for n_bins in n_bins_list:
            disc_func = make_discretized(base_func, n_bins, bounds)
            variants.append((f"BINS_{n_bins}", disc_func))
        
        for variant_name, func in variants:
            key = f"{base_func.__name__}_{variant_name}"
            print(f"\n{base_func.__name__.upper()} ({variant_name})")
            print("-" * 50)
            
            coh_wins, opt_wins, ties = 0, 0, 0
            coh_scores, opt_scores = [], []
            
            for seed in seeds:
                t0 = time.time()
                
                coh_best = run_coherence(func, bounds, budget, seed)
                opt_best = run_optuna(func, bounds, budget, seed)
                
                elapsed = time.time() - t0
                
                # Determina vincitore (1% tolerance)
                if coh_best < opt_best * 0.99:
                    winner = "COH"
                    coh_wins += 1
                elif opt_best < coh_best * 0.99:
                    winner = "OPT"
                    opt_wins += 1
                else:
                    winner = "TIE"
                    ties += 1
                
                coh_scores.append(coh_best)
                opt_scores.append(opt_best)
                
                print(f"  seed={seed:2d} | COH={coh_best:.2e} OPT={opt_best:.2e} -> {winner} ({elapsed:.1f}s)")
            
            winrate = coh_wins / len(seeds) * 100 if len(seeds) > 0 else 0
            print(f"  SUMMARY: COH={coh_wins}, OPT={opt_wins}, TIE={ties} (COH winrate: {winrate:.1f}%)")
            
            results["data"][key] = {
                "coh_wins": coh_wins,
                "opt_wins": opt_wins,
                "ties": ties,
                "coh_winrate": winrate,
                "coh_scores": coh_scores,
                "opt_scores": opt_scores
            }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--bins", type=str, default="10,25,50,100")
    args = parser.parse_args()
    
    # Parse seeds
    if "-" in args.seeds:
        start, end = map(int, args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    
    # Parse bins
    n_bins_list = [int(b) for b in args.bins.split(",")]
    
    results = run_benchmark(args.dim, args.budget, seeds, n_bins_list)
    
    # Summary finale
    print("\n" + "=" * 70)
    print("RIEPILOGO FINALE")
    print("=" * 70)
    
    for key, data in results["data"].items():
        print(f"{key:30s} | COH={data['coh_wins']:2d} OPT={data['opt_wins']:2d} TIE={data['ties']:2d} | winrate={data['coh_winrate']:.1f}%")
    
    # Analisi per tipo
    print("\n" + "-" * 70)
    print("ANALISI: Effetto della discretizzazione")
    print("-" * 70)
    
    for base in ["sphere", "rosenbrock"]:
        smooth_key = f"{base}_SMOOTH"
        if smooth_key in results["data"]:
            smooth_wr = results["data"][smooth_key]["coh_winrate"]
            print(f"\n{base.upper()}:")
            print(f"  SMOOTH: {smooth_wr:.1f}%")
            for n_bins in n_bins_list:
                disc_key = f"{base}_BINS_{n_bins}"
                if disc_key in results["data"]:
                    disc_wr = results["data"][disc_key]["coh_winrate"]
                    delta = disc_wr - smooth_wr
                    print(f"  BINS_{n_bins:3d}: {disc_wr:.1f}% (delta: {delta:+.1f}%)")
    
    # Salva risultati
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = f"/mnt/workspace/thesis/benchmark_results/coherence_smooth_vs_discretized_d{args.dim}_b{args.budget}_{timestamp}.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    
    # Converti per JSON
    results_json = results.copy()
    for k, v in results_json["data"].items():
        v["coh_scores"] = [float(x) for x in v["coh_scores"]]
        v["opt_scores"] = [float(x) for x in v["opt_scores"]]
    
    with open(outpath, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {outpath}")

if __name__ == "__main__":
    main()
