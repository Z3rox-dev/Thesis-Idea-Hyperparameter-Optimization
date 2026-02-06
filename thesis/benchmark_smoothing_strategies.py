#!/usr/bin/env python3
"""
Benchmark: Strategie di Smoothing per funzioni discretizzate

Confronta:
1. BASELINE: funzione discretizzata pura (gradini)
2. RANDOM_NOISE: aggiunge rumore i.i.d. (moto browniano)
3. SPATIAL_NOISE: rumore deterministico basato sulla posizione
4. KNN_SMOOTH: interpolazione pesata sui k vicini osservati

L'idea è "rompere" i plateau per permettere a Coherence di calcolare gradienti.
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
# FUNZIONE BASE E DISCRETIZZAZIONE
# ============================================================================

def sphere(x):
    return np.sum(x ** 2)

def discretize(value, n_bins=25, vmin=0, vmax=375):
    """Discretizza in N bins (simula XGBoost)"""
    value = np.clip(value, vmin, vmax)
    bin_size = (vmax - vmin) / n_bins
    bin_idx = int((value - vmin) / bin_size)
    bin_idx = min(bin_idx, n_bins - 1)
    return vmin + (bin_idx + 0.5) * bin_size

# ============================================================================
# STRATEGIE DI SMOOTHING
# ============================================================================

class SmoothingWrapper:
    """Wrapper che applica smoothing a una funzione discretizzata"""
    
    def __init__(self, base_func, n_bins, bounds, strategy="none", noise_scale=0.1):
        self.base_func = base_func
        self.n_bins = n_bins
        self.bounds = bounds
        self.strategy = strategy
        self.noise_scale = noise_scale
        self.dim = len(bounds)
        
        # Per Sphere in [-5,5]^d
        self.vmax = 25 * self.dim * 1.5
        
        # Storia per k-NN smoothing
        self.observed_X = []
        self.observed_y_raw = []  # valori discretizzati originali
        
    def __call__(self, x):
        x = np.array(x)
        
        # Calcola valore discretizzato
        raw = self.base_func(x)
        disc = discretize(raw, self.n_bins, 0, self.vmax)
        
        if self.strategy == "none":
            # Baseline: solo discretizzazione
            result = disc
            
        elif self.strategy == "random":
            # Moto browniano: rumore i.i.d.
            noise = np.random.normal(0, self.noise_scale * (self.vmax / self.n_bins))
            result = disc + noise
            
        elif self.strategy == "spatial":
            # Rumore deterministico basato sulla posizione
            # Usa hash della posizione per generare rumore reproducibile
            hash_val = np.sum(np.sin(x * 1000 + np.arange(len(x)) * 0.1))
            noise = hash_val * self.noise_scale * (self.vmax / self.n_bins)
            result = disc + noise
            
        elif self.strategy == "knn":
            # Smoothing k-NN sui punti osservati
            if len(self.observed_X) < 5:
                result = disc
            else:
                X_arr = np.array(self.observed_X)
                y_arr = np.array(self.observed_y_raw)
                
                # Calcola distanze
                distances = np.linalg.norm(X_arr - x, axis=1)
                k = min(5, len(distances))
                nearest_idx = np.argsort(distances)[:k]
                
                # Media pesata (più vicino = più peso)
                weights = 1 / (distances[nearest_idx] + 1e-6)
                smooth_val = np.average(y_arr[nearest_idx], weights=weights)
                
                # Mix tra discretizzato e smooth
                alpha = 0.5  # quanto smoothing applicare
                result = (1 - alpha) * disc + alpha * smooth_val
            
            # Aggiorna storia
            self.observed_X.append(x.copy())
            self.observed_y_raw.append(disc)
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return result
    
    def reset(self):
        """Reset per nuova run"""
        self.observed_X = []
        self.observed_y_raw = []

# ============================================================================
# OPTIMIZER RUNNERS
# ============================================================================

def run_coherence(func_wrapper, bounds, budget, seed):
    """Esegue ALBA con Coherence"""
    func_wrapper.reset()
    np.random.seed(seed)
    
    opt = ALBA(bounds=bounds, seed=seed, total_budget=budget)
    
    for _ in range(budget):
        x = opt.ask()
        y = func_wrapper(np.array(x))
        opt.tell(x, y)
    
    return opt.best_y

def run_optuna(func_wrapper, bounds, budget, seed):
    """Esegue Optuna TPE"""
    func_wrapper.reset()
    dim = len(bounds)
    
    def objective(trial):
        x = np.array([trial.suggest_float(f"x{i}", bounds[i][0], bounds[i][1]) 
                      for i in range(dim)])
        return func_wrapper(x)
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    
    return study.best_value

# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(dim, budget, seeds, n_bins, noise_scale):
    bounds = [(-5.0, 5.0)] * dim
    
    strategies = [
        ("DISCRETIZED", "none"),      # Baseline
        ("RANDOM_NOISE", "random"),   # Moto browniano
        ("SPATIAL_NOISE", "spatial"), # Rumore deterministico
        ("KNN_SMOOTH", "knn"),        # Interpolazione k-NN
    ]
    
    results = {}
    
    print(f"SMOOTHING STRATEGIES | dim={dim} | budget={budget} | bins={n_bins} | noise_scale={noise_scale}")
    print(f"seeds={seeds}")
    print("=" * 70)
    
    for name, strategy in strategies:
        print(f"\n{name}")
        print("-" * 50)
        
        coh_wins, opt_wins, ties = 0, 0, 0
        coh_scores, opt_scores = [], []
        
        for seed in seeds:
            t0 = time.time()
            
            # Crea wrapper con strategia
            func = SmoothingWrapper(sphere, n_bins, bounds, strategy, noise_scale)
            
            coh_best = run_coherence(func, bounds, budget, seed)
            
            # Reset e ri-crea per Optuna (per avere storia pulita)
            func = SmoothingWrapper(sphere, n_bins, bounds, strategy, noise_scale)
            opt_best = run_optuna(func, bounds, budget, seed)
            
            elapsed = time.time() - t0
            
            # Determina vincitore
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
        
        winrate = coh_wins / len(seeds) * 100
        print(f"  SUMMARY: COH={coh_wins}, OPT={opt_wins}, TIE={ties} (COH winrate: {winrate:.1f}%)")
        
        results[name] = {
            "coh_wins": coh_wins,
            "opt_wins": opt_wins,
            "ties": ties,
            "winrate": winrate
        }
    
    # Riepilogo
    print("\n" + "=" * 70)
    print("RIEPILOGO")
    print("=" * 70)
    for name, data in results.items():
        print(f"{name:20s} | COH={data['coh_wins']:2d} OPT={data['opt_wins']:2d} TIE={data['ties']:2d} | winrate={data['winrate']:.1f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    args = parser.parse_args()
    
    # Parse seeds
    if "-" in args.seeds:
        start, end = map(int, args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    
    results = run_benchmark(args.dim, args.budget, seeds, args.bins, args.noise_scale)
    
    # Salva risultati
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"benchmark_results/smoothing_strategies_b{args.bins}_n{args.noise_scale}_{timestamp}.json"
    os.makedirs("benchmark_results", exist_ok=True)
    
    with open(outfile, "w") as f:
        json.dump({
            "dim": args.dim,
            "budget": args.budget,
            "seeds": seeds,
            "bins": args.bins,
            "noise_scale": args.noise_scale,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {outfile}")

if __name__ == "__main__":
    main()
