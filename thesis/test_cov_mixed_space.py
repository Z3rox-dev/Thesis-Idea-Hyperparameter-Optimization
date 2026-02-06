#!/usr/bin/env python3
"""
Test: Covariance Matrix su spazio MISTO (continui + categorici).

Obiettivo: Verificare se la covariance aiuta quando ci sono 
           sia dimensioni continue che categoriche.
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import CovarianceLocalSearchSampler, GaussianLocalSearchSampler

# =============================================================================
# FUNZIONI BENCHMARK MISTE
# =============================================================================

def nn_mixed_score(x: np.ndarray) -> float:
    """
    Simula lo score di una configurazione NN con spazio misto.
    
    Dimensioni CONTINUE (0-2):
    - 0: learning_rate (0.0001 - 0.1) log-scale simulato
    - 1: weight_decay (0.0 - 0.1)
    - 2: momentum (0.8 - 0.99)
    
    Dimensioni CATEGORICHE (3-5):
    - 3: activation (relu, tanh, gelu, selu) -> 4 choices
    - 4: optimizer (adam, sgd, rmsprop) -> 3 choices
    - 5: batch_size (16, 32, 64, 128, 256) -> 5 choices
    """
    def discretize(val, n_choices):
        val = max(0.0, min(1.0, val))
        return int(round(val * (n_choices - 1)))
    
    # Continui
    lr = x[0]           # ottimo ~0.3 (che corrisponde a ~0.001)
    wd = x[1]           # ottimo ~0.1
    momentum = x[2]     # ottimo ~0.7 (che corrisponde a 0.95)
    
    # Categorici
    activation = discretize(x[3], 4)    # 0=relu, 1=tanh, 2=gelu, 3=selu
    optimizer = discretize(x[4], 3)     # 0=adam, 1=sgd, 2=rmsprop
    batch_size = discretize(x[5], 5)    # 0=16, 1=32, 2=64, 3=128, 4=256
    
    score = 0.0
    
    # Score continui: Rosenbrock-like valley per lr e momentum
    # Ottimo: lr=0.3, wd=0.1, momentum=0.7
    score -= 10 * (lr - 0.3)**2  
    score -= 5 * (wd - 0.1)**2
    score -= 8 * (momentum - 0.7)**2
    
    # Interazione continui: lr e momentum devono essere correlati
    # Se lr alto, momentum deve essere basso
    score -= 3 * (lr + momentum - 1.0)**2
    
    # Score categorici
    act_scores = [0.0, -0.1, 0.2, -0.05]  # gelu è ottimo
    score += act_scores[activation]
    
    opt_scores = [0.15, -0.2, 0.0]  # adam è ottimo
    score += opt_scores[optimizer]
    
    batch_scores = [-0.1, 0.0, 0.15, 0.1, -0.05]  # 64 è ottimo
    score += batch_scores[batch_size]
    
    # Interazione mista: gelu + lr basso = bonus
    if activation == 2 and lr < 0.4:
        score += 0.1
    
    # adam + momentum alto = bonus
    if optimizer == 0 and momentum > 0.6:
        score += 0.1
    
    # Normalizza
    return score + 1.0  # shift per avere valori positivi


def rosenbrock_with_categorical(x: np.ndarray) -> float:
    """
    Rosenbrock sulle prime 4 dimensioni + 2 categoriche.
    
    Continui (0-3): Rosenbrock 4D
    Categorici (4-5): modificatori
    """
    def discretize(val, n_choices):
        val = max(0.0, min(1.0, val))
        return int(round(val * (n_choices - 1)))
    
    # Rosenbrock sulle continue (scaled to [0,1])
    # Remap to [-2, 2]
    cont = [(x[i] - 0.5) * 4 for i in range(4)]
    rosen = sum(100 * (cont[i+1] - cont[i]**2)**2 + (1 - cont[i])**2 
                for i in range(3))
    
    # Categorici
    cat1 = discretize(x[4], 4)  # 4 choices
    cat2 = discretize(x[5], 3)  # 3 choices
    
    # Modificatori: solo certe combinazioni sono buone
    # Ottimo: cat1=1, cat2=0
    cat_penalty = 0
    if cat1 != 1:
        cat_penalty += 10
    if cat2 != 0:
        cat_penalty += 5
    
    # Minimizziamo quindi neghiamo
    return -(rosen + cat_penalty)


def sphere_with_categorical(x: np.ndarray) -> float:
    """
    Sphere sulle prime 5 dimensioni + 3 categoriche.
    
    Continui (0-4): Sphere 5D
    Categorici (5-7): selezione regime
    """
    def discretize(val, n_choices):
        val = max(0.0, min(1.0, val))
        return int(round(val * (n_choices - 1)))
    
    # Sphere sulle continue (remap to [-1, 1])
    cont = [(x[i] - 0.5) * 2 for i in range(5)]
    sphere = sum(c**2 for c in cont)
    
    # Categorici
    cat1 = discretize(x[5], 3)
    cat2 = discretize(x[6], 4) 
    cat3 = discretize(x[7], 2)
    
    # Regime: modifica il centro ottimale
    # cat1=0, cat2=2, cat3=1 è l'ottimo
    regime_penalty = 0
    if cat1 != 0:
        regime_penalty += 0.5
    if cat2 != 2:
        regime_penalty += 0.3
    if cat3 != 1:
        regime_penalty += 0.2
    
    return -(sphere + regime_penalty)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark_with_sampler(func, dim, budget, n_seeds, cat_dims, use_cov: bool):
    """Esegue benchmark con o senza covariance sampler."""
    results = []
    
    for seed in range(n_seeds):
        if use_cov:
            sampler = CovarianceLocalSearchSampler()
        else:
            sampler = GaussianLocalSearchSampler()
        
        opt = ALBA(
            bounds=[(0.0, 1.0) for _ in range(dim)],
            maximize=True,
            seed=100 + seed,
            total_budget=budget,
            categorical_dims=cat_dims,
            use_potential_field=False,
        )
        
        opt._local_search_sampler = sampler
        
        for _ in range(budget):
            x = opt.ask()
            score = func(x)
            opt.tell(x, score)
        
        results.append(opt.best_y_internal)
    
    return results


def main():
    print("=" * 70)
    print("TEST: Covariance Matrix su Spazio MISTO (continui + categorici)")
    print("=" * 70)
    
    n_seeds = 20
    
    benchmarks = [
        # (func, name, dim, budget, cat_dims)
        # NN Mixed: 3 continui + 3 categorici
        (nn_mixed_score, "NN Mixed (3C+3K)", 6, 200, [(3, 4), (4, 3), (5, 5)]),
        
        # Rosenbrock + cat: 4 continui + 2 categorici
        (rosenbrock_with_categorical, "Rosen+Cat (4C+2K)", 6, 250, [(4, 4), (5, 3)]),
        
        # Sphere + cat: 5 continui + 3 categorici  
        (sphere_with_categorical, "Sphere+Cat (5C+3K)", 8, 200, [(5, 3), (6, 4), (7, 2)]),
    ]
    
    results_table = []
    
    print(f"\nEsecuzione con {n_seeds} seeds per configurazione...")
    print("-" * 70)
    
    for func, name, dim, budget, cat_dims in benchmarks:
        print(f"  Testing {name}...", end=" ", flush=True)
        
        results_cov = run_benchmark_with_sampler(func, dim, budget, n_seeds, cat_dims, use_cov=True)
        results_gauss = run_benchmark_with_sampler(func, dim, budget, n_seeds, cat_dims, use_cov=False)
        
        mean_cov = np.mean(results_cov)
        std_cov = np.std(results_cov)
        mean_gauss = np.mean(results_gauss)
        std_gauss = np.std(results_gauss)
        
        wins_cov = sum(1 for a, b in zip(results_cov, results_gauss) if a > b)
        
        results_table.append({
            'name': name,
            'mean_cov': mean_cov,
            'std_cov': std_cov,
            'mean_gauss': mean_gauss,
            'std_gauss': std_gauss,
            'wins_cov': wins_cov,
        })
        
        print(f"done (COV: {mean_cov:.4f}, GAUSS: {mean_gauss:.4f})")
    
    print("\n" + "=" * 70)
    print("RISULTATI")
    print("=" * 70)
    
    print(f"\n{'Benchmark':<22} {'COV':>10} {'GAUSS':>10} {'Δ%':>8} {'Wins COV':>10}")
    print("-" * 62)
    
    for r in results_table:
        delta = r['mean_cov'] - r['mean_gauss']
        delta_pct = delta / abs(r['mean_gauss']) * 100 if r['mean_gauss'] != 0 else 0
        print(f"{r['name']:<22} {r['mean_cov']:>10.4f} {r['mean_gauss']:>10.4f} {delta_pct:>+7.1f}% {r['wins_cov']:>7}/{n_seeds}")
    
    print("\n" + "=" * 70)
    print("ANALISI")
    print("=" * 70)
    
    total_wins_cov = sum(r['wins_cov'] for r in results_table)
    total_runs = len(results_table) * n_seeds
    
    print(f"\n  Wins Covariance: {total_wins_cov}/{total_runs}")
    print(f"  Wins Gaussian:   {total_runs - total_wins_cov}/{total_runs}")
    
    avg_delta_pct = np.mean([(r['mean_cov'] - r['mean_gauss']) / abs(r['mean_gauss']) * 100 
                             if r['mean_gauss'] != 0 else 0 
                             for r in results_table])
    print(f"\n  Delta % medio: {avg_delta_pct:+.2f}%")
    
    if total_wins_cov > total_runs * 0.55:
        print("\n  Conclusione: Covariance AIUTA su spazi misti")
    elif total_wins_cov < total_runs * 0.45:
        print("\n  Conclusione: Covariance DANNEGGIA su spazi misti")
    else:
        print("\n  Conclusione: Nessuna differenza significativa")


if __name__ == "__main__":
    main()
