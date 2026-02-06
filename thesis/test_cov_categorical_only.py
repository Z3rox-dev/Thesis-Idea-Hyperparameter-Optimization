#!/usr/bin/env python3
"""
Test: Covariance Matrix su benchmark SOLO CATEGORIALE.

Obiettivo: Verificare se la covariance matrix ha senso quando 
           tutte le dimensioni sono categoriche.
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.local_search import CovarianceLocalSearchSampler, GaussianLocalSearchSampler

# =============================================================================
# FUNZIONI BENCHMARK SOLO CATEGORICHE
# =============================================================================

def nn_config_score(x: np.ndarray, categorical_dims: list) -> float:
    """
    Simula lo score di una configurazione NN.
    
    x è in [0,1]^dim, le dimensioni categoriche vengono discretizzate.
    
    Dimensioni:
    - 0: n_layers (1, 2, 3, 4) -> 4 choices
    - 1: activation (relu, tanh, gelu, selu) -> 4 choices  
    - 2: optimizer (adam, sgd, rmsprop) -> 3 choices
    - 3: dropout (0.0, 0.1, 0.2, 0.3, 0.4, 0.5) -> 6 choices
    - 4: batch_size (16, 32, 64, 128, 256) -> 5 choices
    - 5: lr_schedule (constant, step, cosine, exponential) -> 4 choices
    """
    def discretize(val, n_choices):
        val = max(0.0, min(1.0, val))
        return int(round(val * (n_choices - 1)))
    
    # Discretize all dims
    n_layers = discretize(x[0], 4)      # 0,1,2,3 -> 1,2,3,4 layers
    activation = discretize(x[1], 4)    # 0=relu, 1=tanh, 2=gelu, 3=selu
    optimizer = discretize(x[2], 3)     # 0=adam, 1=sgd, 2=rmsprop
    dropout = discretize(x[3], 6)       # 0..5 -> 0.0, 0.1, ..., 0.5
    batch_size = discretize(x[4], 5)    # 0=16, 1=32, 2=64, 3=128, 4=256
    lr_schedule = discretize(x[5], 4)   # 0=constant, 1=step, 2=cosine, 3=exponential
    
    # Score function: penalizza alcune combinazioni, premia altre
    # Ottimo: 2 layers + gelu + adam + dropout=0.2 + batch_size=64 + cosine
    score = 0.0
    
    # n_layers: 2 è ottimo
    layer_scores = [0.7, 0.9, 1.0, 0.6]  # 1,2,3,4 layers
    score += layer_scores[n_layers]
    
    # activation: gelu è ottimo, relu ok
    act_scores = [0.85, 0.7, 1.0, 0.75]  # relu, tanh, gelu, selu
    score += act_scores[activation]
    
    # optimizer: adam è ottimo
    opt_scores = [1.0, 0.6, 0.8]  # adam, sgd, rmsprop
    score += opt_scores[optimizer]
    
    # dropout: 0.2 è ottimo (index 2)
    drop_scores = [0.7, 0.85, 1.0, 0.9, 0.75, 0.6]  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
    score += drop_scores[dropout]
    
    # batch_size: 64 (index 2) è ottimo
    batch_scores = [0.6, 0.8, 1.0, 0.85, 0.7]  # 16, 32, 64, 128, 256
    score += batch_scores[batch_size]
    
    # lr_schedule: cosine (index 2) è ottimo
    lr_scores = [0.7, 0.8, 1.0, 0.85]  # constant, step, cosine, exponential
    score += lr_scores[lr_schedule]
    
    # Interazioni: bonus se combinazioni buone
    # gelu + adam = +0.2
    if activation == 2 and optimizer == 0:
        score += 0.2
    
    # 2 layers + dropout 0.2 = +0.15
    if n_layers == 1 and dropout == 2:
        score += 0.15
    
    # cosine + batch 64 = +0.1
    if lr_schedule == 2 and batch_size == 2:
        score += 0.1
    
    # Normalizza in [0, 1] circa (max teorico ~6.45)
    return score / 6.5


def onemax_categorical(x: np.ndarray, n_choices: int = 4) -> float:
    """
    OneMax categoriale: ogni dimensione ha un valore target.
    Score = numero di dimensioni che matchano il target.
    
    Target: tutte le dimensioni a valore 0.
    """
    def discretize(val, n_ch):
        val = max(0.0, min(1.0, val))
        return int(round(val * (n_ch - 1)))
    
    score = 0.0
    for i in range(len(x)):
        if discretize(x[i], n_choices) == 0:
            score += 1.0
    return score / len(x)


def trap_categorical(x: np.ndarray, n_choices: int = 4) -> float:
    """
    Trap function categoriale: ingannevole.
    
    Se tutte le dim sono a 0 (all-zeros): score = 1.0 (ottimo globale)
    Altrimenti: score = proporzione di NON-zeri (ingannevole!)
    """
    def discretize(val, n_ch):
        val = max(0.0, min(1.0, val))
        return int(round(val * (n_ch - 1)))
    
    n = len(x)
    n_zeros = sum(1 for i in range(n) if discretize(x[i], n_choices) == 0)
    
    if n_zeros == n:
        return 1.0  # Global optimum
    else:
        # Deceptive: più non-zeri = meglio (tranne l'ottimo)
        return (n - n_zeros) / (n + 1)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark_with_sampler(func, name, dim, budget, n_seeds, cat_dims, use_cov: bool):
    """Esegue benchmark con o senza covariance sampler."""
    results = []
    
    for seed in range(n_seeds):
        # Imposta il sampler corretto PRIMA di creare l'ottimizzatore
        if use_cov:
            sampler = CovarianceLocalSearchSampler()
        else:
            sampler = GaussianLocalSearchSampler()
        
        opt = ALBA(
            bounds=[(0.0, 1.0) for _ in range(dim)],
            maximize=True,  # Vogliamo massimizzare
            seed=100 + seed,
            total_budget=budget,
            categorical_dims=cat_dims,
            use_potential_field=False,  # Disabilitato
        )
        
        # Override del sampler
        opt._local_search_sampler = sampler
        
        for _ in range(budget):
            x = opt.ask()
            score = func(x)
            opt.tell(x, score)
        
        results.append(opt.best_y_internal)
    
    return results


def nn_config_wrapper(x):
    """Wrapper per nn_config_score senza categorical_dims."""
    return nn_config_score(x, None)


def main():
    print("=" * 70)
    print("TEST: Covariance Matrix su Benchmark SOLO CATEGORIALE")
    print("=" * 70)
    
    n_seeds = 15
    
    # Benchmark 1: NN Config (6 dimensioni categoriche)
    nn_dim = 6
    nn_cat_dims = [
        (0, 4),  # n_layers
        (1, 4),  # activation
        (2, 3),  # optimizer
        (3, 6),  # dropout
        (4, 5),  # batch_size
        (5, 4),  # lr_schedule
    ]
    
    # Benchmark 2: OneMax (8 dimensioni, 4 scelte ciascuna)
    onemax_dim = 8
    onemax_cat_dims = [(i, 4) for i in range(onemax_dim)]
    
    # Benchmark 3: Trap (6 dimensioni, 4 scelte ciascuna)
    trap_dim = 6
    trap_cat_dims = [(i, 4) for i in range(trap_dim)]
    
    benchmarks = [
        (nn_config_wrapper, "NN Config", nn_dim, 150, nn_cat_dims),
        (lambda x: onemax_categorical(x, 4), "OneMax 8D", onemax_dim, 100, onemax_cat_dims),
        (lambda x: trap_categorical(x, 4), "Trap 6D", trap_dim, 100, trap_cat_dims),
    ]
    
    results_table = []
    
    print(f"\nEsecuzione con {n_seeds} seeds per configurazione...")
    print("-" * 70)
    
    for func, name, dim, budget, cat_dims in benchmarks:
        print(f"  Testing {name}...", end=" ", flush=True)
        
        # COV ON
        results_cov = run_benchmark_with_sampler(func, name, dim, budget, n_seeds, cat_dims, use_cov=True)
        
        # COV OFF (Gaussian)
        results_gauss = run_benchmark_with_sampler(func, name, dim, budget, n_seeds, cat_dims, use_cov=False)
        
        mean_cov = np.mean(results_cov)
        std_cov = np.std(results_cov)
        mean_gauss = np.mean(results_gauss)
        std_gauss = np.std(results_gauss)
        
        # Wins
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
    
    # Stampa risultati
    print("\n" + "=" * 70)
    print("RISULTATI")
    print("=" * 70)
    
    print(f"\n{'Benchmark':<15} {'COV':>10} {'GAUSS':>10} {'Δ':>8} {'Wins COV':>10}")
    print("-" * 55)
    
    for r in results_table:
        delta = r['mean_cov'] - r['mean_gauss']
        delta_pct = delta / abs(r['mean_gauss']) * 100 if r['mean_gauss'] != 0 else 0
        print(f"{r['name']:<15} {r['mean_cov']:>10.4f} {r['mean_gauss']:>10.4f} {delta_pct:>+7.1f}% {r['wins_cov']:>7}/{n_seeds}")
    
    print("\n" + "=" * 70)
    print("ANALISI")
    print("=" * 70)
    
    total_wins_cov = sum(r['wins_cov'] for r in results_table)
    total_runs = len(results_table) * n_seeds
    
    print(f"\n  Wins Covariance: {total_wins_cov}/{total_runs}")
    print(f"  Wins Gaussian:   {total_runs - total_wins_cov}/{total_runs}")
    
    avg_delta = np.mean([r['mean_cov'] - r['mean_gauss'] for r in results_table])
    print(f"\n  Delta medio: {avg_delta:+.4f}")
    
    if total_wins_cov > total_runs / 2:
        print("\n  Conclusione: Covariance AIUTA anche su categorici puri")
    elif total_wins_cov < total_runs / 2:
        print("\n  Conclusione: Covariance è INUTILE/DANNOSA su categorici puri")
    else:
        print("\n  Conclusione: Nessuna differenza significativa")


if __name__ == "__main__":
    main()
