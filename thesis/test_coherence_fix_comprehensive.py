#!/usr/bin/env python3
"""
Test completo delle fix al modulo Coherence su molte funzioni benchmark.
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from alba_framework_potential.optimizer import ALBA

# =============================================================================
# FUNZIONI BENCHMARK
# =============================================================================

def sphere(x):
    """Sphere: semplice, convessa, unimodale."""
    return sum(xi**2 for xi in x)

def rosenbrock(x):
    """Rosenbrock: valle curva, difficile da seguire."""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
               for i in range(len(x)-1))

def rastrigin(x):
    """Rastrigin: altamente multimodale."""
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

def ackley(x):
    """Ackley: multimodale con molti minimi locali."""
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(2 * np.pi * xi) for xi in x)
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

def griewank(x):
    """Griewank: multimodale ma regolare."""
    sum_sq = sum(xi**2 for xi in x)
    prod_cos = np.prod([np.cos(xi / np.sqrt(i+1)) for i, xi in enumerate(x)])
    return sum_sq / 4000 - prod_cos + 1

def levy(x):
    """Levy: multimodale con struttura regolare."""
    n = len(x)
    w = [1 + (xi - 1) / 4 for xi in x]
    term1 = np.sin(np.pi * w[0])**2
    term2 = sum((wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2) for wi in w[:-1])
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def schwefel(x):
    """Schwefel: minimo globale lontano dall'origine."""
    n = len(x)
    return 418.9829 * n - sum(xi * np.sin(np.sqrt(abs(xi))) for xi in x)

def styblinski_tang(x):
    """Styblinski-Tang: multimodale semplice."""
    return sum(xi**4 - 16*xi**2 + 5*xi for xi in x) / 2

def zakharov(x):
    """Zakharov: unimodale ma con termini di ordine superiore."""
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(0.5 * (i+1) * xi for i, xi in enumerate(x))
    return sum1 + sum2**2 + sum2**4

def dixon_price(x):
    """Dixon-Price: valle curva come Rosenbrock."""
    n = len(x)
    term1 = (x[0] - 1)**2
    term2 = sum((i+1) * (2*x[i]**2 - x[i-1])**2 for i in range(1, n))
    return term1 + term2


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(func, name, dim, budget, n_seeds=10, bounds=(-5.0, 5.0)):
    """Esegue benchmark con PF ON e OFF."""
    results_on = []
    results_off = []
    
    for seed in range(n_seeds):
        # PF ON
        opt = ALBA(
            bounds=[(bounds[0], bounds[1]) for _ in range(dim)],
            maximize=False,
            seed=100 + seed,
            total_budget=budget,
            use_potential_field=True
        )
        for _ in range(budget):
            x = opt.ask()
            opt.tell(x, func(x))
        results_on.append(-opt.best_y_internal)
        
        # PF OFF
        opt = ALBA(
            bounds=[(bounds[0], bounds[1]) for _ in range(dim)],
            maximize=False,
            seed=100 + seed,
            total_budget=budget,
            use_potential_field=False
        )
        for _ in range(budget):
            x = opt.ask()
            opt.tell(x, func(x))
        results_off.append(-opt.best_y_internal)
    
    return results_on, results_off


def main():
    print("=" * 70)
    print("TEST COMPLETO FIX COHERENCE - Benchmark su Multiple Funzioni")
    print("=" * 70)
    
    # Configurazione benchmark
    benchmarks = [
        # (funzione, nome, dimensione, budget, bounds)
        (sphere, "Sphere", 5, 200, (-5, 5)),
        (sphere, "Sphere", 10, 300, (-5, 5)),
        (rosenbrock, "Rosenbrock", 5, 300, (-5, 5)),
        (rosenbrock, "Rosenbrock", 10, 400, (-5, 5)),
        (rastrigin, "Rastrigin", 5, 300, (-5, 5)),
        (rastrigin, "Rastrigin", 10, 400, (-5, 5)),
        (ackley, "Ackley", 5, 300, (-5, 5)),
        (ackley, "Ackley", 10, 400, (-5, 5)),
        (griewank, "Griewank", 5, 300, (-5, 5)),
        (griewank, "Griewank", 10, 400, (-5, 5)),
        (levy, "Levy", 5, 300, (-5, 5)),
        (levy, "Levy", 10, 400, (-5, 5)),
        (styblinski_tang, "StyblinskiTang", 5, 300, (-5, 5)),
        (styblinski_tang, "StyblinskiTang", 10, 400, (-5, 5)),
        (zakharov, "Zakharov", 5, 300, (-5, 5)),
        (zakharov, "Zakharov", 10, 400, (-5, 5)),
        (dixon_price, "DixonPrice", 5, 300, (-5, 5)),
        (dixon_price, "DixonPrice", 10, 400, (-5, 5)),
    ]
    
    n_seeds = 10
    results_table = []
    
    print(f"\nEsecuzione con {n_seeds} seeds per configurazione...")
    print("-" * 70)
    
    for func, name, dim, budget, bounds in benchmarks:
        print(f"  Testing {name} {dim}D...", end=" ", flush=True)
        
        results_on, results_off = run_benchmark(
            func, name, dim, budget, n_seeds, bounds
        )
        
        mean_on = np.mean(results_on)
        std_on = np.std(results_on)
        mean_off = np.mean(results_off)
        std_off = np.std(results_off)
        
        # Calcola wins
        wins = sum(1 for a, b in zip(results_on, results_off) if a < b)
        ties = sum(1 for a, b in zip(results_on, results_off) if abs(a - b) < 1e-6)
        
        # Improvement %
        if mean_off != 0:
            improvement = (mean_off - mean_on) / abs(mean_off) * 100
        else:
            improvement = 0
        
        results_table.append({
            'name': f"{name} {dim}D",
            'mean_on': mean_on,
            'std_on': std_on,
            'mean_off': mean_off,
            'std_off': std_off,
            'wins': wins,
            'ties': ties,
            'improvement': improvement
        })
        
        print(f"done (PF: {mean_on:.2f}, Base: {mean_off:.2f})")
    
    # Stampa risultati
    print("\n" + "=" * 70)
    print("RISULTATI")
    print("=" * 70)
    
    print(f"\n{'Benchmark':<20} {'PF ON':>12} {'PF OFF':>12} {'Δ%':>8} {'Wins':>8} {'Status':>8}")
    print("-" * 70)
    
    total_wins = 0
    total_losses = 0
    total_neutral = 0
    
    for r in results_table:
        imp = r['improvement']
        wins = r['wins']
        
        if imp > 5:
            status = "✅ WIN"
            total_wins += 1
        elif imp < -5:
            status = "❌ LOSS"
            total_losses += 1
        else:
            status = "➖ TIE"
            total_neutral += 1
        
        print(f"{r['name']:<20} {r['mean_on']:>12.2f} {r['mean_off']:>12.2f} {imp:>+7.1f}% {wins:>5}/10 {status:>8}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Vittorie PF:    {total_wins}/{len(results_table)}")
    print(f"  Sconfitte PF:   {total_losses}/{len(results_table)}")
    print(f"  Pareggi:        {total_neutral}/{len(results_table)}")
    
    avg_improvement = np.mean([r['improvement'] for r in results_table])
    print(f"\n  Miglioramento medio: {avg_improvement:+.1f}%")
    
    # Breakdown per dimensione
    results_5d = [r for r in results_table if "5D" in r['name']]
    results_10d = [r for r in results_table if "10D" in r['name']]
    
    avg_5d = np.mean([r['improvement'] for r in results_5d])
    avg_10d = np.mean([r['improvement'] for r in results_10d])
    
    print(f"\n  Media 5D:  {avg_5d:+.1f}%")
    print(f"  Media 10D: {avg_10d:+.1f}%")
    
    # Verdetto finale
    print("\n" + "=" * 70)
    if total_wins > total_losses:
        print("🎉 VERDETTO: Il Potential Field AIUTA complessivamente!")
    elif total_wins < total_losses:
        print("⚠️ VERDETTO: Il Potential Field DANNEGGIA complessivamente!")
    else:
        print("➖ VERDETTO: Il Potential Field è NEUTRO complessivamente.")
    print("=" * 70)


if __name__ == "__main__":
    main()
