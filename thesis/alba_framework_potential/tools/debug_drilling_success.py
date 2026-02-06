#!/usr/bin/env python3
"""
Debug approfondito: perché drilling ha 0 successi?
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.drilling import DrillingOptimizer


def debug_drilling_success():
    """Debug step-by-step del drilling."""
    print("="*70)
    print("DEBUG: Perché 0 successi nel drilling?")
    print("="*70)
    
    rng = np.random.default_rng(42)
    dim = 5
    bounds = [(0, 1)] * dim
    
    # Funzione semplice: sfera
    def sphere(x):
        return np.sum((x - 0.5)**2)
    
    start_x = np.array([0.6] * dim)
    start_y = sphere(start_x)
    
    print(f"Start: x = {start_x}")
    print(f"Start: y = {start_y:.6f}")
    print(f"Ottimo: x = [0.5]*5, y = 0.0")
    print()
    
    drill = DrillingOptimizer(start_x, start_y, initial_sigma=0.1, bounds=bounds)
    
    print(f"sigma_init = {drill.sigma}")
    print(f"best_y_init = {drill.best_y}")
    print()
    
    print("-"*50)
    print("Step-by-step:")
    
    for i in range(10):
        x_new = drill.ask(rng)
        y_new = sphere(x_new)
        
        is_better = y_new < drill.best_y
        
        print(f"\nStep {i+1}:")
        print(f"  x_new = {x_new}")
        print(f"  y_new = {y_new:.6f}")
        print(f"  best_y = {drill.best_y:.6f}")
        print(f"  y_new < best_y? {is_better}")
        
        should_continue = drill.tell(x_new, y_new)
        
        print(f"  -> sigma dopo = {drill.sigma:.6f}")
        print(f"  -> best_y dopo = {drill.best_y:.6f}")
        print(f"  -> continue = {should_continue}")
        
        if not should_continue:
            print("  STOP!")
            break


def debug_with_perfect_function():
    """Debug con funzione dove ogni step è garantito migliore."""
    print("\n" + "="*70)
    print("DEBUG: Funzione con miglioramento garantito")
    print("="*70)
    
    rng = np.random.default_rng(42)
    dim = 5
    bounds = [(0, 1)] * dim
    
    start_x = np.array([0.6] * dim)
    start_y = 100.0  # Valore alto
    
    drill = DrillingOptimizer(start_x, start_y, initial_sigma=0.1, bounds=bounds)
    
    # Funzione che SEMPRE migliora (monotona decrescente nel tempo)
    step = [0]
    def always_better(x):
        step[0] += 1
        return 100.0 - step[0] * 0.1  # Decresce di 0.1 ogni volta
    
    n_success = 0
    for i in range(20):
        x_new = drill.ask(rng)
        y_new = always_better(x_new)
        
        is_better = y_new < drill.best_y
        if is_better:
            n_success += 1
        
        should_continue = drill.tell(x_new, y_new)
        
        print(f"Step {i+1}: y={y_new:.2f}, best={drill.best_y:.2f}, success={is_better}, sigma={drill.sigma:.4f}")
        
        if not should_continue:
            print("STOP!")
            break
    
    print(f"\nTotale successi: {n_success}")


def analyze_success_probability():
    """Calcola la probabilità di successo teorica."""
    print("\n" + "="*70)
    print("ANALISI: Probabilità di successo")
    print("="*70)
    
    # Per una sfera, la probabilità di miglioramento dipende da:
    # - Distanza dall'ottimo
    # - Sigma
    # - Dimensionalità
    
    dim = 5
    sigma = 0.1
    dist_to_opt = 0.1 * np.sqrt(dim)  # Distanza da [0.6]*5 a [0.5]*5
    
    print(f"Distanza dall'ottimo: {dist_to_opt:.4f}")
    print(f"Sigma: {sigma}")
    print(f"Dimensionalità: {dim}")
    
    # In alta dimensione, un passo random raramente migliora
    # La probabilità approssimata è:
    # P(success) ≈ 1/2 * (1 - sigma^2 * dim / (2 * dist^2))
    # 
    # Per sfera, miglioramento richiede che il nuovo punto sia più vicino
    
    print("\nSimulazione Monte Carlo:")
    
    rng = np.random.default_rng(42)
    n_trials = 10000
    
    center = np.array([0.5] * dim)
    start = np.array([0.6] * dim)
    start_dist = np.sum((start - center)**2)
    
    n_better = 0
    for _ in range(n_trials):
        noise = rng.normal(0, sigma, dim)
        x_new = start + noise
        new_dist = np.sum((x_new - center)**2)
        if new_dist < start_dist:
            n_better += 1
    
    p_success = n_better / n_trials
    print(f"P(miglioramento) ≈ {p_success:.3f} ({p_success*100:.1f}%)")
    
    if p_success < 0.3:
        print("\n⚠ Probabilità di successo bassa!")
        print("  Con sigma=0.1 e dim=5, è difficile migliorare random.")
        print("  Dopo pochi fallimenti, stagnation_counter > 5 → STOP")


def test_stagnation_counter():
    """Verifica il meccanismo di stagnation."""
    print("\n" + "="*70)
    print("ANALISI: Stagnation Counter")
    print("="*70)
    
    rng = np.random.default_rng(42)
    dim = 5
    bounds = [(0, 1)] * dim
    
    def sphere(x):
        return np.sum((x - 0.5)**2)
    
    start_x = np.array([0.6] * dim)
    start_y = sphere(start_x)
    
    drill = DrillingOptimizer(start_x, start_y, initial_sigma=0.1, bounds=bounds)
    
    print(f"max_stagnation = 5 (hardcoded in tell())")
    print(f"max_steps = {drill.max_steps}")
    print()
    
    for i in range(20):
        x_new = drill.ask(rng)
        y_new = sphere(x_new)
        
        is_success = y_new < drill.best_y
        
        should_continue = drill.tell(x_new, y_new)
        
        print(f"Step {i+1}: success={is_success}, stagnation={drill.stagnation_counter}, continue={should_continue}")
        
        if not should_continue:
            print(f"\nSTOP dopo {i+1} step!")
            print(f"  Motivo: stagnation_counter ({drill.stagnation_counter}) > 5" if drill.stagnation_counter > 5 else 
                  f"  Motivo: altro (sigma={drill.sigma:.2e})")
            break


if __name__ == "__main__":
    debug_drilling_success()
    debug_with_perfect_function()
    analyze_success_probability()
    test_stagnation_counter()
