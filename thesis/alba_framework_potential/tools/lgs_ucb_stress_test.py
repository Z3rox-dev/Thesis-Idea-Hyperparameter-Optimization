#!/usr/bin/env python3
"""
LGS & UCB Selection Stress Test

Verifica scientifica che:
1. LGS predice bene (correlazione mu vs valore reale)
2. UCB seleziona punti effettivamente buoni (regret analysis)
3. Non ci sono bug nel processo di selezione

Approccio:
- Per ogni iterazione, generiamo candidati
- Predichiamo con LGS (mu, sigma)
- Valutiamo TUTTI i candidati con la funzione reale
- Confrontiamo: il punto scelto da UCB è tra i migliori?
"""

import numpy as np
import sys
import os

sys.path.insert(0, '/mnt/workspace/thesis')
os.chdir('/mnt/workspace/thesis/alba_framework_potential')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.cube import Cube
from alba_framework_potential.lgs import fit_lgs_model, predict_bayesian
from alba_framework_potential.candidates import MixtureCandidateGenerator
from alba_framework_potential.acquisition import UCBSoftmaxSelector

from scipy.stats import spearmanr, pearsonr

# =============================================================================
# Test Functions
# =============================================================================

def sphere(x):
    return float(np.sum(x**2))

def rosenbrock(x):
    x = np.array(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

def rastrigin(x):
    x = np.array(x)
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

# =============================================================================
# Test 1: LGS Prediction Quality
# =============================================================================

def test_lgs_prediction_quality():
    """
    Verifica che LGS predica bene il ranking dei candidati.
    
    Metrica: Spearman correlation tra mu predetto e valore reale
    """
    print("=" * 70)
    print("TEST 1: LGS Prediction Quality (Spearman Correlation)")
    print("=" * 70)
    
    DIM = 5
    N_CANDIDATES = 50
    BUDGET = 100
    
    results = []
    
    for func, name in [(sphere, "Sphere"), (rosenbrock, "Rosenbrock"), (rastrigin, "Rastrigin")]:
        correlations = []
        
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            
            # Simula un cubo con dati
            bounds = [(-5.0, 5.0)] * DIM
            cube = Cube(bounds=bounds, parent=None, depth=0)
            
            # Aggiungi osservazioni random per costruire il modello
            for _ in range(30):
                x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
                y = func(x)
                # ALBA usa score interno (negato per minimization)
                score = -y
                cube.add_observation(x, score, gamma=-100)  # gamma basso per accettare tutti
            
            # Fit LGS model
            gamma = np.percentile([s for _, s in cube.tested_pairs], 80)
            cube.fit_lgs_model(gamma, DIM, rng)
            
            if cube.lgs_model is None:
                print(f"  {name} seed {seed}: LGS model failed to fit")
                continue
            
            # Genera candidati
            generator = MixtureCandidateGenerator()
            candidates = generator.generate(cube, DIM, rng, N_CANDIDATES)
            
            # Predici con LGS
            mu, sigma = cube.predict_bayesian(candidates)
            
            # Valuta tutti i candidati con funzione reale
            real_values = np.array([func(c) for c in candidates])
            real_scores = -real_values  # Negato come fa ALBA
            
            # Calcola correlazione (mu vs real_scores)
            # Buon LGS: alta correlazione positiva
            if np.std(mu) > 1e-9 and np.std(real_scores) > 1e-9:
                corr, pval = spearmanr(mu, real_scores)
                correlations.append(corr)
            else:
                correlations.append(0.0)
        
        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        
        status = "✓ GOOD" if mean_corr > 0.3 else ("⚠ WEAK" if mean_corr > 0 else "✗ BAD")
        print(f"  {name:<12}: Spearman ρ = {mean_corr:.3f} ± {std_corr:.3f}  {status}")
        results.append((name, mean_corr, std_corr))
    
    return results

# =============================================================================
# Test 2: UCB Selection Regret
# =============================================================================

def test_ucb_selection_regret():
    """
    Verifica che UCB selezioni punti buoni.
    
    Metrica: Regret = (valore scelto - valore migliore tra candidati)
    Normalized Regret = Regret / (max - min)
    """
    print("\n" + "=" * 70)
    print("TEST 2: UCB Selection Regret Analysis")
    print("=" * 70)
    
    DIM = 5
    N_CANDIDATES = 50
    
    results = []
    
    for func, name in [(sphere, "Sphere"), (rosenbrock, "Rosenbrock")]:
        regrets = []
        rank_positions = []
        
        for seed in [42, 123, 456, 789, 1000]:
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            
            bounds = [(-5.0, 5.0)] * DIM
            cube = Cube(bounds=bounds, parent=None, depth=0)
            
            # Aggiungi osservazioni
            for _ in range(40):
                x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
                y = func(x)
                score = -y
                cube.add_observation(x, score, gamma=-100)
            
            gamma = np.percentile([s for _, s in cube.tested_pairs], 80)
            cube.fit_lgs_model(gamma, DIM, rng)
            
            if cube.lgs_model is None:
                continue
            
            # Genera candidati
            generator = MixtureCandidateGenerator()
            candidates = generator.generate(cube, DIM, rng, N_CANDIDATES)
            
            # Predici
            mu, sigma = cube.predict_bayesian(candidates)
            
            # UCB selection
            selector = UCBSoftmaxSelector()
            selected_idx = selector.select(mu, sigma, rng, novelty_weight=0.4)
            
            # Valuta tutti i candidati
            real_values = np.array([func(c) for c in candidates])
            
            # Regret: quanto peggio è il punto scelto rispetto al migliore?
            best_value = np.min(real_values)
            selected_value = real_values[selected_idx]
            
            # Normalized regret
            value_range = np.max(real_values) - np.min(real_values)
            if value_range > 1e-9:
                norm_regret = (selected_value - best_value) / value_range
            else:
                norm_regret = 0.0
            
            regrets.append(norm_regret)
            
            # Rank position: dove si posiziona il punto scelto?
            sorted_indices = np.argsort(real_values)
            rank = np.where(sorted_indices == selected_idx)[0][0]
            rank_percentile = rank / len(candidates) * 100
            rank_positions.append(rank_percentile)
        
        mean_regret = np.mean(regrets)
        mean_rank = np.mean(rank_positions)
        
        # Buon UCB: regret basso, rank basso (top %)
        status = "✓ GOOD" if mean_rank < 30 else ("⚠ OK" if mean_rank < 50 else "✗ BAD")
        print(f"  {name:<12}: Norm Regret = {mean_regret:.3f}, Avg Rank = {mean_rank:.1f}%  {status}")
        results.append((name, mean_regret, mean_rank))
    
    return results

# =============================================================================
# Test 3: Bug Hunt - NaN/Inf propagation
# =============================================================================

def test_nan_inf_propagation():
    """
    Verifica che NaN/Inf non si propaghino attraverso LGS -> UCB.
    """
    print("\n" + "=" * 70)
    print("TEST 3: NaN/Inf Propagation Check")
    print("=" * 70)
    
    DIM = 5
    rng = np.random.default_rng(42)
    bounds = [(-5.0, 5.0)] * DIM
    
    bugs_found = []
    
    # Test 3a: Candidati con NaN
    print("\n  3a. Candidates containing NaN...")
    cube = Cube(bounds=bounds, parent=None, depth=0)
    for _ in range(20):
        x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        cube.add_observation(x, -sphere(x), gamma=-100)
    cube.fit_lgs_model(-50, DIM, rng)
    
    candidates = [np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
                  np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                  np.array([np.inf, 2.0, 3.0, 4.0, 5.0])]
    
    try:
        mu, sigma = cube.predict_bayesian(candidates)
        if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma)):
            print(f"    ⚠ BUG: NaN/Inf in predictions: mu={mu}, sigma={sigma}")
            bugs_found.append("NaN in predict_bayesian output")
        else:
            print(f"    ✓ predict_bayesian handles NaN candidates gracefully")
    except Exception as e:
        print(f"    ✗ CRASH: {e}")
        bugs_found.append(f"predict_bayesian crash: {e}")
    
    # Test 3b: UCB con mu/sigma NaN
    print("\n  3b. UCB with NaN mu/sigma...")
    selector = UCBSoftmaxSelector()
    
    test_cases = [
        ("NaN in mu", np.array([np.nan, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])),
        ("NaN in sigma", np.array([1.0, 2.0, 3.0]), np.array([np.nan, 1.0, 1.0])),
        ("Inf in mu", np.array([np.inf, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])),
        ("All NaN mu", np.array([np.nan, np.nan, np.nan]), np.array([1.0, 1.0, 1.0])),
        ("All zero sigma", np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0])),
    ]
    
    for name, mu, sigma in test_cases:
        try:
            idx = selector.select(mu, sigma, rng, novelty_weight=0.4)
            if not isinstance(idx, (int, np.integer)) or idx < 0 or idx >= len(mu):
                print(f"    ✗ BUG {name}: Invalid index {idx}")
                bugs_found.append(f"UCB invalid index: {name}")
            else:
                print(f"    ✓ {name}: returned valid index {idx}")
        except Exception as e:
            print(f"    ✗ CRASH {name}: {e}")
            bugs_found.append(f"UCB crash: {name} -> {e}")
    
    # Test 3c: Extreme novelty_weight
    print("\n  3c. Extreme novelty_weight values...")
    mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sigma = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    for nw in [0.0, 1.0, 10.0, -1.0, np.inf, np.nan]:
        try:
            idx = selector.select(mu, sigma, rng, novelty_weight=nw)
            if 0 <= idx < len(mu):
                print(f"    ✓ novelty_weight={nw}: index {idx}")
            else:
                print(f"    ✗ novelty_weight={nw}: invalid index {idx}")
                bugs_found.append(f"Invalid index with nw={nw}")
        except Exception as e:
            print(f"    ✗ novelty_weight={nw} CRASH: {e}")
            bugs_found.append(f"Crash with nw={nw}")
    
    return bugs_found

# =============================================================================
# Test 4: Selection Consistency
# =============================================================================

def test_selection_consistency():
    """
    Verifica che UCB sia deterministico con stesso seed e che
    preferisca punti con alto mu+beta*sigma.
    """
    print("\n" + "=" * 70)
    print("TEST 4: UCB Selection Consistency")
    print("=" * 70)
    
    selector = UCBSoftmaxSelector(beta_multiplier=2.0, softmax_temperature=3.0)
    
    # Test 4a: Determinismo
    print("\n  4a. Determinism with same seed...")
    mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sigma = np.array([0.5, 0.3, 0.2, 0.1, 0.05])
    
    selections = []
    for _ in range(10):
        rng = np.random.default_rng(42)  # Same seed
        idx = selector.select(mu, sigma, rng, novelty_weight=0.4)
        selections.append(idx)
    
    if len(set(selections)) == 1:
        print(f"    ✓ Deterministic: always selects index {selections[0]}")
    else:
        print(f"    ⚠ Non-deterministic: selections = {selections}")
    
    # Test 4b: Preferenza per alto UCB score
    print("\n  4b. Preference for high UCB score...")
    # Crea scenario dove un punto ha chiaramente il miglior UCB
    mu = np.array([0.0, 0.0, 0.0, 10.0, 0.0])  # punto 3 ha mu molto alto
    sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    selections = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        idx = selector.select(mu, sigma, rng, novelty_weight=0.4)
        selections.append(idx)
    
    # Dovrebbe preferire indice 3 (mu=10)
    counts = {i: selections.count(i) for i in range(5)}
    most_selected = max(counts, key=counts.get)
    
    print(f"    Selection counts: {counts}")
    if most_selected == 3:
        print(f"    ✓ Correctly prefers high-mu point (idx=3 selected {counts[3]}%)")
    else:
        print(f"    ⚠ Unexpected preference: idx={most_selected} selected most often")
    
    # Test 4c: Exploration vs Exploitation
    print("\n  4c. Exploration (high sigma) influence...")
    mu = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # Tutti uguali
    sigma = np.array([0.1, 0.1, 0.1, 0.1, 10.0])  # Ultimo ha alta incertezza
    
    selections = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        idx = selector.select(mu, sigma, rng, novelty_weight=0.4)
        selections.append(idx)
    
    counts = {i: selections.count(i) for i in range(5)}
    print(f"    Selection counts: {counts}")
    
    # Con novelty_weight=0.4, beta=0.8, punto 4 dovrebbe avere UCB più alto
    if counts[4] > 50:
        print(f"    ✓ Correctly explores high-uncertainty point (idx=4 selected {counts[4]}%)")
    else:
        print(f"    ⚠ Low exploration: idx=4 selected only {counts[4]}%")

# =============================================================================
# Test 5: End-to-End ALBA Selection Quality
# =============================================================================

def test_e2e_alba_selection():
    """
    Test end-to-end: ALBA dovrebbe migliorare nel tempo.
    Metrica: regret cumulativo dovrebbe diminuire.
    """
    print("\n" + "=" * 70)
    print("TEST 5: End-to-End ALBA Selection Quality")
    print("=" * 70)
    
    DIM = 5
    BUDGET = 100
    
    for func, name, optimal in [(sphere, "Sphere", 0.0), (rosenbrock, "Rosenbrock", 0.0)]:
        print(f"\n  {name} (optimum = {optimal}):")
        
        bounds = [(-2.0, 2.0)] * DIM
        
        regrets_over_time = []
        
        for seed in [42, 123, 456]:
            opt = ALBA(bounds=bounds, maximize=False, total_budget=BUDGET, seed=seed)
            
            iteration_regrets = []
            best_so_far = float('inf')
            
            for i in range(BUDGET):
                x = opt.ask()
                y = func(x)
                opt.tell(x, y)
                
                if y < best_so_far:
                    best_so_far = y
                
                # Regret = best_so_far - optimal
                regret = best_so_far - optimal
                iteration_regrets.append(regret)
            
            regrets_over_time.append(iteration_regrets)
        
        # Calcola regret medio
        regrets_arr = np.array(regrets_over_time)
        mean_regrets = regrets_arr.mean(axis=0)
        
        # Verifica che regret diminuisca
        early_regret = mean_regrets[:20].mean()
        late_regret = mean_regrets[-20:].mean()
        improvement = (early_regret - late_regret) / (early_regret + 1e-9) * 100
        
        print(f"    Early regret (iter 1-20):  {early_regret:.4f}")
        print(f"    Late regret (iter 80-100): {late_regret:.4f}")
        print(f"    Improvement: {improvement:.1f}%")
        
        if improvement > 50:
            print(f"    ✓ GOOD: Significant improvement over time")
        elif improvement > 20:
            print(f"    ⚠ OK: Moderate improvement")
        else:
            print(f"    ✗ BAD: Poor improvement - possible bug?")

# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("LGS & UCB SELECTION STRESS TEST")
    print("=" * 70)
    
    # Run all tests
    lgs_results = test_lgs_prediction_quality()
    ucb_results = test_ucb_selection_regret()
    bugs = test_nan_inf_propagation()
    test_selection_consistency()
    test_e2e_alba_selection()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nLGS Prediction Quality:")
    for name, corr, std in lgs_results:
        status = "✓" if corr > 0.3 else "⚠"
        print(f"  {status} {name}: ρ = {corr:.3f}")
    
    print("\nUCB Selection Regret:")
    for name, regret, rank in ucb_results:
        status = "✓" if rank < 30 else "⚠"
        print(f"  {status} {name}: Avg Rank = {rank:.1f}%")
    
    if bugs:
        print(f"\n⚠ BUGS FOUND: {len(bugs)}")
        for bug in bugs:
            print(f"  - {bug}")
    else:
        print("\n✓ No bugs found in NaN/Inf handling")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
