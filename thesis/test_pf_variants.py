#!/usr/bin/env python3
"""
Test tutte le varianti di Potential Field per trovare la configurazione ottimale.

Varianti testate:
A) Threshold bassa: (coh - 0.3) * 2.0
B) Weight alto: potential_weight = 3.0
C) No gating: coherence_scale = 1.0 sempre
D) Curva aggressiva: coh ** 0.5
E) Combo A+B: threshold bassa + weight alto
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# Importo le classi base
from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.leaf_selection import PotentialAwareLeafSelector
from alba_framework_potential.coherence import CoherenceTracker


# ============================================================================
# VARIANTI DEL LEAF SELECTOR
# ============================================================================

class VariantA_LowThreshold(PotentialAwareLeafSelector):
    """Threshold bassa: PF attivo già da coherence 0.3"""
    
    def _compute_coherence_scale(self, global_coh: float) -> float:
        # (coh - 0.3) * 2.0 → coh 0.3 = 0%, coh 0.8 = 100%
        return max(0.0, min(1.0, (global_coh - 0.3) * 2.0))


class VariantB_HighWeight(PotentialAwareLeafSelector):
    """Weight alto: bonus 3x"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.potential_weight = 3.0


class VariantC_NoGating(PotentialAwareLeafSelector):
    """No gating: PF sempre al 100%"""
    
    def _compute_coherence_scale(self, global_coh: float) -> float:
        return 1.0


class VariantD_AggressiveCurve(PotentialAwareLeafSelector):
    """Curva aggressiva: sqrt(coh)"""
    
    def _compute_coherence_scale(self, global_coh: float) -> float:
        # sqrt rende la curva più "piatta in alto"
        # coh 0.25 → 0.5, coh 0.5 → 0.71, coh 0.8 → 0.89
        return np.sqrt(max(0.0, global_coh))


class VariantE_ComboAB(PotentialAwareLeafSelector):
    """Combo: threshold bassa + weight alto"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.potential_weight = 3.0
    
    def _compute_coherence_scale(self, global_coh: float) -> float:
        return max(0.0, min(1.0, (global_coh - 0.3) * 2.0))


# Monkey-patch per iniettare la variante
def create_alba_with_variant(variant_class, bounds, seed, total_budget):
    """Crea ALBA con una variante specifica del leaf selector."""
    
    opt = ALBA(
        bounds=bounds,
        seed=seed,
        maximize=False,
        total_budget=total_budget,
        use_potential_field=True,
        use_coherence_gating=True,
    )
    
    # Sostituisco il leaf selector con la variante
    if variant_class is not None:
        # Creo nuovo selector con stessi parametri
        old_selector = opt._leaf_selector
        new_selector = variant_class(
            tracker=opt._coherence_tracker,
            potential_weight=getattr(old_selector, 'potential_weight', 1.0),
        )
        opt._leaf_selector = new_selector
    
    return opt


# ============================================================================
# FUNZIONI TEST
# ============================================================================

def sphere(x):
    return float(np.sum(np.array(x)**2))

def rosenbrock(x):
    x = np.array(x)
    return float(sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1)))

def rastrigin(x):
    x = np.array(x)
    return float(10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x))

def levy(x):
    x = np.array(x)
    w = 1 + (x - 1) / 4
    return float(np.sin(np.pi*w[0])**2 + 
                 sum((w[i]-1)**2 * (1 + 10*np.sin(np.pi*w[i]+1)**2) for i in range(len(w)-1)) +
                 (w[-1]-1)**2 * (1 + np.sin(2*np.pi*w[-1])**2))


# ============================================================================
# BENCHMARK
# ============================================================================

def run_comparison(func, bounds, n_trials, seed, variants):
    """Confronta tutte le varianti su una funzione."""
    
    results = {}
    
    # Baseline COV (no PF)
    opt = ALBA(bounds=bounds, seed=seed, maximize=False, total_budget=n_trials,
               use_potential_field=False, use_coherence_gating=True)
    best = float('inf')
    for _ in range(n_trials):
        x = opt.ask()
        y = func(x)
        opt.tell(x, y)
        best = min(best, y)
    results['COV'] = best
    
    # Varianti PF
    for name, variant_class in variants.items():
        opt = create_alba_with_variant(variant_class, bounds, seed, n_trials)
        best = float('inf')
        for _ in range(n_trials):
            x = opt.ask()
            y = func(x)
            opt.tell(x, y)
            best = min(best, y)
        results[name] = best
    
    return results


def main():
    print("=" * 80)
    print("  TEST VARIANTI POTENTIAL FIELD")
    print("=" * 80)
    
    VARIANTS = {
        'PF_orig': None,  # PotentialAwareLeafSelector originale
        'A_LowThr': VariantA_LowThreshold,
        'B_HighWt': VariantB_HighWeight,
        'C_NoGate': VariantC_NoGating,
        'D_Aggres': VariantD_AggressiveCurve,
        'E_Combo': VariantE_ComboAB,
    }
    
    FUNCTIONS = {
        'sphere': (sphere, [(-2.0, 2.0)] * 8),
        'rosenbrock': (rosenbrock, [(-2.0, 2.0)] * 8),
        'rastrigin': (rastrigin, [(-3.0, 3.0)] * 8),
        'levy': (levy, [(-5.0, 5.0)] * 8),
    }
    
    N_TRIALS = 250
    N_SEEDS = 10
    
    print(f"\nConfig: {N_TRIALS} trials, {N_SEEDS} seeds, 8D\n")
    
    # Header
    header = f"{'Func':<12} {'Seed':<6}"
    for v in ['COV'] + list(VARIANTS.keys()):
        header += f" {v:<9}"
    header += "  Winner"
    print(header)
    print("-" * len(header))
    
    # Contatori vittorie
    wins = {k: 0 for k in ['COV'] + list(VARIANTS.keys())}
    all_results = []
    
    for func_name, (func, bounds) in FUNCTIONS.items():
        for seed_off in range(N_SEEDS):
            seed = 42 + seed_off * 1000
            
            results = run_comparison(func, bounds, N_TRIALS, seed, VARIANTS)
            all_results.append((func_name, seed, results))
            
            # Trova winner
            best_val = min(results.values())
            winners = [k for k, v in results.items() if abs(v - best_val) < 1e-6]
            
            if len(winners) == 1:
                wins[winners[0]] += 1
                winner_str = winners[0]
            else:
                winner_str = "TIE"
            
            # Print row
            row = f"{func_name:<12} {seed:<6}"
            for v in ['COV'] + list(VARIANTS.keys()):
                row += f" {results[v]:<9.3f}"
            row += f"  {winner_str}"
            print(row)
    
    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY - Vittorie per variante")
    print("=" * 80)
    
    total = len(FUNCTIONS) * N_SEEDS
    
    for name in ['COV'] + list(VARIANTS.keys()):
        pct = 100 * wins[name] / total
        bar = "█" * int(pct / 5)
        print(f"  {name:<12}: {wins[name]:>3}/{total} ({pct:>5.1f}%) {bar}")
    
    # Dettaglio per funzione
    print("\n" + "-" * 40)
    print("  Per funzione:")
    
    for func_name in FUNCTIONS.keys():
        func_results = [r for r in all_results if r[0] == func_name]
        func_wins = {k: 0 for k in ['COV'] + list(VARIANTS.keys())}
        
        for _, _, results in func_results:
            best_val = min(results.values())
            for k, v in results.items():
                if abs(v - best_val) < 1e-6:
                    func_wins[k] += 1
                    break  # Solo un vincitore
        
        best_variant = max(func_wins.items(), key=lambda x: x[1])
        print(f"  {func_name:<12}: best={best_variant[0]} ({best_variant[1]}/{N_SEEDS})")
    
    # Media delle performance
    print("\n" + "-" * 40)
    print("  Performance media (lower is better):")
    
    for name in ['COV'] + list(VARIANTS.keys()):
        avg = np.mean([r[2][name] for r in all_results])
        print(f"  {name:<12}: {avg:.4f}")


if __name__ == "__main__":
    main()
