#!/usr/bin/env python3
"""
Test varianti PF - Versione 2 con varianti realmente diverse.
Modifica direttamente la formula nel select().
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')

import numpy as np
from typing import List
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA
from alba_framework_potential.leaf_selection import PotentialAwareLeafSelector
from alba_framework_potential.cube import Cube


# ============================================================================
# VARIANTI CON FORMULE DIVERSE
# ============================================================================

class OriginalSelector(PotentialAwareLeafSelector):
    """Originale: (coh - 0.5) * 3.33"""
    pass  # Usa la formula originale


class LowThresholdSelector(PotentialAwareLeafSelector):
    """Threshold bassa: PF attivo da coh 0.3"""
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool,
               rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("No leaves")
        if len(leaves) == 1:
            return leaves[0]
        
        scores = []
        for c in leaves:
            ratio = c.good_ratio()
            if not np.isfinite(ratio):
                ratio = 0.5
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            model_bonus = 0.0
            
            potential_bonus = 0.0
            if self.tracker is not None:
                u = self.tracker.get_potential(c, leaves)
                if not np.isfinite(u):
                    u = 0.5
                global_coh = self.tracker.global_coherence
                # NUOVA FORMULA: (coh - 0.3) * 2.0
                coherence_scale = max(0.0, min(1.0, (global_coh - 0.3) * 2.0))
                effective_weight = self.potential_weight * coherence_scale
                potential_bonus = effective_weight * (1.0 - u)
            
            scores.append(ratio + exploration + model_bonus + potential_bonus)
        
        scores_arr = np.asarray(scores, dtype=float)
        if not np.any(np.isfinite(scores_arr)):
            return leaves[rng.integers(len(leaves))]
        
        min_score = np.nanmin(scores_arr)
        scores_arr = np.where(np.isfinite(scores_arr), scores_arr, min_score)
        scores_arr = scores_arr - scores_arr.max()
        
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        return leaves[int(rng.choice(len(leaves), p=probs))]


class NoGateSelector(PotentialAwareLeafSelector):
    """No gating: PF sempre al 100%"""
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool,
               rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("No leaves")
        if len(leaves) == 1:
            return leaves[0]
        
        scores = []
        for c in leaves:
            ratio = c.good_ratio()
            if not np.isfinite(ratio):
                ratio = 0.5
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            model_bonus = 0.0
            
            potential_bonus = 0.0
            if self.tracker is not None:
                u = self.tracker.get_potential(c, leaves)
                if not np.isfinite(u):
                    u = 0.5
                # SEMPRE 100%
                coherence_scale = 1.0
                effective_weight = self.potential_weight * coherence_scale
                potential_bonus = effective_weight * (1.0 - u)
            
            scores.append(ratio + exploration + model_bonus + potential_bonus)
        
        scores_arr = np.asarray(scores, dtype=float)
        if not np.any(np.isfinite(scores_arr)):
            return leaves[rng.integers(len(leaves))]
        
        min_score = np.nanmin(scores_arr)
        scores_arr = np.where(np.isfinite(scores_arr), scores_arr, min_score)
        scores_arr = scores_arr - scores_arr.max()
        
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        return leaves[int(rng.choice(len(leaves), p=probs))]


class HighWeightSelector(PotentialAwareLeafSelector):
    """Weight 3x"""
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool,
               rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("No leaves")
        if len(leaves) == 1:
            return leaves[0]
        
        scores = []
        for c in leaves:
            ratio = c.good_ratio()
            if not np.isfinite(ratio):
                ratio = 0.5
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            model_bonus = 0.0
            
            potential_bonus = 0.0
            if self.tracker is not None:
                u = self.tracker.get_potential(c, leaves)
                if not np.isfinite(u):
                    u = 0.5
                global_coh = self.tracker.global_coherence
                coherence_scale = max(0.0, min(1.0, (global_coh - 0.5) * 3.33))
                # WEIGHT 3x
                effective_weight = 3.0 * coherence_scale
                potential_bonus = effective_weight * (1.0 - u)
            
            scores.append(ratio + exploration + model_bonus + potential_bonus)
        
        scores_arr = np.asarray(scores, dtype=float)
        if not np.any(np.isfinite(scores_arr)):
            return leaves[rng.integers(len(leaves))]
        
        min_score = np.nanmin(scores_arr)
        scores_arr = np.where(np.isfinite(scores_arr), scores_arr, min_score)
        scores_arr = scores_arr - scores_arr.max()
        
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        return leaves[int(rng.choice(len(leaves), p=probs))]


class ComboSelector(PotentialAwareLeafSelector):
    """Combo: threshold bassa + weight 2x"""
    
    def select(self, leaves: List[Cube], dim: int, stagnating: bool,
               rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("No leaves")
        if len(leaves) == 1:
            return leaves[0]
        
        scores = []
        for c in leaves:
            ratio = c.good_ratio()
            if not np.isfinite(ratio):
                ratio = 0.5
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier
            model_bonus = 0.0
            
            potential_bonus = 0.0
            if self.tracker is not None:
                u = self.tracker.get_potential(c, leaves)
                if not np.isfinite(u):
                    u = 0.5
                global_coh = self.tracker.global_coherence
                # THRESHOLD BASSA + WEIGHT 2x
                coherence_scale = max(0.0, min(1.0, (global_coh - 0.3) * 2.0))
                effective_weight = 2.0 * coherence_scale
                potential_bonus = effective_weight * (1.0 - u)
            
            scores.append(ratio + exploration + model_bonus + potential_bonus)
        
        scores_arr = np.asarray(scores, dtype=float)
        if not np.any(np.isfinite(scores_arr)):
            return leaves[rng.integers(len(leaves))]
        
        min_score = np.nanmin(scores_arr)
        scores_arr = np.where(np.isfinite(scores_arr), scores_arr, min_score)
        scores_arr = scores_arr - scores_arr.max()
        
        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        return leaves[int(rng.choice(len(leaves), p=probs))]


# ============================================================================
# FUNZIONI
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

def run_with_selector(selector_class, func, bounds, n_trials, seed):
    """Run ALBA con un selector specifico."""
    
    opt = ALBA(
        bounds=bounds,
        seed=seed,
        maximize=False,
        total_budget=n_trials,
        use_potential_field=True,
        use_coherence_gating=True,
    )
    
    # Sostituisci il selector
    if selector_class is not None:
        opt._leaf_selector = selector_class(
            tracker=opt._coherence_tracker,
            potential_weight=1.0,
        )
    
    best = float('inf')
    for _ in range(n_trials):
        x = opt.ask()
        y = func(x)
        opt.tell(x, y)
        best = min(best, y)
    
    return best, opt._coherence_tracker.global_coherence


def main():
    print("=" * 85)
    print("  TEST VARIANTI PF - Versione 2")
    print("=" * 85)
    
    VARIANTS = {
        'COV': None,  # use_potential_field=False
        'PF_orig': OriginalSelector,
        'LowThr': LowThresholdSelector,
        'NoGate': NoGateSelector,
        'HighWt': HighWeightSelector,
        'Combo': ComboSelector,
    }
    
    FUNCTIONS = {
        'sphere': (sphere, [(-2.0, 2.0)] * 8),
        'rosenbrock': (rosenbrock, [(-2.0, 2.0)] * 8),
        'rastrigin': (rastrigin, [(-3.0, 3.0)] * 8),
        'levy': (levy, [(-5.0, 5.0)] * 8),
    }
    
    N_TRIALS = 300
    N_SEEDS = 15
    
    print(f"\nConfig: {N_TRIALS} trials, {N_SEEDS} seeds, 8D")
    print(f"Varianti: {list(VARIANTS.keys())}\n")
    
    # Contatori
    wins = {k: 0 for k in VARIANTS.keys()}
    totals = {k: [] for k in VARIANTS.keys()}
    
    for func_name, (func, bounds) in FUNCTIONS.items():
        print(f"\n{'='*40}")
        print(f"  {func_name.upper()}")
        print(f"{'='*40}")
        
        func_wins = {k: 0 for k in VARIANTS.keys()}
        
        for seed_off in range(N_SEEDS):
            seed = 42 + seed_off * 1000
            results = {}
            
            # COV (no PF)
            opt = ALBA(bounds=bounds, seed=seed, maximize=False, total_budget=N_TRIALS,
                       use_potential_field=False, use_coherence_gating=True)
            best = float('inf')
            for _ in range(N_TRIALS):
                x = opt.ask()
                y = func(x)
                opt.tell(x, y)
                best = min(best, y)
            results['COV'] = best
            
            # Varianti PF
            for name, selector_class in list(VARIANTS.items())[1:]:
                val, coh = run_with_selector(selector_class, func, bounds, N_TRIALS, seed)
                results[name] = val
            
            # Winner
            best_val = min(results.values())
            for k, v in results.items():
                totals[k].append(v)
                if abs(v - best_val) < 1e-6:
                    wins[k] += 1
                    func_wins[k] += 1
                    break
            
            # Print compatto
            row = f"  s={seed}: "
            for k in VARIANTS.keys():
                marker = "✓" if results[k] == best_val else " "
                row += f"{k}={results[k]:.2f}{marker} "
            print(row)
        
        # Summary per funzione
        print(f"\n  {func_name} summary: ", end="")
        for k in VARIANTS.keys():
            print(f"{k}={func_wins[k]} ", end="")
        print()
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "=" * 85)
    print("  FINAL SUMMARY")
    print("=" * 85)
    
    total_runs = len(FUNCTIONS) * N_SEEDS
    
    print("\nVittorie:")
    for name in VARIANTS.keys():
        pct = 100 * wins[name] / total_runs
        bar = "█" * int(pct / 3)
        print(f"  {name:<10}: {wins[name]:>3}/{total_runs} ({pct:>5.1f}%) {bar}")
    
    print("\nMedia (lower is better):")
    for name in VARIANTS.keys():
        avg = np.mean(totals[name])
        print(f"  {name:<10}: {avg:.4f}")
    
    # Best variant
    best_variant = min(VARIANTS.keys(), key=lambda k: np.mean(totals[k]))
    best_avg = np.mean(totals[best_variant])
    cov_avg = np.mean(totals['COV'])
    improvement = 100 * (cov_avg - best_avg) / cov_avg
    
    print(f"\n  ⭐ BEST VARIANT: {best_variant} (avg={best_avg:.4f}, -{improvement:.1f}% vs COV)")


if __name__ == "__main__":
    main()
