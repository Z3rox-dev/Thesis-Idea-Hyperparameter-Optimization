#!/usr/bin/env python3
"""
Benchmark: ALBA COV-only vs ALBA COV+PotentialField su NASBench-201

NASBench-201 ha 6 dimensioni TUTTE CATEGORIALI (5 scelte ciascuna).
Questo testa il Potential Field su spazi puramente categoriali.
"""

import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace/HPOBench')

import numpy as np
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from alba_framework_potential.optimizer import ALBA


# ============================================================================
# NASBENCH-201 PARAM SPACE (6 CATEGORICAL DIMS)
# ============================================================================

# NAS operations available for each edge
NAS_OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

# 6 edges in the DAG: (to, from) pairs
EDGES = ['1<-0', '2<-0', '2<-1', '3<-0', '3<-1', '3<-2']


def build_nasbench201_param_space() -> Dict[str, Any]:
    """Build NASBench-201 param space - ALL CATEGORICAL."""
    return {edge: NAS_OPS for edge in EDGES}


def config_to_arch_str(config: Dict[str, str]) -> str:
    """Convert ALBA config dict to NASBench-201 architecture string."""
    # Format: |op1~0|+|op2~0|op3~1|+|op4~0|op5~1|op6~2|
    ops = [config[e] for e in EDGES]
    return f"|{ops[0]}~0|+|{ops[1]}~0|{ops[2]}~1|+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"


# ============================================================================
# SYNTHETIC SURROGATE (NASBench-201 style)
# ============================================================================

class NASBench201Surrogate:
    """
    Synthetic surrogate for NASBench-201 to avoid HPOBench dependencies.
    Simulates realistic accuracy patterns based on known NAS heuristics.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Pre-compute some "good" architectures
        self._setup_landscape()
    
    def _setup_landscape(self):
        """Create a realistic accuracy landscape."""
        # Heuristics: skip_connect and conv ops are generally better than 'none'
        self.op_scores = {
            'none': 0.0,
            'skip_connect': 0.6,
            'nor_conv_1x1': 0.7,
            'nor_conv_3x3': 1.0,
            'avg_pool_3x3': 0.5,
        }
        # Add some random interactions between edges
        self.interactions = self.rng.randn(6, 6) * 0.05
    
    def evaluate(self, config: Dict[str, str]) -> float:
        """Evaluate architecture, return validation accuracy (0-100)."""
        ops = [config[e] for e in EDGES]
        
        # Base score from individual ops
        base_score = sum(self.op_scores[op] for op in ops) / 6.0
        
        # Interaction effects
        op_indices = [NAS_OPS.index(op) for op in ops]
        interaction_score = 0.0
        for i in range(6):
            for j in range(i+1, 6):
                interaction_score += self.interactions[i, j] * (op_indices[i] == op_indices[j])
        
        # Add some noise
        noise = self.rng.randn() * 0.02
        
        # Scale to realistic accuracy range (85-95%)
        accuracy = 85.0 + base_score * 10.0 + interaction_score * 5.0 + noise
        return np.clip(accuracy, 75.0, 97.0)


# ============================================================================
# RUN
# ============================================================================

def run_alba(
    surrogate: NASBench201Surrogate,
    n_trials: int,
    seed: int,
    use_potential_field: bool,
) -> float:
    """Run ALBA on NASBench-201 surrogate, return best valid-acc."""
    
    param_space = build_nasbench201_param_space()
    
    opt = ALBA(
        param_space=param_space,
        seed=seed,
        maximize=True,  # Maximize valid-acc
        total_budget=n_trials,
        use_potential_field=use_potential_field,
        use_coherence_gating=True,
    )
    
    best_y = -np.inf
    for _ in range(n_trials):
        cfg = opt.ask()
        y = surrogate.evaluate(cfg)
        opt.tell(cfg, y)
        
        if y > best_y:
            best_y = y
    
    return best_y


def main():
    print("=" * 75)
    print("  BENCHMARK: ALBA COV-only vs ALBA COV+PotentialField")
    print("  NASBench-201 Surrogate - 6 CATEGORICAL dims - Budget 300")
    print("=" * 75)
    
    BUDGET = 300
    N_SEEDS = 20  # More seeds since it's fast
    N_TASKS = 5   # Different surrogate seeds = different landscapes
    
    all_results = {}
    
    for task_seed in range(N_TASKS):
        task_name = f"landscape_{task_seed}"
        print(f"\n{'='*65}")
        print(f"  NASBench-201 Surrogate (seed={task_seed})")
        print(f"{'='*65}")
        
        surrogate = NASBench201Surrogate(seed=task_seed * 1000)
        
        results_cov = []
        results_pf = []
        
        for seed in range(N_SEEDS):
            print(f"  Seed {seed:2d}: ", end="", flush=True)
            
            # COV-only (PF=False)
            val_cov = run_alba(surrogate, BUDGET, 100 + seed, use_potential_field=False)
            results_cov.append(val_cov)
            print(f"COV={val_cov:.2f}%", end=" ", flush=True)
            
            # COV+PF (PF=True)
            val_pf = run_alba(surrogate, BUDGET, 100 + seed, use_potential_field=True)
            results_pf.append(val_pf)
            print(f"PF={val_pf:.2f}%", end="", flush=True)
            
            # Winner
            if val_cov > val_pf:
                print(" → COV")
            elif val_pf > val_cov:
                print(" → PF")
            else:
                print(" → TIE")
        
        mean_cov = np.mean(results_cov)
        std_cov = np.std(results_cov)
        mean_pf = np.mean(results_pf)
        std_pf = np.std(results_pf)
        
        wins_cov = sum(1 for a, b in zip(results_cov, results_pf) if a > b)
        wins_pf = sum(1 for a, b in zip(results_cov, results_pf) if b > a)
        
        delta_pct = (mean_pf - mean_cov) / abs(mean_cov) * 100 if mean_cov != 0 else 0
        
        all_results[task_name] = {
            'mean_cov': mean_cov,
            'std_cov': std_cov,
            'mean_pf': mean_pf,
            'std_pf': std_pf,
            'wins_cov': wins_cov,
            'wins_pf': wins_pf,
            'delta_pct': delta_pct,
        }
        
        print(f"\n  Summary {task_name}:")
        print(f"    COV-only: {mean_cov:.2f}% ± {std_cov:.2f}")
        print(f"    COV+PF:   {mean_pf:.2f}% ± {std_pf:.2f}")
        print(f"    Delta:    {delta_pct:+.2f}%")
        print(f"    Wins:     COV={wins_cov}, PF={wins_pf}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 75)
    print("  FINAL SUMMARY - NASBench-201 (6 CATEGORICAL dims)")
    print("=" * 75)
    
    print(f"\n{'Landscape':<20} {'COV-only':>12} {'COV+PF':>12} {'Δ%':>8} {'Wins COV':>10} {'Wins PF':>10}")
    print("-" * 75)
    
    total_wins_cov = 0
    total_wins_pf = 0
    
    for task, r in all_results.items():
        status = "✅" if r['wins_cov'] > r['wins_pf'] else ("❌" if r['wins_pf'] > r['wins_cov'] else "➖")
        print(f"{task:<20} {r['mean_cov']:>11.2f}% {r['mean_pf']:>11.2f}% {r['delta_pct']:>+7.2f}% {r['wins_cov']:>7}/{N_SEEDS} {r['wins_pf']:>7}/{N_SEEDS} {status}")
        total_wins_cov += r['wins_cov']
        total_wins_pf += r['wins_pf']
    
    total_runs = len(all_results) * N_SEEDS
    avg_delta = np.mean([r['delta_pct'] for r in all_results.values()])
    
    print("-" * 75)
    print(f"{'TOTALE':<20} {'':<12} {'':<12} {avg_delta:>+7.2f}% {total_wins_cov:>7}/{total_runs} {total_wins_pf:>7}/{total_runs}")
    
    print("\n" + "=" * 75)
    if total_wins_pf > total_wins_cov:
        print(f"🔥 VERDETTO: COV+PF è MIGLIORE (vince {total_wins_pf}/{total_runs})")
    elif total_wins_cov > total_wins_pf:
        print(f"🎉 VERDETTO: COV-only è MIGLIORE (vince {total_wins_cov}/{total_runs})")
    else:
        print("➖ VERDETTO: Sostanzialmente equivalenti")
    print("=" * 75)
    
    print("\n📊 NOTA: Questo benchmark usa 6 dimensioni TUTTE CATEGORIALI.")
    print("   Il CovarianceLocalSearchSampler viene completamente sovrascritto")
    print("   dal CategoricalSampler (Thompson Sampling), quindi questo test")
    print("   misura SOLO l'effetto del Potential Field sulla selezione categoriale.")


if __name__ == "__main__":
    main()
