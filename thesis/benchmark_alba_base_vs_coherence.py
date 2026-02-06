#!/usr/bin/env python3
"""
Benchmark: ALBA Framework (base) vs ALBA Framework (Coherence) su Nevergrad.

Obiettivo: Verificare se l'aggiunta del modulo Coherence migliora le performance
su funzioni smooth in alta dimensionalità.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from nevergrad.functions import ArtificialFunction

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import both framework versions
from thesis.alba_framework.optimizer import ALBA as ALBA_Base
from thesis.alba_framework_coherence.optimizer import ALBA as ALBA_Coherence

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# Nevergrad Function Wrapper
# ============================================================================
class NevergradWrapper:
    """Wrapper per funzioni Nevergrad con bounds [0, 1]^d."""
    
    def __init__(self, ng_func, dim: int, original_bounds: Tuple[float, float] = (-5, 5)):
        self.ng_func = ng_func
        self.dim = dim
        self.low = original_bounds[0]
        self.high = original_bounds[1]
    
    def __call__(self, x: np.ndarray) -> float:
        """x è in [0, 1]^d, lo mappiamo ai bounds originali."""
        x_scaled = self.low + x * (self.high - self.low)
        return float(self.ng_func(x_scaled))


def create_nevergrad_function(name: str, dim: int) -> Tuple[Callable, str]:
    """Crea una funzione Nevergrad."""
    fn = ArtificialFunction(name=name, block_dimension=dim)
    wrapper = NevergradWrapper(fn, dim, original_bounds=(-5, 5))
    return wrapper, f"{name} (dim={dim})"


# ============================================================================
# ALBA Base runner
# ============================================================================
def run_alba_base(objective: Callable, dim: int, budget: int, seed: int) -> Tuple[float, float]:
    """Run ALBA Base optimization."""
    np.random.seed(seed)
    
    bounds = [(0.0, 1.0) for _ in range(dim)]
    
    alba = ALBA_Base(
        bounds=bounds,
        maximize=False,  # Minimize
        seed=seed,
        total_budget=budget,
    )
    
    t0 = time.time()
    best_x, best_y = alba.optimize(objective, budget=budget)
    elapsed = time.time() - t0
    
    return best_y, elapsed


# ============================================================================
# ALBA Coherence runner
# ============================================================================
def run_alba_coherence(objective: Callable, dim: int, budget: int, seed: int) -> Tuple[float, float]:
    """Run ALBA Coherence optimization."""
    np.random.seed(seed)
    
    bounds = [(0.0, 1.0) for _ in range(dim)]
    
    alba = ALBA_Coherence(
        bounds=bounds,
        maximize=False,  # Minimize
        seed=seed,
        total_budget=budget,
    )
    
    t0 = time.time()
    best_x, best_y = alba.optimize(objective, budget=budget)
    elapsed = time.time() - t0
    
    return best_y, elapsed


# ============================================================================
# Optuna runner (for reference)
# ============================================================================
def run_optuna(objective: Callable, dim: int, budget: int, seed: int) -> Tuple[float, float]:
    """Run Optuna TPE optimization."""
    
    def optuna_objective(trial):
        x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)])
        return objective(x)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    
    t0 = time.time()
    study.optimize(optuna_objective, n_trials=budget, show_progress_bar=False)
    elapsed = time.time() - t0
    
    return study.best_value, elapsed


# ============================================================================
# Run single comparison
# ============================================================================
def run_comparison(
    func_name: str,
    dim: int,
    budget: int,
    seed: int,
    include_optuna: bool = False,
) -> Dict:
    """Run ALBA Base vs ALBA Coherence on a single function."""
    
    # Create function
    objective, desc = create_nevergrad_function(func_name, dim)
    
    # Run ALBA Base
    base_best, base_time = run_alba_base(objective, dim, budget, seed)
    
    # Run ALBA Coherence
    coh_best, coh_time = run_alba_coherence(objective, dim, budget, seed)
    
    result = {
        'function': func_name,
        'description': desc,
        'dim': dim,
        'seed': seed,
        'budget': budget,
        'alba_base_best': float(base_best),
        'alba_coherence_best': float(coh_best),
        'alba_base_time': base_time,
        'alba_coherence_time': coh_time,
    }
    
    # Winner (lower is better)
    if coh_best < base_best:
        result['winner'] = 'coherence'
    elif base_best < coh_best:
        result['winner'] = 'base'
    else:
        result['winner'] = 'tie'
    
    # Optuna for reference
    if include_optuna:
        optuna_best, optuna_time = run_optuna(objective, dim, budget, seed)
        result['optuna_best'] = float(optuna_best)
        result['optuna_time'] = optuna_time
    
    return result


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='ALBA Base vs ALBA Coherence on Nevergrad')
    parser.add_argument('--budget', type=int, default=200, help='Budget per optimizer')
    parser.add_argument('--seeds', type=str, default='0-4', help='Seed range (e.g., 0-9)')
    parser.add_argument('--dim', type=int, default=10, help='Dimension')
    parser.add_argument('--functions', type=str, default='all', 
                        help='Comma-separated list or "all"')
    parser.add_argument('--optuna', action='store_true', help='Include Optuna for reference')
    args = parser.parse_args()
    
    # Parse seeds
    if '-' in args.seeds:
        start, end = map(int, args.seeds.split('-'))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(',')]
    
    # Parse functions
    if args.functions == 'all':
        func_list = [
            # Smooth (dove Coherence dovrebbe aiutare)
            'sphere', 'rosenbrock', 'cigar', 'ellipsoid',
            # Multimodal
            'rastrigin', 'ackley', 'griewank',
            # Ill-conditioned  
            'bentcigar', 'discus',
            # Deceptive
            'deceptivemultimodal',
        ]
    elif args.functions == 'quick':
        func_list = ['sphere', 'rosenbrock', 'rastrigin', 'ackley']
    else:
        func_list = [f.strip() for f in args.functions.split(',')]
    
    print("=" * 70)
    print("ALBA BASE vs ALBA COHERENCE on Nevergrad")
    print("=" * 70)
    print(f"Budget: {args.budget}")
    print(f"Seeds: {seeds}")
    print(f"Dimension: {args.dim}")
    print(f"Functions: {len(func_list)}")
    if args.optuna:
        print("Including Optuna for reference")
    print()
    
    # Results storage
    all_results = []
    summary_by_func = {}
    
    total_runs = len(func_list) * len(seeds)
    run_idx = 0
    
    for func_name in func_list:
        print(f"\n{'='*50}")
        print(f"Function: {func_name}")
        print('='*50)
        
        func_results = []
        
        for seed in seeds:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] Seed {seed}...", end=" ", flush=True)
            
            try:
                result = run_comparison(func_name, args.dim, args.budget, seed, args.optuna)
                func_results.append(result)
                all_results.append(result)
                
                base_str = f"Base={result['alba_base_best']:.4f}"
                coh_str = f"Coh={result['alba_coherence_best']:.4f}"
                winner = result['winner'].upper()
                
                if args.optuna:
                    opt_str = f"Optuna={result['optuna_best']:.4f}"
                    print(f"{base_str} vs {coh_str} vs {opt_str} → {winner}")
                else:
                    print(f"{base_str} vs {coh_str} → {winner}")
                    
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary for this function
        if func_results:
            coh_wins = sum(1 for r in func_results if r['winner'] == 'coherence')
            base_wins = sum(1 for r in func_results if r['winner'] == 'base')
            ties = sum(1 for r in func_results if r['winner'] == 'tie')
            
            coh_avg = np.mean([r['alba_coherence_best'] for r in func_results])
            base_avg = np.mean([r['alba_base_best'] for r in func_results])
            
            summary_by_func[func_name] = {
                'coherence_wins': coh_wins,
                'base_wins': base_wins,
                'ties': ties,
                'coherence_avg': float(coh_avg),
                'base_avg': float(base_avg),
                'coherence_winrate': coh_wins / len(func_results) if func_results else 0,
            }
            
            print(f"  → {func_name}: Coherence {coh_wins}/{len(func_results)} "
                  f"({100*coh_wins/len(func_results):.0f}%)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    total_coh = sum(1 for r in all_results if r['winner'] == 'coherence')
    total_base = sum(1 for r in all_results if r['winner'] == 'base')
    total_ties = sum(1 for r in all_results if r['winner'] == 'tie')
    
    print(f"\nOverall: Coherence {total_coh}/{len(all_results)} "
          f"({100*total_coh/len(all_results):.1f}%), "
          f"Base {total_base}/{len(all_results)} "
          f"({100*total_base/len(all_results):.1f}%), "
          f"Ties {total_ties}")
    
    print("\nBy function:")
    for func_name, summary in summary_by_func.items():
        total = summary['coherence_wins'] + summary['base_wins'] + summary['ties']
        print(f"  {func_name:25s}: Coherence {summary['coherence_wins']}/{total} "
              f"({100*summary['coherence_winrate']:.0f}%)")
    
    # Categorize
    print("\nBy category:")
    smooth_funcs = ['sphere', 'rosenbrock', 'cigar', 'ellipsoid']
    multimodal_funcs = ['rastrigin', 'ackley', 'griewank']
    illcond_funcs = ['bentcigar', 'discus']
    deceptive_funcs = ['deceptivemultimodal']
    
    for cat_name, cat_funcs in [('Smooth', smooth_funcs), ('Multimodal', multimodal_funcs), 
                                 ('Ill-conditioned', illcond_funcs), ('Deceptive', deceptive_funcs)]:
        cat_results = [r for r in all_results if r['function'] in cat_funcs]
        if cat_results:
            cat_coh = sum(1 for r in cat_results if r['winner'] == 'coherence')
            print(f"  {cat_name:18s}: Coherence {cat_coh}/{len(cat_results)} "
                  f"({100*cat_coh/len(cat_results):.0f}%)")
    
    # Save results
    os.makedirs('/mnt/workspace/thesis/benchmark_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = f'/mnt/workspace/thesis/benchmark_results/alba_base_vs_coherence_{timestamp}.json'
    
    with open(outfile, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': all_results,
            'summary_by_function': summary_by_func,
            'summary_overall': {
                'coherence_wins': total_coh,
                'base_wins': total_base,
                'ties': total_ties,
                'total': len(all_results),
                'coherence_winrate': total_coh / len(all_results) if all_results else 0,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {outfile}")


if __name__ == '__main__':
    main()
