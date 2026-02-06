#!/usr/bin/env python3
"""
Benchmark: ALBA vs Optuna su funzioni Nevergrad.

Nevergrad offre funzioni realistiche e smooth, ideali per testare
se Coherence/ALBA performa meglio su landscape non discretizzati.

Funzioni testate:
- Classiche smooth: sphere, rosenbrock, cigar, ellipsoid
- Multimodali: rastrigin, lunacek, griewank, ackley
- Ill-conditioned: bentcigar, discus
- Speciali: ARCoating (physics), Photonics (optics)
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
from nevergrad.functions.corefuncs import registry
from nevergrad.functions.arcoating import ARCoating
from nevergrad.functions.photonics import Photonics

# Add parent for ALBA
sys.path.insert(0, str(Path(__file__).parent.parent))
from thesis.ALBA_V1 import ALBA

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
    
    if name == "arcoating":
        # ARCoating ha dimensione fissa
        arc = ARCoating()
        actual_dim = arc.dimension
        wrapper = lambda x: float(arc(x * 100))  # ARCoating usa [0, 100]
        return wrapper, f"ARCoating (dim={actual_dim})"
    
    elif name == "photonics_bragg":
        ph = Photonics("bragg", dim)
        actual_dim = ph.dimension
        wrapper = lambda x: float(ph(x))
        return wrapper, f"Photonics-Bragg (dim={actual_dim})"
    
    elif name == "photonics_morpho":
        ph = Photonics("morpho", dim)
        actual_dim = ph.dimension
        wrapper = lambda x: float(ph(x))
        return wrapper, f"Photonics-Morpho (dim={actual_dim})"
    
    else:
        # Funzioni standard via ArtificialFunction
        fn = ArtificialFunction(name=name, block_dimension=dim)
        wrapper = NevergradWrapper(fn, dim, original_bounds=(-5, 5))
        return wrapper, f"{name} (dim={dim})"


# ============================================================================
# ALBA runner
# ============================================================================
def run_alba(objective: Callable, dim: int, budget: int, seed: int) -> Tuple[float, float]:
    """Run ALBA optimization."""
    np.random.seed(seed)
    
    bounds = [(0.0, 1.0) for _ in range(dim)]
    
    alba = ALBA(
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
# Optuna runner
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
) -> Dict:
    """Run ALBA vs Optuna on a single function."""
    
    # Create function
    objective, desc = create_nevergrad_function(func_name, dim)
    
    # Determine actual dimension
    if func_name == "arcoating":
        actual_dim = ARCoating().dimension
    elif func_name.startswith("photonics"):
        actual_dim = dim
    else:
        actual_dim = dim
    
    # Run ALBA
    alba_best, alba_time = run_alba(objective, actual_dim, budget, seed)
    
    # Run Optuna
    optuna_best, optuna_time = run_optuna(objective, actual_dim, budget, seed)
    
    # Winner (lower is better)
    if alba_best < optuna_best:
        winner = 'alba'
    elif optuna_best < alba_best:
        winner = 'optuna'
    else:
        winner = 'tie'
    
    return {
        'function': func_name,
        'description': desc,
        'dim': actual_dim,
        'seed': seed,
        'budget': budget,
        'alba_best': float(alba_best),
        'optuna_best': float(optuna_best),
        'alba_time': alba_time,
        'optuna_time': optuna_time,
        'winner': winner,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='ALBA vs Optuna on Nevergrad functions')
    parser.add_argument('--budget', type=int, default=200, help='Budget per optimizer')
    parser.add_argument('--seeds', type=str, default='0-4', help='Seed range (e.g., 0-9)')
    parser.add_argument('--dim', type=int, default=10, help='Dimension for standard functions')
    parser.add_argument('--functions', type=str, default='all', 
                        help='Comma-separated list or "all"')
    args = parser.parse_args()
    
    # Parse seeds
    if '-' in args.seeds:
        start, end = map(int, args.seeds.split('-'))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(',')]
    
    # Parse functions
    if args.functions == 'all':
        # TUTTE le funzioni disponibili in Nevergrad
        func_list = [
            # Smooth classiche (convesse o quasi)
            'sphere', 'sphere1', 'sphere2', 'sphere4',
            'rosenbrock', 'ellipsoid', 'cigar', 'altcigar', 'altellipsoid',
            'schwefel_1_2', 'linear', 'slope', 'doublelinearslope',
            # Multimodali
            'rastrigin', 'bucherastrigin', 'lunacek', 'griewank', 'ackley',
            'multipeak', 'hm',
            # Ill-conditioned
            'bentcigar', 'discus',
            # Deceptive / Tricky
            'deceptivemultimodal', 'deceptiveillcond', 'deceptivepath',
            'maxdeceptive', 'sumdeceptive',
            # Step functions (discretized!)
            'stepellipsoid', 'stepdoublelinearslope',
            # Genz functions (integration benchmarks)
            'genzcornerpeak', 'minusgenzcornerpeak',
            # Delayed/Noisy
            'DelayedSphere',
        ]
    elif args.functions == 'quick':
        # Subset veloce per test
        func_list = [
            'sphere', 'rosenbrock', 'rastrigin', 'ackley',
            'bentcigar', 'deceptivemultimodal', 'stepellipsoid',
        ]
    else:
        func_list = [f.strip() for f in args.functions.split(',')]
    
    print("=" * 70)
    print("NEVERGRAD BENCHMARK: ALBA vs Optuna")
    print("=" * 70)
    print(f"Budget: {args.budget}")
    print(f"Seeds: {seeds}")
    print(f"Dimension: {args.dim}")
    print(f"Functions: {len(func_list)}")
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
                result = run_comparison(func_name, args.dim, args.budget, seed)
                func_results.append(result)
                all_results.append(result)
                
                print(f"ALBA={result['alba_best']:.4f} vs Optuna={result['optuna_best']:.4f} "
                      f"→ {result['winner'].upper()}")
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        # Summary for this function
        if func_results:
            alba_wins = sum(1 for r in func_results if r['winner'] == 'alba')
            optuna_wins = sum(1 for r in func_results if r['winner'] == 'optuna')
            ties = sum(1 for r in func_results if r['winner'] == 'tie')
            
            alba_avg = np.mean([r['alba_best'] for r in func_results])
            optuna_avg = np.mean([r['optuna_best'] for r in func_results])
            
            summary_by_func[func_name] = {
                'alba_wins': alba_wins,
                'optuna_wins': optuna_wins,
                'ties': ties,
                'alba_avg': float(alba_avg),
                'optuna_avg': float(optuna_avg),
                'winrate': alba_wins / len(func_results) if func_results else 0,
            }
            
            print(f"  → {func_name}: ALBA {alba_wins}/{len(func_results)} "
                  f"({100*alba_wins/len(func_results):.0f}%)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    total_alba = sum(1 for r in all_results if r['winner'] == 'alba')
    total_optuna = sum(1 for r in all_results if r['winner'] == 'optuna')
    total_ties = sum(1 for r in all_results if r['winner'] == 'tie')
    
    print(f"\nOverall: ALBA {total_alba}/{len(all_results)} "
          f"({100*total_alba/len(all_results):.1f}%), "
          f"Optuna {total_optuna}/{len(all_results)} "
          f"({100*total_optuna/len(all_results):.1f}%), "
          f"Ties {total_ties}")
    
    print("\nBy function:")
    for func_name, summary in summary_by_func.items():
        print(f"  {func_name:25s}: ALBA {summary['alba_wins']}/{summary['alba_wins']+summary['optuna_wins']+summary['ties']} "
              f"({100*summary['winrate']:.0f}%)")
    
    # Categorize results
    print("\nBy category:")
    smooth_funcs = ['sphere', 'sphere1', 'sphere2', 'sphere4', 'rosenbrock', 
                    'ellipsoid', 'cigar', 'altcigar', 'altellipsoid',
                    'schwefel_1_2', 'linear', 'slope', 'doublelinearslope']
    multimodal_funcs = ['rastrigin', 'bucherastrigin', 'lunacek', 'griewank', 
                        'ackley', 'multipeak', 'hm']
    illcond_funcs = ['bentcigar', 'discus']
    deceptive_funcs = ['deceptivemultimodal', 'deceptiveillcond', 'deceptivepath',
                       'maxdeceptive', 'sumdeceptive']
    step_funcs = ['stepellipsoid', 'stepdoublelinearslope']
    genz_funcs = ['genzcornerpeak', 'minusgenzcornerpeak']
    
    for cat_name, cat_funcs in [('Smooth', smooth_funcs), ('Multimodal', multimodal_funcs), 
                                 ('Ill-conditioned', illcond_funcs), ('Deceptive', deceptive_funcs),
                                 ('Step (discretized)', step_funcs), ('Genz', genz_funcs)]:
        cat_results = [r for r in all_results if r['function'] in cat_funcs]
        if cat_results:
            cat_alba = sum(1 for r in cat_results if r['winner'] == 'alba')
            print(f"  {cat_name:18s}: ALBA {cat_alba}/{len(cat_results)} ({100*cat_alba/len(cat_results):.0f}%)")
    
    # Save results
    os.makedirs('/mnt/workspace/thesis/benchmark_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = f'/mnt/workspace/thesis/benchmark_results/nevergrad_{timestamp}.json'
    
    with open(outfile, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': all_results,
            'summary_by_function': summary_by_func,
            'summary_overall': {
                'alba_wins': total_alba,
                'optuna_wins': total_optuna,
                'ties': total_ties,
                'total': len(all_results),
                'alba_winrate': total_alba / len(all_results) if all_results else 0,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {outfile}")


if __name__ == '__main__':
    main()
