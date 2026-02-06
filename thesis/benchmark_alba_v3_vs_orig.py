#!/usr/bin/env python3
"""
Benchmark: ALBA V3 (LHS Fast Start) vs ALBA Original
On JAHS-Bench-201 surrogate (all 3 tasks) with checkpoints.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List

import numpy as np

sys.path.insert(0, '/mnt/workspace/thesis')
from benchmark_jahs import JAHSBenchWrapper
from ALBA_V3 import ALBA as ALBA_V3
from ALBA_V1 import ALBA as ALBA_Original


def run_single_experiment(
    wrapper: JAHSBenchWrapper,
    optimizer_class,
    optimizer_name: str,
    seed: int,
    budget: int,
    checkpoints: List[int],
    categorical_dims: List
) -> Dict:
    """Run a single optimization experiment with checkpoints."""
    bounds = [(0, 1)] * wrapper.dim
    
    opt = optimizer_class(
        bounds=bounds,
        maximize=True,
        seed=seed,
        total_budget=budget,
        categorical_dims=categorical_dims
    )
    
    results = {
        'optimizer': optimizer_name,
        'seed': seed,
        'task': wrapper.task,
        'checkpoints': {},
        'final_best': None,
        'final_best_x': None,
        'runtime': 0
    }
    
    start_time = time.time()
    best_y = -np.inf
    best_x = None
    
    for i in range(budget):
        x = opt.ask()
        y = wrapper.evaluate_array(x)
        opt.tell(x, y)
        
        if y > best_y:
            best_y = y
            best_x = x.copy()
        
        if (i + 1) in checkpoints:
            results['checkpoints'][i + 1] = {
                'best_y': float(best_y),
                'n_leaves': len(opt.leaves) if hasattr(opt, 'leaves') else 1
            }
    
    results['runtime'] = time.time() - start_time
    results['final_best'] = float(best_y)
    results['final_best_x'] = best_x.tolist() if best_x is not None else None
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=['cifar10', 'colorectal_histology', 'fashion_mnist'])
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--budget', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='results/alba_v3_vs_orig')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Checkpoints
    checkpoints = [25, 50, 100, 200, 400, 600, 800, 1000]
    checkpoints = [c for c in checkpoints if c <= args.budget]
    
    # Categorical dimensions for JAHS
    categorical_dims = [(2, 3), (3, 3), (4, 3), (5, 3), (6, 2), (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5)]
    
    # Optimizers
    optimizers = [
        ('Original', ALBA_Original),
        ('V3-LHS', ALBA_V3),
    ]
    
    all_results = []
    
    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")
        
        wrapper = JAHSBenchWrapper(task=task)
        
        for seed in range(args.n_seeds):
            print(f"\n--- Seed {seed} ---")
            
            for opt_name, opt_class in optimizers:
                print(f"  Running {opt_name}...", end=' ', flush=True)
                
                result = run_single_experiment(
                    wrapper=wrapper,
                    optimizer_class=opt_class,
                    optimizer_name=opt_name,
                    seed=seed,
                    budget=args.budget,
                    checkpoints=checkpoints,
                    categorical_dims=categorical_dims
                )
                
                all_results.append(result)
                print(f"done. Best: {result['final_best']:.6f} ({result['runtime']:.1f}s)")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_dir, f'benchmark_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'tasks': args.tasks,
                'n_seeds': args.n_seeds,
                'budget': args.budget,
                'checkpoints': checkpoints
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for task in args.tasks:
        print(f"\n{task}:")
        task_results = [r for r in all_results if r['task'] == task]
        
        for opt_name, _ in optimizers:
            opt_results = [r for r in task_results if r['optimizer'] == opt_name]
            finals = [r['final_best'] for r in opt_results]
            print(f"  {opt_name:12s}: {np.mean(finals):.6f} ± {np.std(finals):.6f}")
        
        # Compare at each checkpoint
        print(f"\n  Checkpoint comparison (V3-LHS - Original):")
        orig_results = [r for r in task_results if r['optimizer'] == 'Original']
        v3_results = [r for r in task_results if r['optimizer'] == 'V3-LHS']
        
        for cp in checkpoints:
            orig_vals = [r['checkpoints'][cp]['best_y'] for r in orig_results]
            v3_vals = [r['checkpoints'][cp]['best_y'] for r in v3_results]
            gap = np.mean(v3_vals) - np.mean(orig_vals)
            wins = sum(v > o for v, o in zip(v3_vals, orig_vals))
            print(f"    {cp:4d}: {gap:+.6f} (V3 wins {wins}/{args.n_seeds})")


if __name__ == '__main__':
    main()
