#!/usr/bin/env python3
"""
Debug ALBA Stagnation - Seed 7 on colorectal_histology
======================================================

Analizza perché ALBA si blocca a 0.0406 dopo ~600 valutazioni.
Logga lo stato interno: cube splits, best values, exploration, etc.
"""

import os
import sys
import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, '/mnt/workspace/thesis')

from benchmark_jahs import JAHSBenchWrapper
from ALBA_V1 import ALBA, Cube


def debug_alba_run(
    task: str = 'colorectal_histology',
    seed: int = 7,
    budget: int = 2000,
    log_every: int = 50
):
    """
    Esegue ALBA con logging dettagliato dello stato interno.
    """
    print("=" * 80)
    print(f"DEBUG ALBA - Task: {task}, Seed: {seed}, Budget: {budget}")
    print("=" * 80)
    
    # Initialize
    wrapper = JAHSBenchWrapper(task=task)
    dim = wrapper.dim
    # JAHS wrapper usa [0,1]^d normalizzato
    bounds = np.array([[0.0, 1.0]] * dim)
    
    print(f"\nDimension: {dim}")
    print(f"Bounds: [0, 1]^{dim}")
    
    # Track history
    history = []
    best_so_far = float('inf')
    stagnation_start = None
    
    def objective(x: np.ndarray) -> float:
        nonlocal best_so_far, stagnation_start
        
        val = wrapper.evaluate_array(x)
        history.append(val)
        
        if val < best_so_far:
            improvement = best_so_far - val
            best_so_far = val
            stagnation_start = len(history)
        
        return val
    
    # Define categorical dimensions for JAHS-Bench
    # HP_ORDER = ['LearningRate', 'WeightDecay', 'N', 'W', 'Resolution', 
    #             'Activation', 'TrivialAugment', 'Op1', 'Op2', 'Op3', 'Op4', 'Op5', 'Op6']
    categorical_dims = [
        (2, 3),   # N: 3 choices
        (3, 3),   # W: 3 choices
        (4, 3),   # Resolution: 3 choices
        (5, 3),   # Activation: 3 choices
        (6, 2),   # TrivialAugment: 2 choices
        (7, 5),   # Op1: 5 choices
        (8, 5),   # Op2: 5 choices
        (9, 5),   # Op3: 5 choices
        (10, 5),  # Op4: 5 choices
        (11, 5),  # Op5: 5 choices
        (12, 5),  # Op6: 5 choices
    ]
    
    # Initialize ALBA con ask/tell API
    optimizer = ALBA(
        bounds=[(0.0, 1.0)] * dim,
        maximize=False,
        seed=seed,
        total_budget=budget,
        categorical_dims=categorical_dims,
    )
    
    print(f"\nALBA initialized with:")
    print(f"  - categorical_dims: {len(categorical_dims)} categorical dimensions")
    print(f"  - maximize: {optimizer.maximize}")
    
    # Detailed logging during optimization
    print("\n" + "-" * 80)
    print("OPTIMIZATION LOG")
    print("-" * 80)
    
    last_best = float('inf')
    last_n_cubes = 1
    stuck_counter = 0
    
    for i in range(budget):
        # Get next point
        x = optimizer.ask()
        y = objective(x)
        optimizer.tell(x, y)
        
        # Periodic logging
        if (i + 1) % log_every == 0 or i < 20:
            n_cubes = len(optimizer.leaves)
            current_best = min(history)
            
            # Check stagnation
            if current_best == last_best:
                stuck_counter += log_every
            else:
                stuck_counter = 0
            
            # Cube analysis
            def cube_volume(c):
                return np.prod([hi - lo for lo, hi in c.bounds])
            
            cube_volumes = [cube_volume(c) for c in optimizer.leaves]
            cube_n_points = [c.n_trials for c in optimizer.leaves]
            
            # Best cube info
            if optimizer.leaves:
                best_cube_idx = np.argmax([c.best_score for c in optimizer.leaves])
                best_cube = optimizer.leaves[best_cube_idx]
                
                print(f"\n[Eval {i+1:4d}] Best: {current_best:.6f} | "
                      f"Cubes: {n_cubes} | Stuck: {stuck_counter}")
                print(f"  Cube volumes: min={min(cube_volumes):.2e}, "
                      f"max={max(cube_volumes):.2e}, mean={np.mean(cube_volumes):.2e}")
                print(f"  Points/cube: min={min(cube_n_points)}, "
                      f"max={max(cube_n_points)}, total={sum(cube_n_points)}")
                print(f"  Best cube: idx={best_cube_idx}, vol={cube_volume(best_cube):.2e}, "
                      f"n_pts={best_cube.n_trials}, best_score={best_cube.best_score:.6f}")
                
                # Show top 3 cubes by best_score
                sorted_cubes = sorted(enumerate(optimizer.leaves), 
                                     key=lambda x: x[1].best_score, reverse=True)[:3]
                print(f"  Top 3 cubes (by best_score):")
                for rank, (idx, cube) in enumerate(sorted_cubes, 1):
                    print(f"    {rank}. cube[{idx}]: best={cube.best_score:.6f}, "
                          f"vol={cube_volume(cube):.2e}, pts={cube.n_trials}")
            
            last_best = current_best
            last_n_cubes = n_cubes
    
    # Final analysis
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS")
    print("=" * 80)
    
    final_best = min(history)
    print(f"\nFinal best: {final_best:.6f}")
    print(f"Stagnation started at eval: {stagnation_start}")
    print(f"Evaluations without improvement: {budget - stagnation_start}")
    
    # Find when we hit 0.0406
    target = 0.0406
    hit_target_at = None
    for i, val in enumerate(history):
        if val <= target + 0.0001:
            hit_target_at = i + 1
            break
    
    if hit_target_at:
        print(f"\nHit ~0.0406 at evaluation: {hit_target_at}")
    
    # Cube state at end
    print(f"\nFinal cube count: {len(optimizer.leaves)}")
    
    # Distribution of cube volumes
    def cube_volume(c):
        return np.prod([hi - lo for lo, hi in c.bounds])
    
    volumes = [cube_volume(c) for c in optimizer.leaves]
    print(f"Volume distribution:")
    print(f"  min: {min(volumes):.2e}")
    print(f"  max: {max(volumes):.2e}")
    print(f"  mean: {np.mean(volumes):.2e}")
    print(f"  median: {np.median(volumes):.2e}")
    
    # Check if best region is over-explored
    best_cube_idx = np.argmax([c.best_score for c in optimizer.leaves])
    best_cube = optimizer.leaves[best_cube_idx]
    
    print(f"\nBest cube analysis:")
    print(f"  Volume: {cube_volume(best_cube):.2e}")
    print(f"  Points inside: {best_cube.n_trials}")
    print(f"  Best value (internal score): {best_cube.best_score:.6f}")
    print(f"  Depth: {best_cube.depth}")
    print(f"  Bounds:")
    for d in range(dim):
        lo, hi = best_cube.bounds[d]
        print(f"    dim {d}: [{lo:.4f}, {hi:.4f}] "
              f"(width: {hi - lo:.4f})")
    
    # Show the best configuration found
    print(f"\n  Best x found:")
    print(f"    {best_cube.best_x}")
    
    # Convert to dict to understand what HP values this represents
    config = wrapper._array_to_dict(best_cube.best_x)
    print(f"\n  Best config (decoded):")
    for k, v in config.items():
        print(f"    {k}: {v}")
    
    # Convergence curve analysis
    print("\n" + "-" * 60)
    print("CONVERGENCE ANALYSIS")
    print("-" * 60)
    
    # Best-so-far curve
    best_curve = []
    running_best = float('inf')
    for val in history:
        running_best = min(running_best, val)
        best_curve.append(running_best)
    
    # Check improvement rate at different stages
    checkpoints = [100, 200, 400, 600, 800, 1000, 1500, 2000]
    print("\nCheckpoint analysis:")
    prev_best = best_curve[0]
    for cp in checkpoints:
        if cp <= len(best_curve):
            cp_best = best_curve[cp - 1]
            improvement = prev_best - cp_best
            print(f"  @{cp:4d}: {cp_best:.6f} (Δ = {improvement:.6f})")
            prev_best = cp_best
    
    # Find plateau regions
    print("\nPlateau analysis:")
    plateau_start = None
    plateau_threshold = 100  # evals without improvement
    
    for i in range(1, len(best_curve)):
        if best_curve[i] == best_curve[i-1]:
            if plateau_start is None:
                plateau_start = i
        else:
            if plateau_start is not None:
                plateau_len = i - plateau_start
                if plateau_len >= plateau_threshold:
                    print(f"  Plateau: evals {plateau_start} to {i} "
                          f"(length: {plateau_len}, value: {best_curve[i-1]:.6f})")
            plateau_start = None
    
    # Check final plateau
    if plateau_start is not None:
        plateau_len = len(best_curve) - plateau_start
        if plateau_len >= plateau_threshold:
            print(f"  Plateau: evals {plateau_start} to {len(best_curve)} "
                  f"(length: {plateau_len}, value: {best_curve[-1]:.6f})")
    
    return history, optimizer


if __name__ == '__main__':
    history, optimizer = debug_alba_run(
        task='colorectal_histology',
        seed=7,
        budget=2000,
        log_every=100
    )
