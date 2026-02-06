#!/usr/bin/env python3
"""
DEBUG: Perché Copula genera candidati più lontani ma converge meglio?
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
np.random.seed(42)

if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int

from alba_framework_potential import ALBA as ALBA_LGS
from alba_framework_copula.optimizer import ALBA as ALBA_Copula

def sphere(x):
    return float(np.sum((x - 0.5)**2))

dim = 5
bounds = [(0, 1)] * dim

# Run with detailed logging
print("="*80)
print("DETAILED STEP-BY-STEP COMPARISON")
print("="*80)

opt_lgs = ALBA_LGS(bounds=bounds, seed=42, total_budget=50, maximize=False)
opt_cop = ALBA_Copula(bounds=bounds, seed=42, total_budget=50, maximize=False)

history_lgs = []
history_cop = []
points_lgs = []
points_cop = []

for i in range(50):
    x_lgs = opt_lgs.ask()
    y_lgs = sphere(x_lgs)
    opt_lgs.tell(x_lgs, y_lgs)
    
    x_cop = opt_cop.ask()
    y_cop = sphere(x_cop)
    opt_cop.tell(x_cop, y_cop)
    
    history_lgs.append(opt_lgs.best_y)
    history_cop.append(opt_cop.best_y)
    points_lgs.append((x_lgs.copy(), y_lgs))
    points_cop.append((x_cop.copy(), y_cop))

# Analyze when each improved
print("\nImprovement events:")
print(f"{'Step':<6} {'Who':<8} {'New Best':<12} {'Point y':<12} {'Dist to opt':<12}")

prev_lgs = float('inf')
prev_cop = float('inf')
for i in range(50):
    x_l, y_l = points_lgs[i]
    x_c, y_c = points_cop[i]
    
    lgs_improved = history_lgs[i] < prev_lgs
    cop_improved = history_cop[i] < prev_cop
    
    if lgs_improved:
        dist = np.linalg.norm(x_l - 0.5)
        print(f"{i:<6} {'LGS':<8} {history_lgs[i]:<12.6f} {y_l:<12.6f} {dist:<12.4f}")
        prev_lgs = history_lgs[i]
    
    if cop_improved:
        dist = np.linalg.norm(x_c - 0.5)
        print(f"{i:<6} {'COPULA':<8} {history_cop[i]:<12.6f} {y_c:<12.6f} {dist:<12.4f}")
        prev_cop = history_cop[i]

# Points near optimum
print("\n" + "="*80)
print("Points found near optimum (dist < 0.3)")
print("="*80)

print("\nLGS points near opt:")
for i, (x, y) in enumerate(points_lgs):
    dist = np.linalg.norm(x - 0.5)
    if dist < 0.3:
        print(f"  Step {i}: dist={dist:.4f}, y={y:.6f}")

print("\nCopula points near opt:")
for i, (x, y) in enumerate(points_cop):
    dist = np.linalg.norm(x - 0.5)
    if dist < 0.3:
        print(f"  Step {i}: dist={dist:.4f}, y={y:.6f}")

# Distribution of samples
print("\n" + "="*80)
print("Distribution analysis")
print("="*80)

lgs_dists = [np.linalg.norm(x - 0.5) for x, y in points_lgs]
cop_dists = [np.linalg.norm(x - 0.5) for x, y in points_cop]

print(f"\nLGS distances to opt:")
print(f"  Mean: {np.mean(lgs_dists):.4f}")
print(f"  Std:  {np.std(lgs_dists):.4f}")
print(f"  Min:  {np.min(lgs_dists):.4f}")
print(f"  <0.3: {sum(1 for d in lgs_dists if d < 0.3)} points")
print(f"  <0.2: {sum(1 for d in lgs_dists if d < 0.2)} points")

print(f"\nCopula distances to opt:")
print(f"  Mean: {np.mean(cop_dists):.4f}")
print(f"  Std:  {np.std(cop_dists):.4f}")
print(f"  Min:  {np.min(cop_dists):.4f}")
print(f"  <0.3: {sum(1 for d in cop_dists if d < 0.3)} points")
print(f"  <0.2: {sum(1 for d in cop_dists if d < 0.2)} points")

# Copula sampling effect
print("\n" + "="*80)
print("Copula vs LGS: Sample analysis first 10 vs last 10 steps")
print("="*80)

print("\nFirst 10 steps (exploration phase):")
print(f"  LGS mean dist:    {np.mean(lgs_dists[:10]):.4f}")
print(f"  Copula mean dist: {np.mean(cop_dists[:10]):.4f}")

print("\nLast 10 steps (exploitation phase):")
print(f"  LGS mean dist:    {np.mean(lgs_dists[-10:]):.4f}")
print(f"  Copula mean dist: {np.mean(cop_dists[-10:]):.4f}")

# Score distribution
print("\n" + "="*80)
print("Score distribution")
print("="*80)

lgs_scores = [y for x, y in points_lgs]
cop_scores = [y for x, y in points_cop]

print(f"\nLGS scores:")
print(f"  Mean: {np.mean(lgs_scores):.4f}")
print(f"  Best: {np.min(lgs_scores):.6f}")
print(f"  <0.1: {sum(1 for s in lgs_scores if s < 0.1)} points")

print(f"\nCopula scores:")
print(f"  Mean: {np.mean(cop_scores):.4f}")
print(f"  Best: {np.min(cop_scores):.6f}")
print(f"  <0.1: {sum(1 for s in cop_scores if s < 0.1)} points")

# Check what copula sampling adds
print("\n" + "="*80)
print("CRITICAL: Are samples identical in first 10 steps?")
print("="*80)

identical = 0
for i in range(10):
    x_l, _ = points_lgs[i]
    x_c, _ = points_cop[i]
    if np.allclose(x_l, x_c):
        identical += 1

print(f"First 10 steps: {identical}/10 identical")
print("(Same because both use same seed and same init sampling)")

print("\n" + "="*80)
print("After init phase, are samples different?")
print("="*80)

for i in range(10, 20):
    x_l, y_l = points_lgs[i]
    x_c, y_c = points_cop[i]
    dist_l = np.linalg.norm(x_l - 0.5)
    dist_c = np.linalg.norm(x_c - 0.5)
    same = "SAME" if np.allclose(x_l, x_c) else "DIFF"
    print(f"Step {i}: LGS dist={dist_l:.3f}, y={y_l:.4f} | Cop dist={dist_c:.3f}, y={y_c:.4f} | {same}")
