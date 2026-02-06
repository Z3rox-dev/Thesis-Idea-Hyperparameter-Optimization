#!/usr/bin/env python3
"""
DEEP DEBUG: Analisi numerica dettagliata di cosa succede dentro ALBA.
Mostra ogni singolo passaggio con i numeri reali.
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

# Import entrambi i framework
from alba_framework_potential import ALBA as ALBA_LGS
from alba_framework_copula.optimizer import ALBA as ALBA_Copula

print("="*80)
print("DEEP DEBUG: Analisi Numerica Interna ALBA")
print("="*80)

# Funzione semplice per debugging
def sphere(x):
    """Sphere function - minimum at [0.5, 0.5, ...]"""
    return float(np.sum((x - 0.5)**2))

dim = 5
bounds = [(0, 1)] * dim

# ============================================================================
# TEST 1: Inizializzazione e primi step
# ============================================================================
print("\n" + "="*80)
print("TEST 1: Confronto inizializzazione")
print("="*80)

opt_lgs = ALBA_LGS(bounds=bounds, seed=42, total_budget=50, maximize=False)
opt_cop = ALBA_Copula(bounds=bounds, seed=42, total_budget=50, maximize=False)

print(f"\nALBA-LGS initialized:")
print(f"  dim: {opt_lgs.dim}")
print(f"  bounds: {opt_lgs.bounds[:2]}... (first 2)")
print(f"  total_budget: {opt_lgs.total_budget}")

print(f"\nALBA-Copula initialized:")
print(f"  dim: {opt_cop.dim}")
print(f"  bounds: {opt_cop.bounds[:2]}... (first 2)")
print(f"  total_budget: {opt_cop.total_budget}")

# ============================================================================
# TEST 2: Primi 10 step - cosa campiona?
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Primi 10 step - punti campionati")
print("="*80)

print("\nALBA-LGS sampling:")
for i in range(10):
    x = opt_lgs.ask()
    y = sphere(x)
    opt_lgs.tell(x, y)
    print(f"  Step {i}: x=[{x[0]:.3f}, {x[1]:.3f}, ...] -> y={y:.4f}")

print(f"\n  Best so far: {opt_lgs.best_y:.4f}")

print("\nALBA-Copula sampling:")
for i in range(10):
    x = opt_cop.ask()
    y = sphere(x)
    opt_cop.tell(x, y)
    print(f"  Step {i}: x=[{x[0]:.3f}, {x[1]:.3f}, ...] -> y={y:.4f}")

print(f"\n  Best so far: {opt_cop.best_y:.4f}")

# ============================================================================
# TEST 3: Analisi del modello LGS interno
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Analisi modello LGS dopo 10 punti")
print("="*80)

# Access the root cube
root_lgs = opt_lgs.root
root_cop = opt_cop.root

print(f"\nLGS Root Cube:")
print(f"  n_trials: {root_lgs.n_trials}")
print(f"  n_good: {root_lgs.n_good}")
print(f"  best_score: {root_lgs.best_score:.4f}")
print(f"  depth: {root_lgs.depth}")

if root_lgs.lgs_model is not None:
    model = root_lgs.lgs_model
    print(f"\n  LGS Model:")
    print(f"    gradient_dir: {model.get('gradient_dir')}")
    print(f"    y_mean: {model.get('y_mean', 'N/A'):.4f}" if model.get('y_mean') else "    y_mean: N/A")
    print(f"    noise_var: {model.get('noise_var', 'N/A'):.6f}" if model.get('noise_var') else "    noise_var: N/A")
    print(f"    top_k_pts shape: {np.array(model.get('top_k_pts', [])).shape}")
    if model.get('inv_cov') is not None:
        print(f"    inv_cov diagonal: {np.diag(model['inv_cov'])}")
else:
    print("  No LGS model yet")

print(f"\nCopula Root Cube:")
print(f"  n_trials: {root_cop.n_trials}")
print(f"  n_good: {root_cop.n_good}")
print(f"  best_score: {root_cop.best_score:.4f}")
print(f"  depth: {root_cop.depth}")

if root_cop.lgs_model is not None:
    model = root_cop.lgs_model
    print(f"\n  Copula Model (hybrid):")
    print(f"    gradient_dir: {model.get('gradient_dir')}")
    print(f"    y_mean: {model.get('y_mean', 'N/A'):.4f}" if model.get('y_mean') else "    y_mean: N/A")
    print(f"    noise_var: {model.get('noise_var', 'N/A'):.6f}" if model.get('noise_var') else "    noise_var: N/A")
    print(f"    L (Cholesky): {'present' if model.get('L') is not None else 'None'}")
    print(f"    marginal_params: {model.get('marginal_params')[:2] if model.get('marginal_params') else 'None'}...")
    print(f"    elite_pts shape: {np.array(model.get('elite_pts', [])).shape}")
else:
    print("  No model yet")

# ============================================================================
# TEST 4: Prediction comparison
# ============================================================================
print("\n" + "="*80)
print("TEST 4: Confronto prediction mu/sigma")
print("="*80)

# Test points
test_points = np.array([
    [0.5, 0.5, 0.5, 0.5, 0.5],  # Optimum
    [0.0, 0.0, 0.0, 0.0, 0.0],  # Corner (bad)
    [1.0, 1.0, 1.0, 1.0, 1.0],  # Opposite corner (bad)
    [0.3, 0.4, 0.5, 0.6, 0.7],  # Near optimum
])

print("\nTest points and their true values:")
for i, p in enumerate(test_points):
    print(f"  Point {i}: [{p[0]:.1f}, {p[1]:.1f}, ...] -> true_y={sphere(p):.4f}")

# Get predictions from LGS
if root_lgs.lgs_model is not None and root_lgs.lgs_model.get('inv_cov') is not None:
    from alba_framework_potential.lgs import predict_bayesian
    mu_lgs, sigma_lgs = predict_bayesian(root_lgs.lgs_model, test_points)
    
    print("\nLGS predictions:")
    for i in range(len(test_points)):
        ucb = mu_lgs[i] - 2.0 * sigma_lgs[i]  # Minimize -> -sigma
        print(f"  Point {i}: mu={mu_lgs[i]:.4f}, sigma={sigma_lgs[i]:.4f}, UCB={ucb:.4f}")
else:
    print("\nLGS: No model for prediction")

# Get predictions from Copula (using LGS prediction in hybrid)
if root_cop.lgs_model is not None and root_cop.lgs_model.get('inv_cov') is not None:
    from alba_framework_potential.lgs import predict_bayesian
    mu_cop, sigma_cop = predict_bayesian(root_cop.lgs_model, test_points)
    
    print("\nCopula (hybrid) predictions:")
    for i in range(len(test_points)):
        ucb = mu_cop[i] - 2.0 * sigma_cop[i]
        print(f"  Point {i}: mu={mu_cop[i]:.4f}, sigma={sigma_cop[i]:.4f}, UCB={ucb:.4f}")
else:
    print("\nCopula: No model for prediction")

# ============================================================================
# TEST 5: Candidate generation analysis
# ============================================================================
print("\n" + "="*80)
print("TEST 5: Analisi generazione candidati")
print("="*80)

# Generate candidates from both
from alba_framework_potential.candidates import MixtureCandidateGenerator as LGSGen
from alba_framework_copula.candidates import MixtureCandidateGenerator as CopulaGen

rng = np.random.default_rng(123)
n_cands = 20

lgs_gen = LGSGen()
cop_gen = CopulaGen()

cands_lgs = lgs_gen.generate(root_lgs, dim, rng, n_cands)
cands_cop = cop_gen.generate(root_cop, dim, rng, n_cands)

print(f"\nLGS candidates ({len(cands_lgs)} generated):")
print(f"  Mean position: [{np.mean([c[0] for c in cands_lgs]):.3f}, {np.mean([c[1] for c in cands_lgs]):.3f}, ...]")
print(f"  Std position:  [{np.std([c[0] for c in cands_lgs]):.3f}, {np.std([c[1] for c in cands_lgs]):.3f}, ...]")
print(f"  Range dim 0:   [{min(c[0] for c in cands_lgs):.3f}, {max(c[0] for c in cands_lgs):.3f}]")

# Distance to optimum
dists_lgs = [np.linalg.norm(c - 0.5) for c in cands_lgs]
print(f"  Dist to opt:   mean={np.mean(dists_lgs):.3f}, min={np.min(dists_lgs):.3f}")

print(f"\nCopula candidates ({len(cands_cop)} generated):")
print(f"  Mean position: [{np.mean([c[0] for c in cands_cop]):.3f}, {np.mean([c[1] for c in cands_cop]):.3f}, ...]")
print(f"  Std position:  [{np.std([c[0] for c in cands_cop]):.3f}, {np.std([c[1] for c in cands_cop]):.3f}, ...]")
print(f"  Range dim 0:   [{min(c[0] for c in cands_cop):.3f}, {max(c[0] for c in cands_cop):.3f}]")

dists_cop = [np.linalg.norm(c - 0.5) for c in cands_cop]
print(f"  Dist to opt:   mean={np.mean(dists_cop):.3f}, min={np.min(dists_cop):.3f}")

# ============================================================================
# TEST 6: Continuation - cosa succede dopo 30 step?
# ============================================================================
print("\n" + "="*80)
print("TEST 6: Evoluzione dopo 30 step totali")
print("="*80)

# Continue optimization
for i in range(20):  # 20 more steps (30 total)
    x_lgs = opt_lgs.ask()
    y_lgs = sphere(x_lgs)
    opt_lgs.tell(x_lgs, y_lgs)
    
    x_cop = opt_cop.ask()
    y_cop = sphere(x_cop)
    opt_cop.tell(x_cop, y_cop)

print(f"\nAfter 30 steps:")
print(f"  LGS best:    {opt_lgs.best_y:.6f}")
print(f"  Copula best: {opt_cop.best_y:.6f}")

# Check cube splitting
def count_leaves(cube):
    kids = getattr(cube, 'children', None) or getattr(cube, '_children', [])
    if not kids:
        return 1
    return sum(count_leaves(c) for c in kids)

n_leaves_lgs = count_leaves(opt_lgs.root)
n_leaves_cop = count_leaves(opt_cop.root)

print(f"\nCube structure:")
print(f"  LGS leaves:    {n_leaves_lgs}")
print(f"  Copula leaves: {n_leaves_cop}")

# ============================================================================
# TEST 7: Analisi di un singolo step in dettaglio
# ============================================================================
print("\n" + "="*80)
print("TEST 7: Analisi di UN SINGOLO STEP in dettaglio")
print("="*80)

print("\n--- LGS Step 31 ---")

# Before ask
print(f"Before ask: iteration={opt_lgs.iteration}")

x_lgs = opt_lgs.ask()
print(f"Asked point: [{x_lgs[0]:.4f}, {x_lgs[1]:.4f}, {x_lgs[2]:.4f}, {x_lgs[3]:.4f}, {x_lgs[4]:.4f}]")

# Which cube was selected?
cube_selected = getattr(opt_lgs, '_last_cube', opt_lgs.root)
print(f"Selected cube depth: {cube_selected.depth}")
print(f"Cube bounds[0]: [{cube_selected.bounds[0][0]:.3f}, {cube_selected.bounds[0][1]:.3f}]")
print(f"Cube n_trials: {cube_selected.n_trials}")

y_lgs = sphere(x_lgs)
print(f"Evaluated: y={y_lgs:.6f}")

opt_lgs.tell(x_lgs, y_lgs)
print(f"After tell: best_y={opt_lgs.best_y:.6f}")

print("\n--- Copula Step 31 ---")

x_cop = opt_cop.ask()
print(f"Asked point: [{x_cop[0]:.4f}, {x_cop[1]:.4f}, {x_cop[2]:.4f}, {x_cop[3]:.4f}, {x_cop[4]:.4f}]")

cube_selected = getattr(opt_cop, '_last_cube', opt_cop.root)
print(f"Selected cube depth: {cube_selected.depth}")
print(f"Cube bounds[0]: [{cube_selected.bounds[0][0]:.3f}, {cube_selected.bounds[0][1]:.3f}]")
print(f"Cube n_trials: {cube_selected.n_trials}")

y_cop = sphere(x_cop)
print(f"Evaluated: y={y_cop:.6f}")

opt_cop.tell(x_cop, y_cop)
print(f"After tell: best_y={opt_cop.best_y:.6f}")

# ============================================================================
# TEST 8: Full run comparison
# ============================================================================
print("\n" + "="*80)
print("TEST 8: Run completo 50 step - convergence")
print("="*80)

# Fresh optimizers
opt_lgs2 = ALBA_LGS(bounds=bounds, seed=99, total_budget=50, maximize=False)
opt_cop2 = ALBA_Copula(bounds=bounds, seed=99, total_budget=50, maximize=False)

history_lgs = []
history_cop = []

for i in range(50):
    x = opt_lgs2.ask()
    y = sphere(x)
    opt_lgs2.tell(x, y)
    history_lgs.append(opt_lgs2.best_y)
    
    x = opt_cop2.ask()
    y = sphere(x)
    opt_cop2.tell(x, y)
    history_cop.append(opt_cop2.best_y)

print("\nConvergence curve:")
print(f"{'Step':<6} {'LGS':<12} {'Copula':<12} {'Diff':<12}")
for i in [0, 5, 10, 20, 30, 40, 49]:
    diff = history_lgs[i] - history_cop[i]
    marker = "←LGS" if diff < 0 else "←COP" if diff > 0 else "="
    print(f"{i:<6} {history_lgs[i]:<12.6f} {history_cop[i]:<12.6f} {diff:+.6f} {marker}")

print(f"\nFinal results:")
print(f"  LGS:    {history_lgs[-1]:.6f}")
print(f"  Copula: {history_cop[-1]:.6f}")
print(f"  Winner: {'LGS' if history_lgs[-1] < history_cop[-1] else 'Copula'}")

# ============================================================================
# TEST 9: Analisi statistica candidati UCB
# ============================================================================
print("\n" + "="*80)
print("TEST 9: Analisi UCB acquisition su candidati")
print("="*80)

# Generate candidates and compute UCB
if root_lgs.lgs_model is not None and root_lgs.lgs_model.get('inv_cov') is not None:
    cands = lgs_gen.generate(root_lgs, dim, rng, 50)
    cands_arr = np.array(cands)
    
    mu, sigma = predict_bayesian(root_lgs.lgs_model, cands_arr)
    ucb = mu - 2.0 * sigma  # Minimize
    
    print("\nLGS: 50 candidates UCB analysis")
    print(f"  mu range:    [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"  sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
    print(f"  UCB range:   [{ucb.min():.4f}, {ucb.max():.4f}]")
    
    best_idx = np.argmin(ucb)
    print(f"\n  Best candidate (idx={best_idx}):")
    print(f"    x = [{cands[best_idx][0]:.3f}, {cands[best_idx][1]:.3f}, ...]")
    print(f"    mu={mu[best_idx]:.4f}, sigma={sigma[best_idx]:.4f}, UCB={ucb[best_idx]:.4f}")
    print(f"    true_y = {sphere(cands[best_idx]):.4f}")
    print(f"    dist_to_opt = {np.linalg.norm(cands[best_idx] - 0.5):.4f}")

print("\n" + "="*80)
print("DEBUG COMPLETO")
print("="*80)
