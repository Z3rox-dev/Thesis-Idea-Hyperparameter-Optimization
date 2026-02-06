#!/usr/bin/env python3
"""
DEBUG INTENSIVO: Cosa succede dentro ALBA-Copula?
Stampa tutti i numeri per capire cosa si rompe.
"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
np.random.seed(42)

# Fix numpy compatibility
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int

from scipy.stats import norm
from scipy.stats import gaussian_kde

# =============================================================================
# VERSIONE STANDALONE delle funzioni copula per debug
# =============================================================================

def fit_copula_standalone(X: np.ndarray, scores: np.ndarray, gamma: float = 0.2, noise_var: float = 0.01):
    """Fit copula model from raw X, scores arrays (standalone for debug)."""
    n, dim = X.shape
    
    # Select elite points
    n_elite = max(3, int(gamma * n))
    elite_idx = np.argsort(scores)[-n_elite:]
    elite_points = X[elite_idx]
    elite_scores = scores[elite_idx]
    
    result = {
        "n_elite": n_elite,
        "elite_idx": elite_idx,
        "elite_points": elite_points,
        "elite_scores": elite_scores,
        "all_points": X,
        "all_scores": scores,
        "copula": None,
        "corr_matrix": None,
        "marginals": None,
        "noise_var": noise_var,
    }
    
    if n_elite < 3:
        return result
    
    # Compute empirical marginals (percentile ranks)
    try:
        # Transform to uniform marginals using rank
        u_elite = np.zeros_like(elite_points)
        for d in range(dim):
            ranks = np.argsort(np.argsort(elite_points[:, d]))
            u_elite[:, d] = (ranks + 0.5) / n_elite
        
        # Clip to avoid infinities in norm.ppf
        u_elite = np.clip(u_elite, 0.001, 0.999)
        
        # Transform to normal
        z_elite = norm.ppf(u_elite)
        
        # Compute correlation matrix
        if n_elite > dim:
            corr_matrix = np.corrcoef(z_elite.T)
            # Ensure valid correlation matrix
            corr_matrix = np.clip(corr_matrix, -0.999, 0.999)
            np.fill_diagonal(corr_matrix, 1.0)
        else:
            corr_matrix = np.eye(dim)
        
        result["copula"] = True  # Mark as fitted
        result["corr_matrix"] = corr_matrix
        result["marginals"] = {"elite_points": elite_points}
        
    except Exception as e:
        print(f"    [DEBUG] Copula fit failed: {e}")
        result["corr_matrix"] = np.eye(dim)
    
    return result


def sample_copula_standalone(model: dict, n_samples: int, bounds: np.ndarray, rng=None):
    """Sample from copula model (standalone for debug)."""
    if rng is None:
        rng = np.random.default_rng()
    
    dim = bounds.shape[0]
    elite_points = model["elite_points"]
    corr_matrix = model["corr_matrix"]
    
    if corr_matrix is None or model["copula"] is None:
        # Fallback: perturb elite points
        samples = []
        for _ in range(n_samples):
            base = elite_points[rng.integers(len(elite_points))]
            noise = rng.normal(0, 0.1, dim)
            sample = np.clip(base + noise, bounds[:, 0], bounds[:, 1])
            samples.append(sample)
        return np.array(samples)
    
    # Sample from multivariate normal with correlation structure
    try:
        z_samples = rng.multivariate_normal(np.zeros(dim), corr_matrix, size=n_samples)
        u_samples = norm.cdf(z_samples)  # Transform to uniform
        
        # Map back to original space using elite point ranges
        samples = np.zeros((n_samples, dim))
        for d in range(dim):
            lo, hi = bounds[d]
            elite_d = elite_points[:, d]
            # Use interpolation between elite min/max
            elite_min, elite_max = elite_d.min(), elite_d.max()
            # Expand slightly
            margin = 0.1 * (elite_max - elite_min + 0.01)
            effective_lo = max(lo, elite_min - margin)
            effective_hi = min(hi, elite_max + margin)
            samples[:, d] = effective_lo + u_samples[:, d] * (effective_hi - effective_lo)
        
        return np.clip(samples, bounds[:, 0], bounds[:, 1])
        
    except Exception as e:
        print(f"    [DEBUG] Copula sample failed: {e}")
        return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, dim))


def predict_copula_standalone(model: dict, X: np.ndarray):
    """Predict mu, sigma using k-NN on all points (standalone for debug)."""
    all_points = model["all_points"]
    all_scores = model["all_scores"]
    noise_var = model.get("noise_var", 0.01)
    
    n_test = X.shape[0]
    mu = np.zeros(n_test)
    sigma = np.zeros(n_test)
    
    k = min(5, len(all_points))
    
    # Global score stats
    score_std = np.std(all_scores) if len(all_scores) > 1 else 1.0
    score_std = max(score_std, 0.01)
    
    for i in range(n_test):
        dists = np.linalg.norm(all_points - X[i], axis=1)
        knn_idx = np.argsort(dists)[:k]
        knn_dists = dists[knn_idx]
        knn_scores = all_scores[knn_idx]
        
        # Inverse distance weighting
        if knn_dists[0] < 1e-8:
            weights = np.zeros(k)
            weights[0] = 1.0
        else:
            weights = 1.0 / (knn_dists + 1e-8)
            weights /= weights.sum()
        
        mu[i] = np.dot(weights, knn_scores)
        
        # FIXED: Cap sigma to prevent exploration dominating
        if k > 1:
            local_var = np.std(knn_scores)
            sigma[i] = min(local_var, 0.5 * score_std) + 0.01
        else:
            sigma[i] = np.sqrt(noise_var)
        
        sigma[i] = max(sigma[i], 0.01)
    
    return mu, sigma

# Aliases for the rest of the script
fit_copula_model = fit_copula_standalone
sample_from_copula = sample_copula_standalone
predict_copula = predict_copula_standalone

print("="*70)
print("DEBUG SESSIONE 1: Funzione sintetica semplice (Sphere)")
print("="*70)

# Creiamo dati sintetici - funzione sphere
dim = 5
n_points = 30

# Punti random nel cubo [0,1]^5
X = np.random.rand(n_points, dim)
# Scores: sphere function (minimo in 0.5, 0.5, ...)
scores = -np.sum((X - 0.5)**2, axis=1)  # Negativo perché massimizziamo

print(f"\n[1] DATI INPUT:")
print(f"    Dimensioni: {dim}")
print(f"    Punti: {n_points}")
print(f"    X shape: {X.shape}")
print(f"    Scores range: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"    Best score: {scores.max():.4f} at index {scores.argmax()}")
print(f"    Best point: {X[scores.argmax()]}")

# Fit copula model
print(f"\n[2] FIT COPULA MODEL:")
model = fit_copula_model(X, scores, noise_var=0.01)

print(f"    Model keys: {list(model.keys())}")
print(f"    n_elite: {model['n_elite']}")
print(f"    Elite indices: {model['elite_idx'][:10]}..." if len(model['elite_idx']) > 10 else f"    Elite indices: {model['elite_idx']}")
print(f"    Elite scores: {model['elite_scores'][:5]}..." if len(model['elite_scores']) > 5 else f"    Elite scores: {model['elite_scores']}")

if model['copula'] is not None:
    print(f"    Copula fitted: YES")
    print(f"    Copula type: {type(model['copula'])}")
else:
    print(f"    Copula fitted: NO (fallback mode)")

print(f"    Correlation matrix shape: {model['corr_matrix'].shape if model['corr_matrix'] is not None else 'None'}")
if model['corr_matrix'] is not None:
    print(f"    Correlation matrix:\n{model['corr_matrix']}")

# Sample from copula
print(f"\n[3] SAMPLE FROM COPULA:")
n_samples = 20
samples = sample_from_copula(model, n_samples, bounds=np.array([[0,1]]*dim))

print(f"    Requested samples: {n_samples}")
print(f"    Got samples shape: {samples.shape}")
print(f"    Samples range per dim:")
for d in range(dim):
    print(f"      Dim {d}: [{samples[:,d].min():.3f}, {samples[:,d].max():.3f}] mean={samples[:,d].mean():.3f}")

# Verifica: i campioni sono vicini agli elite?
elite_points = model['elite_points']
print(f"\n    Distance from samples to nearest elite:")
for i in range(min(5, n_samples)):
    dists = np.linalg.norm(elite_points - samples[i], axis=1)
    print(f"      Sample {i}: min_dist={dists.min():.4f}, mean_dist={dists.mean():.4f}")

# Predict
print(f"\n[4] PREDICT (mu, sigma):")
test_points = np.random.rand(10, dim)
mu, sigma = predict_copula(model, test_points)

print(f"    Test points: {len(test_points)}")
print(f"    Mu range: [{mu.min():.4f}, {mu.max():.4f}]")
print(f"    Sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
print(f"    Mu values: {mu}")
print(f"    Sigma values: {sigma}")

# Test su punto ottimo vs punto pessimo
optimal_point = np.array([[0.5]*dim])
bad_point = np.array([[0.0]*dim])

mu_opt, sigma_opt = predict_copula(model, optimal_point)
mu_bad, sigma_bad = predict_copula(model, bad_point)

print(f"\n    Punto ottimo [0.5, 0.5, ...]: mu={mu_opt[0]:.4f}, sigma={sigma_opt[0]:.4f}")
print(f"    Punto pessimo [0.0, 0.0, ...]: mu={mu_bad[0]:.4f}, sigma={sigma_bad[0]:.4f}")

if mu_opt[0] > mu_bad[0]:
    print(f"    ✓ CORRETTO: mu(ottimo) > mu(pessimo)")
else:
    print(f"    ✗ ERRORE: mu(ottimo) <= mu(pessimo) - la predizione è sbagliata!")

# UCB check
beta = 2.0
ucb_opt = mu_opt[0] + beta * sigma_opt[0]
ucb_bad = mu_bad[0] + beta * sigma_bad[0]
print(f"\n    UCB (beta=2): ottimo={ucb_opt:.4f}, pessimo={ucb_bad:.4f}")
if ucb_opt > ucb_bad:
    print(f"    ✓ UCB preferisce punto ottimo")
else:
    print(f"    ✗ UCB preferisce punto pessimo!")

print("\n" + "="*70)
print("DEBUG SESSIONE 2: Cosa succede con pochi punti?")
print("="*70)

# Solo 5 punti
X_small = np.random.rand(5, dim)
scores_small = -np.sum((X_small - 0.5)**2, axis=1)

print(f"\n[1] Solo 5 punti:")
model_small = fit_copula_model(X_small, scores_small, noise_var=0.01)
print(f"    Copula fitted: {model_small['copula'] is not None}")
print(f"    n_elite: {model_small['n_elite']}")

samples_small = sample_from_copula(model_small, 10, bounds=np.array([[0,1]]*dim))
print(f"    Samples shape: {samples_small.shape}")
print(f"    Samples are NaN: {np.isnan(samples_small).any()}")
print(f"    Samples are Inf: {np.isinf(samples_small).any()}")

print("\n" + "="*70)
print("DEBUG SESSIONE 3: Alta dimensionalità (come ParamNet)")
print("="*70)

dim_high = 8
n_points_high = 50

X_high = np.random.rand(n_points_high, dim_high)
scores_high = -np.sum((X_high - 0.5)**2, axis=1)

print(f"\n[1] 8D con 50 punti:")
model_high = fit_copula_model(X_high, scores_high, noise_var=0.01)
print(f"    Copula fitted: {model_high['copula'] is not None}")
print(f"    n_elite: {model_high['n_elite']}")
print(f"    Correlation matrix diagonal: {np.diag(model_high['corr_matrix']) if model_high['corr_matrix'] is not None else 'None'}")

# Controlliamo se la matrice di correlazione è valida
if model_high['corr_matrix'] is not None:
    eigvals = np.linalg.eigvalsh(model_high['corr_matrix'])
    print(f"    Eigenvalues of corr matrix: {eigvals}")
    if np.any(eigvals < 0):
        print(f"    ✗ PROBLEMA: Matrice di correlazione NON positiva semi-definita!")
    else:
        print(f"    ✓ Matrice di correlazione valida")

samples_high = sample_from_copula(model_high, 20, bounds=np.array([[0,1]]*dim_high))
print(f"    Samples NaN: {np.isnan(samples_high).any()}")
print(f"    Samples out of bounds: {(samples_high < 0).any() or (samples_high > 1).any()}")

print("\n" + "="*70)
print("DEBUG SESSIONE 4: Simulazione iterazione ALBA")
print("="*70)

# Simuliamo cosa succede in un loop di ottimizzazione
dim = 5
budget = 50

# Init random
X_history = np.random.rand(10, dim)
scores_history = -np.sum((X_history - 0.5)**2, axis=1)

print(f"\nSimulazione {budget} iterazioni:")
for iteration in range(budget):
    model = fit_copula_model(X_history, scores_history, noise_var=0.01)
    
    # Genera candidati
    n_copula = 5
    n_random = 15
    
    if model['copula'] is not None:
        copula_samples = sample_from_copula(model, n_copula, bounds=np.array([[0,1]]*dim))
    else:
        copula_samples = np.random.rand(n_copula, dim)
    
    random_samples = np.random.rand(n_random, dim)
    candidates = np.vstack([copula_samples, random_samples])
    
    # Predici e seleziona con UCB
    mu, sigma = predict_copula(model, candidates)
    ucb = mu + 2.0 * sigma
    
    # Check for NaN/Inf
    if np.isnan(ucb).any() or np.isinf(ucb).any():
        print(f"  Iter {iteration}: ✗ UCB contiene NaN/Inf!")
        print(f"    mu: {mu}")
        print(f"    sigma: {sigma}")
        break
    
    best_idx = np.argmax(ucb)
    next_point = candidates[best_idx]
    next_score = -np.sum((next_point - 0.5)**2)
    
    X_history = np.vstack([X_history, next_point])
    scores_history = np.append(scores_history, next_score)
    
    if iteration % 10 == 0 or iteration == budget - 1:
        print(f"  Iter {iteration}: best_so_far={scores_history.max():.4f}, n_elite={model['n_elite']}, copula={'YES' if model['copula'] else 'NO'}")

print(f"\nRisultato finale:")
print(f"  Best score: {scores_history.max():.4f} (ottimo teorico = 0.0)")
print(f"  Best point: {X_history[scores_history.argmax()]}")
print(f"  Distance from optimum: {np.linalg.norm(X_history[scores_history.argmax()] - 0.5):.4f}")

print("\n" + "="*70)
print("FINE DEBUG")
print("="*70)
