"""
Indagine profonda: PERCHÉ il gradiente LGS esplode?

Ipotesi da verificare:
1. y_std molto grande scala il gradiente
2. Matrice XtWX mal condizionata → inv_cov esplode
3. Weights sbilanciati
4. Correlazione tra punti (collinearità)
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis/alba_framework_potential')

# Disabilita temporaneamente il clipping per vedere il problema reale
# Copio la logica di fit_lgs_model manualmente

def analyze_lgs_numerics(all_pts, all_scores, widths, center, dim):
    """Analizza passo-passo cosa succede nel fitting LGS."""
    
    report = {"steps": []}
    
    # Step 1: Normalizzazione X
    X_norm = (all_pts - center) / widths
    report["X_norm_range"] = (X_norm.min(), X_norm.max())
    report["X_norm_cond"] = np.linalg.cond(X_norm.T @ X_norm + 1e-6 * np.eye(dim))
    
    # Step 2: Normalizzazione y
    y_mean = all_scores.mean()
    y_std = all_scores.std() + 1e-6
    y_centered = (all_scores - y_mean) / y_std
    
    report["y_raw_range"] = (all_scores.min(), all_scores.max())
    report["y_std"] = y_std
    report["y_centered_range"] = (y_centered.min(), y_centered.max())
    
    # Step 3: Calcolo pesi
    dists_sq = np.sum(X_norm**2, axis=1)
    sigma_sq = np.mean(dists_sq) + 1e-6
    weights = np.exp(-dists_sq / (2 * sigma_sq))
    
    rank_weights = 1.0 + 0.5 * (all_scores - all_scores.min()) / (all_scores.ptp() + 1e-9)
    weights = weights * rank_weights
    
    report["weights_range"] = (weights.min(), weights.max())
    report["weights_ratio"] = weights.max() / (weights.min() + 1e-12)
    
    W = np.diag(weights)
    
    # Step 4: XtWX e regolarizzazione
    n_pts = len(all_pts)
    lambda_base = 0.1 * (1 + dim / max(n_pts - dim, 1))
    XtWX = X_norm.T @ W @ X_norm
    
    report["XtWX_diag"] = np.diag(XtWX)
    report["XtWX_cond_raw"] = np.linalg.cond(XtWX)
    report["lambda_base"] = lambda_base
    
    XtWX_reg = XtWX + lambda_base * np.eye(dim)
    report["XtWX_reg_cond"] = np.linalg.cond(XtWX_reg)
    
    # Step 5: Inversione
    inv_cov = np.linalg.inv(XtWX_reg)
    report["inv_cov_max"] = np.abs(inv_cov).max()
    report["inv_cov_diag"] = np.diag(inv_cov)
    
    # Step 6: Calcolo gradiente
    XtWy = X_norm.T @ W @ y_centered
    report["XtWy_norm"] = np.linalg.norm(XtWy)
    
    grad_normalized = inv_cov @ XtWy  # Questo è in spazio normalizzato
    report["grad_before_ystd"] = np.linalg.norm(grad_normalized)
    
    grad = grad_normalized * y_std  # QUESTO È IL MOLTIPLICATORE CHIAVE
    report["grad_after_ystd"] = np.linalg.norm(grad)
    
    # Fattorizzazione del problema
    report["explosion_factor"] = y_std  # Se y_std è enorme, il gradiente esplode
    
    return report, grad


def test_function_gradient_analysis(func_name, func, dim=5, n_samples=50):
    """Testa una funzione e analizza da dove viene l'esplosione."""
    
    print(f"\n{'='*70}")
    print(f"  ANALISI: {func_name}")
    print(f"{'='*70}")
    
    # Genera punti random
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (n_samples, dim))
    scores = np.array([func(p) for p in pts])
    
    widths = np.ones(dim)
    center = np.full(dim, 0.5)
    
    report, grad = analyze_lgs_numerics(pts, scores, widths, center, dim)
    
    print(f"\n📊 DATI RAW:")
    print(f"   y range: [{report['y_raw_range'][0]:.2e}, {report['y_raw_range'][1]:.2e}]")
    print(f"   y_std:   {report['y_std']:.2e}")
    
    print(f"\n📐 NORMALIZZAZIONE:")
    print(f"   X_norm range: [{report['X_norm_range'][0]:.4f}, {report['X_norm_range'][1]:.4f}]")
    print(f"   y_centered range: [{report['y_centered_range'][0]:.4f}, {report['y_centered_range'][1]:.4f}]")
    
    print(f"\n⚖️ PESI:")
    print(f"   weights range: [{report['weights_range'][0]:.4f}, {report['weights_range'][1]:.4f}]")
    print(f"   weights ratio (max/min): {report['weights_ratio']:.2e}")
    
    print(f"\n🔢 MATRICI:")
    print(f"   XtWX cond (raw):  {report['XtWX_cond_raw']:.2e}")
    print(f"   lambda_base:      {report['lambda_base']:.4f}")
    print(f"   XtWX_reg cond:    {report['XtWX_reg_cond']:.2e}")
    print(f"   inv_cov max elem: {report['inv_cov_max']:.2e}")
    
    print(f"\n📈 GRADIENTE (decomposizione):")
    print(f"   ||XtWy||:              {report['XtWy_norm']:.4e}")
    print(f"   ||grad_normalized||:   {report['grad_before_ystd']:.4e}")
    print(f"   y_std (moltiplicatore): {report['y_std']:.4e}")
    print(f"   ||grad_final||:        {report['grad_after_ystd']:.4e}")
    
    # DIAGNOSI
    print(f"\n🔍 DIAGNOSI:")
    if report['y_std'] > 1e3:
        print(f"   ⚠️  y_std={report['y_std']:.2e} >> 1 → IL PROBLEMA È LO SCALING DELLE Y!")
        print(f"       Il gradiente in spazio normalizzato è {report['grad_before_ystd']:.2e}")
        print(f"       Ma viene moltiplicato per y_std, portandolo a {report['grad_after_ystd']:.2e}")
    if report['XtWX_reg_cond'] > 1e6:
        print(f"   ⚠️  Matrice mal condizionata: cond={report['XtWX_reg_cond']:.2e}")
    if report['weights_ratio'] > 1e6:
        print(f"   ⚠️  Pesi sbilanciati: ratio={report['weights_ratio']:.2e}")
    
    return report


# Funzioni di test
def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

def ill_conditioned(x):
    """Funzione con scaling esponenziale tra dimensioni."""
    scales = np.array([10**i for i in range(len(x))])
    return np.sum(scales * x**2)

def rastrigin(x):
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)


if __name__ == "__main__":
    funcs = [
        ("Sphere", sphere),
        ("Rosenbrock", rosenbrock),
        ("IllConditioned", ill_conditioned),
        ("Rastrigin", rastrigin),
    ]
    
    reports = {}
    for name, func in funcs:
        reports[name] = test_function_gradient_analysis(name, func)
    
    print("\n" + "="*70)
    print("  CONFRONTO FINALE")
    print("="*70)
    print(f"\n{'Funzione':<20} {'y_std':>12} {'grad_norm':>12} {'grad/y_std':>12}")
    print("-"*60)
    for name, r in reports.items():
        ratio = r['grad_after_ystd'] / r['y_std'] if r['y_std'] > 0 else 0
        print(f"{name:<20} {r['y_std']:>12.2e} {r['grad_after_ystd']:>12.2e} {ratio:>12.4f}")
    
    print("\n" + "="*70)
    print("  SOLUZIONE PROPOSTA")
    print("="*70)
    print("""
Il problema NON è nel gradiente in sé - è nel fatto che 
`grad = grad_normalized * y_std` moltiplica per la scala delle y.

Ma poi questo gradiente viene usato in predict_bayesian per calcolare:
   mu = y_mean + C_norm @ grad

Qui C_norm è in [-1, 1]^d, quindi il prodotto scalare ha valori ragionevoli.
Ma grad ha magnitudine proporzionale a y_std!

SOLUZIONI POSSIBILI (senza bias):
1. Lavorare sempre in spazio normalizzato (y_centered) e de-normalizzare solo alla fine
2. Dividere grad per y_std in predict_bayesian (annulla la moltiplicazione)
3. Usare gradient_dir (già normalizzato) + magnitudine separata clippata

La soluzione 1 è la più pulita: tutto in spazio [0,1] normalizzato.
""")
