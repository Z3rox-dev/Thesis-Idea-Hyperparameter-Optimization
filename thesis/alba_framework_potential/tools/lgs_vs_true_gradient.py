#!/usr/bin/env python3
"""
VERIFICA STIMA LGS vs GRADIENTE VERO
=====================================

Confronto tra:
- Gradiente vero (analitico) di Rastrigin
- Gradiente stimato da LGS con punti campionati

Se LGS stima bene → problema strutturale (multimodalità)
Se LGS stima male → possibile bug
"""

import numpy as np


def rastrigin(x):
    A = 10
    x = np.array(x)
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def grad_rastrigin(x):
    """Gradiente analitico di Rastrigin."""
    x = np.array(x)
    return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)


def estimate_gradient_lgs_style(center, samples_x, samples_y, dim=2):
    """Stima il gradiente come fa LGS."""
    
    # Normalizza
    X = np.array(samples_x)
    y = np.array(samples_y)
    
    center = np.array(center)
    widths = np.ones(dim) * 0.3  # Simula larghezza cubo
    
    X_norm = (X - center) / widths
    y_mean = y.mean()
    y_std = y.std() + 1e-6
    y_centered = (y - y_mean) / y_std
    
    # Pesi
    dists_sq = np.sum(X_norm**2, axis=1)
    sigma_sq = np.mean(dists_sq) + 1e-6
    weights = np.exp(-dists_sq / (2 * sigma_sq))
    
    # Boost per migliori
    rank_weights = 1.0 + 0.5 * (y - y.min()) / (y.ptp() + 1e-9)
    weights = weights * rank_weights
    W = np.diag(weights)
    
    # Regressione
    lambda_reg = 0.1
    XtWX = X_norm.T @ W @ X_norm
    XtWX_reg = XtWX + lambda_reg * np.eye(dim)
    inv_cov = np.linalg.inv(XtWX_reg)
    grad = inv_cov @ (X_norm.T @ W @ y_centered)
    
    # Normalizza
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-9:
        grad_dir = grad / grad_norm
    else:
        grad_dir = grad
    
    return grad_dir, grad


def test_lgs_vs_true_gradient():
    """Confronta LGS con gradiente vero in varie situazioni."""
    
    print("=" * 70)
    print("CONFRONTO LGS vs GRADIENTE VERO SU RASTRIGIN")
    print("=" * 70)
    
    np.random.seed(42)
    
    test_cases = [
        {"name": "Vicino origine (0.1, 0.1)", "center": [0.1, 0.1]},
        {"name": "A metà (0.5, 0.5)", "center": [0.5, 0.5]},
        {"name": "Minimo locale (1.0, 0.0)", "center": [1.0, 0.0]},
        {"name": "Tra minimi (0.5, 0.0)", "center": [0.5, 0.0]},
    ]
    
    for case in test_cases:
        center = np.array(case["center"])
        
        print(f"\n--- {case['name']} ---")
        
        # Gradiente vero
        true_grad = grad_rastrigin(center)
        true_grad_norm = np.linalg.norm(true_grad)
        if true_grad_norm > 1e-9:
            true_grad_dir = true_grad / true_grad_norm
        else:
            true_grad_dir = true_grad
        
        print(f"Gradiente VERO: {true_grad}")
        print(f"Direzione VERA: {true_grad_dir}")
        
        # Genera campioni intorno al centro
        n_samples = 15
        radius = 0.2
        samples_x = []
        samples_y = []
        
        for _ in range(n_samples):
            x = center + np.random.uniform(-radius, radius, 2)
            y = rastrigin(x)
            samples_x.append(x)
            samples_y.append(y)
        
        # Stima LGS
        lgs_grad_dir, lgs_grad = estimate_gradient_lgs_style(center, samples_x, samples_y)
        
        print(f"Gradiente LGS:  {lgs_grad}")
        print(f"Direzione LGS:  {lgs_grad_dir}")
        
        # Alignment
        alignment = np.dot(true_grad_dir, lgs_grad_dir)
        print(f"Alignment (LGS vs VERO): {alignment:.4f}")
        
        if alignment > 0.8:
            print("✓ LGS stima BENE il gradiente")
        elif alignment > 0.3:
            print("~ LGS stima APPROSSIMATIVAMENTE")
        elif alignment > -0.3:
            print("? LGS stima ORTOGONALE (molto diverso)")
        else:
            print("✗ LGS stima OPPOSTO!")


def test_effect_of_sample_density():
    """Verifica se più campioni migliorano la stima."""
    
    print("\n" + "=" * 70)
    print("EFFETTO DENSITÀ CAMPIONI")
    print("=" * 70)
    
    np.random.seed(42)
    center = np.array([0.1, 0.1])
    
    true_grad = grad_rastrigin(center)
    true_grad_dir = true_grad / np.linalg.norm(true_grad)
    
    print(f"Centro: {center}")
    print(f"Gradiente VERO: {true_grad}")
    print(f"Direzione VERA: {true_grad_dir}")
    print()
    
    for n_samples in [5, 10, 20, 50, 100]:
        alignments = []
        
        for trial in range(10):  # Media su 10 prove
            samples_x = []
            samples_y = []
            
            for _ in range(n_samples):
                x = center + np.random.uniform(-0.2, 0.2, 2)
                y = rastrigin(x)
                samples_x.append(x)
                samples_y.append(y)
            
            lgs_grad_dir, _ = estimate_gradient_lgs_style(center, samples_x, samples_y)
            alignment = np.dot(true_grad_dir, lgs_grad_dir)
            alignments.append(alignment)
        
        mean_align = np.mean(alignments)
        std_align = np.std(alignments)
        
        if mean_align > 0.8:
            status = "✓ BUONO"
        elif mean_align > 0.3:
            status = "~ OK"
        else:
            status = "✗ CATTIVO"
        
        print(f"N={n_samples:3d}: alignment = {mean_align:.3f} ± {std_align:.3f}  {status}")


def diagnose_lgs_on_rastrigin():
    """Diagnosi finale."""
    
    print("\n" + "=" * 70)
    print("DIAGNOSI LGS SU RASTRIGIN")
    print("=" * 70)
    
    print("""
CONCLUSIONI:

1. LGS stima il gradiente dai PUNTI CAMPIONATI, non dal vero gradiente
2. Su Rastrigin, la funzione oscilla rapidamente (termine coseno)
3. LGS fa una MEDIA pesata dei punti → smooth out le oscillazioni
4. Questo può dare un gradiente che punta verso il trend LOCALE

DOMANDA CHIAVE: LGS stima il gradiente correttamente?

Se SÌ → Il problema è che il gradiente locale non porta al globale
        (limite strutturale, non bug)
        
Se NO → C'è un problema nella stima LGS
        (bug da investigare)
""")


if __name__ == "__main__":
    test_lgs_vs_true_gradient()
    test_effect_of_sample_density()
    diagnose_lgs_on_rastrigin()
