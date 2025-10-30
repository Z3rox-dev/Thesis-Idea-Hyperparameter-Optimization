#!/usr/bin/env python3
"""
Esempio Dimostrativo del Teorema di Split Basato su Anisotropia e Curvatura
============================================================================

Questo script mostra come funzionano i due teoremi in un caso semplice.
"""

import numpy as np
import sys

# Mock ParamSpace per permettere l'import
class ParamSpace:
    pass

sys.modules['cube_hpo'] = type(sys)('cube_hpo')
sys.modules['cube_hpo'].ParamSpace = ParamSpace

from HPO_QuadTree_v1 import QuadCube

def esempio_anisotropia():
    """Dimostra il calcolo dell'anisotropia via PCA."""
    print("=" * 70)
    print("ESEMPIO 1: Calcolo Anisotropia")
    print("=" * 70)
    
    # Crea un cubo 2D
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    cube = QuadCube(bounds)
    
    # Simula una "valle" diagonale: punti migliori lungo x + y ≈ 1
    # Questo dovrebbe dare alta anisotropia (λ₁ >> λ₂)
    np.random.seed(42)
    punti_valle = []
    for _ in range(15):
        t = np.random.uniform(0.3, 0.7)
        noise = np.random.normal(0, 0.05, 2)
        x = np.array([t, 1.0 - t]) + noise
        x = np.clip(x, 0.0, 1.0)
        # Score più alto vicino alla valle
        score = 1.0 - np.abs((x[0] + x[1]) - 1.0) + np.random.normal(0, 0.1)
        punti_valle.append((x, score))
    
    cube._tested_pairs = punti_valle
    
    # Calcola PCA e anisotropia
    R, mu, eigvals, ok = cube._principal_axes(min_points=10, anisotropy_threshold=1.4)
    
    print(f"\nPunti testati: {len(punti_valle)}")
    print(f"Centro (μ): [{mu[0]:.3f}, {mu[1]:.3f}]")
    print(f"Autovalori (λ): [{eigvals[0]:.4f}, {eigvals[1]:.4f}]")
    
    ratio = eigvals[0] / max(eigvals[1], 1e-9)
    print(f"\nRatio anisotropia: λ₁/λ₂ = {ratio:.2f}")
    print(f"Soglia: 1.4")
    print(f"Anisotropia OK: {ok} ({'SPLIT LUNGO PCA' if ok else 'SPLIT LUNGO ASSI ORIGINALI'})")
    
    if ok:
        print(f"\nPrimo componente principale (PC1): [{R[0,0]:.3f}, {R[1,0]:.3f}]")
        print(f"Direzione: ~{np.arctan2(R[1,0], R[0,0]) * 180 / np.pi:.1f}°")
        print("(Dovrebbe essere ~-45° per la valle diagonale)")


def esempio_curvatura():
    """Dimostra il calcolo del punto di massima curvatura."""
    print("\n" + "=" * 70)
    print("ESEMPIO 2: Punto di Massima Curvatura")
    print("=" * 70)
    
    # Crea un cubo 1D
    bounds = [(0.0, 1.0)]
    cube = QuadCube(bounds)
    
    # Simula una parabola: y = -(x - 0.6)² + 1
    # Massimo in x = 0.6
    np.random.seed(42)
    punti_parabola = []
    for _ in range(20):
        x = np.random.uniform(0.0, 1.0)
        y = -(x - 0.6)**2 + 1.0 + np.random.normal(0, 0.05)
        punti_parabola.append((np.array([x]), y))
    
    cube._tested_pairs = punti_parabola
    
    # Calcola il punto di taglio
    R = np.eye(1)
    mu = np.array([0.5])
    t_cut = cube._quad_cut_along_axis(0, R, mu)
    
    print(f"\nPunti testati: {len(punti_parabola)}")
    print(f"Funzione simulata: y = -(x - 0.6)² + 1")
    print(f"Massimo teorico: x = 0.6")
    print(f"Punto di taglio calcolato: x = {t_cut:.3f}")
    print(f"Errore: {abs(t_cut - 0.6):.3f}")
    
    if abs(t_cut - 0.6) < 0.1:
        print("✓ Il fit quadratico ha identificato correttamente il massimo!")
    

def esempio_split_quad():
    """Dimostra lo split quadruplo con PCA."""
    print("\n" + "=" * 70)
    print("ESEMPIO 3: Split Quadruplo (4-way)")
    print("=" * 70)
    
    # Crea un cubo 2D
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    cube = QuadCube(bounds)
    
    # Simula dati con valle diagonale e punto di massimo
    np.random.seed(42)
    punti = []
    for _ in range(30):
        x = np.random.uniform(0.0, 1.0, 2)
        # Valle lungo x + y ≈ 1, con massimo in (0.6, 0.4)
        score = 1.0 - ((x[0] - 0.6)**2 + (x[1] - 0.4)**2) - 0.5 * abs((x[0] + x[1]) - 1.0)
        score += np.random.normal(0, 0.1)
        punti.append((x, score))
    
    cube._tested_pairs = punti
    cube._tested_points = [p for p, _ in punti]
    
    # Verifica se dovrebbe fare split
    split_type = cube.should_split(min_trials=5, min_points=10, gamma=0.0)
    print(f"\nPunti testati: {len(punti)}")
    print(f"Tipo di split raccomandato: {split_type}")
    
    if split_type == 'quad':
        print("\n✓ Split QUAD raccomandato (anisotropia sufficiente)")
        
        # Esegui lo split
        children = cube.split4()
        print(f"Numero di figli creati: {len(children)}")
        
        for i, child in enumerate(children):
            n_pts = len(child._tested_pairs)
            print(f"  Quadrante {i+1}: {n_pts} punti")
    elif split_type == 'binary':
        print("\n→ Split BINARY raccomandato (anisotropia insufficiente)")
    else:
        print("\n→ Nessuno split raccomandato")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("DIMOSTRAZIONE TEOREMA SPLIT ANISOTROPIA + CURVATURA")
    print("=" * 70)
    
    esempio_anisotropia()
    esempio_curvatura()
    esempio_split_quad()
    
    print("\n" + "=" * 70)
    print("Fine Esempi")
    print("=" * 70)
    print("\nVedere README.md per spiegazione matematica completa.")
