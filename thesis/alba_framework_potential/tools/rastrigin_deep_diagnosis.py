#!/usr/bin/env python3
"""
VERIFICA DETTAGLIATA: Cosa DOVREBBE fare ALBA vs cosa FA
=========================================================

Verifichiamo se ci sono meccanismi in ALBA che dovrebbero
aiutare su multimodali ma non stanno funzionando.
"""

import sys
import numpy as np
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

import os
os.chdir(parent_dir)


def rastrigin_2d(x):
    A = 10
    x = np.array(x)
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def verify_basin_of_attraction():
    """Verifica più attenta del basin of attraction."""
    
    print("=" * 70)
    print("VERIFICA BASIN OF ATTRACTION")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Test 1: Quanto è grande il basin del globale?
    print("\n1. Dimensione basin del minimo globale:")
    
    radii = [0.1, 0.2, 0.3, 0.4, 0.5]
    for r in radii:
        successes = 0
        n_trials = 50
        
        for _ in range(n_trials):
            # Parti da un punto a distanza r dall'origine
            angle = np.random.uniform(0, 2*np.pi)
            x = np.array([r * np.cos(angle), r * np.sin(angle)])
            
            # Gradient descent
            for step in range(200):
                eps = 0.001
                grad = np.zeros(2)
                for d in range(2):
                    x_plus = x.copy()
                    x_plus[d] += eps
                    x_minus = x.copy()
                    x_minus[d] -= eps
                    grad[d] = (rastrigin_2d(x_plus) - rastrigin_2d(x_minus)) / (2 * eps)
                
                x = x - 0.01 * grad  # Learning rate più piccolo
                x = np.clip(x, -5.12, 5.12)
            
            if rastrigin_2d(x) < 0.01:
                successes += 1
        
        print(f"  Raggio {r}: {successes}/{n_trials} ({100*successes/n_trials:.0f}%) → globale")
    
    # Test 2: Da posizioni specifiche
    print("\n2. Da posizioni specifiche:")
    
    test_points = [
        ([0.0, 0.0], "Origine (globale)"),
        ([0.1, 0.1], "Molto vicino"),
        ([0.3, 0.3], "Vicino"),
        ([0.5, 0.0], "Metà strada verso (1,0)"),
        ([0.5, 0.5], "Centro quadrante"),
        ([0.9, 0.0], "Vicino a minimo locale (1,0)"),
        ([1.0, 0.0], "Minimo locale (1,0)"),
        ([2.0, 2.0], "Lontano"),
    ]
    
    for start, desc in test_points:
        x = np.array(start, dtype=float)
        
        for step in range(500):
            eps = 0.001
            grad = np.zeros(2)
            for d in range(2):
                x_plus = x.copy()
                x_plus[d] += eps
                x_minus = x.copy()
                x_minus[d] -= eps
                grad[d] = (rastrigin_2d(x_plus) - rastrigin_2d(x_minus)) / (2 * eps)
            
            x = x - 0.01 * grad
            x = np.clip(x, -5.12, 5.12)
        
        final_y = rastrigin_2d(x)
        final_x = x
        
        if final_y < 0.01:
            status = "→ GLOBALE ✓"
        else:
            status = f"→ locale y={final_y:.2f}"
        
        print(f"  {desc}: {start} → [{final_x[0]:.2f}, {final_x[1]:.2f}] {status}")


def check_alba_exploration_mechanisms():
    """Verifica i meccanismi di esplorazione di ALBA."""
    
    print("\n" + "=" * 70)
    print("MECCANISMI DI ESPLORAZIONE IN ALBA")
    print("=" * 70)
    
    # Leggi optimizer.py per capire i meccanismi
    optimizer_path = Path(parent_dir) / "optimizer.py"
    
    with open(optimizer_path, 'r') as f:
        content = f.read()
    
    # Cerca parametri chiave
    exploration_params = [
        "exploration",
        "random",
        "global",
        "restart",
        "diversity",
        "should_exploit",
        "should_explore",
    ]
    
    print("\nParametri/metodi legati all'esplorazione trovati:")
    print("-" * 50)
    
    for param in exploration_params:
        count = content.lower().count(param.lower())
        if count > 0:
            print(f"  '{param}': {count} occorrenze")
    
    # Verifica se c'è un meccanismo di restart o diversificazione
    print("\n" + "-" * 50)
    print("Analisi meccanismi anti-stagnazione:")
    
    if "restart" in content.lower():
        print("  ✓ Meccanismo di restart presente")
    else:
        print("  ✗ Nessun meccanismo di restart")
    
    if "stagnation" in content.lower():
        print("  ✓ Rilevamento stagnazione presente")
    else:
        print("  ✗ Nessun rilevamento stagnazione")
    
    if "latin" in content.lower() or "lhs" in content.lower():
        print("  ✓ Latin Hypercube sampling presente")
    else:
        print("  ✗ Nessun Latin Hypercube sampling")
    
    if "multi_start" in content.lower() or "multistart" in content.lower():
        print("  ✓ Multi-start presente")
    else:
        print("  ✗ Nessun multi-start")


def analyze_coherence_behavior_on_rastrigin():
    """Analizza come si comporta la coherence su Rastrigin."""
    
    print("\n" + "=" * 70)
    print("COMPORTAMENTO COHERENCE SU RASTRIGIN")
    print("=" * 70)
    
    print("""
Scenario: ALBA ha esplorato e trovato 3 regioni su Rastrigin

Regione A: intorno a (0,0) - minimo GLOBALE
Regione B: intorno a (1,0) - minimo locale
Regione C: intorno a (2,2) - minimo locale

COSA VEDE LA COHERENCE:
""")
    
    # Simula gradienti in ogni regione
    regions = {
        "A (globale)": {"center": [0.0, 0.0], "points": [[0.1, 0.1], [-0.1, 0.1], [0.1, -0.1]]},
        "B (locale)":  {"center": [1.0, 0.0], "points": [[0.9, 0.1], [1.1, 0.1], [1.0, -0.1]]},
        "C (locale)":  {"center": [2.0, 2.0], "points": [[1.9, 2.1], [2.1, 1.9], [2.0, 2.1]]},
    }
    
    gradients = {}
    
    for name, region in regions.items():
        center = np.array(region["center"])
        
        # Stima gradiente al centro
        eps = 0.01
        grad = np.zeros(2)
        for d in range(2):
            c_plus = center.copy()
            c_plus[d] += eps
            c_minus = center.copy()
            c_minus[d] -= eps
            grad[d] = (rastrigin_2d(c_plus) - rastrigin_2d(c_minus)) / (2 * eps)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6:
            grad_unit = grad / grad_norm
        else:
            grad_unit = np.zeros(2)
        
        gradients[name] = grad_unit
        
        print(f"{name}:")
        print(f"  Centro: {center}")
        print(f"  Gradiente: {grad}")
        print(f"  |grad|: {grad_norm:.4f}")
        print(f"  Direzione: {grad_unit}")
        print()
    
    # Calcola alignment tra coppie
    print("Alignment tra regioni (cosa vede coherence):")
    print("-" * 50)
    
    region_names = list(gradients.keys())
    for i in range(len(region_names)):
        for j in range(i+1, len(region_names)):
            g_i = gradients[region_names[i]]
            g_j = gradients[region_names[j]]
            alignment = np.dot(g_i, g_j)
            
            # Interpretazione
            if alignment > 0.7:
                interp = "paralleli (stessa direzione)"
            elif alignment < -0.7:
                interp = "opposti (VALLE tra loro!)"
            else:
                interp = "ortogonali"
            
            print(f"  {region_names[i]} ↔ {region_names[j]}: {alignment:.2f} ({interp})")
    
    print("\n" + "-" * 50)
    print("OSSERVAZIONE CHIAVE:")
    print("-" * 50)
    print("""
Al minimo globale (0,0), il gradiente è ~0 (piatto).
Ai minimi locali, il gradiente è anche ~0 (piatto).

La coherence NON può distinguerli perché:
- Entrambi hanno gradienti piccoli
- Entrambi sono "valli" coerenti
- Non c'è informazione sul VALORE della funzione!

La coherence misura SOLO la geometria (direzione gradienti),
non il valore assoluto. Quindi ogni valle sembra ugualmente buona.
""")


def final_diagnosis():
    """Diagnosi finale."""
    
    print("\n" + "=" * 70)
    print("DIAGNOSI FINALE")
    print("=" * 70)
    
    print("""
DOMANDA: È un BUG o un LIMITE STRUTTURALE?

RISPOSTA: È un LIMITE STRUTTURALE, ma con AGGRAVANTI:

1. LIMITE FONDAMENTALE (non risolvibile):
   - Metodi gradient-based non possono distinguere minimi locali da globali
   - Il gradiente punta sempre verso il minimo PIÙ VICINO
   - Questo è vero per GD, LGS, e qualsiasi metodo basato su derivate

2. POSSIBILE AGGRAVANTE (verificare):
   - ALBA potrebbe non avere abbastanza esplorazione GLOBALE
   - I meccanismi di exploration potrebbero non essere sufficienti
   - Potrebbe mancare un meccanismo di restart/diversificazione

RACCOMANDAZIONE:
----------------
Verificare i parametri di esplorazione:
- exploration_ratio: dovrebbe essere >0.2 su funzioni multimodali
- n_initial: dovrebbe essere alto per buona copertura
- Considerare aggiunta di restart quando best stagna

Rastrigin è un TEST CASE ESTREMO:
- ~100 minimi locali
- Basin del globale <0.5 di raggio
- Anche ottimizzatori avanzati faticano

ALBA non è "rotto" su Rastrigin, è semplicemente un metodo
gradient-based su una funzione dove i gradienti sono fuorvianti.
""")


if __name__ == "__main__":
    verify_basin_of_attraction()
    check_alba_exploration_mechanisms()
    analyze_coherence_behavior_on_rastrigin()
    final_diagnosis()
