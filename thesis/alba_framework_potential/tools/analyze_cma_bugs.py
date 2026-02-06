#!/usr/bin/env python3
"""
Test focalizzato sui BUG identificati nelle matrici CMA.

BUG IDENTIFICATI:
1. sigma esplode (8e6 dopo 100 successi) - manca sigma_max
2. C non si adatta all'ellitticità - learning rate troppo basso o pochi step?
3. sigma growth 1.2x troppo aggressivo senza damping

CORREZIONI PROPOSTE:
1. Aggiungere sigma_max (es. 2.0 o range_width)
2. Aggiungere decay rate su failure più aggressivo
3. Usare 1/5th success rule con smoothing
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/workspace/thesis')

from alba_framework_potential.drilling import DrillingOptimizer


def analyze_sigma_explosion():
    """Analizza il problema dell'esplosione di sigma."""
    print("="*70)
    print("ANALISI: Esplosione di Sigma")
    print("="*70)
    
    rng = np.random.default_rng(42)
    dim = 5
    bounds = [(0, 1)] * dim
    
    drill = DrillingOptimizer(np.array([0.5]*dim), 1.0, initial_sigma=0.1, bounds=bounds)
    
    print(f"sigma_init = {drill.sigma}")
    print(f"Growth factor = 1.2 (on success)")
    print(f"Shrink factor = 0.85 (on failure)")
    
    # Calcola sigma dopo N successi consecutivi
    print("\nSimulazione successi consecutivi:")
    sigma = 0.1
    for n in [10, 20, 50, 100]:
        sigma_after = 0.1 * (1.2 ** n)
        print(f"  Dopo {n} successi: sigma = {sigma_after:.2e}")
    
    print("\nPROBLEMA: sigma cresce esponenzialmente senza limite!")
    print("In 100 successi: 0.1 * 1.2^100 = 8.28e+06")
    
    # Calcolo equilibrio
    print("\nEquilibrio 1/5th rule:")
    print("  Con 20% successo, 80% fallimento:")
    print("  Fattore medio = 1.2^0.2 * 0.85^0.8 = {:.4f}".format((1.2**0.2) * (0.85**0.8)))
    print("  Se < 1: sigma decresce. Se > 1: sigma cresce.")
    
    factor = (1.2**0.2) * (0.85**0.8)
    print(f"  Fattore = {factor:.4f} → {'STABILE (decresce)' if factor < 1 else 'INSTABILE (cresce)'}")


def test_with_sigma_cap():
    """Testa comportamento con sigma capped."""
    print("\n" + "="*70)
    print("TEST: Comportamento con sigma_max")
    print("="*70)
    
    rng = np.random.default_rng(42)
    dim = 5
    bounds = [(0, 1)] * dim
    
    # Simula drilling con cap
    class DrillWithCap:
        def __init__(self):
            self.sigma = 0.1
            self.sigma_max = 1.0  # Cap a 1.0 (larghezza del bounds)
            self.sigma_min = 1e-6
        
        def success(self):
            self.sigma = min(self.sigma * 1.2, self.sigma_max)
        
        def failure(self):
            self.sigma = max(self.sigma * 0.85, self.sigma_min)
    
    drill = DrillWithCap()
    
    print("Con sigma_max = 1.0:")
    for n in [10, 20, 50, 100]:
        d = DrillWithCap()
        for _ in range(n):
            d.success()
        print(f"  Dopo {n} successi: sigma = {d.sigma:.4f}")
    
    print("\n✓ sigma resta bounded!")


def analyze_covariance_learning():
    """Analizza perché C non impara la geometria ellittica."""
    print("\n" + "="*70)
    print("ANALISI: Learning della Covarianza")
    print("="*70)
    
    rng = np.random.default_rng(42)
    dim = 5
    bounds = [(0, 1)] * dim
    
    def elliptic(x):
        weights = np.arange(1, dim + 1) ** 2  # 1, 4, 9, 16, 25
        return np.sum(weights * (x - 0.5)**2)
    
    # Hessian della funzione ellittica
    # H = diag(2*1, 2*4, 2*9, 2*16, 2*25) = diag(2, 8, 18, 32, 50)
    # Direzione "facile" = ultima (peso più grande = curvatura maggiore)
    # Direzione "difficile" = prima (peso minore = curvatura minore)
    # 
    # CMA dovrebbe imparare che muoversi lungo dim[0] è "economico"
    # quindi C[0,0] dovrebbe diventare > C[4,4]
    
    print("Funzione: f = sum(i^2 * (x_i - 0.5)^2)")
    print("Hessian diagonale: [2, 8, 18, 32, 50]")
    print("Direzione facile: dim[0] (curvatura bassa)")
    print("Direzione difficile: dim[4] (curvatura alta)")
    print()
    print("C ottimale dovrebbe essere proporzionale a H^{-1}:")
    print("C* ~ diag(1/2, 1/8, 1/18, 1/32, 1/50)")
    print("Normalizzato: C* ~ diag(1.0, 0.25, 0.11, 0.06, 0.04)")
    
    # Test con più iterazioni
    print("\n" + "-"*50)
    print("Test con drilling reale:")
    
    start_x = np.array([0.6] * dim)
    start_y = elliptic(start_x)
    
    for max_iter in [20, 50, 100, 200]:
        drill = DrillingOptimizer(start_x.copy(), start_y, initial_sigma=0.1, bounds=bounds)
        drill.max_steps = max_iter
        drill.step_cap = max_iter + 10
        
        n_success = 0
        for i in range(max_iter):
            x_new = drill.ask(rng)
            y_new = elliptic(x_new)
            should_continue = drill.tell(x_new, y_new)
            if y_new < drill.best_y:
                n_success += 1
            if not should_continue:
                break
        
        # Analizza C
        diag_C = np.diag(drill.C)
        normalized_diag = diag_C / diag_C[0]
        
        print(f"\n  Dopo {i+1} iter ({n_success} successi):")
        print(f"    diag(C) = {diag_C}")
        print(f"    normalizzato (rispetto a dim[0]) = {normalized_diag}")
        print(f"    c_cov = {drill.c_cov:.4f}")
        print(f"    sigma = {drill.sigma:.2e}")


def propose_fixes():
    """Propone correzioni concrete."""
    print("\n" + "="*70)
    print("CORREZIONI PROPOSTE")
    print("="*70)
    
    print("""
1. SIGMA CAPPING:
   - Aggiungere sigma_max = initial_sigma * 10 (o bounds-based)
   - In tell(): self.sigma = min(self.sigma * 1.2, self.sigma_max)

2. SIGMA ADAPTATION PIÙ ROBUSTA (1/5th rule con smoothing):
   - Tracciare success_rate con exponential moving average
   - Usare: sigma *= exp((success_rate - 0.2) / d_sigma)
   - Invece di: sigma *= 1.2 (success) o 0.85 (failure)

3. C LEARNING:
   - c_cov = 2/(dim^2 + 6) è molto piccolo per dim=5: c_cov = 0.065
   - Dopo 100 successi, C è solo 1 - 0.935^100 = 99.8% learned
   - MA: ogni update è rank-1, quindi C converge a rank-1 se tutti
     i successi sono nella stessa direzione
   
   FIX: Aggiungere evolution path pc per accumulare info:
   - pc = (1-cc)*pc + sqrt(cc*(2-cc)) * y
   - C = (1-c_cov)*C + c_cov * pc * pc^T

4. EARLY STOP PIÙ AGGRESSIVO:
   - Se sigma > bounds_width: stop (siamo fuori scala)
   - Se C condition number > 1e6: reset C = I

PRIORITÀ:
1. Sigma capping (bug critico)
2. Early stop su sigma grande
3. Evolution path (nice to have)
""")


def main():
    analyze_sigma_explosion()
    test_with_sigma_cap()
    analyze_covariance_learning()
    propose_fixes()


if __name__ == "__main__":
    main()
