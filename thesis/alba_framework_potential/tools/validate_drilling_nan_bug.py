#!/usr/bin/env python3
"""
VALIDAZIONE DRILLING BUG - Test Inline (no import ciclici)

Simula il flusso reale con classi mock per evitare import issues.
"""

import numpy as np

print("=" * 70)
print("VALIDAZIONE: Il bug drilling è reale o teorico?")
print("=" * 70)

# Copia inline della logica DrillingOptimizer (senza import)
class DrillingOptimizer:
    """Minimal copy for testing"""
    def __init__(self, start_x, start_y=0.0, initial_sigma=0.1, bounds=None):
        self.dim = len(start_x)
        self.mu = np.array(start_x, dtype=float)
        self.best_x = self.mu.copy()
        self.best_y = float(start_y)
        self.sigma = float(initial_sigma)
        self.bounds = bounds or [(0.0, 1.0)] * self.dim
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.iteration = 0
    
    def ask(self, rng):
        # Sample from N(mu, sigma * C)
        z = rng.standard_normal(self.dim)
        try:
            L = np.linalg.cholesky(self.C)
            x = self.mu + self.sigma * (L @ z)
        except np.linalg.LinAlgError:
            x = self.mu + self.sigma * z
        # Clip to bounds
        for i, (lo, hi) in enumerate(self.bounds):
            x[i] = np.clip(x[i], lo, hi)
        return x
    
    def tell(self, x, y):
        self.iteration += 1
        if y < self.best_y:  # Minimization
            self.best_y = y
            self.best_x = x.copy()
            self.mu = x.copy()
        self.sigma *= 0.95
        return self.iteration < 100 and self.sigma > 1e-8

rng = np.random.default_rng(42)

# Test 1: NaN in start_x - IL BUG
print("\nTEST 1: NaN in start_x propagato?")
print("-" * 50)

start_x_nan = np.array([0.5, np.nan, 0.5])
driller = DrillingOptimizer(start_x_nan, start_y=1.0)
x = driller.ask(rng)

if np.any(np.isnan(x)):
    print(f"  ❌ BUG CONFERMATO: NaN in x = {x}")
    bug_confirmed = True
else:
    print(f"  ✓ NaN gestito: x = {x}")
    bug_confirmed = False

# Test 2: MA - può mai start_x avere NaN nel contesto reale?
print("\nTEST 2: Può start_x avere NaN nel contesto ALBA?")
print("-" * 50)

print("""
Analisi del flusso:

1. DrillingOptimizer è creato in optimizer.py con:
   start_x = x (il punto appena valutato)

2. x viene da ask() → _sample_in_cube() → CandidateGenerator

3. CandidateGenerator post-fix (Finding 23) sanitizza i candidati:
   - gradient_dir NaN → skip strategia gradient
   - Tutti i candidati sono clippati ai bounds
   - NaN non può passare

4. QUINDI: nel flusso normale, start_x NON può avere NaN

CONCLUSIONE:
""")

if bug_confirmed:
    print("""
Il bug esiste tecnicamente, MA:
- Nel contesto ALBA non si manifesta
- CandidateGenerator (post-fix) previene NaN
- È DIFESA IN PROFONDITÀ, non bug critico

RACCOMANDAZIONE:
- Fix opzionale per robustezza API pubblica
- Priorità: ★★☆☆☆ (2/5)
- Pattern: Sanitize start_x in __init__
""")

# Test 3: Verifica che il bug è facile da fixare
print("\nTEST 3: Fix proposta")
print("-" * 50)

class DrillingOptimizerFixed:
    """Con fix per NaN"""
    def __init__(self, start_x, start_y=0.0, initial_sigma=0.1, bounds=None):
        self.dim = len(start_x)
        start_x = np.array(start_x, dtype=float)
        
        # FIX: Sanitize start_x
        if not np.all(np.isfinite(start_x)):
            # Replace NaN/Inf with bounds center
            bounds = bounds or [(0.0, 1.0)] * self.dim
            centers = np.array([(lo + hi) / 2 for lo, hi in bounds])
            mask = ~np.isfinite(start_x)
            start_x[mask] = centers[mask]
        
        self.mu = start_x.copy()
        self.best_x = self.mu.copy()
        self.best_y = float(start_y) if np.isfinite(start_y) else 0.0
        self.sigma = float(initial_sigma)
        self.bounds = bounds or [(0.0, 1.0)] * self.dim
        self.C = np.eye(self.dim)
    
    def ask(self, rng):
        z = rng.standard_normal(self.dim)
        x = self.mu + self.sigma * z
        for i, (lo, hi) in enumerate(self.bounds):
            x[i] = np.clip(x[i], lo, hi)
        return x

driller_fixed = DrillingOptimizerFixed(start_x_nan, start_y=1.0, bounds=[(0,1)]*3)
x_fixed = driller_fixed.ask(rng)

if np.any(np.isnan(x_fixed)):
    print(f"  ❌ Fix non funziona: x = {x_fixed}")
else:
    print(f"  ✓ Fix funziona: NaN → center = {driller_fixed.mu}")
    print(f"     ask() → {x_fixed}")

print("\n" + "=" * 70)
print("VERDETTO FINALE")
print("=" * 70)
print("""
Finding 26: DrillingOptimizer NaN propagation
- Severity: LOW (difesa in profondità)
- Impact: Solo se usato standalone con input invalidi
- Fix: Sanitize start_x in __init__
- Decision: FIX per API robustness, ma non blocca il framework
""")
