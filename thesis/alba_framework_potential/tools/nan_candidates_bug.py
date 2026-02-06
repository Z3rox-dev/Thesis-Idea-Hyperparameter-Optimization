#!/usr/bin/env python3
"""
BUG ANALYSIS: NaN in Candidates from gradient_dir
===================================================

Bug trovato: quando gradient_dir contiene NaN, i candidati generati
dalla strategia "gradient" contengono NaN.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MixtureCandidateGenerator:
    """Copy of the original generator."""
    
    sigma_topk: float = 0.15
    sigma_gradient_noise: float = 0.05
    sigma_center: float = 0.2
    step_min: float = 0.05
    step_max: float = 0.3

    def generate(self, cube, dim: int, rng: np.random.Generator, n: int) -> List[np.ndarray]:
        candidates: List[np.ndarray] = []
        widths = cube.widths()
        center = cube.center()
        model = cube.lgs_model

        for _ in range(n):
            strategy = float(rng.random())

            if strategy < 0.25 and model is not None and len(model["top_k_pts"]) > 0:
                idx = int(rng.integers(len(model["top_k_pts"])))
                x = model["top_k_pts"][idx] + rng.normal(0, self.sigma_topk, dim) * widths
            elif strategy < 0.40 and model is not None and model["gradient_dir"] is not None:
                # BUG: se gradient_dir è NaN, x diventa NaN
                top_center = model["top_k_pts"].mean(axis=0)
                step = float(rng.uniform(self.step_min, self.step_max))
                x = top_center + step * model["gradient_dir"] * widths
                x = x + rng.normal(0, self.sigma_gradient_noise, dim) * widths
            elif strategy < 0.55:
                x = center + rng.normal(0, self.sigma_center, dim) * widths
            else:
                x = np.array([rng.uniform(lo, hi) for lo, hi in cube.bounds], dtype=float)

            # clip to cube - NON risolve il problema perché clip(NaN) = NaN
            x = np.array([np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1]) for i in range(dim)], dtype=float)
            candidates.append(x)

        return candidates


class MockCube:
    def __init__(self, bounds, lgs_model=None):
        self.bounds = bounds
        self.lgs_model = lgs_model
    
    def center(self):
        return np.array([(lo + hi) / 2 for lo, hi in self.bounds])
    
    def widths(self):
        return np.array([hi - lo for lo, hi in self.bounds])


def test_nan_gradient():
    print("=" * 70)
    print("BUG: NaN in gradient_dir → NaN in candidati")
    print("=" * 70)
    
    gen = MixtureCandidateGenerator()
    
    # Seed per far entrare nella strategia gradient (0.25 <= rng < 0.40)
    # La strategia gradient è tra 0.25 e 0.40 = 15% delle volte
    
    cube = MockCube(
        [(0, 1), (0, 1)],
        lgs_model={
            "top_k_pts": np.array([[0.5, 0.5]]),
            "gradient_dir": np.array([float('nan'), float('nan')])
        }
    )
    
    rng = np.random.default_rng(42)
    candidates = gen.generate(cube, dim=2, rng=rng, n=1000)
    
    nan_candidates = [c for c in candidates if np.any(np.isnan(c))]
    
    print(f"\nTotale candidati: {len(candidates)}")
    print(f"Candidati con NaN: {len(nan_candidates)}")
    print(f"Percentuale: {100 * len(nan_candidates) / len(candidates):.1f}%")
    
    # La percentuale dovrebbe essere circa 15% (strategia gradient)
    expected_pct = 15  # 0.40 - 0.25 = 0.15 = 15%
    actual_pct = 100 * len(nan_candidates) / len(candidates)
    
    print(f"\nExpected ~{expected_pct}% (strategia gradient)")
    print(f"Actual: {actual_pct:.1f}%")
    
    print("\n" + "=" * 70)
    print("ROOT CAUSE")
    print("=" * 70)
    print("""
1. Quando strategy è tra 0.25 e 0.40, usa gradient_dir
2. Se gradient_dir contiene NaN, x diventa NaN:
   x = top_center + step * model["gradient_dir"] * widths
   
3. Il clipping NON risolve il problema:
   np.clip(NaN, lo, hi) = NaN
   
4. Risultato: candidati con NaN vengono valutati dalla black-box
   che potrebbe crashare o dare risultati errati.
""")


def propose_fix():
    print("\n" + "=" * 70)
    print("PROPOSTA FIX")
    print("=" * 70)
    
    fix_code = '''
# Prima di usare gradient_dir, verificare che sia valido:
elif strategy < 0.40 and model is not None and model.get("gradient_dir") is not None:
    grad_dir = model["gradient_dir"]
    # FIX: Skip gradient strategy if gradient contains NaN/Inf
    if not np.all(np.isfinite(grad_dir)):
        # Fallback to center perturbation
        x = center + rng.normal(0, self.sigma_center, dim) * widths
    else:
        top_center = model["top_k_pts"].mean(axis=0)
        step = float(rng.uniform(self.step_min, self.step_max))
        x = top_center + step * grad_dir * widths
        x = x + rng.normal(0, self.sigma_gradient_noise, dim) * widths
'''
    print(fix_code)
    
    print("\n" + "=" * 70)
    print("ADDITIONAL FIX: top_k_pts vuoto")
    print("=" * 70)
    
    print("""
Quando top_k_pts è vuoto ma gradient_dir esiste, la linea:
    top_center = model["top_k_pts"].mean(axis=0)
    
Produce un warning e top_center diventa [nan, nan].

Fix: verificare che top_k_pts non sia vuoto prima di usare gradient strategy.
""")


if __name__ == "__main__":
    test_nan_gradient()
    propose_fix()
