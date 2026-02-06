"""ALBA Framework (Copula variant) - Candidate Generation

This module defines how ALBA-Copula generates candidate points inside a Cube.

Uses copula-based sampling instead of gradient direction:
- 30%: copula samples (correlated elite-based)
- 25%: top-k perturbation
- 20%: center perturbation
- 25%: uniform random
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

from .cube import Cube
from .copula_model import sample_from_copula


class CandidateGenerator(Protocol):
    def generate(self, cube: Cube, dim: int, rng: np.random.Generator, n: int) -> List[np.ndarray]:
        ...


@dataclass(frozen=True)
class MixtureCandidateGenerator:
    """Hybrid candidate generator: matches LGS distribution but adds copula option."""

    sigma_topk: float = 0.15
    sigma_gradient_noise: float = 0.05
    sigma_center: float = 0.2
    step_min: float = 0.05
    step_max: float = 0.3
    copula_fraction: float = 0.10  # 10% copula replaces 10% of uniform

    def generate(self, cube: Cube, dim: int, rng: np.random.Generator, n: int) -> List[np.ndarray]:
        candidates: List[np.ndarray] = []
        widths = cube.widths()
        center = cube.center()
        model = cube.lgs_model

        # Pre-generate copula samples
        n_copula = int(n * self.copula_fraction)
        copula_samples = []
        if model is not None and model.get("L") is not None:
            copula_samples = sample_from_copula(model, n_copula, rng)
        
        copula_idx = 0
        for _ in range(n):
            strategy = float(rng.random())

            # 25%: top-k perturbation (same as original)
            if strategy < 0.25 and model is not None and len(model.get("top_k_pts", [])) > 0:
                idx = int(rng.integers(len(model["top_k_pts"])))
                x = model["top_k_pts"][idx] + rng.normal(0, self.sigma_topk, dim) * widths
            # 15%: gradient direction (same as original)
            elif strategy < 0.40 and model is not None and model.get("gradient_dir") is not None:
                top_center = model["top_k_pts"].mean(axis=0) if len(model.get("top_k_pts", [])) > 0 else center
                step = float(rng.uniform(self.step_min, self.step_max))
                x = top_center + step * model["gradient_dir"] * widths
                x = x + rng.normal(0, self.sigma_gradient_noise, dim) * widths
            # 15%: center perturbation (same as original)
            elif strategy < 0.55:
                x = center + rng.normal(0, self.sigma_center, dim) * widths
            # 10%: copula samples (NEW - replaces part of uniform)
            elif strategy < 0.65 and copula_idx < len(copula_samples):
                x = copula_samples[copula_idx]
                copula_idx += 1
            # 35%: uniform (slightly reduced from 45%)
            else:
                x = np.array([rng.uniform(lo, hi) for lo, hi in cube.bounds], dtype=float)

            # clip to cube
            x = np.array([np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1]) for i in range(dim)], dtype=float)
            candidates.append(x)

        return candidates

        return candidates
