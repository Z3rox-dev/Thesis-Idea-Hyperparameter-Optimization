"""ALBA Framework - Candidate Generation

This module defines how ALBA generates candidate points inside a Cube.

Default generator matches ALBA_V1: mixture of strategies based on top-k
points, gradient direction, cube center perturbation, and uniform sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

from .cube import Cube


class CandidateGenerator(Protocol):
    def generate(self, cube: Cube, dim: int, rng: np.random.Generator, n: int) -> List[np.ndarray]:
        ...


@dataclass(frozen=True)
class MixtureCandidateGenerator:
    """Default candidate generator (matches ALBA_V1)."""

    sigma_topk: float = 0.18  # Increased for more exploration on RF
    sigma_gradient_noise: float = 0.05
    sigma_center: float = 0.25  # Increased
    step_min: float = 0.05
    step_max: float = 0.3

    def generate(self, cube: Cube, dim: int, rng: np.random.Generator, n: int) -> List[np.ndarray]:
        candidates: List[np.ndarray] = []
        widths = cube.widths()
        center = cube.center()
        model = cube.lgs_model

        for _ in range(n):
            strategy = float(rng.random())

            # MODIFIED: Remove gradient sampling (unreliable on RF surfaces)
            # Give those candidates to top-k instead
            # Original: 25% topk, 15% gradient, 15% center, 45% uniform
            # New:      40% topk, 0% gradient, 15% center, 45% uniform
            
            if strategy < 0.40 and model is not None and len(model["top_k_pts"]) > 0:
                # Top-k perturbation (expanded from 25% to 40%)
                idx = int(rng.integers(len(model["top_k_pts"])))
                x = model["top_k_pts"][idx] + rng.normal(0, self.sigma_topk, dim) * widths
            elif strategy < 0.55:
                # Center perturbation (15%)
                x = center + rng.normal(0, self.sigma_center, dim) * widths
            else:
                # Uniform random (45%)
                x = np.array([rng.uniform(lo, hi) for lo, hi in cube.bounds], dtype=float)

            # clip to cube
            x = np.array([np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1]) for i in range(dim)], dtype=float)
            candidates.append(x)

        return candidates
