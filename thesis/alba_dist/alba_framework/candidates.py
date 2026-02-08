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

    sigma_topk: float = 0.15
    sigma_gradient_noise: float = 0.05
    sigma_center: float = 0.2
    step_min: float = 0.05
    step_max: float = 0.3

    def generate(self, cube: Cube, dim: int, rng: np.random.Generator, n: int) -> List[np.ndarray]:
        candidates: List[np.ndarray] = []
        widths = cube.widths()
        center = cube.center()
        model = cube.lgs_model

        for _ in range(n):
            strategy = float(rng.random())

            if strategy < 0.25 and model is not None and len(model["top_k_pts"]) > 0:
                idx = int(rng.integers(len(model["top_k_pts"])))
                x = model["top_k_pts"][idx] + rng.normal(0, self.sigma_topk, dim) * widths
            elif strategy < 0.40 and model is not None and model.get("gradient_dir") is not None:
                grad_dir = model["gradient_dir"]
                top_k_pts = model.get("top_k_pts", np.array([]))
                # BUG FIX: Skip gradient strategy if gradient contains NaN/Inf or top_k_pts is empty
                if not np.all(np.isfinite(grad_dir)) or len(top_k_pts) == 0:
                    # Fallback to center perturbation
                    x = center + rng.normal(0, self.sigma_center, dim) * widths
                else:
                    top_center = top_k_pts.mean(axis=0)
                    step = float(rng.uniform(self.step_min, self.step_max))
                    x = top_center + step * grad_dir * widths
                    x = x + rng.normal(0, self.sigma_gradient_noise, dim) * widths
            elif strategy < 0.55:
                x = center + rng.normal(0, self.sigma_center, dim) * widths
            else:
                x = np.array([rng.uniform(lo, hi) for lo, hi in cube.bounds], dtype=float)

            # clip to cube
            x = np.array([np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1]) for i in range(dim)], dtype=float)
            candidates.append(x)

        return candidates
