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
        try:
            w = cube.get_warp_multipliers()
        except Exception:
            w = None
        if w is not None and getattr(w, "shape", None) == (dim,):
            step_widths = widths / np.asarray(w, dtype=float)
        else:
            step_widths = widths
        center = cube.center()
        model = cube.lgs_model

        for _ in range(n):
            strategy = float(rng.random())

            if strategy < 0.25 and model is not None and len(model["top_k_pts"]) > 0:
                idx = int(rng.integers(len(model["top_k_pts"])))
                x = model["top_k_pts"][idx] + rng.normal(0, self.sigma_topk, dim) * step_widths
            elif strategy < 0.40 and model is not None and model["gradient_dir"] is not None:
                top_center = model["top_k_pts"].mean(axis=0)
                step = float(rng.uniform(self.step_min, self.step_max))
                x = top_center + step * model["gradient_dir"] * step_widths
                x = x + rng.normal(0, self.sigma_gradient_noise, dim) * step_widths
            elif strategy < 0.55:
                x = center + rng.normal(0, self.sigma_center, dim) * step_widths
            else:
                x = np.array([rng.uniform(lo, hi) for lo, hi in cube.bounds], dtype=float)

            # clip to cube
            x = np.array([np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1]) for i in range(dim)], dtype=float)
            candidates.append(x)

        return candidates
