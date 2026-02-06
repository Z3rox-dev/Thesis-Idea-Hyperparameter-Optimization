"""ALBA Framework - Leaf Selection

This module defines how ALBA selects a leaf Cube to sample from during
exploration/local-search tree-guided steps.

Default implementation matches ALBA_V1: a softmax over a UCB-like score
based on good_ratio, exploration bonus, and a small model bonus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

from .cube import Cube


class LeafSelector(Protocol):
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        ...


@dataclass(frozen=True)
class UCBSoftmaxLeafSelector:
    """Default leaf selection policy (matches ALBA_V1)."""

    base_exploration: float = 0.3
    stagnation_exploration_multiplier: float = 2.0
    model_bonus: float = 0.1
    model_bonus_min_points_offset: int = 2
    temperature_normal: float = 3.0
    temperature_stagnating: float = 1.5

    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        if not leaves:
            raise ValueError("LeafSelector requires at least one leaf")

        scores = []
        for c in leaves:
            ratio = c.good_ratio()
            exploration = self.base_exploration / np.sqrt(1 + c.n_trials)
            if stagnating:
                exploration *= self.stagnation_exploration_multiplier

            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= dim + self.model_bonus_min_points_offset:
                    model_bonus = self.model_bonus

            scores.append(ratio + exploration + model_bonus)

        scores_arr = np.asarray(scores, dtype=float)
        scores_arr = scores_arr - scores_arr.max()

        greedy_prob = 0.25 + (0.15 if stagnating else 0.0)
        if rng.random() < greedy_prob:
            return leaves[int(np.argmax(scores_arr))]

        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(leaves), p=probs))
        return leaves[idx]
