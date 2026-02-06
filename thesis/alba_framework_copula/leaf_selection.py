"""ALBA Framework - Leaf Selection

This module defines how ALBA selects a leaf Cube to sample from during
exploration/local-search tree-guided steps.

Default implementation matches ALBA_V1: a softmax over a UCB-like score
based on good_ratio, exploration bonus, and a small model bonus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Any

import numpy as np

from .cube import Cube


class LeafSelector(Protocol):
    def select(self, leaves: List[Cube], dim: int, stagnating: bool, rng: np.random.Generator) -> Cube:
        ...


@dataclass
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

        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(leaves), p=probs))
        return leaves[idx]


@dataclass
class PotentialAwareLeafSelector(UCBSoftmaxLeafSelector):
    """
    Leaf selector that incorporates Global Potential Field from CoherenceTracker.
    
    Potential u in [0, 1] (lower is better).
    Bonus = weight * (1.0 - u).
    """
    
    potential_weight: float = 0.5
    tracker: Any = None  # Injected CoherenceTracker

    def set_tracker(self, tracker: Any):
        self.tracker = tracker

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

            potential_bonus = 0.0
            if self.tracker is not None:
                # Potential is in [0, 1], lower is better. Convert to bonus.
                u = self.tracker.get_potential(c, leaves)
                potential_bonus = self.potential_weight * (1.0 - u)

            scores.append(ratio + exploration + model_bonus + potential_bonus)

        scores_arr = np.asarray(scores, dtype=float)
        scores_arr = scores_arr - scores_arr.max()

        temperature = self.temperature_stagnating if stagnating else self.temperature_normal
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(leaves), p=probs))
        return leaves[idx]
