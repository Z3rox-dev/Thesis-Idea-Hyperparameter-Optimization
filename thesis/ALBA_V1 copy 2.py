#!/usr/bin/env python3
"""
ALBA (Adaptive Local Bayesian Algorithm) - Version 1.0

A novel hyperparameter optimization algorithm based on:
- Adaptive cube partitioning of the search space
- Local Gradient Surrogate (LGS) models within each cube
- UCB-style acquisition with exploration bonuses
- Two-phase optimization: exploration + local search refinement

Key Features:
- Beta prior for good_ratio estimation (avoids single-point dominance)
- Stagnation-aware exploration (increases randomness when stuck)
- Gradient-guided candidate generation
- Bayesian uncertainty estimation for acquisition
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


@dataclass(eq=False)
class Cube:
    """
    A hyperrectangle region of the search space with local surrogate model.
    
    Meta-objective fields track learning progress (model improvement):
    - lgs_quality: Current LGS model quality (negative weighted MSE)
    - m_trials, m_sum, m_sq_sum: Running stats for meta-objective (learning delta)
    - m_best: Best learning improvement seen in this cube
    """
    bounds: List[Tuple[float, float]]
    parent: Optional["Cube"] = None
    n_trials: int = 0
    n_good: int = 0
    best_score: float = -np.inf
    best_x: Optional[np.ndarray] = None
    _tested_pairs: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    # Per-evaluation learning progress (aligned with _tested_pairs)
    _tested_m: List[float] = field(default_factory=list)
    lgs_model: Optional[dict] = field(default=None, init=False)
    depth: int = 0
    # Categorical stats: {dim_idx: {value_idx: (n_good, n_total)}}
    cat_stats: dict = field(default_factory=dict)
    # Meta-based categorical stats: {dim_idx: {value_idx: (n_meta_good, n_total)}}
    cat_stats_meta: dict = field(default_factory=dict)
    
    # === META-OBJECTIVE FIELDS (learning progress tracking) ===
    lgs_quality: Optional[float] = None  # Current model quality (neg weighted MSE)
    m_trials: int = 0    # Number of meta-objective observations
    m_sum: float = 0.0   # Sum of learning deltas
    m_sq_sum: float = 0.0  # Sum of squared learning deltas (for variance)
    m_best: float = field(default_factory=lambda: -np.inf)  # Best meta value seen

    def _widths(self) -> np.ndarray:
        return np.array([abs(hi - lo) for lo, hi in self.bounds], dtype=float)

    def center(self) -> np.ndarray:
        return np.array([(lo + hi) / 2 for lo, hi in self.bounds], dtype=float)

    def contains(self, x: np.ndarray) -> bool:
        for i, (lo, hi) in enumerate(self.bounds):
            if x[i] < lo - 1e-9 or x[i] > hi + 1e-9:
                return False
        return True

    def good_ratio(self) -> float:
        """Beta prior estimation: (good + 1) / (trials + 2)"""
        return (self.n_good + 1) / (self.n_trials + 2)

    # === META-OBJECTIVE METHODS ===
    
    def compute_lgs_quality(self) -> Optional[float]:
        """
        Compute LGS model quality as negative weighted MSE.
        Higher is better (less error = better model).
        Returns None if no valid model exists.
        """
        if (
            self.lgs_model is None
            or self.lgs_model.get("inv_cov") is None
            or self.lgs_model.get("grad") is None
        ):
            return None
        
        model = self.lgs_model
        all_pts = model.get("all_pts")
        if all_pts is None or len(all_pts) < 3:
            return None
        
        # Get predictions and compute weighted MSE
        widths = model["widths"]
        center = model["center"]
        grad = model["grad"]
        y_mean = model["y_mean"]
        
        # Reconstruct predictions
        X_norm = (all_pts - center) / widths
        
        # Get actual scores aligned with the points used to fit the model.
        # NOTE: fit_lgs_model may backfill points from the parent; therefore
        # using self._tested_pairs here can misalign scores and points.
        scores = model.get("all_scores")
        if scores is None or len(scores) == 0:
            return None
        
        y_pred = y_mean + X_norm @ grad
        residuals = scores - y_pred
        
        # Compute weighted MSE (closer points weighted more)
        dists_sq = np.sum(X_norm**2, axis=1)
        sigma_sq = np.mean(dists_sq) + 1e-6
        weights = np.exp(-dists_sq / (2 * sigma_sq))
        weights = weights / (weights.sum() + 1e-9)
        
        wmse = np.sum(weights * residuals**2)
        
        # Return negative MSE (higher = better)
        return -wmse
    
    def m_mean(self) -> float:
        """Mean learning progress (meta-objective)."""
        if self.m_trials == 0:
            return 0.0
        return self.m_sum / self.m_trials
    
    def m_var(self) -> float:
        """Variance of learning progress."""
        if self.m_trials < 2:
            return 1.0  # High uncertainty with few samples
        mean = self.m_mean()
        return max(0.0, self.m_sq_sum / self.m_trials - mean**2)
    
    def m_ucb(self, beta: float = 1.0) -> float:
        """
        UCB-style score for meta-objective (learning potential).
        Balances mean learning progress with uncertainty.
        """
        mean = self.m_mean()
        std = np.sqrt(self.m_var())
        exploration = beta / np.sqrt(1 + self.m_trials)
        return mean + exploration * std

    def fit_lgs_model(self, gamma: float, dim: int, rng: np.random.Generator = None) -> None:
        """Fit Local Gradient Surrogate model using weighted linear regression."""
        pairs = list(self._tested_pairs)

        # Parent backfill with shuffle for diversity
        if self.parent and len(pairs) < 3 * dim:
            parent_pairs = getattr(self.parent, "_tested_pairs", [])
            extra = [pp for pp in parent_pairs if self.contains(pp[0])]
            needed = 3 * dim - len(pairs)
            if needed > 0 and extra:
                if rng is not None:
                    extra = list(extra)
                    rng.shuffle(extra)
                pairs = pairs + extra[:needed]

        if len(pairs) < dim + 2:
            self.lgs_model = None
            return

        all_pts = np.array([p for p, s in pairs])
        all_scores = np.array([s for p, s in pairs])

        k = max(3, len(pairs) // 5)
        top_k_idx = np.argsort(all_scores)[-k:]
        top_k_pts = all_pts[top_k_idx]

        gradient_dir = None
        grad = None
        inv_cov = None
        y_mean = 0.0
        noise_var = 1.0
        
        widths = np.maximum(self._widths(), 1e-9)
        center = self.center()

        if len(pairs) >= dim + 3:
            X_norm = (all_pts - center) / widths
            y_mean = all_scores.mean()
            y_centered = all_scores - y_mean
            
            try:
                dists_sq = np.sum(X_norm**2, axis=1)
                sigma_sq = np.mean(dists_sq) + 1e-6
                weights = np.exp(-dists_sq / (2 * sigma_sq))
                W = np.diag(weights)

                lambda_reg = 0.1
                XtWX = X_norm.T @ W @ X_norm + lambda_reg * np.eye(dim)
                inv_cov = np.linalg.inv(XtWX)
                grad = inv_cov @ (X_norm.T @ W @ y_centered)
                
                y_pred = X_norm @ grad
                residuals = y_centered - y_pred
                noise_var = np.clip(
                    np.average(residuals**2, weights=weights) + 1e-6,
                    1e-4, 10.0
                )
                
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-9:
                    gradient_dir = grad / grad_norm
            except Exception:
                pass

        self.lgs_model = {
            "all_pts": all_pts,
            "all_scores": all_scores,
            "top_k_pts": top_k_pts,
            "gradient_dir": gradient_dir,
            "grad": grad,
            "inv_cov": inv_cov,
            "y_mean": y_mean,
            "noise_var": noise_var,
            "widths": widths,
            "center": center,
        }

    def predict_bayesian(self, candidates: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and uncertainty for candidate points."""
        if (
            self.lgs_model is None
            or self.lgs_model.get("inv_cov") is None
            or self.lgs_model.get("grad") is None
        ):
            return np.zeros(len(candidates)), np.ones(len(candidates))

        model = self.lgs_model
        widths = model["widths"]
        center = model["center"]
        grad = model["grad"]
        inv_cov = model["inv_cov"]
        noise_var = model["noise_var"]
        y_mean = model["y_mean"]

        C_norm = (np.array(candidates) - center) / widths
        mu = y_mean + C_norm @ grad
        
        model_var = np.clip(np.sum((C_norm @ inv_cov) * C_norm, axis=1), 0, 10.0)
        total_var = noise_var * (1.0 + model_var)
        sigma = np.sqrt(total_var)
        
        return mu, sigma

    def get_split_axis(self) -> int:
        """Choose split axis based on gradient direction or widest dimension."""
        if self.lgs_model is not None and self.lgs_model["gradient_dir"] is not None:
            return int(np.argmax(np.abs(self.lgs_model["gradient_dir"])))
        return int(np.argmax(self._widths()))

    def split(self, gamma: float, dim: int, rng: np.random.Generator = None) -> List["Cube"]:
        """Split cube into two children along the chosen axis."""
        axis = self.get_split_axis()
        lo, hi = self.bounds[axis]
        good_pts = [p[axis] for p, s in self._tested_pairs if s >= gamma]

        if len(good_pts) >= 3:
            cut = float(np.median(good_pts))
            margin = 0.15 * (hi - lo)
            cut = np.clip(cut, lo + margin, hi - margin)
        else:
            cut = (lo + hi) / 2

        bounds_lo = list(self.bounds)
        bounds_hi = list(self.bounds)
        bounds_lo[axis] = (lo, cut)
        bounds_hi[axis] = (cut, hi)

        child_lo = Cube(bounds=bounds_lo, parent=self)
        child_hi = Cube(bounds=bounds_hi, parent=self)
        child_lo.depth = self.depth + 1
        child_hi.depth = self.depth + 1

        for idx, (pt, sc) in enumerate(self._tested_pairs):
            m_val = self._tested_m[idx] if idx < len(self._tested_m) else 0.0
            child = child_lo if pt[axis] < cut else child_hi
            child._tested_pairs.append((pt.copy(), sc))
            child._tested_m.append(float(m_val))
            child.n_trials += 1
            if sc >= gamma:
                child.n_good += 1
            # Update child's best score
            if sc > child.best_score:
                child.best_score = sc
                child.best_x = pt.copy()

        for ch in (child_lo, child_hi):
            # Rebuild meta aggregates from stored m values
            if ch._tested_m:
                ch.m_trials = len(ch._tested_m)
                ch.m_sum = float(np.sum(ch._tested_m))
                ch.m_sq_sum = float(np.sum(np.square(ch._tested_m)))
                ch.m_best = float(np.max(ch._tested_m))
            else:
                ch.m_trials = 0
                ch.m_sum = 0.0
                ch.m_sq_sum = 0.0
                ch.m_best = -np.inf

            ch.fit_lgs_model(gamma, dim, rng)
            ch.lgs_quality = ch.compute_lgs_quality()

        return [child_lo, child_hi]


class ALBA:
    """
    ALBA (Adaptive Local Bayesian Allocator) for Hyperparameter Optimization.
    
    Parameters
    ----------
    bounds : List[Tuple[float, float]]
        Search space bounds for each dimension.
    maximize : bool, default=False
        Whether to maximize (True) or minimize (False) the objective.
    seed : int, default=42
        Random seed for reproducibility.
    gamma_quantile : float, default=0.20
        Final quantile threshold for "good" points.
    gamma_quantile_start : float, default=0.15
        Initial quantile threshold (more selective at start).
    local_search_ratio : float, default=0.30
        Fraction of budget for local search phase.
    n_candidates : int, default=25
        Number of candidates to generate per iteration.
    split_trials_min : int, default=15
        Minimum trials before cube can split.
    split_depth_max : int, default=8
        Maximum tree depth.
    split_trials_factor : float, default=3.0
        Factor for split threshold: factor * dim + offset.
    split_trials_offset : int, default=6
        Offset for split threshold.
    novelty_weight : float, default=0.4
        Weight for exploration in UCB acquisition.
    total_budget : int, default=200
        Total optimization budget.
    global_random_prob : float, default=0.05
        Probability of pure random sampling for diversity.
    stagnation_threshold : int, default=50
        Iterations without improvement before increasing exploration.
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        maximize: bool = False,
        seed: int = 42,
        gamma_quantile: float = 0.20,
        gamma_quantile_start: float = 0.15,
        local_search_ratio: float = 0.30,
        n_candidates: int = 25,
        split_trials_min: int = 15,
        split_depth_max: int = 16,
        split_trials_factor: float = 3.0,
        split_trials_offset: int = 6,
        novelty_weight: float = 0.4,
        total_budget: int = 200,
        global_random_prob: float = 0.05,
        stagnation_threshold: int = 50,
        categorical_dims: List[Tuple[int, int]] = None,  # [(dim_idx, n_choices), ...]

        # --- Hierarchical / lexicographic selection (reward first, then knowledge) ---
        leaf_selection_mode: str = "hybrid",  # "hybrid" | "lexi_topk"
        leaf_reward_topk: int = 5,
        cat_selection_mode: str = "curiosity",  # "curiosity" | "lexi_topk"
        cat_reward_topk: int = 3,

        # During studying, ignore reward filtering and follow knowledge/interest.
        passion_during_study: bool = False,

        # --- Self-adapting controller (infrastructure) ---
        # If enabled, the optimizer learns online when "studying" is paying off.
        # It can switch study<->exploit and adapt a reward guardrail automatically.
        controller_enabled: bool = False,
        controller_check_every: int = 25,
        controller_min_history: int = 4,
        controller_quantile_low: float = 0.40,
        controller_quantile_high: float = 0.60,
        controller_guardrail_init: float = 0.50,
        controller_guardrail_step: float = 0.10,
        controller_guardrail_min: float = 0.20,
        controller_guardrail_max: float = 0.90,

        # --- Budget-adaptive switch: "study" then "exploit" ---
        # If enabled, meta influence is automatically reduced to ~0 after a
        # study budget computed from total_budget. This is critical for small
        # budgets (e.g., 400/500) where pure curiosity may not pay back.
        adaptive_budget: bool = False,
        study_budget_fraction: float = 0.25,
        study_budget_min: int = 50,
        study_budget_max_fraction: float = 0.60,

        # --- Performance-adaptive switch: stop studying if not improving ---
        adaptive_performance: bool = False,
        performance_patience: int = 75,

        # --- Debug ---
        debug: bool = False,
        debug_every: int = 100,

        # --- Meta-objective controls ---
        meta_good_mode: str = "positive",  # "positive" | "quantile"
        gamma_m_quantile: float = 0.50,     # used only if meta_good_mode=="quantile"
        meta_weight_start: float = 0.35,
        meta_weight_end: float = 0.0,
        meta_beta: float = 1.0,
        meta_stagnation_boost: float = 0.25,

        # --- Categorical TS mixing (reward vs meta) ---
        cat_meta_mix_start: float = 0.50,
        cat_meta_mix_end: float = 0.0,
        cat_meta_mix_stagnation_boost: float = 0.25,
    ) -> None:
        self.bounds = bounds
        self.dim = len(bounds)
        self.maximize = maximize
        self.rng = np.random.default_rng(seed)
        self.gamma_quantile = gamma_quantile
        self.gamma_quantile_start = gamma_quantile_start
        self.local_search_ratio = local_search_ratio
        self.n_candidates = n_candidates
        self.total_budget = total_budget
        self.categorical_dims = categorical_dims or []  # List of (dim_idx, n_choices)
        
        # === CURIOSITY-DRIVEN + ELITE CROSSOVER ===
        self._cat_visit_counts = {}  # Track how often each cat combo is visited
        self._elite_configs = []  # Top categorical configurations: [(cat_tuple, score)]
        self._elite_size = 10
        self._curiosity_bonus = 0.3  # Bonus for unseen combinations (tuned)
        self._crossover_rate = 0.15  # Probability of elite crossover (tuned)

        # Global categorical stats (empirical-Bayes prior for TS)
        # {dim_idx: {val_idx: (n_good, n_trials)}}
        self._global_cat_stats: Dict[int, Dict[int, Tuple[int, int]]] = {}

        # === META-OBJECTIVE (LEARNING PROGRESS) TRACKING ===
        # The meta-objective measures how much each evaluation improves model quality.
        # Early exploration is guided by learning progress; later by external objective.
        self.m_all: List[float] = []  # All learning progress values
        self.gamma_m: float = 0.0  # Threshold for "meta-good" (learning progress)
        self._meta_good_mode: str = meta_good_mode
        self._gamma_m_quantile: float = float(gamma_m_quantile)
        self._meta_weight_start: float = float(meta_weight_start)
        self._meta_weight_end: float = float(meta_weight_end)
        self._meta_beta: float = float(meta_beta)
        self._meta_stagnation_boost: float = float(meta_stagnation_boost)

        self._cat_meta_mix_start: float = float(cat_meta_mix_start)
        self._cat_meta_mix_end: float = float(cat_meta_mix_end)
        self._cat_meta_mix_stagnation_boost: float = float(cat_meta_mix_stagnation_boost)

        self._leaf_selection_mode: str = str(leaf_selection_mode)
        self._leaf_reward_topk: int = int(leaf_reward_topk)
        self._cat_selection_mode: str = str(cat_selection_mode)
        self._cat_reward_topk: int = int(cat_reward_topk)
        self._passion_during_study: bool = bool(passion_during_study)

        self._controller_enabled: bool = bool(controller_enabled)
        self._controller_check_every: int = int(controller_check_every)
        self._controller_min_history: int = int(controller_min_history)
        self._controller_q_low: float = float(controller_quantile_low)
        self._controller_q_high: float = float(controller_quantile_high)
        self._controller_guardrail: float = float(controller_guardrail_init)
        self._controller_guardrail_step: float = float(controller_guardrail_step)
        self._controller_guardrail_min: float = float(controller_guardrail_min)
        self._controller_guardrail_max: float = float(controller_guardrail_max)

        # Controller internal state
        self._mode: str = "study"  # "study" | "exploit"
        self._last_ctrl_iter: int = 0
        self._last_ctrl_best_y: float = -np.inf
        self._delta_best_hist: List[float] = []
        self._m_mean_hist: List[float] = []

        self._adaptive_budget: bool = bool(adaptive_budget)
        self._study_budget_fraction: float = float(study_budget_fraction)
        self._study_budget_min: int = int(study_budget_min)
        self._study_budget_max_fraction: float = float(study_budget_max_fraction)

        self._adaptive_performance: bool = bool(adaptive_performance)
        self._performance_patience: int = int(performance_patience)

        self._debug: bool = bool(debug)
        self._debug_every: int = int(debug_every)
        self._last_debug_iter: int = -1

        self.root = Cube(bounds=list(bounds))
        self.leaves: List[Cube] = [self.root]

        self.X_all: List[np.ndarray] = []
        self.y_all: List[float] = []  # Always stored as "higher is better"
        self.best_y_internal = -np.inf  # Internal score (always maximize)
        self.best_x: Optional[np.ndarray] = None
        self.gamma = 0.0
        self.iteration = 0
        
        self.stagnation = 0
        self.last_improvement_iter = 0

        self.exploration_budget = int(total_budget * (1 - local_search_ratio))
        self.local_search_budget = total_budget - self.exploration_budget

        self.global_widths = np.array([hi - lo for lo, hi in bounds])
        self.last_cube: Optional[Cube] = None

        self._split_trials_min = split_trials_min
        self._split_depth_max = split_depth_max
        self._split_trials_factor = split_trials_factor
        self._split_trials_offset = split_trials_offset
        self._novelty_weight = novelty_weight
        self._global_random_prob = global_random_prob
        self._stagnation_threshold = stagnation_threshold

        self._study_budget = self._compute_study_budget()

    def _compute_study_budget(self) -> int:
        tb = int(self.total_budget) if hasattr(self, "total_budget") else 0
        if tb <= 0:
            return int(self._study_budget_min)
        frac = float(np.clip(self._study_budget_fraction, 0.0, 1.0))
        max_frac = float(np.clip(self._study_budget_max_fraction, 0.0, 1.0))
        raw = int(round(frac * tb))
        lo = int(max(0, self._study_budget_min))
        hi = int(round(max_frac * tb))
        if hi <= 0:
            hi = tb
        return int(np.clip(raw, lo, hi))

    def _is_studying(self) -> bool:
        # Controller overrides other heuristics when enabled.
        if getattr(self, "_controller_enabled", False):
            return getattr(self, "_mode", "study") == "study"

        budget_ok = True
        perf_ok = True

        if self._adaptive_budget:
            budget_ok = int(self.iteration) < int(self._study_budget)

        if self._adaptive_performance:
            pat = int(self._performance_patience)
            if pat > 0:
                perf_ok = int(self.stagnation) < pat

        # If neither adaptation is enabled, we always "study".
        if not self._adaptive_budget and not self._adaptive_performance:
            return True

        return bool(budget_ok and perf_ok)

    def _controller_tick(self) -> None:
        """Update controller mode and guardrail periodically.

        Uses two internal signals:
        - reward progress: delta in best_y_internal since last check
        - learning progress: mean of recent meta progress values
        Decision is quantile-based on the history collected so far.
        """
        if not self._controller_enabled:
            return
        every = max(1, int(self._controller_check_every))
        it = int(self.iteration)
        if it - int(self._last_ctrl_iter) < every:
            return

        # Reward progress (>=0)
        cur_best = float(self.best_y_internal) if np.isfinite(self.best_y_internal) else -np.inf
        prev_best = float(self._last_ctrl_best_y) if np.isfinite(self._last_ctrl_best_y) else -np.inf
        delta_best = float(max(0.0, cur_best - prev_best)) if np.isfinite(cur_best) and np.isfinite(prev_best) else 0.0

        # Learning progress (recent m mean)
        recent_m = self.m_all[-every:] if len(self.m_all) >= 1 else []
        m_mean = float(np.mean(recent_m)) if recent_m else 0.0

        self._delta_best_hist.append(delta_best)
        self._m_mean_hist.append(m_mean)

        self._last_ctrl_iter = it
        self._last_ctrl_best_y = cur_best

        # Not enough history => stay in study but start adapting guardrail.
        if len(self._delta_best_hist) < max(1, self._controller_min_history):
            # If reward isn't improving early, tighten guardrail a bit.
            if delta_best <= 0.0:
                self._controller_guardrail = float(
                    np.clip(
                        self._controller_guardrail + self._controller_guardrail_step,
                        self._controller_guardrail_min,
                        self._controller_guardrail_max,
                    )
                )
            return

        ql = float(np.clip(self._controller_q_low, 0.0, 1.0))
        qh = float(np.clip(self._controller_q_high, 0.0, 1.0))
        # Ensure qh > ql
        if qh <= ql:
            qh = min(1.0, ql + 0.1)

        d_low = float(np.quantile(self._delta_best_hist, ql))
        d_high = float(np.quantile(self._delta_best_hist, qh))
        m_low = float(np.quantile(self._m_mean_hist, ql))
        m_high = float(np.quantile(self._m_mean_hist, qh))

        reward_bad = delta_best <= d_low
        reward_good = delta_best >= d_high
        learn_bad = m_mean <= m_low
        learn_good = m_mean >= m_high

        # Adapt guardrail: if reward is bad, tighten; if reward is good, loosen.
        if reward_bad and not reward_good:
            self._controller_guardrail = float(
                np.clip(
                    self._controller_guardrail + self._controller_guardrail_step,
                    self._controller_guardrail_min,
                    self._controller_guardrail_max,
                )
            )
        elif reward_good and not reward_bad:
            self._controller_guardrail = float(
                np.clip(
                    self._controller_guardrail - self._controller_guardrail_step,
                    self._controller_guardrail_min,
                    self._controller_guardrail_max,
                )
            )

        # Mode transitions
        if self._mode == "study":
            # If both learning and reward are bad, stop studying.
            if reward_bad and learn_bad:
                self._mode = "exploit"
        else:
            # If exploit is not yielding reward but learning signal is strong, go back to study.
            if reward_bad and learn_good:
                self._mode = "study"

    def _current_alpha(self) -> float:
        progress = min(1.0, self.iteration / max(1, self.exploration_budget))
        alpha = self._meta_weight_start + progress * (self._meta_weight_end - self._meta_weight_start)
        if self.stagnation > self._stagnation_threshold:
            alpha = min(1.0, alpha + self._meta_stagnation_boost)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if self._adaptive_budget and not self._is_studying():
            alpha = 0.0
        return alpha

    def _current_cat_lambda(self) -> float:
        progress = min(1.0, self.iteration / max(1, self.exploration_budget))
        lam = self._cat_meta_mix_start + progress * (self._cat_meta_mix_end - self._cat_meta_mix_start)
        if self.stagnation > self._stagnation_threshold:
            lam = min(1.0, lam + self._cat_meta_mix_stagnation_boost)
        lam = float(np.clip(lam, 0.0, 1.0))
        if self._adaptive_budget and not self._is_studying():
            lam = 0.0
        return lam

    def _maybe_debug(self, cube: "Cube" = None) -> None:
        if not self._debug:
            return
        if self._debug_every <= 0:
            return
        it = int(self.iteration)
        if it == self._last_debug_iter:
            return
        if it % int(self._debug_every) != 0:
            return
        self._last_debug_iter = it

        best_raw = float(self._to_raw(self.best_y_internal)) if np.isfinite(self.best_y_internal) else float("nan")
        study = self._is_studying()
        alpha = self._current_alpha()
        lam = self._current_cat_lambda()
        n_leaves = len(self.leaves) if hasattr(self, "leaves") else 0
        recent = self.m_all[-50:] if len(self.m_all) >= 1 else []
        if recent:
            recent_mean = float(np.mean(recent))
            recent_pos = float(np.mean([1.0 if m > 0 else 0.0 for m in recent]))
        else:
            recent_mean = float("nan")
            recent_pos = float("nan")

        mode = getattr(self, "_mode", "n/a")
        guard = getattr(self, "_controller_guardrail", float("nan"))
        msg = (
            f"[ALBA DEBUG] it={it}/{int(self.total_budget)} study_budget={int(self._study_budget)} "
            f"studying={study} mode={mode} guard_q={guard:.2f} best_f={best_raw:.6f} stagn={int(self.stagnation)} "
            f"perf_pat={int(self._performance_patience)} leaves={n_leaves} "
            f"gamma={float(self.gamma):.6f} gamma_m={float(self.gamma_m):.6f} alpha={alpha:.3f} lam={lam:.3f} "
            f"m50_mean={recent_mean:.5f} m50_pos={recent_pos:.2f}"
        )
        if cube is not None:
            c_best = float(self._to_raw(cube.best_score)) if np.isfinite(cube.best_score) else float("nan")
            msg += (
                f" | cube(depth={cube.depth}, n={cube.n_trials}, best_f={c_best:.6f}, "
                f"m_mean={cube.m_mean():.5f}, m_var={cube.m_var():.5f})"
            )
        print(msg, flush=True)

    def _to_internal(self, y_raw: float) -> float:
        """Convert raw objective value to internal score (higher is better)."""
        return y_raw if self.maximize else -y_raw

    def _to_raw(self, y_internal: float) -> float:
        """Convert internal score back to raw objective value."""
        return y_internal if self.maximize else -y_internal

    def ask(self) -> np.ndarray:
        """Suggest the next point to evaluate."""
        self.iteration = len(self.X_all)

        # Controller update (may switch mode and/or guardrail)
        self._controller_tick()

        # Global random for diversity - but still assign to a cube
        if self.rng.random() < self._global_random_prob:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
            self.last_cube = self._find_containing_leaf(x)
            return x

        if self.iteration < self.exploration_budget:
            self._update_gamma()
            self._recount_good()

            if self.iteration % 5 == 0:
                self._update_all_models()

            self.last_cube = self._select_leaf()
            return self._sample_in_cube(self.last_cube)

        # Local search phase with tree integration
        ls_iter = self.iteration - self.exploration_budget
        progress = ls_iter / max(1, self.local_search_budget - 1)
        local_search_prob = 0.5 + 0.4 * progress
        
        if self.rng.random() < local_search_prob:
            x = self._local_search_sample(progress)
            self.last_cube = self._find_containing_leaf(x)  # Assign to tree
        else:
            self._update_gamma()
            self._recount_good()
            self.last_cube = self._select_leaf()
            x = self._sample_in_cube(self.last_cube)
        
        return x

    def tell(self, x: np.ndarray, y_raw: float) -> None:
        """Report the objective value for a point.
        
        This method now also computes the META-OBJECTIVE (learning progress):
        m = new_quality - prev_quality (how much this eval improved our model).
        """
        y = self._to_internal(y_raw)
        
        # Update best (internal score, always maximize) - external objective drives final result
        if y > self.best_y_internal:
            self.best_y_internal = y
            self.best_x = x.copy()
            self.stagnation = 0
            self.last_improvement_iter = self.iteration
        else:
            self.stagnation += 1

        self.X_all.append(x.copy())
        self.y_all.append(y)
        
        # Update curiosity tracking and elite pool (y-based, not meta-based)
        if self.categorical_dims:
            cat_key = self._get_cat_key(x)
            self._cat_visit_counts[cat_key] = self._cat_visit_counts.get(cat_key, 0) + 1
            
            # Update elite pool
            self._elite_configs.append((cat_key, y))
            self._elite_configs.sort(key=lambda p: p[1], reverse=True)  # Higher is better (internal)
            self._elite_configs = self._elite_configs[:self._elite_size]

        if self.last_cube is not None:
            cube = self.last_cube
            
            # === META-OBJECTIVE: Compute learning progress ===
            # Save previous model quality before updating
            prev_quality = cube.compute_lgs_quality()
            
            # Add the new point to cube
            cube._tested_pairs.append((x.copy(), y))
            cube.n_trials += 1
            is_good = y >= self.gamma
            if is_good:
                cube.n_good += 1
            
            # Update cube's best score
            if y > cube.best_score:
                cube.best_score = y
                cube.best_x = x.copy()
            
            # Update categorical stats (still y-based for Thompson sampling base)
            for dim_idx, n_choices in self.categorical_dims:
                val_idx = self._discretize_cat(x[dim_idx], n_choices)
                if dim_idx not in cube.cat_stats:
                    cube.cat_stats[dim_idx] = {}
                n_g, n_t = cube.cat_stats[dim_idx].get(val_idx, (0, 0))
                cube.cat_stats[dim_idx][val_idx] = (n_g + (1 if is_good else 0), n_t + 1)

                # Update global stats (used as a stabilizing prior)
                if dim_idx not in self._global_cat_stats:
                    self._global_cat_stats[dim_idx] = {}
                gg, gt = self._global_cat_stats[dim_idx].get(val_idx, (0, 0))
                self._global_cat_stats[dim_idx][val_idx] = (gg + (1 if is_good else 0), gt + 1)

            # Refit model and compute new quality
            cube.fit_lgs_model(self.gamma, self.dim, self.rng)
            new_quality = cube.compute_lgs_quality()
            
            # Update cube's stored quality
            cube.lgs_quality = new_quality
            
            # Compute learning progress (meta-objective)
            if prev_quality is not None and new_quality is not None:
                m = new_quality - prev_quality  # Positive = model improved
            elif new_quality is not None:
                m = 0.1  # First valid model: small positive signal
            else:
                m = 0.0  # No model yet
            
            # Update cube meta stats
            cube.m_trials += 1
            cube.m_sum += m
            cube.m_sq_sum += m * m
            if m > cube.m_best:
                cube.m_best = m
            
            # Update global meta tracking
            self.m_all.append(m)
            self._update_gamma_m()

            # Store per-evaluation learning progress aligned with this point
            # (Used to build correct per-category TS stats for meta_good)
            cube._tested_m.append(float(m))

            # Periodic debug to verify whether we're "studying" (learning progress)
            # or mostly exploiting reward.
            self._maybe_debug(cube)

            # Check split (now includes meta-based triggers)
            if self._should_split(cube):
                children = cube.split(self.gamma, self.dim, self.rng)
                # Propagate categorical stats to children
                for child in children:
                    self._recompute_cat_stats(child)
                if cube in self.leaves:
                    self.leaves.remove(cube)
                    self.leaves.extend(children)

            self.last_cube = None
    
    def _update_gamma_m(self) -> None:
        """Update meta-objective threshold (gamma_m) from m_all quantile."""
        if len(self.m_all) < 10:
            self.gamma_m = 0.0
            return
        self.gamma_m = float(np.percentile(self.m_all, 100 * (1 - self._gamma_m_quantile)))

    def _update_gamma(self) -> None:
        if len(self.y_all) < 10:
            self.gamma = 0.0
            return
        progress = min(1.0, self.iteration / max(1, self.exploration_budget * 0.5))
        current_quantile = self.gamma_quantile_start - progress * (
            self.gamma_quantile_start - self.gamma_quantile
        )
        self.gamma = float(np.percentile(self.y_all, 100 * (1 - current_quantile)))

    def _recount_good(self) -> None:
        for leaf in self.leaves:
            leaf.n_good = sum(1 for _, s in leaf._tested_pairs if s >= self.gamma)
            # Recompute cat_stats with current gamma to avoid stale stats
            self._recompute_cat_stats(leaf)

    def _select_leaf(self) -> Cube:
        """
        Select a leaf cube for exploration.
        
        HYBRID META + EXTERNAL OBJECTIVE SELECTION:
        - meta_score: UCB-style score for learning potential (curiosity-driven)
        - reward_proxy: normalized cube best score (external objective)
        - alpha: weight schedule (high early = explore learnable regions first)
        """
        if not self.leaves:
            return self.root

        # === LEXICOGRAPHIC (HIERARCHICAL) SELECTION ===
        # 1) Filter by reward (proxy for f)
        # 2) Within Top-K, choose by knowledge (meta UCB + exploration bonuses)
        if self._leaf_selection_mode == "lexi_topk":
            scores = []
            studying = self._is_studying()

            if studying and self._passion_during_study:
                # Passion-first study: explore what seems learnable/interesting,
                # but optionally apply an adaptive reward guardrail.
                candidates = list(self.leaves)
            else:
                reward_keys = np.array(
                    [c.best_score if np.isfinite(c.best_score) else -np.inf for c in self.leaves],
                    dtype=float,
                )
                if not np.any(np.isfinite(reward_keys)):
                    # No reward signal yet: explore least-tried region.
                    idx = int(np.argmin([c.n_trials for c in self.leaves]))
                    return self.leaves[idx]

                # If controller is enabled, use its adaptive quantile guardrail in BOTH phases.
                # Otherwise, fall back to fixed Top-K.
                if self._controller_enabled:
                    finite = reward_keys[np.isfinite(reward_keys)]
                    if len(finite) > 0:
                        thr = float(np.quantile(finite, float(np.clip(self._controller_guardrail, 0.0, 1.0))))
                        candidates = [c for c in self.leaves if np.isfinite(c.best_score) and c.best_score >= thr]
                    else:
                        candidates = list(self.leaves)
                else:
                    k = int(np.clip(self._leaf_reward_topk, 1, len(self.leaves)))
                    top_idx = np.argsort(reward_keys)[::-1][:k]
                    candidates = [self.leaves[int(i)] for i in top_idx]

            if not candidates:
                candidates = list(self.leaves)

            for c in candidates:
                exploration = 0.3 / np.sqrt(1 + c.n_trials)
                if self.stagnation > self._stagnation_threshold:
                    exploration *= 2.0

                model_bonus = 0.0
                if c.lgs_model is not None:
                    n_pts = len(c.lgs_model.get("all_pts", []))
                    if n_pts >= self.dim + 2:
                        model_bonus = 0.1

                if studying:
                    # STUDY: choose by knowledge (optionally within reward-acceptable set)
                    meta_score = c.m_ucb(self._meta_beta)
                    if self.stagnation > self._stagnation_threshold:
                        meta_score *= (1.0 + self._meta_stagnation_boost)
                    total = meta_score + exploration + model_bonus
                else:
                    # EXPLOIT: choose by reward within the same reward-acceptable set
                    total = float(c.best_score) + exploration + model_bonus

                if not np.isfinite(total):
                    total = exploration
                scores.append(total)

            scores_arr = np.array(scores, dtype=float)
            scores_arr = scores_arr - np.max(scores_arr)
            temperature = 1.5 if self.stagnation > self._stagnation_threshold else 3.0
            probs = np.exp(scores_arr * temperature)
            probs = probs / probs.sum()
            idx = self.rng.choice(len(candidates), p=probs)
            return candidates[int(idx)]

        # === HYBRID (DEFAULT) SELECTION ===
        # Compute meta weight (alpha) based on progress
        # Early: strong meta dominance; Late: mostly external objective
        alpha = self._current_alpha()
        
        # Normalize best scores for reward proxy (handle inf/nan)
        all_bests = [c.best_score for c in self.leaves]
        # Filter out inf values for normalization
        finite_bests = [b for b in all_bests if np.isfinite(b)]
        if finite_bests:
            best_min, best_max = min(finite_bests), max(finite_bests)
        else:
            best_min, best_max = 0.0, 1.0
        best_range = best_max - best_min if best_max > best_min else 1.0

        scores = []
        for c in self.leaves:
            # === META SCORE: Learning potential (UCB on meta-objective) ===
            meta_score = c.m_ucb(self._meta_beta)
            
            # === REWARD PROXY: Normalized external objective ===
            if np.isfinite(c.best_score):
                reward_proxy = (c.best_score - best_min) / best_range if best_range > 0 else 0.5
            else:
                reward_proxy = 0.0  # No valid score yet
            
            # === EXPLORATION BONUS ===
            exploration = 0.3 / np.sqrt(1 + c.n_trials)
            if self.stagnation > self._stagnation_threshold:
                exploration *= 2.0
            
            # Model bonus (has learned something)
            model_bonus = 0.0
            if c.lgs_model is not None:
                n_pts = len(c.lgs_model.get("all_pts", []))
                if n_pts >= self.dim + 2:
                    model_bonus = 0.1
            
            # === HYBRID SCORE ===
            # Alpha blends meta (learning) vs reward (exploitation)
            hybrid = alpha * meta_score + (1 - alpha) * reward_proxy
            
            total_score = hybrid + exploration + model_bonus
            
            # Safety check for NaN
            if not np.isfinite(total_score):
                total_score = exploration  # Fallback to exploration bonus
            
            scores.append(total_score)

        scores_arr = np.array(scores)
        scores_arr = scores_arr - scores_arr.max()
        
        temperature = 1.5 if self.stagnation > self._stagnation_threshold else 3.0
        
        probs = np.exp(scores_arr * temperature)
        probs = probs / probs.sum()
        idx = self.rng.choice(len(self.leaves), p=probs)
        return self.leaves[int(idx)]

    def _clip_to_cube(self, x: np.ndarray, cube: Cube) -> np.ndarray:
        return np.array([
            np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1])
            for i in range(self.dim)
        ])

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(x[i], self.bounds[i][0], self.bounds[i][1])
            for i in range(self.dim)
        ])

    def _find_containing_leaf(self, x: np.ndarray) -> Cube:
        """Find the leaf cube that contains point x."""
        for leaf in self.leaves:
            if leaf.contains(x):
                return leaf
        # Fallback: return leaf with closest center (shouldn't happen normally)
        min_dist = float('inf')
        closest = self.leaves[0] if self.leaves else self.root
        for leaf in self.leaves:
            dist = np.linalg.norm(x - leaf.center())
            if dist < min_dist:
                min_dist = dist
                closest = leaf
        return closest

    def _generate_candidates(self, cube: Cube, n: int) -> List[np.ndarray]:
        candidates: List[np.ndarray] = []
        widths = cube._widths()
        center = cube.center()
        model = cube.lgs_model

        for _ in range(n):
            strategy = self.rng.random()

            if strategy < 0.25 and model is not None and len(model["top_k_pts"]) > 0:
                idx = self.rng.integers(len(model["top_k_pts"]))
                x = model["top_k_pts"][idx] + self.rng.normal(0, 0.15, self.dim) * widths
            elif strategy < 0.40 and model is not None and model["gradient_dir"] is not None:
                top_center = model["top_k_pts"].mean(axis=0)
                step = self.rng.uniform(0.05, 0.3)
                x = top_center + step * model["gradient_dir"] * widths
                x = x + self.rng.normal(0, 0.05, self.dim) * widths
            elif strategy < 0.55:
                x = center + self.rng.normal(0, 0.2, self.dim) * widths
            else:
                x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])

            x = self._clip_to_cube(x, cube)
            candidates.append(x)

        return candidates

    def _sample_with_lgs(self, cube: Cube) -> np.ndarray:
        if cube.lgs_model is None or cube.n_trials < 5:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])

        candidates = self._generate_candidates(cube, self.n_candidates)
        mu, sigma = cube.predict_bayesian(candidates)
        
        beta = self._novelty_weight * 2.0
        score = mu + beta * sigma
        
        if score.std() > 1e-9:
            score_z = (score - score.mean()) / score.std()
        else:
            score_z = np.zeros_like(score)

        probs = np.exp(score_z * 3.0)
        probs = probs / probs.sum()
        idx = self.rng.choice(len(candidates), p=probs)
        return candidates[idx]

    def _discretize_cat(self, x_val: float, n_choices: int) -> int:
        """Convert continuous [0,1] value to discrete category index."""
        return min(int(round(x_val * (n_choices - 1))), n_choices - 1)
    
    def _cat_value_to_continuous(self, val_idx: int, n_choices: int) -> float:
        """Convert discrete category index back to continuous [0,1]."""
        return val_idx / (n_choices - 1) if n_choices > 1 else 0.5
    
    def _get_cat_key(self, x: np.ndarray) -> tuple:
        """Get tuple of categorical values from x."""
        return tuple(self._discretize_cat(x[dim_idx], n_ch) 
                     for dim_idx, n_ch in self.categorical_dims)
    
    def _cat_key_to_x(self, cat_key: tuple, x: np.ndarray) -> np.ndarray:
        """Apply categorical key to x vector."""
        x = x.copy()
        for i, (dim_idx, n_ch) in enumerate(self.categorical_dims):
            x[dim_idx] = self._cat_value_to_continuous(cat_key[i], n_ch)
        return x
    
    def _elite_crossover(self) -> tuple:
        """Create new categorical config by crossing over two elites."""
        if len(self._elite_configs) < 2:
            return None
        
        # Select two parents (bias towards better ones)
        n = len(self._elite_configs)
        weights = np.array([1.0 / (i + 1) for i in range(n)])
        weights = weights / weights.sum()
        
        idx1, idx2 = self.rng.choice(n, size=2, replace=False, p=weights)
        parent1 = self._elite_configs[idx1][0]
        parent2 = self._elite_configs[idx2][0]
        
        # Adaptive mutation rate: higher when stagnating
        mutation_rate = 0.1
        if self.stagnation > self._stagnation_threshold:
            mutation_rate = 0.25
        
        # Uniform crossover with mutation
        child = []
        for i, (dim_idx, n_ch) in enumerate(self.categorical_dims):
            if self.rng.random() < 0.5:
                val = parent1[i]
            else:
                val = parent2[i]
            
            if self.rng.random() < mutation_rate:
                val = self.rng.integers(0, n_ch)
            
            child.append(val)
        
        return tuple(child)
    
    def _recompute_cat_stats(self, cube: Cube) -> None:
        """
        Recompute categorical stats for a cube from its tested pairs.
        
        META-OBJECTIVE UPDATE: Use meta_good (learning progress >= gamma_m)
        instead of y-based good. This makes Thompson Sampling estimate
        probability of meta improvement, not just reward.
        """
        cube.cat_stats = {}
        cube.cat_stats_meta = {}  # Track meta-based stats separately
        for dim_idx, n_choices in self.categorical_dims:
            cube.cat_stats[dim_idx] = {}
            cube.cat_stats_meta[dim_idx] = {}
        
        for idx, (pt, sc) in enumerate(cube._tested_pairs):
            # Original y-based good (kept for backward compatibility)
            is_good_y = sc >= self.gamma

            # Meta-based good MUST be per-evaluation:
            # meta_good := (m >= gamma_m) where m = Δ(model quality) from that evaluation.
            if idx < len(getattr(cube, "_tested_m", [])):
                m_val = float(cube._tested_m[idx])
                if self._meta_good_mode == "quantile":
                    is_good_meta = m_val >= self.gamma_m
                else:
                    # Default: any positive learning progress counts as meta-good
                    is_good_meta = m_val > 0.0
            else:
                # Fallback for legacy history without per-point m
                is_good_meta = False
            
            for dim_idx, n_choices in self.categorical_dims:
                val_idx = self._discretize_cat(pt[dim_idx], n_choices)
                if dim_idx not in cube.cat_stats:
                    cube.cat_stats[dim_idx] = {}
                if dim_idx not in cube.cat_stats_meta:
                    cube.cat_stats_meta[dim_idx] = {}
                
                # Y-based stats (original)
                n_g, n_t = cube.cat_stats[dim_idx].get(val_idx, (0, 0))
                cube.cat_stats[dim_idx][val_idx] = (n_g + (1 if is_good_y else 0), n_t + 1)
                
                # Meta-based stats (new)
                n_gm, n_tm = cube.cat_stats_meta[dim_idx].get(val_idx, (0, 0))
                cube.cat_stats_meta[dim_idx][val_idx] = (n_gm + (1 if is_good_meta else 0), n_tm + 1)
    
    def _apply_categorical_sampling(self, x: np.ndarray, cube: Cube) -> np.ndarray:
        """
        CURIOSITY-DRIVEN categorical sampling with ELITE CROSSOVER.
        - Bonus for less-visited combinations (curiosity)
        - Occasional crossover from elite pool
        - TPE-like base probabilities
        """
        if not self.categorical_dims:
            return x
        
        x = x.copy()
        
        # === ELITE CROSSOVER: occasionally use crossover from top configs ===
        crossover_prob = self._crossover_rate
        if self.stagnation > self._stagnation_threshold:
            crossover_prob *= 2  # More crossover when stuck
        
        if self.rng.random() < crossover_prob and len(self._elite_configs) >= 2:
            child_key = self._elite_crossover()
            if child_key:
                return self._cat_key_to_x(child_key, x)
        
        # === CURIOSITY-DRIVEN SAMPLING ===
        # Thompson Sampling per categoria usando stats META-OBJECTIVE!
        # Stima la probabilità di meta improvement, non solo reward.
        
        # First, do Thompson Sampling per dimension
        exploration_boost = 2.0 if self.stagnation > self._stagnation_threshold else 1.0
        
        # Mix reward-based and meta-based stats (meta early / when stuck)
        has_meta_stats = hasattr(cube, "cat_stats_meta") and bool(cube.cat_stats_meta)
        has_reward_stats = hasattr(cube, "cat_stats") and bool(cube.cat_stats)

        lam = self._current_cat_lambda()
        
        # Generate multiple candidate categorical configs
        n_candidates = 5
        candidates = []
        cand_reward_scores: List[float] = []
        cand_meta_scores: List[float] = []
        
        for _ in range(n_candidates):
            cat_vals = []
            cand_reward = 0.0
            cand_meta = 0.0
            for dim_idx, n_choices in self.categorical_dims:
                stats_reward = cube.cat_stats.get(dim_idx, {}) if has_reward_stats else {}
                stats_meta = cube.cat_stats_meta.get(dim_idx, {}) if has_meta_stats else {}

                # Thompson Sampling: sample from Beta distribution for each category
                # Reward-based: estimates P(y >= gamma | category)
                # Meta-based: estimates P(meta_good | category)
                # We blend the sampled probabilities: sample = (1-lam)*sample_reward + lam*sample_meta
                samples = []
                samples_r = []
                samples_m = []
                K = n_choices * exploration_boost
                for v in range(n_choices):
                    n_gr, n_tr = stats_reward.get(v, (0, 0))
                    n_gm, n_tm = stats_meta.get(v, (0, 0))

                    # Empirical-Bayes prior from global stats to stabilize sparse cube stats.
                    g_gr, g_tr = self._global_cat_stats.get(dim_idx, {}).get(v, (0, 0))
                    if g_tr > 0:
                        g_rate = float(g_gr) / float(g_tr)
                        prior_strength = float(min(10.0, np.sqrt(float(g_tr))))
                        prior_a = 0.5 + g_rate * prior_strength
                        prior_b = 0.5 + (1.0 - g_rate) * prior_strength
                    else:
                        prior_a = 1.0
                        prior_b = 1.0

                    # Reward posterior sample
                    alpha_r = float(n_gr) + prior_a
                    beta_r = float(n_tr - n_gr) + float(K) + prior_b
                    sample_r = self.rng.beta(alpha_r, beta_r)

                    # Meta posterior sample (if unavailable, fall back to reward)
                    if has_meta_stats:
                        alpha_m = n_gm + 1
                        beta_m = (n_tm - n_gm) + K
                        sample_m = self.rng.beta(alpha_m, beta_m)
                    else:
                        sample_m = sample_r

                    sample = (1.0 - lam) * sample_r + lam * sample_m
                    samples.append(sample)
                    samples_r.append(sample_r)
                    samples_m.append(sample_m)
                
                # Pick the category with highest sampled value
                chosen = int(np.argmax(samples))
                cat_vals.append(chosen)

                cand_reward += float(samples_r[chosen])
                cand_meta += float(samples_m[chosen])
            
            candidates.append(tuple(cat_vals))
            cand_reward_scores.append(float(cand_reward))
            cand_meta_scores.append(float(cand_meta))
        
        # Score candidates by curiosity (inverse visit count)
        curiosity_scores = []
        for cat_key in candidates:
            visit_count = self._cat_visit_counts.get(cat_key, 0)
            curiosity = self._curiosity_bonus / (1 + visit_count)
            curiosity_scores.append(float(curiosity))

        if self._cat_selection_mode == "lexi_topk":
            reward_arr = np.array(cand_reward_scores, dtype=float)
            meta_arr = np.array(cand_meta_scores, dtype=float)
            cur_arr = np.array(curiosity_scores, dtype=float)

            studying = self._is_studying()

            # Optionally ignore reward filtering (passion) only when controller is disabled.
            if studying and getattr(self, "_passion_during_study", False) and not self._controller_enabled:
                top_idx = np.arange(len(candidates))
            else:
                # If controller is enabled, use its adaptive quantile guardrail in BOTH phases.
                if self._controller_enabled:
                    thr = float(np.quantile(reward_arr, float(np.clip(self._controller_guardrail, 0.0, 1.0))))
                    top_idx = np.where(reward_arr >= thr)[0]
                    if len(top_idx) == 0:
                        top_idx = np.argsort(reward_arr)[::-1][: int(np.clip(self._cat_reward_topk, 1, len(candidates)))]
                else:
                    k = int(np.clip(self._cat_reward_topk, 1, len(candidates)))
                    top_idx = np.argsort(reward_arr)[::-1][:k]

            # Knowledge stage depends on whether we're still studying.
            # - STUDY: meta (learning potential) + curiosity
            # - EXPLOIT: reward + curiosity (no extra meta preference)
            if studying:
                score2 = meta_arr[top_idx] + cur_arr[top_idx]
            else:
                score2 = reward_arr[top_idx] + cur_arr[top_idx]
            score2 = score2 - np.max(score2)
            probs = np.exp(score2 * 3.0)
            probs = probs / probs.sum()
            chosen_rel = self.rng.choice(len(top_idx), p=probs)
            chosen_key = candidates[int(top_idx[int(chosen_rel)])]
        else:
            # Default: select purely by curiosity (softmax)
            scores = np.array(curiosity_scores, dtype=float)
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            probs = np.exp(scores * 3)
            probs = probs / probs.sum()
            chosen_idx = self.rng.choice(len(candidates), p=probs)
            chosen_key = candidates[int(chosen_idx)]
        
        # Apply chosen categorical config to x
        return self._cat_key_to_x(chosen_key, x)

    def _sample_in_cube(self, cube: Cube) -> np.ndarray:
        if self.iteration < 15:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        else:
            x = self._sample_with_lgs(cube)
        
        # Apply categorical sampling based on cube stats
        x = self._apply_categorical_sampling(x, cube)
        
        # Re-clip to cube bounds (categorical sampling may have moved outside)
        x = self._clip_to_cube(x, cube)
        return x

    def _local_search_sample(self, progress: float) -> np.ndarray:
        if self.best_x is None:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
        radius = 0.15 * (1 - progress) + 0.03
        noise = self.rng.normal(0, radius, self.dim) * self.global_widths
        x = self.best_x + noise

        # Discrete-heavy local search: mutate categorical dimensions near incumbent.
        # This is critical for JAHS-like spaces where most gain comes from categorical moves.
        if self.categorical_dims:
            # More conservative over time (later = smaller chance of categorical jumps).
            p_mut = 0.18 * (1 - progress) + 0.04
            for dim_idx, n_choices in self.categorical_dims:
                if self.rng.random() < p_mut:
                    cur_idx = self._discretize_cat(float(self.best_x[dim_idx]), int(n_choices))
                    # Pick a different category uniformly.
                    if n_choices > 1:
                        new_idx = int(self.rng.integers(0, n_choices - 1))
                        if new_idx >= cur_idx:
                            new_idx += 1
                    else:
                        new_idx = cur_idx
                    # Map category index back into [0,1).
                    x[dim_idx] = min(0.999999, (new_idx + 0.5) / float(n_choices))

        return self._clip_to_bounds(x)

    def _should_split(self, cube: Cube) -> bool:
        """
        Decide if a cube should be split.
        
        META-OBJECTIVE TRIGGERS (in addition to standard trial-based):
        1. High m_var → heterogeneous region → split to specialize
        2. Low m_mean after many trials → learning saturated → split to restart
        """
        # Basic constraints
        if cube.n_trials < self._split_trials_min:
            return False
        if cube.depth >= self._split_depth_max:
            return False
        
        # === META-BASED SPLIT TRIGGERS ===
        if cube.m_trials >= 5:  # Need enough meta observations
            m_var = cube.m_var()
            m_mean = cube.m_mean()
            
            # 1. High variance in learning progress → heterogeneous region
            # Different parts of this cube learn at very different rates
            # Splitting can create more homogeneous subregions
            if m_var > 0.05:  # Variance threshold (tunable)
                return True
            
            # 2. Low mean learning progress after many trials → saturated
            # The model in this cube has stopped improving
            # Splitting gives new subregions fresh models to fit
            if cube.m_trials >= 10 and m_mean < 0.001:  # Near-zero learning
                return True
        
        # === STANDARD TRIAL-BASED SPLIT ===
        return cube.n_trials >= self._split_trials_factor * self.dim + self._split_trials_offset

    def _update_all_models(self) -> None:
        for leaf in self.leaves:
            leaf.fit_lgs_model(self.gamma, self.dim, self.rng)

    def optimize(self, objective: Callable[[np.ndarray], float], budget: int = 100) -> Tuple[np.ndarray, float]:
        """
        Run optimization loop.
        
        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Function to optimize.
        budget : int, default=100
            Number of function evaluations.
            
        Returns
        -------
        best_x : np.ndarray
            Best configuration found.
        best_y : float
            Best objective value found.
        """
        if budget != self.total_budget:
            self.total_budget = budget
            self.exploration_budget = int(budget * (1 - self.local_search_ratio))
            self.local_search_budget = budget - self.exploration_budget

        for _ in range(budget):
            x = self.ask()
            y_raw = objective(x)
            self.tell(x, y_raw)

        return self.best_x, self._to_raw(self.best_y_internal)
