"""
ALBA Framework - Optimizer Module

This module implements the main ALBA optimizer class, orchestrating:
- Adaptive cube partitioning of the search space
- Local Gradient Surrogate (LGS) models for each region
- UCB-style acquisition with exploration bonuses
- Two-phase optimization: exploration + local search refinement
- Categorical handling with curiosity-driven sampling

ALBA (Adaptive Local Bayesian Algorithm) is designed for efficient
hyperparameter optimization in mixed continuous-categorical spaces.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .cube import Cube
from .categorical import CategoricalSampler
from .gamma import GammaScheduler, QuantileAnnealedGammaScheduler
from .grid import LeafGridState
from .leaf_selection import LeafSelector, UCBSoftmaxLeafSelector
from .local_search import GaussianLocalSearchSampler, LocalSearchSampler
from .param_space import ParamSpaceHandler
from .splitting import (
    CubeIntrinsicSplitPolicy,
    SplitDecider,
    SplitPolicy,
    ThresholdSplitDecider,
)

warnings.filterwarnings("ignore")


class ALBA:
    """
    ALBA (Adaptive Local Bayesian Allocator) for Hyperparameter Optimization.

    ALBA combines adaptive space partitioning with local surrogate models to
    efficiently optimize both continuous and categorical hyperparameters.

    Parameters
    ----------
    bounds : Optional[List[Tuple[float, float]]]
        Search space bounds for each dimension. Required if param_space is not provided.
    param_space : Optional[Dict[str, Any]]
        Typed parameter space specification. If provided, ALBA handles
        encoding/decoding automatically.
    param_order : Optional[List[str]]
        Optional explicit ordering of parameters when using param_space.
    maximize : bool
        Whether to maximize (True) or minimize (False) the objective.
    seed : int
        Random seed for reproducibility.
    gamma_quantile : float
        Final quantile threshold for "good" points.
    gamma_quantile_start : float
        Initial quantile threshold (more selective at start).
    local_search_ratio : float
        Fraction of budget for local search phase.
    split_trials_min : int
        Minimum trials before cube can split.
    split_depth_max : int
        Maximum tree depth.
    split_trials_factor : float
        Factor for split threshold: factor * dim + offset.
    split_trials_offset : int
        Offset for split threshold.
    novelty_weight : float
        Weight for exploration in UCB acquisition.
    total_budget : int
        Total optimization budget.
    global_random_prob : float
        Probability of pure random sampling for diversity.
    stagnation_threshold : int
        Iterations without improvement before increasing exploration.
    categorical_dims : Optional[List[Tuple[int, int]]]
        List of (dim_idx, n_choices) for categorical dimensions.
        Only needed when using bounds directly.

    Examples
    --------
    Using param_space (recommended):

    >>> param_space = {
    ...     'learning_rate': (1e-4, 1e-1, 'log'),
    ...     'hidden_size': (32, 512, 'int'),
    ...     'activation': ['relu', 'tanh', 'gelu'],
    ... }
    >>> opt = ALBA(param_space=param_space, seed=42)
    >>> for _ in range(100):
    ...     config = opt.ask()  # Returns dict
    ...     score = evaluate(config)
    ...     opt.tell(config, score)

    Using bounds directly:

    >>> bounds = [(0, 1), (0, 1), (0, 1)]
    >>> categorical_dims = [(2, 3)]  # Third dim has 3 categories
    >>> opt = ALBA(bounds=bounds, categorical_dims=categorical_dims)
    >>> for _ in range(100):
    ...     x = opt.ask()  # Returns np.ndarray
    ...     score = evaluate(x)
    ...     opt.tell(x, score)
    """

    def __init__(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        param_space: Optional[Dict[str, Any]] = None,
        param_order: Optional[List[str]] = None,
        maximize: bool = False,
        seed: int = 42,
        # Strategy components (optional). If not provided, defaults match ALBA_V1.
        gamma_scheduler: Optional[GammaScheduler] = None,
        leaf_selector: Optional[LeafSelector] = None,
        split_decider: Optional[SplitDecider] = None,
        split_policy: Optional[SplitPolicy] = None,
        local_search_sampler: Optional[LocalSearchSampler] = None,
        gamma_quantile: float = 0.20,
        gamma_quantile_start: float = 0.15,
        local_search_ratio: float = 0.30,
        split_trials_min: int = 15,
        split_depth_max: int = 16,
        split_trials_factor: float = 3.0,
        split_trials_offset: int = 6,
        novelty_weight: float = 0.4,
        total_budget: int = 200,
        global_random_prob: float = 0.05,
        stagnation_threshold: int = 50,
        categorical_dims: Optional[List[Tuple[int, int]]] = None,
        # Grid + heatmap sampling
        grid_bins: int = 8,
        grid_batch_size: int = 512,
        grid_batches: int = 4,
        grid_sampling: str = "grid_random",
        grid_jitter: bool = True,
        grid_penalty_lambda: float = 0.06,
        heatmap_ucb_beta: float = 1.0,
        heatmap_ucb_explore_prob: float = 0.25,
        heatmap_ucb_temperature: float = 1.0,
        categorical_sampling: bool = True,
        categorical_stage: str = "auto",
        categorical_pre_n: int = 8,
        heatmap_blend_tau: float = 1e9,
        heatmap_soft_assignment: bool = True,
        heatmap_multi_resolution: bool = True,
        trace_top_k: int = 0,
        trace_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        trace_hook_tell: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        # Initialize param space handler if using param_space mode
        self._param_space_handler: Optional[ParamSpaceHandler] = None
        self._param_space_mode = param_space is not None

        if self._param_space_mode:
            self._param_space_handler = ParamSpaceHandler(param_space, param_order)
            bounds = self._param_space_handler.get_bounds()
            categorical_dims = self._param_space_handler.categorical_dims

        if bounds is None:
            raise TypeError("ALBA requires either bounds=... or param_space=...")

        # Core attributes
        self.bounds = bounds
        self.dim = len(bounds)
        self.maximize = maximize
        self.rng = np.random.default_rng(seed)

        # Gamma (good threshold) parameters
        self.gamma_quantile = gamma_quantile
        self.gamma_quantile_start = gamma_quantile_start
        self.gamma = 0.0

        # Budget allocation
        self.total_budget = total_budget
        self.local_search_ratio = local_search_ratio
        self.exploration_budget = int(total_budget * (1 - local_search_ratio))
        self.local_search_budget = total_budget - self.exploration_budget

        # Sampling parameters
        self._novelty_weight = novelty_weight
        self._global_random_prob = global_random_prob

        # Grid sampling configuration
        self._grid_bins = int(grid_bins)
        self._grid_batch_size = int(grid_batch_size)
        self._grid_batches = int(grid_batches)
        self._grid_sampling = str(grid_sampling)
        self._grid_jitter = bool(grid_jitter)
        self._grid_penalty_lambda = float(grid_penalty_lambda)
        self._heatmap_ucb_beta = float(heatmap_ucb_beta)
        self._heatmap_ucb_explore_prob = float(heatmap_ucb_explore_prob)
        self._heatmap_ucb_temperature = float(heatmap_ucb_temperature)
        self._categorical_sampling = bool(categorical_sampling)
        self._categorical_stage = str(categorical_stage).strip().lower()
        self._categorical_pre_n = int(categorical_pre_n)
        self._heatmap_blend_tau = float(heatmap_blend_tau)
        self._heatmap_soft_assignment = bool(heatmap_soft_assignment)
        self._heatmap_multi_resolution = bool(heatmap_multi_resolution)
        self._trace_top_k = max(0, int(trace_top_k))
        self._trace_hook = trace_hook
        self._trace_hook_tell = trace_hook_tell
        self._pending_trace: Optional[Dict[str, Any]] = None
        self._last_sample_cat_already_applied = False

        # Residual filtering: reduce the impact of very noisy/outlier residuals.
        # - Clip stored residuals at tell-time relative to the base surrogate sigma.
        # This is intentionally conservative to avoid degrading tasks with noisy residuals.
        self._heatmap_resid_clip_sigma = 2.5

        # Global sigma calibration (online, branchless):
        # We keep two EMAs:
        # - `ema_z2` (acquisition): updated on z-scores computed with the *acquisition*
        #   sigma multiplier. This intentionally under-corrects and is more robust for
        #   optimization under selection bias (winner's curse).
        # - `ema_z2_raw` (full): updated on z-scores computed against the *raw* surrogate
        #   sigma, and used for tell-time residual clipping and diagnostics.
        self._sigma_calib_alpha = 0.05
        # Cap the z^2 contribution to keep the EMA robust while still allowing
        # calibration to react when the surrogate is severely underconfident.
        self._sigma_calib_z2_cap = 100.0  # cap at |z|<=10
        self._sigma_calib_ema_z2 = 1.0
        self._sigma_calib_ema_z2_raw = 1.0
        self._sigma_calib_scale_min = 0.2
        self._sigma_calib_scale_max = 10.0

        # Visit-based novelty (count-based exploration):
        # Add an extra uncertainty term that is *in sigma units* (scale-safe), decaying with visits.
        # Mixed categorical spaces already get diversity via categorical handling, so we keep the
        # novelty scaling weaker there (internal, deterministic; no extra user knobs).
        self._visit_novelty_tau = 10.0
        self._visit_novelty_kappa_scale = 10.0

        # Taylor-aware local policy (deterministic defaults).
        # Used to reduce winner's-curse behavior and force smaller local regions
        # when the LGS linear fit is clearly poor (e.g., curved valleys).
        # rel_mse ~= 1 means the linear model is no better than a constant mean model.
        self._taylor_rel_mse_threshold = 1.0
        # Candidate pool size for selection. Kept independent of tracing so that enabling
        # trace hooks does not change optimization behavior.
        self._taylor_select_top_k = 64
        self._tr_min_scale = 0.02
        self._tr_expand = 1.3
        self._tr_shrink = 0.7

        if self._grid_bins < 1:
            raise ValueError("grid_bins must be >= 1")
        if self._grid_batch_size < 1:
            raise ValueError("grid_batch_size must be >= 1")
        if self._grid_batches < 1:
            raise ValueError("grid_batches must be >= 1")
        if self._grid_penalty_lambda < 0.0:
            raise ValueError("grid_penalty_lambda must be >= 0")
        if self._heatmap_ucb_beta < 0.0:
            raise ValueError("heatmap_ucb_beta must be >= 0")
        if not (0.0 <= self._heatmap_ucb_explore_prob <= 1.0):
            raise ValueError("heatmap_ucb_explore_prob must be in [0,1]")
        if self._heatmap_ucb_temperature < 0.0:
            raise ValueError("heatmap_ucb_temperature must be >= 0")
        if self._categorical_stage not in {"post", "pre", "auto"}:
            raise ValueError("categorical_stage must be one of {'post','pre','auto'}")
        if self._categorical_pre_n < 1:
            raise ValueError("categorical_pre_n must be >= 1")
        if self._heatmap_blend_tau <= 0.0:
            raise ValueError("heatmap_blend_tau must be > 0")
        if self._grid_sampling not in {"grid_random", "grid_halton", "halton", "heatmap_ucb"}:
            raise ValueError(
                "grid_sampling must be one of {'grid_random','grid_halton','halton','heatmap_ucb'} "
                f"(got '{self._grid_sampling}')"
            )
        if self._trace_hook is not None and not callable(self._trace_hook):
            raise TypeError("trace_hook must be callable or None")
        if self._trace_hook_tell is not None and not callable(self._trace_hook_tell):
            raise TypeError("trace_hook_tell must be callable or None")

        # Split parameters
        self._split_trials_min = split_trials_min
        self._split_depth_max = split_depth_max
        self._split_trials_factor = split_trials_factor
        self._split_trials_offset = split_trials_offset

        # Stagnation tracking
        self._stagnation_threshold = stagnation_threshold
        self.stagnation = 0
        self.last_improvement_iter = 0

        # Continuous vs categorical dimensions
        cat_dims = sorted({int(i) for i, _ in (categorical_dims or [])})
        for i in cat_dims:
            if i < 0 or i >= self.dim:
                raise ValueError(f"categorical_dims contains out-of-range dim_idx={i} for dim={self.dim}")
        self._cat_dim_indices: List[int] = cat_dims
        self._cont_dim_indices: List[int] = [i for i in range(self.dim) if i not in set(cat_dims)]

        # Initialize cube tree
        self.root = Cube(bounds=list(bounds))
        # Attach categorical dimension metadata to cubes so the LGS can condition on cat keys.
        # This does not change the external API: categorical_dims is already provided to ALBA.
        try:
            self.root.categorical_dims = list(categorical_dims or [])  # type: ignore[attr-defined]
        except Exception:
            pass
        self.leaves: List[Cube] = [self.root]
        self._last_cube: Optional[Cube] = None
        if self._cont_dim_indices:
            cont_bounds = [bounds[i] for i in self._cont_dim_indices]
            index_dims = self._choose_heatmap_index_dims(self.root)
            self.root.grid_state = LeafGridState(
                bounds=list(cont_bounds),
                B=self._grid_bins,
                B_index=self._choose_heatmap_index_bins(index_dims),
                index_dims=index_dims,
                soft_assignment=self._heatmap_soft_assignment,
                multi_resolution=self._heatmap_multi_resolution,
            )
        else:
            self.root.grid_state = None

        # Observation history
        self.X_all: List[np.ndarray] = []
        self.y_all: List[float] = []  # Always stored as "higher is better"
        self.best_y_internal = -np.inf
        self.best_x: Optional[np.ndarray] = None

        # Global geometry
        self._global_widths = np.array([hi - lo for lo, hi in bounds])

        # Initialize categorical sampler
        self._cat_sampler = CategoricalSampler(
            categorical_dims=categorical_dims or [],
            elite_size=10,
            curiosity_bonus=0.3,
            crossover_rate=0.15,
        )

        # Strategy components (defaults reproduce the current baseline behavior)
        self._gamma_scheduler: GammaScheduler = (
            gamma_scheduler
            if gamma_scheduler is not None
            else QuantileAnnealedGammaScheduler(
                gamma_quantile=self.gamma_quantile,
                gamma_quantile_start=self.gamma_quantile_start,
            )
        )
        self._leaf_selector: LeafSelector = (
            leaf_selector if leaf_selector is not None else UCBSoftmaxLeafSelector()
        )
        self._split_decider: SplitDecider = (
            split_decider
            if split_decider is not None
            else ThresholdSplitDecider(
                split_trials_min=self._split_trials_min,
                split_depth_max=self._split_depth_max,
                split_trials_factor=self._split_trials_factor,
                split_trials_offset=self._split_trials_offset,
            )
        )
        self._split_policy: SplitPolicy = (
            split_policy if split_policy is not None else CubeIntrinsicSplitPolicy()
        )
        self._local_search_sampler: LocalSearchSampler = (
            local_search_sampler if local_search_sampler is not None else GaussianLocalSearchSampler()
        )

        # Iteration counter
        self.iteration = 0

    # -------------------------------------------------------------------------
    # Grid helpers
    # -------------------------------------------------------------------------

    def _lgs_misfit(self, cube: Cube) -> Tuple[bool, Optional[float]]:
        """Heuristic: decide whether the local linear surrogate is a poor fit."""
        model = cube.lgs_model
        if not isinstance(model, dict):
            return False, None

        n_pts = int(len(model.get("all_pts", [])))

        rel_mse_f: Optional[float] = None
        misfit = False

        rel_mse = model.get("rel_mse")
        if rel_mse is not None:
            try:
                rel_mse_f = float(rel_mse)
                if np.isfinite(rel_mse_f) and n_pts >= 8 and rel_mse_f > float(self._taylor_rel_mse_threshold):
                    misfit = True
            except Exception:
                rel_mse_f = None

        try:
            sigma_scale = float(model.get("sigma_scale", 1.0))
            if np.isfinite(sigma_scale) and sigma_scale >= 4.5:
                misfit = True
        except Exception:
            pass

        return bool(misfit), rel_mse_f

    def _choose_heatmap_index_dims(self, cube: Cube) -> np.ndarray:
        """Choose a fixed subset of continuous dimensions to index the heatmap on.

        This is intentionally deterministic (no random projections) and aims to keep
        the number of indexed cells roughly constant as dim grows, so that per-cell
        statistics (visits/residuals) get enough collisions to be informative.
        """
        cont_dim = int(len(self._cont_dim_indices))
        if cont_dim <= 0:
            return np.zeros(0, dtype=np.int64)

        # Target number of indexed cells. With default B=8 this yields k≈2 (64 cells).
        target_cells = 64.0
        B = float(max(int(self._grid_bins), 1))
        if B <= 1.0:
            k = 1
        else:
            k = int(np.round(np.log(target_cells) / np.log(B)))
            k = max(1, k)
        k = int(min(cont_dim, k))

        # Build a ranking of continuous dims (indices in [0..cont_dim-1]).
        full_to_cont = {int(fi): int(ci) for ci, fi in enumerate(self._cont_dim_indices)}
        order: List[int] = []

        model = cube.lgs_model
        if isinstance(model, dict):
            grad = model.get("grad")
            if grad is not None:
                g = np.asarray(grad, dtype=float).ravel()
                if g.size == self.dim:
                    scores = np.abs(g[np.array(self._cont_dim_indices, dtype=np.int64)])
                    if np.any(np.isfinite(scores)) and float(np.max(scores)) > 0.0:
                        order = [int(i) for i in np.argsort(-scores)]
            if not order:
                active = model.get("active_dims")
                if active is not None:
                    active_full = np.asarray(active, dtype=np.int64).ravel().tolist()
                    for fi in active_full:
                        ci = full_to_cont.get(int(fi))
                        if ci is None:
                            continue
                        order.append(ci)

        if not order:
            pairs = list(cube.tested_pairs)
            if len(pairs) >= 5:
                Xc = np.stack(
                    [np.asarray(p, dtype=float)[self._cont_dim_indices] for p, _ in pairs],
                    axis=0,
                )
                y = np.asarray([s for _, s in pairs], dtype=float)
                y0 = y - float(np.mean(y))
                y_std = float(np.std(y0))
                if y_std > 1e-12:
                    X0 = Xc - np.mean(Xc, axis=0, keepdims=True)
                    cov = np.mean(X0 * y0.reshape(-1, 1), axis=0)
                    x_std = np.std(Xc, axis=0)
                    corr = np.abs(cov) / (x_std * y_std + 1e-12)
                    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                    order = [int(i) for i in np.argsort(-corr)]

        if not order:
            widths = np.array([cube.bounds[i][1] - cube.bounds[i][0] for i in self._cont_dim_indices], dtype=float)
            if widths.size:
                order = [int(i) for i in np.argsort(-widths)]

        if not order:
            order = list(range(cont_dim))

        # Deduplicate while preserving order, then fill any missing dims.
        seen = set()
        uniq: List[int] = []
        for ci in order:
            if ci < 0 or ci >= cont_dim:
                continue
            if ci in seen:
                continue
            seen.add(ci)
            uniq.append(ci)
        for ci in range(cont_dim):
            if ci not in seen:
                uniq.append(ci)

        return np.array(uniq[:k], dtype=np.int64)

    def _choose_heatmap_index_bins(self, index_dims: np.ndarray) -> int:
        """Choose the number of bins used for heatmap indexing (B_index).

        The sampling grid resolution stays at B (self._grid_bins), but the heatmap
        groups points into fewer, coarser cells via B_index to increase collisions.
        """
        B_sample = int(max(int(self._grid_bins), 1))
        if B_sample <= 1:
            return 1

        k = int(np.asarray(index_dims, dtype=np.int64).size) if index_dims is not None else 0
        if k <= 0:
            return B_sample

        # Target a small number of indexed cells so residual stats stabilize quickly.
        target_cells = 16.0
        b = int(np.round(target_cells ** (1.0 / float(k))))
        b = max(2, b)
        # Keep indexing coarse (2..4) and never finer than sampling resolution.
        b = min(b, 4, B_sample)
        return int(max(1, b))

    def _ensure_grid_state(self, cube: Cube) -> LeafGridState:
        if not self._cont_dim_indices:
            raise RuntimeError("Grid sampling requires at least one continuous dimension")
        cont_bounds = [cube.bounds[i] for i in self._cont_dim_indices]
        if cube.grid_state is None or cube.grid_state.B != self._grid_bins or cube.grid_state.bounds != cont_bounds:
            index_dims = self._choose_heatmap_index_dims(cube)
            cube.grid_state = LeafGridState(
                bounds=list(cont_bounds),
                B=self._grid_bins,
                B_index=self._choose_heatmap_index_bins(index_dims),
                index_dims=index_dims,
                soft_assignment=self._heatmap_soft_assignment,
                multi_resolution=self._heatmap_multi_resolution,
            )
            if cube.n_trials > 0:
                self._rebuild_grid_state_from_pairs(cube, cube.grid_state)
        return cube.grid_state

    def _rebuild_grid_state_from_pairs(self, cube: Cube, grid_state: LeafGridState) -> None:
        grid_state.stats = {}
        grid_state.total_visits = 0.0
        for x_full, y_internal in cube.tested_pairs:
            x_cont = np.asarray(x_full, dtype=float)[self._cont_dim_indices]
            grid_state.update(x_cont, float(y_internal), self.gamma)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def best_y(self) -> float:
        """Best objective value found (in original scale)."""
        return self._to_raw(self.best_y_internal)

    @property
    def n_observations(self) -> int:
        """Number of observations so far."""
        return len(self.X_all)

    @property
    def is_stagnating(self) -> bool:
        """Check if optimization is currently stagnating."""
        return self.stagnation > self._stagnation_threshold

    # -------------------------------------------------------------------------
    # Internal score conversion
    # -------------------------------------------------------------------------

    def _to_internal(self, y_raw: float) -> float:
        """Convert raw objective value to internal score (higher is better)."""
        return y_raw if self.maximize else -y_raw

    def _to_raw(self, y_internal: float) -> float:
        """Convert internal score back to raw objective value."""
        return y_internal if self.maximize else -y_internal

    def _sigma_calibration_scale(self) -> float:
        """Return the current sigma multiplier used for acquisition (>=0)."""
        try:
            ema_z2 = float(getattr(self, "_sigma_calib_ema_z2", 1.0))
            s = float(np.sqrt(max(ema_z2, 1e-12)))
            s_min = float(getattr(self, "_sigma_calib_scale_min", 0.2))
            s_max = float(getattr(self, "_sigma_calib_scale_max", 5.0))
            return float(np.clip(s, s_min, s_max))
        except Exception:
            return 1.0

    def _sigma_calibration_scale_full(self) -> float:
        """Return the current sigma multiplier used for tell-time clipping/diagnostics (>=0)."""
        try:
            ema_z2 = float(
                getattr(self, "_sigma_calib_ema_z2_raw", getattr(self, "_sigma_calib_ema_z2", 1.0))
            )
            s = float(np.sqrt(max(ema_z2, 1e-12)))
            s_min = float(getattr(self, "_sigma_calib_scale_min", 0.2))
            s_max = float(getattr(self, "_sigma_calib_scale_max", 5.0))
            return float(np.clip(s, s_min, s_max))
        except Exception:
            return 1.0

    def _heatmap_effective_tau(self) -> Tuple[float, Dict[str, Any]]:
        """Return (tau_eff, info) for heatmap sigma inflation.

        We keep a single global heatmap knob (`heatmap_blend_tau`) but gate its use
        deterministically to improve robustness across tasks:
        - Only enable heatmap during the exploration phase
        - Only when the optimizer is stagnating (no improvements for a while)
        """
        tau_on = float(self._heatmap_blend_tau)
        enabled = bool(self.iteration < self.exploration_budget and self.stagnation > self._stagnation_threshold)
        tau_eff = tau_on if enabled else 1e9
        info: Dict[str, Any] = {
            "enabled": bool(enabled),
            "tau_on": float(tau_on),
            "tau_eff": float(tau_eff),
            "stagnation": int(self.stagnation),
            "stagnation_threshold": int(self._stagnation_threshold),
            "phase": ("exploration" if self.iteration < self.exploration_budget else "local_search"),
        }
        return float(tau_eff), info

    # -------------------------------------------------------------------------
    # Ask interface
    # -------------------------------------------------------------------------

    def ask(self) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Suggest the next point to evaluate.

        Returns
        -------
        Union[np.ndarray, Dict[str, Any]]
            - If constructed with bounds=..., returns a normalized np.ndarray.
            - If constructed with param_space=..., returns a dict config.
        """
        x = self.ask_array()
        if self._param_space_mode:
            return self.decode(x)
        return x

    def ask_array(self) -> np.ndarray:
        """
        Suggest next point in the internal normalized space ([0,1]^d).

        Returns
        -------
        np.ndarray
            Next point to evaluate in normalized coordinates.
        """
        self.iteration = len(self.X_all)

        # Global random for diversity
        if self.rng.random() < self._global_random_prob:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
            self._last_cube = self._find_containing_leaf(x)
            return x

        if self.iteration < self.exploration_budget:
            # Exploration phase
            self._update_gamma()
            self._recount_good()

            if self.iteration % 5 == 0:
                self._update_all_models()

            self._last_cube = self._select_leaf()
            return self._sample_in_cube(self._last_cube)

        # Local search phase with tree integration
        ls_iter = self.iteration - self.exploration_budget
        progress = ls_iter / max(1, self.local_search_budget - 1)
        local_search_prob = 0.5 + 0.4 * progress

        if self.rng.random() < local_search_prob:
            x = self._local_search_sample(progress)
            self._last_cube = self._find_containing_leaf(x)
        else:
            self._update_gamma()
            self._recount_good()
            self._last_cube = self._select_leaf()
            x = self._sample_in_cube(self._last_cube)

        return x

    # -------------------------------------------------------------------------
    # Tell interface
    # -------------------------------------------------------------------------

    def tell(self, x: Union[np.ndarray, Dict[str, Any]], y_raw: float) -> None:
        """
        Report the objective value for a point.

        Parameters
        ----------
        x : Union[np.ndarray, Dict[str, Any]]
            The evaluated point. Can be dict (if using param_space) or array.
        y_raw : float
            The objective value (in original scale).
        """
        if self._param_space_mode and isinstance(x, dict):
            x = self.encode(x)
        x = np.asarray(x, dtype=float)
        y = self._to_internal(y_raw)

        # Update best
        if y > self.best_y_internal:
            self.best_y_internal = y
            self.best_x = x.copy()
            self.stagnation = 0
            self.last_improvement_iter = self.iteration
        else:
            self.stagnation += 1

        # Record observation
        self.X_all.append(x.copy())
        self.y_all.append(y)

        # Update categorical tracking
        self._cat_sampler.record_observation(x, y)

        # Update cube
        if self._last_cube is not None:
            cube = self._last_cube
            prev_best_score = float(cube.best_score)

            # Compute base LGS prediction BEFORE adding the point (avoid self-neighbor effects).
            mu_pred_base: Optional[float] = None
            sigma_pred_base: Optional[float] = None
            sigma_pred_base_raw: Optional[float] = None
            sigma_mul_used_acq: Optional[float] = None
            sigma_mul_used_full: Optional[float] = None
            if cube.grid_state is not None and self._cont_dim_indices:
                x_row = x.reshape(1, -1)
                try:
                    if cube.n_trials >= 5 and cube.lgs_model is not None and cube.lgs_model.get("inv_cov") is not None:
                        mu0, s0 = cube.predict_bayesian(x_row)
                        mu_pred_base = float(mu0[0])
                        sigma_pred_base_raw = float(s0[0])
                        sigma_mul_used_acq = float(self._sigma_calibration_scale())
                        sigma_mul_used_full = float(self._sigma_calibration_scale_full())
                        sigma_pred_base = float(sigma_pred_base_raw) * float(sigma_mul_used_full)
                except Exception:
                    mu_pred_base = None
                    sigma_pred_base = None

            resid_for_grid: Optional[float] = None
            resid_raw: Optional[float] = None
            resid_clip: Optional[float] = None
            resid_clipped: Optional[bool] = None
            z_val: Optional[float] = None
            z_raw_val: Optional[float] = None
            z_acq_val: Optional[float] = None
            if mu_pred_base is not None and np.isfinite(mu_pred_base):
                try:
                    resid_raw = float(y - float(mu_pred_base))
                    resid = float(resid_raw)
                    if sigma_pred_base_raw is not None and np.isfinite(sigma_pred_base_raw):
                        z_raw_val = float(float(resid_raw) / max(float(sigma_pred_base_raw), 1e-12))
                        if sigma_mul_used_acq is not None and np.isfinite(sigma_mul_used_acq):
                            z_acq_val = float(
                                float(resid_raw)
                                / max(float(sigma_pred_base_raw) * float(sigma_mul_used_acq), 1e-12)
                            )
                    if sigma_pred_base is not None and np.isfinite(sigma_pred_base):
                        clip = float(self._heatmap_resid_clip_sigma) * float(max(float(sigma_pred_base), 1e-12))
                        resid_clip = float(clip)
                        resid_clipped = bool(abs(float(resid)) > float(clip))
                        resid = float(np.clip(resid, -clip, clip))
                        z_val = float(float(resid_raw) / max(float(sigma_pred_base), 1e-12))
                    resid_for_grid = resid if np.isfinite(resid) else None
                except Exception:
                    resid_for_grid = None

            # Online global sigma calibration update based on realized z-scores.
            # We keep two EMAs:
            # - Full: update on z_raw (against raw sigma) for diagnostics/clipping.
            # - Acquisition: update on z_acq (against raw sigma * acq scale) to be robust
            #   under selection bias (this converges to an intentionally smaller correction).
            try:
                alpha = float(getattr(self, "_sigma_calib_alpha", 0.05))
                cap = float(getattr(self, "_sigma_calib_z2_cap", 25.0))
            except Exception:
                alpha, cap = 0.05, 25.0

            if z_raw_val is not None and np.isfinite(z_raw_val):
                try:
                    z2 = float(z_raw_val) * float(z_raw_val)
                    if np.isfinite(cap) and cap > 0.0:
                        z2 = float(min(z2, cap))
                    prev = float(getattr(self, "_sigma_calib_ema_z2_raw", 1.0))
                    if not np.isfinite(prev) or prev <= 0.0:
                        prev = 1.0
                    if np.isfinite(alpha) and 0.0 < alpha <= 1.0 and np.isfinite(z2) and z2 > 0.0:
                        self._sigma_calib_ema_z2_raw = (1.0 - alpha) * prev + alpha * z2
                except Exception:
                    pass

            if z_acq_val is not None and np.isfinite(z_acq_val):
                try:
                    z2 = float(z_acq_val) * float(z_acq_val)
                    if np.isfinite(cap) and cap > 0.0:
                        z2 = float(min(z2, cap))
                    prev = float(getattr(self, "_sigma_calib_ema_z2", 1.0))
                    if not np.isfinite(prev) or prev <= 0.0:
                        prev = 1.0
                    if np.isfinite(alpha) and 0.0 < alpha <= 1.0 and np.isfinite(z2) and z2 > 0.0:
                        self._sigma_calib_ema_z2 = (1.0 - alpha) * prev + alpha * z2
                except Exception:
                    pass

            hook_tell = getattr(self, "_trace_hook_tell", None)
            if hook_tell is not None and callable(hook_tell):
                try:
                    tt: Dict[str, Any] = {
                        "event": "tell",
                        "iteration": int(self.iteration),
                        "gamma": float(self.gamma),
                        "y_raw": float(y_raw),
                        "y_internal": float(y),
                        "x": x.copy(),
                        "cube_depth": int(cube.depth),
                        "cube_n_trials_pre": int(cube.n_trials),
                        "mu_pred_base": (float(mu_pred_base) if mu_pred_base is not None else None),
                        "sigma_pred_base_raw": (float(sigma_pred_base_raw) if sigma_pred_base_raw is not None else None),
                        "sigma_pred_base": (float(sigma_pred_base) if sigma_pred_base is not None else None),
                        "resid_raw": (float(resid_raw) if resid_raw is not None else None),
                        "resid_clip": (float(resid_clip) if resid_clip is not None else None),
                        "resid_clipped": (bool(resid_clipped) if resid_clipped is not None else None),
                        "resid_for_grid": (float(resid_for_grid) if resid_for_grid is not None else None),
                        "sigma_calib_scale": (
                            float(sigma_mul_used_full)
                            if sigma_mul_used_full is not None
                            else float(self._sigma_calibration_scale_full())
                        ),
                        "sigma_calib_scale_acq": (
                            float(sigma_mul_used_acq)
                            if sigma_mul_used_acq is not None
                            else float(self._sigma_calibration_scale())
                        ),
                    }
                    if (
                        resid_raw is not None
                        and sigma_pred_base is not None
                        and np.isfinite(resid_raw)
                        and np.isfinite(sigma_pred_base)
                    ):
                        tt["z"] = float(z_val) if z_val is not None else float(float(resid_raw) / max(float(sigma_pred_base), 1e-12))
                    if (
                        resid_raw is not None
                        and sigma_pred_base_raw is not None
                        and np.isfinite(resid_raw)
                        and np.isfinite(sigma_pred_base_raw)
                    ):
                        tt["z_raw"] = float(z_raw_val) if z_raw_val is not None else float(float(resid_raw) / max(float(sigma_pred_base_raw), 1e-12))
                    if cube.grid_state is not None and self._cont_dim_indices:
                        try:
                            x_cont = np.asarray(x, dtype=float)[self._cont_dim_indices]
                            tt["cell_key_fine"] = list(cube.grid_state.cell_index(x_cont))
                        except Exception:
                            pass
                    hook_tell(tt)
                except Exception:
                    pass

            cube.add_observation(x, y, self.gamma)
            improved_in_cube = bool(y > prev_best_score)

            # Update per-leaf heatmap
            if cube.grid_state is not None and self._cont_dim_indices:
                cube.grid_state.update(
                    x[self._cont_dim_indices],
                    y,
                    self.gamma,
                    resid=resid_for_grid,
                )

            # Update categorical stats in cube
            for dim_idx, n_choices in self._cat_sampler.categorical_dims:
                val_idx = self._cat_sampler.discretize(x[dim_idx], n_choices)
                if dim_idx not in cube.cat_stats:
                    cube.cat_stats[dim_idx] = {}
                n_g, n_t = cube.cat_stats[dim_idx].get(val_idx, (0, 0))
                is_good = y >= self.gamma
                cube.cat_stats[dim_idx][val_idx] = (
                    n_g + (1 if is_good else 0),
                    n_t + 1,
                )

            cube.fit_lgs_model(self.gamma, self.dim, self.rng)

            # Taylor-aware trust region update: only activate when local fit is clearly poor.
            misfit, _ = self._lgs_misfit(cube)
            if misfit:
                if improved_in_cube:
                    cube.tr_scale = float(min(1.0, float(cube.tr_scale) * float(self._tr_expand)))
                else:
                    cube.tr_scale = float(
                        max(float(self._tr_min_scale), float(cube.tr_scale) * float(self._tr_shrink))
                    )
            else:
                cube.tr_scale = 1.0

            # Check for split
            if self._should_split(cube):
                children = self._split_policy.split(cube, self.gamma, self.dim, self.rng)
                for child in children:
                    self._cat_sampler.recompute_cube_cat_stats(child, self.gamma)
                    # Rebuild children's heatmap from tested pairs (robust to gamma + semantics).
                    if self._cont_dim_indices:
                        cont_bounds = [child.bounds[i] for i in self._cont_dim_indices]
                        index_dims = self._choose_heatmap_index_dims(child)
                        child.grid_state = LeafGridState(
                            bounds=list(cont_bounds),
                            B=self._grid_bins,
                            B_index=self._choose_heatmap_index_bins(index_dims),
                            index_dims=index_dims,
                            soft_assignment=self._heatmap_soft_assignment,
                            multi_resolution=self._heatmap_multi_resolution,
                        )
                        self._rebuild_grid_state_from_pairs(child, child.grid_state)
                    else:
                        child.grid_state = None
                if cube in self.leaves:
                    self.leaves.remove(cube)
                    self.leaves.extend(children)

            self._last_cube = None

    # -------------------------------------------------------------------------
    # Encode / Decode (param_space mode)
    # -------------------------------------------------------------------------

    def encode(self, config: Dict[str, Any]) -> np.ndarray:
        """
        Convert typed config dict to internal normalized vector.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        np.ndarray
            Normalized vector in [0, 1]^dim.

        Raises
        ------
        RuntimeError
            If not in param_space mode.
        """
        if self._param_space_handler is None:
            raise RuntimeError("encode() requires param_space mode")
        return self._param_space_handler.encode(config)

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Convert internal normalized vector to typed config dict.

        Parameters
        ----------
        x : np.ndarray
            Normalized vector in [0, 1]^dim.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.

        Raises
        ------
        RuntimeError
            If not in param_space mode.
        """
        if self._param_space_handler is None:
            raise RuntimeError("decode() requires param_space mode")
        return self._param_space_handler.decode(x)

    # -------------------------------------------------------------------------
    # Gamma (threshold) management
    # -------------------------------------------------------------------------

    def _update_gamma(self) -> None:
        """Update the gamma threshold based on current observations."""
        self.gamma = float(
            self._gamma_scheduler.compute(self.y_all, self.iteration, self.exploration_budget)
        )

    def _recount_good(self) -> None:
        """Recount good points in all leaves with current gamma."""
        for leaf in self.leaves:
            leaf.n_good = sum(1 for _, s in leaf.tested_pairs if s >= self.gamma)
            self._cat_sampler.recompute_cube_cat_stats(leaf, self.gamma)

    # -------------------------------------------------------------------------
    # Leaf selection
    # -------------------------------------------------------------------------

    def _select_leaf(self) -> Cube:
        """
        Select a leaf cube for the next sample using UCB-like scoring.

        Returns
        -------
        Cube
            Selected leaf cube.
        """
        if not self.leaves:
            return self.root
        return self._leaf_selector.select(self.leaves, self.dim, self.is_stagnating, self.rng)

    # -------------------------------------------------------------------------
    # Sampling within cubes
    # -------------------------------------------------------------------------

    def _sample_in_cube(self, cube: Cube) -> np.ndarray:
        """
        Sample a point within the given cube.

        Parameters
        ----------
        cube : Cube
            The cube to sample from.

        Returns
        -------
        np.ndarray
            Sampled point.
        """
        if self.iteration < 15:
            x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
        else:
            x = self._sample_with_lgs(cube)

        # Apply categorical sampling
        cat_already_applied = bool(self._last_sample_cat_already_applied)
        self._last_sample_cat_already_applied = False

        if self._categorical_sampling and (self._categorical_stage == "post" or not cat_already_applied):
            x = self._cat_sampler.sample(x, cube, self.rng, self.is_stagnating)

        # Re-clip to cube bounds
        x = self._clip_to_cube(x, cube)

        # Optional: deliver post-categorical point to a pending trace collected during LGS selection.
        if self._pending_trace is not None:
            trace = self._pending_trace
            self._pending_trace = None
            trace["x_final"] = x.copy()
            if self._cat_sampler.has_categoricals:
                x_scored = trace.get("x_chosen_raw")
                if x_scored is not None:
                    key_scored = self._cat_sampler.get_cat_key(np.asarray(x_scored, dtype=float))
                    trace.setdefault("categorical", {})["key_scored"] = list(key_scored)
                key_final = self._cat_sampler.get_cat_key(x)
                trace.setdefault("categorical", {})["key_final"] = list(key_final)
                if x_scored is not None:
                    trace.setdefault("categorical", {})["key_mismatch"] = bool(
                        tuple(trace["categorical"].get("key_scored", [])) != tuple(key_final)
                    )

            # Predict again after categorical application (helps detect scoring mismatch).
            try:
                beta = float(trace.get("beta", 0.0))
            except Exception:
                beta = 0.0

            mu_after = 0.0
            sigma_base_after = 1.0
            sigma_base_after_raw = 1.0
            sigma_mul_after = float(self._sigma_calibration_scale())
            try:
                mu0, s0 = cube.predict_bayesian(x.reshape(1, -1))
                mu_after = float(mu0[0])
                sigma_base_after_raw = float(s0[0])
                sigma_base_after = float(sigma_base_after_raw) * float(max(float(sigma_mul_after), 1e-12))
            except Exception:
                pass

            sigma_adj_after = float(sigma_base_after)
            cell_after = None
            visits_after = 0.0
            if cube.grid_state is not None and self._cont_dim_indices:
                try:
                    extra_var, cell_after = self._heatmap_adjustment_for_cont(
                        cube.grid_state,
                        np.asarray(x, dtype=float)[self._cont_dim_indices],
                    )
                    sigma_adj_after = float(
                        np.sqrt(max(float(sigma_base_after) * float(sigma_base_after) + float(extra_var), 1e-12))
                    )
                    if isinstance(cell_after, dict):
                        visits_after = float(cell_after.get("visits", 0.0))
                except Exception:
                    pass

            ucb_after = float(mu_after) + float(beta) * float(sigma_adj_after)

            # Keep after-cat score consistent with the scoring-time visit-based novelty.
            try:
                lam = float(trace.get("grid_penalty_lambda", self._grid_penalty_lambda))
            except Exception:
                lam = float(self._grid_penalty_lambda)
            kappa_scale = float(getattr(self, "_visit_novelty_kappa_scale", 1.0))
            if self._categorical_sampling and self._cat_sampler.has_categoricals:
                kappa_scale = 1.0
            kappa = float(lam) * float(kappa_scale)
            tau_v = float(getattr(self, "_visit_novelty_tau", 10.0))

            sigma_ref = 0.0
            try:
                sigma_ref = float(trace.get("visit_novelty", {}).get("sigma_ref_chosen", 0.0))
            except Exception:
                sigma_ref = 0.0
            if not np.isfinite(sigma_ref) or sigma_ref <= 0.0:
                try:
                    sigma_ref = float(trace.get("batch_stats", {}).get("sigma_adj", {}).get("p50", 0.0))
                except Exception:
                    sigma_ref = 0.0
            if not np.isfinite(sigma_ref) or sigma_ref <= 0.0:
                sigma_ref = float(sigma_adj_after)
            sigma_ref = float(max(float(sigma_ref), 1e-12))

            w_v = float(tau_v) / float(max(float(visits_after), 0.0) + float(tau_v))
            sigma_nov_after = float(kappa) * float(sigma_ref) * float(np.sqrt(max(float(w_v), 0.0)))
            sigma_tot_after = float(
                np.sqrt(max(float(sigma_adj_after) * float(sigma_adj_after) + sigma_nov_after * sigma_nov_after, 1e-12))
            )
            score_after = float(mu_after) + float(beta) * float(sigma_tot_after)
            novelty_bonus_after = float(score_after) - float(ucb_after)
            sigma_ratio_after = float(sigma_adj_after / max(float(sigma_base_after), 1e-12))
            sigma_ratio_tot_after = float(sigma_tot_after / max(float(sigma_base_after), 1e-12))

            trace["chosen_pred_after_cat"] = {
                "mu": float(mu_after),
                "sigma_calib_scale": float(sigma_mul_after),
                "sigma_base_raw": float(sigma_base_after_raw),
                "sigma_base": float(sigma_base_after),
                "sigma": float(sigma_adj_after),
                "sigma_ratio": float(sigma_ratio_after),
                "sigma_nov": float(sigma_nov_after),
                "sigma_tot": float(sigma_tot_after),
                "sigma_ratio_tot": float(sigma_ratio_tot_after),
                "ucb": float(ucb_after),
                "novelty_bonus": float(novelty_bonus_after),
                "score": float(score_after),
                "visits": float(visits_after),
            }
            trace["chosen_cell_after_cat"] = cell_after
            try:
                prev = trace.get("chosen_pred", {})
                if isinstance(prev, dict) and "score" in prev:
                    trace.setdefault("categorical", {})["delta_score_after_cat"] = float(score_after) - float(
                        prev.get("score", 0.0)
                    )
            except Exception:
                pass
            hook = self._trace_hook
            if hook is not None:
                hook(trace)
        return x

    def _sample_with_lgs(self, cube: Cube) -> np.ndarray:
        """
        Sample using the LGS model with batched grid candidate scoring.

        Parameters
        ----------
        cube : Cube
            The cube to sample from.

        Returns
        -------
        np.ndarray
            Sampled point.
        """
        if cube.lgs_model is None or cube.n_trials < 5:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])

        return self._sample_with_grid_lgs(cube)

    def _predict_candidates(
        self,
        cube: Cube,
        X: np.ndarray,
        *,
        Xc: np.ndarray,
        grid_state: LeafGridState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        mu_lgs, sigma_base = cube.predict_bayesian(X)
        # Apply global sigma calibration for acquisition.
        # This uses the "acquisition" EMA which is intentionally conservative under
        # selection bias (winner's curse).
        sigma_base = np.asarray(sigma_base, dtype=float)
        sigma_mul_acq = float(self._sigma_calibration_scale())
        sigma_base = sigma_base * float(max(sigma_mul_acq, 1e-12))
        if Xc.size == 0:
            return (
                mu_lgs,
                sigma_base,
                sigma_base.copy(),
                {
                    "counts_used": np.zeros(0, dtype=float),
                    "vars_r_used": np.zeros(0, dtype=float),
                    "counts_fine": np.zeros(0, dtype=float),
                    "vars_r_fine": np.zeros(0, dtype=float),
                    "used_mode": np.zeros(0, dtype=np.int8),
                    "w_used": np.zeros(0, dtype=float),
                },
            )

        vars_r_used = np.zeros(Xc.shape[0], dtype=float)
        counts_used = np.zeros(Xc.shape[0], dtype=float)
        vars_r_fine = np.zeros(Xc.shape[0], dtype=float)
        counts_fine = np.zeros(Xc.shape[0], dtype=float)
        used_mode = np.zeros(Xc.shape[0], dtype=np.int8)  # 0 none, 1 fine, 2 coarse, 3 fine_weak
        use_multi = bool(getattr(grid_state, "multi_resolution", False))
        B_coarse = int(getattr(grid_state, "B_index_coarse", 0) or getattr(grid_state, "B_index", 0) or grid_state.B)
        for i in range(Xc.shape[0]):
            st = None
            key = grid_state.cell_index(Xc[i])
            st_f = grid_state.stats.get(key)
            if st_f is not None:
                counts_fine[i] = float(getattr(st_f, "n_r", 0.0))
                vars_r_fine[i] = float(st_f.var_r())
            if st_f is not None and float(getattr(st_f, "n_r", 0.0)) >= 3.0:
                st = st_f
                used_mode[i] = 1
            elif use_multi:
                key_c = grid_state.cell_index(Xc[i], B_index=B_coarse)
                st_c = getattr(grid_state, "stats_coarse", {}).get(key_c)
                if st_c is not None:
                    st = st_c
                    used_mode[i] = 2
                elif st_f is not None:
                    st = st_f
                    used_mode[i] = 3
            elif st_f is not None:
                st = st_f
                used_mode[i] = 3
            if st is None:
                continue
            counts_used[i] = float(getattr(st, "n_r", 0.0))
            vars_r_used[i] = float(st.var_r())

        tau, _ = self._heatmap_effective_tau()
        w = counts_used / (counts_used + tau)
        # X is built as: [base0 × Xc, base1 × Xc, ...] so per-Xc vectors must be tiled.
        n_bases = max(1, int(X.shape[0] // max(1, Xc.shape[0])))
        w_rep = np.tile(w, n_bases)[: X.shape[0]]
        v_rep = np.tile(vars_r_used, n_bases)[: X.shape[0]]
        sigma2 = sigma_base * sigma_base
        sigma_adj = np.sqrt(np.maximum(sigma2 + w_rep * v_rep, 1e-12))
        return (
            mu_lgs,
            sigma_base,
            sigma_adj,
            {
                "counts_used": counts_used,
                "vars_r_used": vars_r_used,
                "counts_fine": counts_fine,
                "vars_r_fine": vars_r_fine,
                "used_mode": used_mode,
                "w_used": w,
            },
        )

    def _heatmap_adjustment_for_cont(
        self,
        grid_state: LeafGridState,
        x_cont: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute heatmap sigma inflation term for one continuous point."""
        x_cont = np.asarray(x_cont, dtype=float).reshape(-1)
        key_f = grid_state.cell_index(x_cont)
        st_f = grid_state.stats.get(key_f)

        use_multi = bool(getattr(grid_state, "multi_resolution", False))
        B_coarse = int(
            getattr(grid_state, "B_index_coarse", 0) or getattr(grid_state, "B_index", 0) or grid_state.B
        )
        key_c = grid_state.cell_index(x_cont, B_index=B_coarse) if use_multi else None
        st_c = getattr(grid_state, "stats_coarse", {}).get(key_c) if (use_multi and key_c is not None) else None

        st_used = None
        mode = "none"
        if st_f is not None and float(getattr(st_f, "n_r", 0.0)) >= 3.0:
            st_used = st_f
            mode = "fine"
        elif st_c is not None:
            st_used = st_c
            mode = "coarse"
        elif st_f is not None:
            st_used = st_f
            mode = "fine_weak"

        tau, _ = self._heatmap_effective_tau()
        n_r_used = float(getattr(st_used, "n_r", 0.0)) if st_used is not None else 0.0
        var_r_used = float(st_used.var_r()) if st_used is not None else 0.0
        w = n_r_used / (n_r_used + tau)

        extra_var = float(w * var_r_used)
        info: Dict[str, Any] = {
            "mode": str(mode),
            "tau": float(tau),
            "key_fine": list(key_f),
            "key_coarse": (list(key_c) if key_c is not None else None),
            "n_r": float(n_r_used),
            "var_r": float(var_r_used),
            "w": float(w),
            "visits": float(getattr(st_f, "n", 0.0)) if st_f is not None else 0.0,
        }
        return extra_var, info

    def _propose_cat_keys(self, cube: Cube, x_template: np.ndarray, n_keys: int) -> List[Tuple[int, ...]]:
        if not self._cat_sampler.has_categoricals:
            return [tuple()]

        want = max(1, int(n_keys))
        # Always leave a small fraction of slots for exploratory keys so categorical
        # search does not collapse too early onto a tiny elite set (key-space is huge).
        n_explore = 0
        if want >= 4:
            n_explore = max(1, want // 4)
        target_before_explore = max(0, want - n_explore)

        seen: set[Tuple[int, ...]] = set()
        keys: List[Tuple[int, ...]] = []

        def _add(key: Tuple[int, ...]) -> None:
            if key in seen:
                return
            seen.add(key)
            keys.append(key)

        _add(self._cat_sampler.get_cat_key(x_template))
        if cube.best_x is not None:
            _add(self._cat_sampler.get_cat_key(cube.best_x))
        if self.best_x is not None:
            _add(self._cat_sampler.get_cat_key(self.best_x))

        for key in self._cat_sampler.elite_keys():
            _add(key)
            if len(keys) >= target_before_explore:
                break

        attempts = 0
        max_attempts = want * 25
        while len(keys) < want and attempts < max_attempts:
            x_s = self._cat_sampler.sample(x_template, cube, self.rng, self.is_stagnating)
            _add(self._cat_sampler.get_cat_key(x_s))
            attempts += 1

        return keys[:want]

    def _sample_with_grid_lgs(self, cube: Cube) -> np.ndarray:
        """
        Grid + heatmap sampling with batched LGS scoring.

        Notes
        -----
        - Generates candidates from an implicit per-leaf grid (no B^d materialization).
        - Scores many candidates cheaply with LGS (mu/sigma).
        - Optionally adds visit-based novelty as extra uncertainty (count-based exploration).
        - Returns exactly 1 point to evaluate.
        """
        if not self._cont_dim_indices:
            # Purely categorical space (or caller provided no continuous dims).
            # Fall back to categorical sampling around cube center.
            return cube.center()

        grid_state = self._ensure_grid_state(cube)

        best_x: Optional[np.ndarray] = None
        best_score = -np.inf
        best_mu = 0.0
        best_sigma = 0.0
        best_sigma_base = 0.0
        best_sigma_nov = 0.0
        best_sigma_tot = 0.0
        best_ucb = 0.0
        best_novelty_bonus = 0.0
        best_visits = 0.0
        best_sigma_ref = 0.0

        beta = float(self._novelty_weight) * 2.0  # keep consistent with UCB definition
        n_scored = 0

        want_trace = self._trace_hook is not None
        want_top = want_trace and self._trace_top_k > 0
        top_x_batches: List[np.ndarray] = []
        top_mu_batches: List[np.ndarray] = []
        top_sigma_base_batches: List[np.ndarray] = []
        top_sigma_batches: List[np.ndarray] = []
        top_sigma_nov_batches: List[np.ndarray] = []
        top_sigma_tot_batches: List[np.ndarray] = []
        top_sigma_ref_batches: List[np.ndarray] = []
        top_ucb_batches: List[np.ndarray] = []
        top_novelty_bonus_batches: List[np.ndarray] = []
        top_score_batches: List[np.ndarray] = []
        top_visits_batches: List[np.ndarray] = []

        # Optional diagnostics over the whole scored batch.
        diag_mu_batches: List[np.ndarray] = []
        diag_sigma_base_batches: List[np.ndarray] = []
        diag_sigma_adj_batches: List[np.ndarray] = []
        diag_sigma_nov_batches: List[np.ndarray] = []
        diag_sigma_tot_batches: List[np.ndarray] = []
        diag_sigma_ref_batches: List[np.ndarray] = []
        diag_ucb_batches: List[np.ndarray] = []
        diag_novelty_bonus_batches: List[np.ndarray] = []
        diag_score_batches: List[np.ndarray] = []
        diag_visits_batches: List[np.ndarray] = []
        diag_counts_used_batches: List[np.ndarray] = []
        diag_vars_r_used_batches: List[np.ndarray] = []
        diag_used_mode_batches: List[np.ndarray] = []
        diag_unique_cell_rates: List[float] = []
        diag_unique_cell_keys: set[Tuple[int, ...]] = set()
        diag_visits_cont_batches: List[np.ndarray] = []

        # Keep categorical dimensions fixed during grid scoring.
        x_template = cube.best_x.copy() if cube.best_x is not None else cube.center()

        misfit, rel_mse = self._lgs_misfit(cube)

        # Candidate selection: sample from a small top-k pool instead of taking a
        # hard argmax over very large scored batches (reduces winner's-curse bias).
        selection_mode = "softmax"
        select_top_k = int(max(2, int(self._taylor_select_top_k)))
        center_cont = np.asarray(x_template, dtype=float)[self._cont_dim_indices]
        if misfit:
            tr_scale = float(getattr(cube, "tr_scale", 1.0))
            tr_scale = float(np.clip(tr_scale, float(self._tr_min_scale), 1.0))

            half = 0.5 * tr_scale * grid_state._widths
            lo0 = grid_state._lo
            hi0 = grid_state._hi
            tr_lo = center_cont - half
            tr_hi = center_cont + half

            shift_up = np.maximum(lo0 - tr_lo, 0.0)
            tr_lo = tr_lo + shift_up
            tr_hi = tr_hi + shift_up
            shift_dn = np.maximum(tr_hi - hi0, 0.0)
            tr_lo = tr_lo - shift_dn
            tr_hi = tr_hi - shift_dn

            tr_lo = np.maximum(tr_lo, lo0)
            tr_hi = np.minimum(tr_hi, hi0)
        else:
            tr_scale = 1.0
            tr_lo = grid_state._lo
            tr_hi = grid_state._hi

        cat_keys: List[Tuple[int, ...]] = []
        bases: np.ndarray
        stage_eff = str(self._categorical_stage)

        cat_mode = "none"  # {"none","enum","sample"}
        if self._categorical_sampling and self._cat_sampler.has_categoricals:
            if self._categorical_stage == "pre":
                cat_mode = "enum"
                stage_eff = "pre_enum"
            elif self._categorical_stage == "auto":
                # Keep a single, consistent policy: enumerate a small set of
                # categorical keys and score them jointly with continuous candidates.
                cat_mode = "enum"
                stage_eff = "auto_enum"

        if cat_mode == "enum":
            cat_keys = self._propose_cat_keys(cube, x_template, self._categorical_pre_n)
            bases = np.stack(
                [self._clip_to_cube(self._cat_sampler.apply_cat_key(k, x_template), cube) for k in cat_keys],
                axis=0,
            )
        elif cat_mode == "sample":
            x_base = self._cat_sampler.sample(x_template, cube, self.rng, self.is_stagnating)
            x_base = self._clip_to_cube(x_base, cube)
            cat_keys = [self._cat_sampler.get_cat_key(x_base)]
            bases = x_base.reshape(1, -1)
        else:
            bases = x_template.reshape(1, -1)

        chosen_cat_key: Optional[Tuple[int, ...]] = None
        cat_bandit: Optional[Dict[str, Any]] = None
        chosen_base_idx: Optional[int] = None
        if cat_mode == "enum" and self._categorical_stage == "auto" and cat_keys:
            # Hierarchical categorical handling: choose a categorical key first (bandit),
            # then do continuous selection within that key. Still score cross-key so
            # we can trace/top-K diagnostics.
            try:
                chosen_cat_key, cat_bandit = self._cat_sampler.choose_key_ts_goodrate(
                    cat_keys, gamma=float(self.gamma), rng=self.rng
                )
                chosen_base_idx = int(cat_keys.index(chosen_cat_key))
                stage_eff = "auto_enum_ts_masked"
            except Exception:
                chosen_cat_key = None
                cat_bandit = None
                chosen_base_idx = None

        # Keep total scored candidates roughly stable even when enumerating multiple
        # categorical bases: score ~grid_batch_size points per batch overall (not per base).
        n_cont_per_batch = int(max(1, int(self._grid_batch_size) // max(1, int(bases.shape[0]))))

        # Always build a small top-k pool for robust selection.
        use_pool = True
        # IMPORTANT: keep selection behavior independent of tracing.
        # - selection pool uses select_top_k
        # - trace top-K uses trace_top_k (stored separately)
        pool_k_sel = int(max(select_top_k, 1))
        pool_k_trace = int(max(self._trace_top_k, 0)) if want_trace else 0
        pool_x_batches: List[np.ndarray] = []
        pool_mu_batches: List[np.ndarray] = []
        pool_sigma_base_batches: List[np.ndarray] = []
        pool_sigma_batches: List[np.ndarray] = []
        pool_sigma_nov_batches: List[np.ndarray] = []
        pool_sigma_tot_batches: List[np.ndarray] = []
        pool_sigma_ref_batches: List[np.ndarray] = []
        pool_ucb_batches: List[np.ndarray] = []
        pool_novelty_bonus_batches: List[np.ndarray] = []
        pool_score_batches: List[np.ndarray] = []
        pool_visits_batches: List[np.ndarray] = []

        for _ in range(max(1, self._grid_batches)):
            if self._grid_sampling == "heatmap_ucb":
                Xc = grid_state.sample_candidates_heatmap_ucb(
                    self.rng,
                    n_cont_per_batch,
                    beta=float(self._heatmap_ucb_beta),
                    explore_prob=float(self._heatmap_ucb_explore_prob),
                    temperature=float(self._heatmap_ucb_temperature),
                    jitter=self._grid_jitter,
                    lo=tr_lo,
                    hi=tr_hi,
                )
            else:
                Xc = grid_state.sample_candidates(
                    self.rng,
                    n_cont_per_batch,
                    mode=self._grid_sampling,
                    jitter=self._grid_jitter,
                    lo=tr_lo,
                    hi=tr_hi,
                )
            if Xc.size == 0:
                continue

            n = int(Xc.shape[0])
            C = int(bases.shape[0])

            # Embed continuous candidates into full vectors for each categorical base.
            X = np.repeat(bases, repeats=n, axis=0)
            X[:, self._cont_dim_indices] = np.tile(Xc, (C, 1))
            n_scored += int(X.shape[0])

            mu, sigma_base, sigma_adj, hmap = self._predict_candidates(cube, X, Xc=Xc, grid_state=grid_state)
            ucb = mu + beta * sigma_adj

            # Visit-based novelty as extra uncertainty (scale-safe and branchless in scoring units).
            # sigma_tot^2 = sigma_adj^2 + sigma_nov^2
            if self._grid_penalty_lambda > 0.0 or want_trace:
                visits = grid_state.visits_for_points(Xc)
            else:
                visits = np.zeros(Xc.shape[0], dtype=float)

            visits_rep = np.tile(visits, C)
            kappa_scale = float(getattr(self, "_visit_novelty_kappa_scale", 1.0))
            if self._categorical_sampling and self._cat_sampler.has_categoricals:
                kappa_scale = 1.0
            kappa = float(self._grid_penalty_lambda) * float(kappa_scale)
            tau_v = float(getattr(self, "_visit_novelty_tau", 10.0))
            sigma_ref = float(np.median(sigma_adj[np.isfinite(sigma_adj)])) if sigma_adj.size else 0.0
            sigma_ref = float(max(sigma_ref, 1e-12))
            w_v = tau_v / (np.maximum(visits, 0.0) + tau_v)
            sigma_nov = (kappa * sigma_ref) * np.sqrt(np.maximum(w_v, 0.0))
            sigma_nov_rep = np.tile(sigma_nov, C)
            sigma_ref_rep = np.full_like(sigma_nov_rep, float(sigma_ref), dtype=float)
            sigma_tot = np.sqrt(np.maximum(sigma_adj * sigma_adj + sigma_nov_rep * sigma_nov_rep, 1e-12))
            score = mu + beta * sigma_tot
            novelty_bonus_rep = score - ucb

            if want_trace:
                diag_mu_batches.append(mu.copy())
                diag_sigma_base_batches.append(sigma_base.copy())
                diag_sigma_adj_batches.append(sigma_adj.copy())
                diag_sigma_nov_batches.append(sigma_nov_rep.copy())
                diag_sigma_tot_batches.append(sigma_tot.copy())
                diag_sigma_ref_batches.append(sigma_ref_rep.copy())
                diag_ucb_batches.append(ucb.copy())
                diag_novelty_bonus_batches.append(novelty_bonus_rep.copy())
                diag_score_batches.append(score.copy())
                diag_visits_batches.append(visits_rep.copy())
                diag_counts_used_batches.append(hmap.get("counts_used", np.zeros(0, dtype=float)).copy())
                diag_vars_r_used_batches.append(hmap.get("vars_r_used", np.zeros(0, dtype=float)).copy())
                diag_used_mode_batches.append(hmap.get("used_mode", np.zeros(0, dtype=np.int8)).copy())
                diag_visits_cont_batches.append(visits.copy())

                try:
                    keys_batch = [grid_state.cell_index(Xc[i]) for i in range(int(Xc.shape[0]))]
                    if keys_batch:
                        diag_unique_cell_rates.append(float(len(set(keys_batch)) / float(len(keys_batch))))
                        diag_unique_cell_keys.update(keys_batch)
                except Exception:
                    pass

            if use_pool and X.shape[0] > 0:
                # Selection pool (potentially masked to a chosen categorical key).
                sel_lo = 0
                sel_hi = int(score.shape[0])
                if chosen_base_idx is not None and int(bases.shape[0]) > 1:
                    try:
                        lo = int(chosen_base_idx) * int(n)
                        hi = lo + int(n)
                        if 0 <= lo < hi <= int(score.shape[0]):
                            sel_lo, sel_hi = lo, hi
                    except Exception:
                        sel_lo, sel_hi = 0, int(score.shape[0])

                k_sel = min(pool_k_sel, max(0, int(sel_hi - sel_lo)))
                if k_sel >= 1:
                    score_sel = score[sel_lo:sel_hi]
                    idxs_local = np.argpartition(-score_sel, k_sel - 1)[:k_sel]
                    idxs_local = idxs_local[np.argsort(-score_sel[idxs_local])]
                    idxs_sel = sel_lo + idxs_local
                    pool_x_batches.append(X[idxs_sel].copy())
                    pool_mu_batches.append(mu[idxs_sel].copy())
                    pool_sigma_base_batches.append(sigma_base[idxs_sel].copy())
                    pool_sigma_batches.append(sigma_adj[idxs_sel].copy())
                    pool_sigma_nov_batches.append(sigma_nov_rep[idxs_sel].copy())
                    pool_sigma_tot_batches.append(sigma_tot[idxs_sel].copy())
                    pool_sigma_ref_batches.append(sigma_ref_rep[idxs_sel].copy())
                    pool_ucb_batches.append(ucb[idxs_sel].copy())
                    pool_novelty_bonus_batches.append(novelty_bonus_rep[idxs_sel].copy())
                    pool_score_batches.append(score[idxs_sel].copy())
                    pool_visits_batches.append(visits_rep[idxs_sel].copy())

                # Trace top-K (cross-key, unmasked).
                if want_top:
                    k_top = min(pool_k_trace, int(X.shape[0]))
                    if k_top >= 1:
                        idxs_top = np.argpartition(-score, k_top - 1)[:k_top]
                        idxs_top = idxs_top[np.argsort(-score[idxs_top])]
                        top_x_batches.append(X[idxs_top].copy())
                        top_mu_batches.append(mu[idxs_top].copy())
                        top_sigma_base_batches.append(sigma_base[idxs_top].copy())
                        top_sigma_batches.append(sigma_adj[idxs_top].copy())
                        top_sigma_nov_batches.append(sigma_nov_rep[idxs_top].copy())
                        top_sigma_tot_batches.append(sigma_tot[idxs_top].copy())
                        top_sigma_ref_batches.append(sigma_ref_rep[idxs_top].copy())
                        top_ucb_batches.append(ucb[idxs_top].copy())
                        top_novelty_bonus_batches.append(novelty_bonus_rep[idxs_top].copy())
                        top_score_batches.append(score[idxs_top].copy())
                        top_visits_batches.append(visits_rep[idxs_top].copy())

            # Track best-by-score within the (possibly masked) selection slice.
            idx = int(np.argmax(score))
            if chosen_base_idx is not None and int(bases.shape[0]) > 1:
                try:
                    lo = int(chosen_base_idx) * int(n)
                    hi = lo + int(n)
                    if 0 <= lo < hi <= int(score.shape[0]):
                        idx = lo + int(np.argmax(score[lo:hi]))
                except Exception:
                    idx = int(np.argmax(score))
            sc = float(score[idx])
            if sc > best_score or best_x is None:
                best_score = sc
                best_x = X[idx].copy()
                best_mu = float(mu[idx])
                best_sigma_base = float(sigma_base[idx])
                best_sigma = float(sigma_adj[idx])
                best_sigma_nov = float(sigma_nov_rep[idx])
                best_sigma_tot = float(sigma_tot[idx])
                best_ucb = float(ucb[idx])
                best_novelty_bonus = float(novelty_bonus_rep[idx])
                best_visits = float(visits_rep[idx])
                best_sigma_ref = float(sigma_ref_rep[idx])

        if use_pool and pool_x_batches:
            X_pool = np.concatenate(pool_x_batches, axis=0)
            mu_pool = np.concatenate(pool_mu_batches, axis=0)
            sigma_base_pool = np.concatenate(pool_sigma_base_batches, axis=0)
            sigma_pool = np.concatenate(pool_sigma_batches, axis=0)
            sigma_nov_pool = np.concatenate(pool_sigma_nov_batches, axis=0)
            sigma_tot_pool = np.concatenate(pool_sigma_tot_batches, axis=0)
            sigma_ref_pool = np.concatenate(pool_sigma_ref_batches, axis=0)
            ucb_pool = np.concatenate(pool_ucb_batches, axis=0)
            novelty_bonus_pool = np.concatenate(pool_novelty_bonus_batches, axis=0)
            score_pool = np.concatenate(pool_score_batches, axis=0)
            visits_pool = np.concatenate(pool_visits_batches, axis=0)

            if score_pool.size > 1:
                # Softmax selection over standardized scores (scale-free).
                score_std = float(np.std(score_pool))
                if score_std > 1e-12:
                    score_z = (score_pool - float(np.mean(score_pool))) / score_std
                else:
                    score_z = np.zeros_like(score_pool, dtype=float)
                logits = 3.0 * score_z
                logits = logits - float(np.max(logits))
                probs = np.exp(logits)
                probs_sum = float(np.sum(probs))
                if probs_sum > 0.0 and np.isfinite(probs_sum):
                    probs = probs / probs_sum
                    pick = int(self.rng.choice(int(score_pool.size), p=probs))
                else:
                    pick = int(np.argmax(score_pool))
            else:
                pick = 0

            best_x = X_pool[pick].copy()
            best_mu = float(mu_pool[pick])
            best_sigma_base = float(sigma_base_pool[pick])
            best_sigma = float(sigma_pool[pick])
            best_sigma_nov = float(sigma_nov_pool[pick])
            best_sigma_tot = float(sigma_tot_pool[pick])
            best_sigma_ref = float(sigma_ref_pool[pick])
            best_ucb = float(ucb_pool[pick])
            best_novelty_bonus = float(novelty_bonus_pool[pick])
            best_score = float(score_pool[pick])
            best_visits = float(visits_pool[pick])

        if best_x is None:
            best_x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])

        if cat_keys and chosen_cat_key is None:
            chosen_cat_key = self._cat_sampler.get_cat_key(best_x)

        if want_trace:
            eps = 1e-12
            sigma_ratio = float(best_sigma / max(best_sigma_base, eps))
            sigma_ratio_tot = float(best_sigma_tot / max(best_sigma_base, eps))
            chosen_x_cont = (
                np.asarray(best_x, dtype=float)[self._cont_dim_indices] if (best_x is not None) else None
            )
            chosen_cell = None
            if chosen_x_cont is not None:
                try:
                    _, chosen_cell = self._heatmap_adjustment_for_cont(grid_state, chosen_x_cont)
                except Exception:
                    chosen_cell = None

            def _pct(x: np.ndarray) -> Dict[str, float]:
                x = np.asarray(x, dtype=float).reshape(-1)
                x = x[np.isfinite(x)]
                if x.size == 0:
                    return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
                p50, p90, p99 = np.percentile(x, [50, 90, 99])
                return {
                    "mean": float(np.mean(x)),
                    "p50": float(p50),
                    "p90": float(p90),
                    "p99": float(p99),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                }

            mu_all = np.concatenate(diag_mu_batches, axis=0) if diag_mu_batches else np.zeros(0, dtype=float)
            sigma_base_all = (
                np.concatenate(diag_sigma_base_batches, axis=0) if diag_sigma_base_batches else np.zeros(0, dtype=float)
            )
            sigma_adj_all = (
                np.concatenate(diag_sigma_adj_batches, axis=0) if diag_sigma_adj_batches else np.zeros(0, dtype=float)
            )
            sigma_nov_all = (
                np.concatenate(diag_sigma_nov_batches, axis=0) if diag_sigma_nov_batches else np.zeros(0, dtype=float)
            )
            sigma_tot_all = (
                np.concatenate(diag_sigma_tot_batches, axis=0) if diag_sigma_tot_batches else np.zeros(0, dtype=float)
            )
            sigma_ref_all = (
                np.concatenate(diag_sigma_ref_batches, axis=0) if diag_sigma_ref_batches else np.zeros(0, dtype=float)
            )
            ucb_all = np.concatenate(diag_ucb_batches, axis=0) if diag_ucb_batches else np.zeros(0, dtype=float)
            novelty_bonus_all = (
                np.concatenate(diag_novelty_bonus_batches, axis=0)
                if diag_novelty_bonus_batches
                else np.zeros(0, dtype=float)
            )
            score_all = np.concatenate(diag_score_batches, axis=0) if diag_score_batches else np.zeros(0, dtype=float)
            visits_all = (
                np.concatenate(diag_visits_batches, axis=0) if diag_visits_batches else np.zeros(0, dtype=float)
            )
            counts_used_all = (
                np.concatenate(diag_counts_used_batches, axis=0)
                if diag_counts_used_batches
                else np.zeros(0, dtype=float)
            )
            vars_r_used_all = (
                np.concatenate(diag_vars_r_used_batches, axis=0)
                if diag_vars_r_used_batches
                else np.zeros(0, dtype=float)
            )
            used_mode_all = (
                np.concatenate(diag_used_mode_batches, axis=0)
                if diag_used_mode_batches
                else np.zeros(0, dtype=np.int8)
            )
            visits_cont_all = (
                np.concatenate(diag_visits_cont_batches, axis=0)
                if diag_visits_cont_batches
                else np.zeros(0, dtype=float)
            )

            sigma_ratio_all = sigma_adj_all / np.maximum(sigma_base_all, eps)
            novelty_to_abs_ucb_all = novelty_bonus_all / np.maximum(np.abs(ucb_all), eps)

            cells_total = int(len(getattr(grid_state, "stats", {})))
            cells_with_resid = 0
            cells_with_resid_ge3 = 0
            try:
                for st in getattr(grid_state, "stats", {}).values():
                    n_r = float(getattr(st, "n_r", 0.0))
                    if n_r > 0.0:
                        cells_with_resid += 1
                    if n_r >= 3.0:
                        cells_with_resid_ge3 += 1
            except Exception:
                cells_with_resid = 0
                cells_with_resid_ge3 = 0

            cells_total_coarse = int(len(getattr(grid_state, "stats_coarse", {}))) if grid_state.multi_resolution else 0
            cells_with_resid_coarse = 0
            try:
                if grid_state.multi_resolution:
                    for st in getattr(grid_state, "stats_coarse", {}).values():
                        if float(getattr(st, "n_r", 0.0)) > 0.0:
                            cells_with_resid_coarse += 1
            except Exception:
                cells_with_resid_coarse = 0

            tau_eff, heatmap_gate = self._heatmap_effective_tau()

            trace: Dict[str, Any] = {
                "event": "ask",
                "iteration": int(self.iteration),
                "cube_depth": int(cube.depth),
                "cube_n_trials": int(cube.n_trials),
                "gamma": float(self.gamma),
                "beta": float(beta),
                "grid_bins": int(self._grid_bins),
                "grid_batch_size": int(self._grid_batch_size),
                "grid_batches": int(self._grid_batches),
                "grid_sampling": str(self._grid_sampling),
                "grid_jitter": bool(self._grid_jitter),
                "grid_penalty_lambda": float(self._grid_penalty_lambda),
                "visit_novelty": {
                    "tau": float(getattr(self, "_visit_novelty_tau", 10.0)),
                    "kappa_scale": float(getattr(self, "_visit_novelty_kappa_scale", 1.0)),
                    "sigma_ref_chosen": float(best_sigma_ref),
                },
                "heatmap_ucb": {
                    "beta": float(self._heatmap_ucb_beta),
                    "explore_prob": float(self._heatmap_ucb_explore_prob),
                    "temperature": float(self._heatmap_ucb_temperature),
                },
                "heatmap_index": {
                    "B_index": int(getattr(grid_state, "B_index", grid_state.B)),
                    "B_index_coarse": (
                        int(getattr(grid_state, "B_index_coarse", 0) or 0) if grid_state.multi_resolution else None
                    ),
                    "index_dims_cont": (
                        grid_state.index_dims.tolist() if grid_state.index_dims is not None else None
                    ),
                    "index_dims_full": (
                        [self._cont_dim_indices[int(i)] for i in grid_state.index_dims.tolist()]
                        if grid_state.index_dims is not None
                        else list(self._cont_dim_indices)
                    ),
                    "soft_assignment": bool(getattr(grid_state, "soft_assignment", False)),
                    "multi_resolution": bool(getattr(grid_state, "multi_resolution", False)),
                    "unique_cells": int(len(grid_state.stats)),
                    "avg_occ": float(
                        float(grid_state.total_visits) / float(max(1, len(grid_state.stats)))
                    ),
                    "cells_with_resid": int(cells_with_resid),
                    "cells_with_resid_ge3": int(cells_with_resid_ge3),
                    "frac_cells_with_resid": float(cells_with_resid) / float(max(1, cells_total)),
                    "unique_cells_coarse": int(cells_total_coarse),
                    "cells_with_resid_coarse": int(cells_with_resid_coarse),
                    "frac_cells_with_resid_coarse": float(cells_with_resid_coarse)
                    / float(max(1, cells_total_coarse)),
                },
                "taylor": {
                    "misfit": bool(misfit),
                    "rel_mse": (float(rel_mse) if rel_mse is not None else None),
                    "tr_scale": float(tr_scale),
                    "tr_lo_cont": tr_lo.copy(),
                    "tr_hi_cont": tr_hi.copy(),
                    "selection_mode": str(selection_mode),
                    "select_top_k": int(select_top_k),
                },
                "heatmap_gate": heatmap_gate,
                "surrogate": {
                    "kind": "lgs_heatmap_sigma",
                    "heatmap_tau_eff": float(tau_eff),
                    "heatmap_tau": float(self._heatmap_blend_tau),
                    "resid_clip_sigma": float(self._heatmap_resid_clip_sigma),
                    "sigma_calib_scale": float(self._sigma_calibration_scale()),
                    "sigma_calib_scale_full": float(self._sigma_calibration_scale_full()),
                },
                "n_scored": int(n_scored),
                "batch_stats": {
                    "mu": _pct(mu_all),
                    "sigma_base": _pct(sigma_base_all),
                    "sigma_adj": _pct(sigma_adj_all),
                    "sigma_nov": _pct(sigma_nov_all),
                    "sigma_tot": _pct(sigma_tot_all),
                    "sigma_ref": _pct(sigma_ref_all),
                    "sigma_ratio": _pct(sigma_ratio_all),
                    "ucb": _pct(ucb_all),
                    "novelty_bonus": _pct(novelty_bonus_all),
                    "score": _pct(score_all),
                    "visits": _pct(visits_all),
                    "novelty_to_abs_ucb": _pct(novelty_to_abs_ucb_all),
                },
                "heatmap_usage": {
                    "frac_candidates_n_r_gt0": float(np.mean(counts_used_all > 0.0))
                    if counts_used_all.size
                    else 0.0,
                    "frac_candidates_n_r_ge3": float(np.mean(counts_used_all >= 3.0))
                    if counts_used_all.size
                    else 0.0,
                    "frac_candidates_used_fine": float(np.mean(used_mode_all == 1))
                    if used_mode_all.size
                    else 0.0,
                    "frac_candidates_used_coarse": float(np.mean(used_mode_all == 2))
                    if used_mode_all.size
                    else 0.0,
                    "frac_candidates_used_fine_weak": float(np.mean(used_mode_all == 3))
                    if used_mode_all.size
                    else 0.0,
                    "unique_cell_rate_overall": float(len(diag_unique_cell_keys))
                    / float(max(1, int(counts_used_all.size))),
                    "unique_cell_rate_per_batch": _pct(np.asarray(diag_unique_cell_rates, dtype=float))
                    if diag_unique_cell_rates
                    else _pct(np.zeros(0, dtype=float)),
                    "frac_candidates_visits0": float(np.mean(visits_cont_all <= 0.0))
                    if visits_cont_all.size
                    else 0.0,
                    "n_candidates_cont": int(visits_cont_all.size),
                    "counts_used": _pct(counts_used_all),
                    "vars_r_used": _pct(vars_r_used_all),
                },
                "chosen_cell": chosen_cell,
                "categorical": {
                    "sampling_enabled": bool(self._categorical_sampling),
                    "stage": str(self._categorical_stage),
                    "stage_effective": str(stage_eff),
                    "pre_n": int(self._categorical_pre_n),
                    "n_bases": int(bases.shape[0]),
                    "keys_considered": [list(k) for k in cat_keys],
                    "key_chosen": (list(chosen_cat_key) if chosen_cat_key is not None else None),
                    "bandit": (cat_bandit if cat_bandit is not None else None),
                },
                "x_chosen_raw": best_x.copy(),
                "chosen_pred": {
                    "mu": float(best_mu),
                    "sigma_base": float(best_sigma_base),
                    "sigma": float(best_sigma),
                    "sigma_ratio": float(sigma_ratio),
                    "sigma_nov": float(best_sigma_nov),
                    "sigma_tot": float(best_sigma_tot),
                    "sigma_ratio_tot": float(sigma_ratio_tot),
                    "ucb": float(best_ucb),
                    "novelty_bonus": float(best_novelty_bonus),
                    "score": float(best_score),
                    "visits": float(best_visits),
                },
            }
            if cube.lgs_model is not None:
                grad = cube.lgs_model.get("grad")
                grad_norm = float(np.linalg.norm(grad)) if grad is not None else 0.0
                trace["lgs"] = {
                    "n_pts": int(len(cube.lgs_model.get("all_pts", []))),
                    "noise_var": float(cube.lgs_model.get("noise_var", 0.0)),
                    "sigma_scale": float(cube.lgs_model.get("sigma_scale", 1.0)),
                    "feature_kind": (
                        str(cube.lgs_model.get("feature_kind")) if cube.lgs_model.get("feature_kind") is not None else None
                    ),
                    "selected_kind": (
                        str(cube.lgs_model.get("selected_kind")) if cube.lgs_model.get("selected_kind") is not None else None
                    ),
                    "gcv": (float(cube.lgs_model.get("gcv")) if cube.lgs_model.get("gcv") is not None else None),
                    "df": (float(cube.lgs_model.get("df")) if cube.lgs_model.get("df") is not None else None),
                    "n_feat": cube.lgs_model.get("n_feat"),
                    "candidates_gcv": cube.lgs_model.get("candidates_gcv"),
                    "loo_regret": (
                        float(cube.lgs_model.get("loo_regret")) if cube.lgs_model.get("loo_regret") is not None else None
                    ),
                    "loo_topk_overlap": (
                        float(cube.lgs_model.get("loo_topk_overlap"))
                        if cube.lgs_model.get("loo_topk_overlap") is not None
                        else None
                    ),
                    "loo_spearman": (
                        float(cube.lgs_model.get("loo_spearman")) if cube.lgs_model.get("loo_spearman") is not None else None
                    ),
                    "candidates_loo_regret": cube.lgs_model.get("candidates_loo_regret"),
                    "candidates_loo_topk_overlap": cube.lgs_model.get("candidates_loo_topk_overlap"),
                    "candidates_loo_spearman": cube.lgs_model.get("candidates_loo_spearman"),
                    "rel_mse": (
                        float(cube.lgs_model.get("rel_mse")) if cube.lgs_model.get("rel_mse") is not None else None
                    ),
                    "grad_norm": grad_norm,
                    "has_gradient_dir": cube.lgs_model.get("gradient_dir") is not None,
                }

            if want_top and top_x_batches:
                X_top = np.concatenate(top_x_batches, axis=0)
                mu_top = np.concatenate(top_mu_batches, axis=0)
                sigma_base_top = np.concatenate(top_sigma_base_batches, axis=0)
                sigma_top = np.concatenate(top_sigma_batches, axis=0)
                sigma_nov_top = np.concatenate(top_sigma_nov_batches, axis=0)
                sigma_tot_top = np.concatenate(top_sigma_tot_batches, axis=0)
                sigma_ref_top = np.concatenate(top_sigma_ref_batches, axis=0)
                ucb_top = np.concatenate(top_ucb_batches, axis=0)
                novelty_bonus_top = np.concatenate(top_novelty_bonus_batches, axis=0)
                score_top = np.concatenate(top_score_batches, axis=0)
                visits_top = np.concatenate(top_visits_batches, axis=0)

                order = np.argsort(-score_top)
                k = min(self._trace_top_k, int(order.shape[0]))
                order = order[:k]
                trace["top"] = {
                    "x": X_top[order],
                    "mu": mu_top[order],
                    "sigma_base": sigma_base_top[order],
                    "sigma": sigma_top[order],
                    "sigma_nov": sigma_nov_top[order],
                    "sigma_tot": sigma_tot_top[order],
                    "sigma_ref": sigma_ref_top[order],
                    "ucb": ucb_top[order],
                    "novelty_bonus": novelty_bonus_top[order],
                    "score": score_top[order],
                    "visits": visits_top[order],
                }

            # Also expose the within-key pool used for selection (helps decompose regret).
            if use_pool and pool_x_batches:
                try:
                    order_pool = np.argsort(-score_pool)
                    k_pool = min(self._trace_top_k, int(order_pool.shape[0]))
                    order_pool = order_pool[:k_pool]
                    trace["top_within_key"] = {
                        "x": X_pool[order_pool],
                        "mu": mu_pool[order_pool],
                        "sigma_base": sigma_base_pool[order_pool],
                        "sigma": sigma_pool[order_pool],
                        "sigma_nov": sigma_nov_pool[order_pool],
                        "sigma_tot": sigma_tot_pool[order_pool],
                        "sigma_ref": sigma_ref_pool[order_pool],
                        "ucb": ucb_pool[order_pool],
                        "novelty_bonus": novelty_bonus_pool[order_pool],
                        "score": score_pool[order_pool],
                        "visits": visits_pool[order_pool],
                    }
                except Exception:
                    pass

            self._pending_trace = trace

        if (
            self._categorical_sampling
            and self._categorical_stage in {"pre", "auto"}
            and self._cat_sampler.has_categoricals
            and cat_mode in {"enum", "sample"}
        ):
            # Prevent post-hoc categorical modification in _sample_in_cube.
            self._last_sample_cat_already_applied = True

        return best_x

    def _local_search_sample(self, progress: float) -> np.ndarray:
        """
        Sample near the best point for local search refinement.

        Parameters
        ----------
        progress : float
            Progress through local search phase in [0, 1].

        Returns
        -------
        np.ndarray
            Sampled point.
        """
        x = self._local_search_sampler.sample(
            self.best_x,
            list(self.bounds),
            self._global_widths,
            float(progress),
            self.rng,
        )
        return self._clip_to_bounds(x)

    # -------------------------------------------------------------------------
    # Clipping helpers
    # -------------------------------------------------------------------------

    def _clip_to_cube(self, x: np.ndarray, cube: Cube) -> np.ndarray:
        """Clip point to cube bounds."""
        return np.array(
            [np.clip(x[i], cube.bounds[i][0], cube.bounds[i][1]) for i in range(self.dim)]
        )

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip point to global bounds."""
        return np.array(
            [np.clip(x[i], self.bounds[i][0], self.bounds[i][1]) for i in range(self.dim)]
        )

    # -------------------------------------------------------------------------
    # Cube tree management
    # -------------------------------------------------------------------------

    def _find_containing_leaf(self, x: np.ndarray) -> Cube:
        """
        Find the leaf cube that contains point x.

        Parameters
        ----------
        x : np.ndarray
            Point to locate.

        Returns
        -------
        Cube
            Containing leaf cube.
        """
        for leaf in self.leaves:
            if leaf.contains(x):
                return leaf
        # Fallback: return leaf with closest center
        min_dist = float("inf")
        closest = self.leaves[0] if self.leaves else self.root
        for leaf in self.leaves:
            dist = np.linalg.norm(x - leaf.center())
            if dist < min_dist:
                min_dist = dist
                closest = leaf
        return closest

    def _should_split(self, cube: Cube) -> bool:
        """Check if a cube should be split."""
        return bool(self._split_decider.should_split(cube, self.dim))

    def _update_all_models(self) -> None:
        """Update LGS models for all leaf cubes."""
        for leaf in self.leaves:
            leaf.fit_lgs_model(self.gamma, self.dim, self.rng)

    # -------------------------------------------------------------------------
    # High-level optimization interface
    # -------------------------------------------------------------------------

    def optimize(
        self,
        objective: Callable[[Union[np.ndarray, Dict[str, Any]]], float],
        budget: int = 100,
    ) -> Tuple[Union[np.ndarray, Dict[str, Any]], float]:
        """
        Run optimization loop.

        Parameters
        ----------
        objective : Callable
            Function to optimize. Takes array or dict (matching ask() output).
        budget : int
            Number of function evaluations.

        Returns
        -------
        best_x : Union[np.ndarray, Dict[str, Any]]
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

        if self._param_space_mode and self.best_x is not None:
            return self.decode(self.best_x), self.best_y
        return self.best_x, self.best_y

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current optimization statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary with optimization statistics.
        """
        return {
            "n_observations": self.n_observations,
            "n_leaves": len(self.leaves),
            "max_depth": max((l.depth for l in self.leaves), default=0),
            "best_y": self.best_y,
            "gamma": self.gamma,
            "stagnation": self.stagnation,
            "is_stagnating": self.is_stagnating,
            "iteration": self.iteration,
            "phase": "exploration" if self.iteration < self.exploration_budget else "local_search",
        }

    def __repr__(self) -> str:
        mode = "param_space" if self._param_space_mode else "bounds"
        return (
            f"ALBA(dim={self.dim}, mode={mode}, n_obs={self.n_observations}, "
            f"n_leaves={len(self.leaves)}, best_y={self.best_y:.4f})"
        )
