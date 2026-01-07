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
        grid_penalty_lambda: float = 0.10,
        heatmap_ucb_beta: float = 1.0,
        heatmap_ucb_explore_prob: float = 0.25,
        heatmap_ucb_temperature: float = 1.0,
        categorical_sampling: bool = True,
        categorical_stage: str = "auto",
        categorical_pre_n: int = 8,
        surrogate: str = "lgs",
        knn_k: int = 15,
        knn_cat_weight: float = 1.0,
        knn_sigma_mode: str = "var",
        knn_dist_scale: float = 0.25,
        catadd_smoothing: float = 1.0,
        heatmap_blend_tau: float = 3.0,
        trace_top_k: int = 0,
        trace_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
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
        self._surrogate = str(surrogate).strip().lower()
        self._knn_k = int(knn_k)
        self._knn_cat_weight = float(knn_cat_weight)
        self._knn_sigma_mode = str(knn_sigma_mode).strip().lower()
        self._knn_dist_scale = float(knn_dist_scale)
        self._catadd_smoothing = float(catadd_smoothing)
        self._heatmap_blend_tau = float(heatmap_blend_tau)
        self._trace_top_k = max(0, int(trace_top_k))
        self._trace_hook = trace_hook
        self._pending_trace: Optional[Dict[str, Any]] = None
        self._last_sample_cat_already_applied = False

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
        if self._surrogate not in {
            "lgs",
            "knn",
            "knn_lgs",
            "lgs_catadd",
            "lgs_heatmap_sigma",
            "lgs_heatmap_blend",
        }:
            raise ValueError(
                "surrogate must be one of "
                "{'lgs','knn','knn_lgs','lgs_catadd','lgs_heatmap_sigma','lgs_heatmap_blend'}"
            )
        if self._knn_k < 1:
            raise ValueError("knn_k must be >= 1")
        if self._knn_cat_weight < 0.0:
            raise ValueError("knn_cat_weight must be >= 0")
        if self._knn_sigma_mode not in {"var", "dist", "var+dist"}:
            raise ValueError("knn_sigma_mode must be one of {'var','dist','var+dist'}")
        if self._knn_dist_scale < 0.0:
            raise ValueError("knn_dist_scale must be >= 0")
        if self._catadd_smoothing < 0.0:
            raise ValueError("catadd_smoothing must be >= 0")
        if self._heatmap_blend_tau <= 0.0:
            raise ValueError("heatmap_blend_tau must be > 0")
        if self._grid_sampling not in {"grid_random", "grid_halton", "halton", "heatmap_ucb"}:
            raise ValueError(
                "grid_sampling must be one of {'grid_random','grid_halton','halton','heatmap_ucb'} "
                f"(got '{self._grid_sampling}')"
            )
        if self._trace_hook is not None and not callable(self._trace_hook):
            raise TypeError("trace_hook must be callable or None")

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
        self.leaves: List[Cube] = [self.root]
        self._last_cube: Optional[Cube] = None
        if self._cont_dim_indices:
            cont_bounds = [bounds[i] for i in self._cont_dim_indices]
            self.root.grid_state = LeafGridState(
                bounds=list(cont_bounds),
                B=self._grid_bins,
                index_dims=self._choose_heatmap_index_dims(self.root),
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

    def _ensure_grid_state(self, cube: Cube) -> LeafGridState:
        if not self._cont_dim_indices:
            raise RuntimeError("Grid sampling requires at least one continuous dimension")
        cont_bounds = [cube.bounds[i] for i in self._cont_dim_indices]
        if cube.grid_state is None or cube.grid_state.B != self._grid_bins or cube.grid_state.bounds != cont_bounds:
            cube.grid_state = LeafGridState(
                bounds=list(cont_bounds),
                B=self._grid_bins,
                index_dims=self._choose_heatmap_index_dims(cube),
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

            # Compute base surrogate prediction BEFORE adding the point (avoid self-neighbor effects).
            mu_pred_base: Optional[float] = None
            if cube.grid_state is not None and self._cont_dim_indices:
                x_row = x.reshape(1, -1)
                try:
                    kind = self._surrogate
                    if kind in {"lgs", "lgs_heatmap_sigma", "lgs_heatmap_blend", "lgs_catadd", "knn_lgs"}:
                        if cube.n_trials >= 5 and cube.lgs_model is not None and cube.lgs_model.get("inv_cov") is not None:
                            mu0, _ = cube.predict_bayesian(x_row)
                            mu_pred_base = float(mu0[0])
                            if kind == "lgs_catadd":
                                mu_pred_base += float(self._catadd_adjustment(cube, x_row)[0])

                    if kind in {"knn", "knn_lgs"} and cube.n_trials >= 3:
                        mu0, _ = self._predict_knn(cube, x_row)
                        mu_pred_base = float(mu0[0])
                except Exception:
                    mu_pred_base = None

            cube.add_observation(x, y, self.gamma)

            # Update per-leaf heatmap
            if cube.grid_state is not None and self._cont_dim_indices:
                cube.grid_state.update(x[self._cont_dim_indices], y, self.gamma, y_pred=mu_pred_base)

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

            # Check for split
            if self._should_split(cube):
                children = self._split_policy.split(cube, self.gamma, self.dim, self.rng)
                for child in children:
                    self._cat_sampler.recompute_cube_cat_stats(child, self.gamma)
                    # Rebuild children's heatmap from tested pairs (robust to gamma + semantics).
                    if self._cont_dim_indices:
                        cont_bounds = [child.bounds[i] for i in self._cont_dim_indices]
                        child.grid_state = LeafGridState(
                            bounds=list(cont_bounds),
                            B=self._grid_bins,
                            index_dims=self._choose_heatmap_index_dims(child),
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
        if self._surrogate in {"knn", "knn_lgs"}:
            if cube.n_trials < 3:
                return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])
            return self._sample_with_grid_lgs(cube)

        if cube.lgs_model is None or cube.n_trials < 5:
            return np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])

        return self._sample_with_grid_lgs(cube)

    def _discretize_vec(self, x: np.ndarray, n_choices: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if n_choices <= 1:
            return np.zeros(x.shape[0], dtype=np.int64)
        idx = np.rint(np.clip(x, 0.0, 1.0) * float(n_choices - 1)).astype(np.int64)
        return np.clip(idx, 0, n_choices - 1)

    def _predict_knn(self, cube: Cube, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pairs = list(cube.tested_pairs)
        if len(pairs) < 3:
            return np.zeros(X.shape[0], dtype=float), np.ones(X.shape[0], dtype=float)

        X_tr = np.array([p for p, _ in pairs], dtype=float)
        y_tr = np.array([s for _, s in pairs], dtype=float)
        n_tr = int(X_tr.shape[0])

        center = cube.center()
        widths = np.maximum(cube.widths(), 1e-9)

        if self._cont_dim_indices:
            Xc = (X[:, self._cont_dim_indices] - center[self._cont_dim_indices]) / widths[self._cont_dim_indices]
            Tc = (X_tr[:, self._cont_dim_indices] - center[self._cont_dim_indices]) / widths[self._cont_dim_indices]
            a2 = np.sum(Xc * Xc, axis=1, keepdims=True)
            b2 = np.sum(Tc * Tc, axis=1, keepdims=True).T
            d2 = a2 + b2 - 2.0 * (Xc @ Tc.T)
            d2 = np.maximum(d2, 0.0)
        else:
            d2 = np.zeros((X.shape[0], n_tr), dtype=float)

        if self._cat_dim_indices and self._knn_cat_weight > 0.0:
            mismatch = np.zeros_like(d2, dtype=float)
            for dim_idx, n_choices in self._cat_sampler.categorical_dims:
                if dim_idx not in self._cat_dim_indices:
                    continue
                cand_cat = self._discretize_vec(X[:, dim_idx], n_choices)
                train_cat = self._discretize_vec(X_tr[:, dim_idx], n_choices)
                mismatch += (cand_cat[:, None] != train_cat[None, :]).astype(float)
            d2 = d2 + float(self._knn_cat_weight) * mismatch

        k = min(int(self._knn_k), n_tr)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        d2k = np.take_along_axis(d2, idx, axis=1)
        yk = y_tr[idx]

        h2 = np.median(d2k, axis=1) + 1e-9
        w = np.exp(-0.5 * (d2k / h2[:, None]))
        w_sum = np.sum(w, axis=1) + 1e-12
        mu = np.sum(w * yk, axis=1) / w_sum
        var_y = np.sum(w * (yk - mu[:, None]) ** 2, axis=1) / w_sum
        sigma_var = np.sqrt(np.maximum(var_y, 1e-8))
        sigma_dist = float(self._knn_dist_scale) * np.sqrt(np.maximum(h2, 0.0))

        if self._knn_sigma_mode == "var":
            sigma = sigma_var
        elif self._knn_sigma_mode == "dist":
            sigma = np.maximum(sigma_dist, 1e-8)
        else:
            sigma = sigma_var + sigma_dist
        return mu, sigma

    def _catadd_adjustment(self, cube: Cube, X: np.ndarray) -> np.ndarray:
        if not self._cat_sampler.has_categoricals or not self._cat_dim_indices:
            return np.zeros(X.shape[0], dtype=float)

        pairs = list(cube.tested_pairs)
        if len(pairs) < 5:
            return np.zeros(X.shape[0], dtype=float)

        X_tr = np.array([p for p, _ in pairs], dtype=float)
        y_tr = np.array([s for _, s in pairs], dtype=float)
        mu_tr, _ = cube.predict_bayesian(X_tr)
        resid = y_tr - mu_tr

        alpha = float(self._catadd_smoothing)
        adj = np.zeros(X.shape[0], dtype=float)
        for dim_idx, n_choices in self._cat_sampler.categorical_dims:
            if dim_idx not in self._cat_dim_indices:
                continue
            tr_cat = self._discretize_vec(X_tr[:, dim_idx], n_choices)
            sums = np.bincount(tr_cat, weights=resid, minlength=n_choices).astype(float)
            cnt = np.bincount(tr_cat, minlength=n_choices).astype(float)
            eff = sums / (cnt + alpha)
            cand_cat = self._discretize_vec(X[:, dim_idx], n_choices)
            adj += eff[cand_cat]
        return adj

    def _predict_candidates(
        self,
        cube: Cube,
        X: np.ndarray,
        *,
        Xc: np.ndarray,
        grid_state: LeafGridState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        kind = self._surrogate
        if kind == "lgs":
            return cube.predict_bayesian(X)
        if kind == "knn":
            return self._predict_knn(cube, X)
        if kind == "knn_lgs":
            mu, _ = self._predict_knn(cube, X)
            _, sigma = cube.predict_bayesian(X)
            return mu, sigma
        if kind == "lgs_catadd":
            mu, sigma = cube.predict_bayesian(X)
            mu = mu + self._catadd_adjustment(cube, X)
            return mu, sigma
        if kind == "lgs_heatmap_sigma":
            mu_lgs, sigma = cube.predict_bayesian(X)
            if Xc.size == 0:
                return mu_lgs, sigma
            vars_r = np.zeros(Xc.shape[0], dtype=float)
            counts = np.zeros(Xc.shape[0], dtype=float)
            for i in range(Xc.shape[0]):
                key = grid_state.cell_index(Xc[i])
                st = grid_state.stats.get(key)
                if st is None:
                    continue
                counts[i] = float(st.n_r)
                vars_r[i] = float(st.var_r())
            tau = float(self._heatmap_blend_tau)
            w = counts / (counts + tau)
            rep = max(1, int(X.shape[0] // max(1, Xc.shape[0])))
            w_rep = np.repeat(w, repeats=rep)
            v_rep = np.repeat(vars_r, repeats=rep)
            sigma2 = sigma * sigma
            sigma = np.sqrt(np.maximum(sigma2 + w_rep * v_rep, 1e-12))
            return mu_lgs, sigma
        if kind == "lgs_heatmap_blend":
            mu_lgs, sigma = cube.predict_bayesian(X)
            if Xc.size == 0:
                return mu_lgs, sigma
            means_r = np.zeros(Xc.shape[0], dtype=float)
            vars_r = np.zeros(Xc.shape[0], dtype=float)
            counts = np.zeros(Xc.shape[0], dtype=float)
            for i in range(Xc.shape[0]):
                key = grid_state.cell_index(Xc[i])
                st = grid_state.stats.get(key)
                if st is None:
                    continue
                counts[i] = float(st.n_r)
                means_r[i] = float(st.mean_r())
                vars_r[i] = float(st.var_r())
            tau = float(self._heatmap_blend_tau)
            w = counts / (counts + tau)
            # Repeat across categorical bases.
            rep = max(1, int(X.shape[0] // max(1, Xc.shape[0])))
            w_rep = np.repeat(w, repeats=rep)
            r_rep = np.repeat(means_r, repeats=rep)
            v_rep = np.repeat(vars_r, repeats=rep)

            mu = mu_lgs + w_rep * r_rep
            sigma2 = sigma * sigma
            sigma = np.sqrt(np.maximum(sigma2 + w_rep * v_rep, 1e-12))
            return mu, sigma

        # Should be unreachable due to validation.
        return cube.predict_bayesian(X)

    def _propose_cat_keys(self, cube: Cube, x_template: np.ndarray, n_keys: int) -> List[Tuple[int, ...]]:
        if not self._cat_sampler.has_categoricals:
            return [tuple()]

        want = max(1, int(n_keys))
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
            if len(keys) >= want:
                return keys[:want]

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
        - Optionally penalizes already-visited cells using the leaf heatmap.
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
        best_ucb = 0.0
        best_penalty = 0.0
        best_visits = 0.0

        beta = float(self._novelty_weight) * 2.0  # keep consistent with UCB definition
        n_scored = 0

        want_trace = self._trace_hook is not None
        want_top = want_trace and self._trace_top_k > 0
        top_x_batches: List[np.ndarray] = []
        top_mu_batches: List[np.ndarray] = []
        top_sigma_batches: List[np.ndarray] = []
        top_ucb_batches: List[np.ndarray] = []
        top_penalty_batches: List[np.ndarray] = []
        top_score_batches: List[np.ndarray] = []
        top_visits_batches: List[np.ndarray] = []

        # Keep categorical dimensions fixed during grid scoring.
        x_template = cube.best_x.copy() if cube.best_x is not None else cube.center()

        cat_keys: List[Tuple[int, ...]] = []
        bases: np.ndarray
        stage_eff = str(self._categorical_stage)

        cat_mode = "none"  # {"none","enum","sample"}
        if self._categorical_sampling and self._cat_sampler.has_categoricals:
            if self._categorical_stage == "pre":
                cat_mode = "enum"
                stage_eff = "pre_enum"
            elif self._categorical_stage == "auto":
                cont_dim = int(len(self._cont_dim_indices))
                cat_dim = int(len(self._cat_dim_indices))
                if cont_dim <= 2 and cat_dim >= 4:
                    cat_mode = "sample"
                    stage_eff = "auto_sample"
                else:
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
            cat_keys = [self._cat_sampler.get_cat_key(x_base)]
            bases = x_base.reshape(1, -1)
        else:
            bases = x_template.reshape(1, -1)

        for _ in range(max(1, self._grid_batches)):
            if self._grid_sampling == "heatmap_ucb":
                Xc = grid_state.sample_candidates_heatmap_ucb(
                    self.rng,
                    self._grid_batch_size,
                    beta=float(self._heatmap_ucb_beta),
                    explore_prob=float(self._heatmap_ucb_explore_prob),
                    temperature=float(self._heatmap_ucb_temperature),
                    jitter=self._grid_jitter,
                )
            else:
                Xc = grid_state.sample_candidates(
                    self.rng,
                    self._grid_batch_size,
                    mode=self._grid_sampling,
                    jitter=self._grid_jitter,
                )
            if Xc.size == 0:
                continue

            n = int(Xc.shape[0])
            C = int(bases.shape[0])

            # Embed continuous candidates into full vectors for each categorical base.
            X = np.repeat(bases, repeats=n, axis=0)
            X[:, self._cont_dim_indices] = np.tile(Xc, (C, 1))
            n_scored += int(X.shape[0])

            mu, sigma = self._predict_candidates(cube, X, Xc=Xc, grid_state=grid_state)
            ucb = mu + beta * sigma

            if self._grid_penalty_lambda > 0.0 or want_trace:
                visits = grid_state.visits_for_points(Xc)
                penalty = float(self._grid_penalty_lambda) * np.log1p(visits)
            else:
                visits = np.zeros(Xc.shape[0], dtype=float)
                penalty = np.zeros(Xc.shape[0], dtype=float)

            visits_rep = np.tile(visits, C)
            penalty_rep = np.tile(penalty, C)
            score = ucb - penalty_rep

            if want_top and X.shape[0] > 0:
                k = min(self._trace_top_k, int(X.shape[0]))
                if k >= 1:
                    idxs = np.argpartition(-score, k - 1)[:k]
                    idxs = idxs[np.argsort(-score[idxs])]
                    top_x_batches.append(X[idxs].copy())
                    top_mu_batches.append(mu[idxs].copy())
                    top_sigma_batches.append(sigma[idxs].copy())
                    top_ucb_batches.append(ucb[idxs].copy())
                    top_penalty_batches.append(penalty_rep[idxs].copy())
                    top_score_batches.append(score[idxs].copy())
                    top_visits_batches.append(visits_rep[idxs].copy())

            idx = int(np.argmax(score))
            sc = float(score[idx])
            if sc > best_score or best_x is None:
                best_score = sc
                best_x = X[idx].copy()
                best_mu = float(mu[idx])
                best_sigma = float(sigma[idx])
                best_ucb = float(ucb[idx])
                best_penalty = float(penalty_rep[idx])
                best_visits = float(visits_rep[idx])

        if best_x is None:
            best_x = np.array([self.rng.uniform(lo, hi) for lo, hi in cube.bounds])

        chosen_cat_key: Optional[Tuple[int, ...]] = None
        if cat_keys:
            chosen_cat_key = self._cat_sampler.get_cat_key(best_x)

        if want_trace:
            trace: Dict[str, Any] = {
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
                "heatmap_ucb": {
                    "beta": float(self._heatmap_ucb_beta),
                    "explore_prob": float(self._heatmap_ucb_explore_prob),
                    "temperature": float(self._heatmap_ucb_temperature),
                },
                "heatmap_index": {
                    "index_dims_cont": (
                        grid_state.index_dims.tolist() if grid_state.index_dims is not None else None
                    ),
                    "index_dims_full": (
                        [self._cont_dim_indices[int(i)] for i in grid_state.index_dims.tolist()]
                        if grid_state.index_dims is not None
                        else list(self._cont_dim_indices)
                    ),
                    "unique_cells": int(len(grid_state.stats)),
                    "avg_occ": float(
                        float(grid_state.total_visits) / float(max(1, len(grid_state.stats)))
                    ),
                },
                "surrogate": {
                    "kind": str(self._surrogate),
                    "knn_k": int(self._knn_k),
                    "knn_cat_weight": float(self._knn_cat_weight),
                    "knn_sigma_mode": str(self._knn_sigma_mode),
                    "knn_dist_scale": float(self._knn_dist_scale),
                    "catadd_smoothing": float(self._catadd_smoothing),
                    "heatmap_blend_tau": float(self._heatmap_blend_tau),
                },
                "n_scored": int(n_scored),
                "categorical": {
                    "sampling_enabled": bool(self._categorical_sampling),
                    "stage": str(self._categorical_stage),
                    "stage_effective": str(stage_eff),
                    "pre_n": int(self._categorical_pre_n),
                    "n_bases": int(bases.shape[0]),
                    "keys_considered": [list(k) for k in cat_keys],
                    "key_chosen": (list(chosen_cat_key) if chosen_cat_key is not None else None),
                },
                "x_chosen_raw": best_x.copy(),
                "chosen_pred": {
                    "mu": float(best_mu),
                    "sigma": float(best_sigma),
                    "ucb": float(best_ucb),
                    "penalty": float(best_penalty),
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
                    "grad_norm": grad_norm,
                    "has_gradient_dir": cube.lgs_model.get("gradient_dir") is not None,
                }

            if want_top and top_x_batches:
                X_top = np.concatenate(top_x_batches, axis=0)
                mu_top = np.concatenate(top_mu_batches, axis=0)
                sigma_top = np.concatenate(top_sigma_batches, axis=0)
                ucb_top = np.concatenate(top_ucb_batches, axis=0)
                penalty_top = np.concatenate(top_penalty_batches, axis=0)
                score_top = np.concatenate(top_score_batches, axis=0)
                visits_top = np.concatenate(top_visits_batches, axis=0)

                order = np.argsort(-score_top)
                k = min(self._trace_top_k, int(order.shape[0]))
                order = order[:k]
                trace["top"] = {
                    "x": X_top[order],
                    "mu": mu_top[order],
                    "sigma": sigma_top[order],
                    "ucb": ucb_top[order],
                    "penalty": penalty_top[order],
                    "score": score_top[order],
                    "visits": visits_top[order],
                }

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
