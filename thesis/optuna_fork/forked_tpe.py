"""
Local-Gravity TPE Sampler — fork sperimentale.

Idea: TPE standard sceglie i candidati in base a EI = log p(x|below) - log p(x|above).
Noi aggiungiamo un termine di *località* che favorisce candidati vicini al best attuale,
ispirato al "cube gravity" di ALBA.

    acquisition(x) = EI(x) + λ * locality(x, x_best)

Dove locality decresce con la distanza dal best (trust-region soft).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

try:
    import optuna
    from optuna.distributions import BaseDistribution
    from optuna.samplers import TPESampler
    from optuna.trial import FrozenTrial, TrialState

    # Internal helper used by Optuna's own TPE.
    from optuna.samplers._tpe.sampler import _split_trials  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "LocalGravityTPESampler requires optuna to be installed"
    ) from e


LocalityMode = Literal["gaussian", "inverse_sq", "none"]


@dataclass
class LocalGravityConfig:
    """Configuration for the local-gravity extension."""

    # Locality injection
    locality_weight: float = 0.5  # λ: how much locality matters vs pure EI
    locality_mode: LocalityMode = "gaussian"  # how distance translates to bonus
    locality_sigma: float = 0.2  # σ for gaussian mode (in normalized coords)

    # Adaptive: increase locality as optimization progresses
    adaptive: bool = True
    adaptive_min_weight: float = 0.1
    adaptive_max_weight: float = 1.0

    # Gravity drift: bias samples toward best (before EI ranking)
    gravity_drift: float = 0.0  # 0 = off, >0 = drift toward best


@dataclass
class _RuntimeState:
    """Tracks best solution and iteration for adaptive weighting."""

    best_x: np.ndarray | None = None
    best_y: float = float("inf")
    n_completed: int = 0


class LocalGravityTPESampler(TPESampler):
    """
    TPE + locality/gravity: favorisce candidati vicini al best attuale.

    Questa è una modifica sperimentale per vedere se aggiungere un
    "trust region" soft a TPE migliora la convergenza.
    """

    def __init__(
        self,
        *,
        local_gravity: LocalGravityConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Parse config
        if local_gravity is None:
            self._lg = LocalGravityConfig()
        elif isinstance(local_gravity, LocalGravityConfig):
            self._lg = local_gravity
        elif isinstance(local_gravity, dict):
            self._lg = LocalGravityConfig(
                locality_weight=float(local_gravity.get("locality_weight", 0.5)),
                locality_mode=str(local_gravity.get("locality_mode", "gaussian")),  # type: ignore
                locality_sigma=float(local_gravity.get("locality_sigma", 0.2)),
                adaptive=bool(local_gravity.get("adaptive", True)),
                adaptive_min_weight=float(local_gravity.get("adaptive_min_weight", 0.1)),
                adaptive_max_weight=float(local_gravity.get("adaptive_max_weight", 1.0)),
                gravity_drift=float(local_gravity.get("gravity_drift", 0.0)),
            )
        else:
            raise TypeError(f"local_gravity must be LocalGravityConfig|dict|None, got {type(local_gravity)}")

        self._state = _RuntimeState()

    # -------------------------------------------------------------------------
    # Locality helpers
    # -------------------------------------------------------------------------

    def _compute_locality_bonus(
        self,
        samples: dict[str, np.ndarray],
        search_space: dict[str, BaseDistribution],
    ) -> np.ndarray:
        """
        Compute locality bonus for each candidate sample.

        Returns array of shape (n_candidates,) with bonus values.
        Higher = closer to best.
        """
        n_candidates = next(iter(samples.values())).size
        if self._state.best_x is None or n_candidates == 0:
            return np.zeros(n_candidates)

        # Build candidate matrix in normalized [0,1]^d space
        param_names = list(search_space.keys())
        d = len(param_names)
        X = np.zeros((n_candidates, d))

        for i, name in enumerate(param_names):
            dist = search_space[name]
            internal_vals = samples[name]
            # Normalize to [0,1] based on distribution bounds
            lo, hi = self._get_bounds(dist)
            if hi > lo:
                X[:, i] = (internal_vals - lo) / (hi - lo)
            else:
                X[:, i] = 0.5

        # Best point in normalized space
        best_norm = np.zeros(d)
        for i, name in enumerate(param_names):
            dist = search_space[name]
            lo, hi = self._get_bounds(dist)
            if hi > lo:
                best_norm[i] = (self._state.best_x[i] - lo) / (hi - lo)
            else:
                best_norm[i] = 0.5

        # Distance from best
        dists = np.linalg.norm(X - best_norm, axis=1)

        # Convert distance to bonus
        mode = self._lg.locality_mode
        if mode == "gaussian":
            sigma = max(self._lg.locality_sigma, 0.01)
            bonus = np.exp(-(dists ** 2) / (2 * sigma ** 2))
        elif mode == "inverse_sq":
            bonus = 1.0 / (1.0 + dists ** 2)
        else:  # "none"
            bonus = np.zeros(n_candidates)

        return bonus

    def _get_bounds(self, dist: BaseDistribution) -> tuple[float, float]:
        """Extract (low, high) bounds from a distribution."""
        # Works for Float/Int distributions
        lo = getattr(dist, "low", 0.0)
        hi = getattr(dist, "high", 1.0)
        return float(lo), float(hi)

    def _get_adaptive_weight(self) -> float:
        """Compute locality weight based on progress."""
        if not self._lg.adaptive:
            return self._lg.locality_weight

        # Ramp up locality as we get more samples
        n = self._state.n_completed
        # Simple linear ramp: start at min, reach max around 100 samples
        progress = min(1.0, n / 100.0)
        w = self._lg.adaptive_min_weight + progress * (
            self._lg.adaptive_max_weight - self._lg.adaptive_min_weight
        )
        return w

    def _update_state(self, study: "optuna.study.Study") -> None:
        """Update runtime state with best solution from study."""
        trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        self._state.n_completed = len(trials)

        if not trials:
            return

        # Find best trial
        direction = study.direction
        if direction == optuna.study.StudyDirection.MINIMIZE:
            best_trial = min(trials, key=lambda t: t.value if t.value is not None else float("inf"))
        else:
            best_trial = max(trials, key=lambda t: t.value if t.value is not None else float("-inf"))

        if best_trial.value is None:
            return

        # Store best_x as array (internal repr)
        param_names = sorted(best_trial.params.keys())
        best_x = []
        for name in param_names:
            val = best_trial.params[name]
            # Convert to internal repr if needed
            if name in best_trial.distributions:
                dist = best_trial.distributions[name]
                val = dist.to_internal_repr(val)
            best_x.append(float(val))

        self._state.best_x = np.array(best_x, dtype=float)
        self._state.best_y = float(best_trial.value)

    # -------------------------------------------------------------------------
    # Override _sample to inject locality
    # -------------------------------------------------------------------------

    def _sample(
        self,
        study: "optuna.study.Study",
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        """
        Sample parameters with locality-augmented acquisition.
        """
        # Update state with current best
        self._update_state(study)

        # --- Standard TPE setup ---
        if getattr(self, "_constant_liar", False):
            states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
            use_cache = False
        else:
            states = [TrialState.COMPLETE, TrialState.PRUNED]
            use_cache = True

        trials = study._get_trials(deepcopy=False, states=states, use_cache=use_cache)

        if getattr(self, "_constant_liar", False):
            trials = [t for t in trials if t.number != trial.number]

        n_finished = sum(t.state != TrialState.RUNNING for t in trials)
        n_below = self._gamma(int(n_finished))
        use_constraints = getattr(self, "_constraints_func", None) is not None
        below, above = _split_trials(study, trials, n_below, use_constraints)

        mpe_below = self._build_parzen_estimator(study, search_space, below, handle_below=True)
        mpe_above = self._build_parzen_estimator(study, search_space, above, handle_below=False)

        n_candidates = int(getattr(self, "_n_ei_candidates", 24))
        samples = mpe_below.sample(self._rng.rng, n_candidates)

        # --- Standard EI ---
        ei = self._compute_acquisition_func(samples, mpe_below, mpe_above)

        # --- Locality bonus ---
        locality_bonus = self._compute_locality_bonus(samples, search_space)
        locality_weight = self._get_adaptive_weight()

        # Combine: scale locality to similar magnitude as EI
        if locality_bonus.std() > 0 and ei.std() > 0:
            # Normalize locality to have same scale as EI
            locality_bonus = locality_bonus * (ei.std() / locality_bonus.std())

        acquisition = ei + locality_weight * locality_bonus

        # --- Select best candidate ---
        idx = int(np.argmax(acquisition))

        picked_internal: dict[str, Any] = {k: v[idx].item() for k, v in samples.items()}
        picked_external: dict[str, Any] = {}
        for name, dist in search_space.items():
            picked_external[name] = dist.to_external_repr(picked_internal[name])

        return picked_external
