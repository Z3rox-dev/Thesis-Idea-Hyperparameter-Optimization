"""
ALBA Framework - Parameter Space Module

This module provides utilities for defining and handling search spaces
with support for continuous, integer, log-scale, and categorical parameters.

The ParamSpaceHandler class enables ALBA to work with typed parameter
specifications, automatically inferring categorical dimensions and providing
encode/decode transformations between user-facing configs and internal
normalized representations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ParamSpaceHandler:
    """
    Handler for typed parameter space definitions.

    Converts user-friendly parameter specifications into internal normalized
    representations and vice versa.

    Supported parameter types:
    - Categorical: list of choices -> encoded as index / (n_choices - 1)
    - Continuous: (low, high) tuple -> linear interpolation
    - Log-scale: (low, high, 'log') -> log-linear interpolation
    - Integer: (low, high, 'int') -> rounded linear interpolation
    - Fixed: single value -> not included in optimization

    Examples
    --------
    >>> param_space = {
    ...     'learning_rate': (1e-4, 1e-1, 'log'),
    ...     'hidden_size': (32, 512, 'int'),
    ...     'activation': ['relu', 'tanh', 'gelu'],
    ...     'dropout': (0.0, 0.5),
    ...     'optimizer': ['adam'],  # Fixed (single choice)
    ... }
    >>> handler = ParamSpaceHandler(param_space)
    >>> x = np.array([0.5, 0.5, 0.5, 0.5])  # internal normalized
    >>> config = handler.decode(x)
    >>> x_back = handler.encode(config)

    Attributes
    ----------
    specs : List[Dict[str, Any]]
        Internal specification for each optimizable parameter.
    fixed : Dict[str, Any]
        Fixed parameters (not optimized).
    param_order : List[str]
        Order of all parameters (optimizable + fixed).
    dim : int
        Number of optimizable dimensions.
    categorical_dims : List[Tuple[int, int]]
        List of (dim_index, n_choices) for categorical parameters.
    """

    def __init__(
        self,
        param_space: Dict[str, Any],
        param_order: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the parameter space handler.

        Parameters
        ----------
        param_space : Dict[str, Any]
            Dictionary mapping parameter names to their specifications.
        param_order : Optional[List[str]]
            Optional explicit ordering of parameters.

        Raises
        ------
        TypeError
            If param_space is not a non-empty dict.
        ValueError
            If param_order contains missing keys or specs are invalid.
        """
        if not isinstance(param_space, dict) or not param_space:
            raise TypeError("param_space must be a non-empty dict")

        if param_order is None:
            param_order = list(param_space.keys())
        else:
            missing = [k for k in param_order if k not in param_space]
            if missing:
                raise ValueError(f"param_order contains missing keys: {missing}")

        self.specs: List[Dict[str, Any]] = []
        self.fixed: Dict[str, Any] = {}
        self._parse_param_space(param_space, param_order)

        # Build final parameter order
        self.param_order = [s["name"] for s in self.specs] + list(self.fixed.keys())

        # Derived attributes
        self.dim = len(self.specs)
        self.categorical_dims = self._extract_categorical_dims()

    def _parse_param_space(
        self,
        param_space: Dict[str, Any],
        param_order: List[str],
    ) -> None:
        """Parse parameter specifications from user-provided dict."""
        for name in param_order:
            spec = param_space[name]

            # Check if it's a choices list (categorical)
            if self._is_choices_spec(spec):
                choices = list(spec)
                if len(choices) == 0:
                    raise ValueError(f"Empty choices list for '{name}'")
                if len(choices) == 1:
                    self.fixed[name] = choices[0]
                    continue
                self.specs.append({
                    "name": name,
                    "type": "categorical",
                    "choices": choices,
                })
                continue

            # Check if it's a range tuple
            if isinstance(spec, tuple) and len(spec) in (2, 3):
                low = float(spec[0])
                high = float(spec[1])
                if not (high > low):
                    raise ValueError(f"Invalid range for '{name}': {spec}")

                mode = "linear"
                if len(spec) == 3:
                    mode = str(spec[2]).lower()

                if mode in ("log", "logscale"):
                    if low <= 0:
                        raise ValueError(
                            f"Log-scale requires positive bounds for '{name}'"
                        )
                    self.specs.append({
                        "name": name,
                        "type": "float",
                        "low": low,
                        "high": high,
                        "log": True,
                    })
                elif mode in ("int", "integer"):
                    self.specs.append({
                        "name": name,
                        "type": "int",
                        "low": low,
                        "high": high,
                        "log": False,
                    })
                else:
                    self.specs.append({
                        "name": name,
                        "type": "float",
                        "low": low,
                        "high": high,
                        "log": False,
                    })
                continue

            # Scalar => fixed
            self.fixed[name] = spec

    def _is_choices_spec(self, spec: Any) -> bool:
        """Check if spec is a choices list (not a numeric range tuple)."""
        if isinstance(spec, list):
            return True
        if isinstance(spec, tuple):
            # It's a range tuple if (low, high) or (low, high, mode) with numeric bounds
            if len(spec) in (2, 3):
                if isinstance(spec[0], (int, float)) and isinstance(spec[1], (int, float)):
                    return False
            return True
        return False

    def _extract_categorical_dims(self) -> List[Tuple[int, int]]:
        """Extract list of (dim_index, n_choices) for categorical parameters."""
        categorical_dims: List[Tuple[int, int]] = []
        for i, s in enumerate(self.specs):
            if s["type"] == "categorical":
                categorical_dims.append((i, len(s["choices"])))
        return categorical_dims

    # -------------------------------------------------------------------------
    # Encode / Decode
    # -------------------------------------------------------------------------

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
            Configuration dictionary with proper types.

        Raises
        ------
        ValueError
            If x has wrong length.
        """
        x = np.asarray(x, dtype=float)
        if x.shape[0] != len(self.specs):
            raise ValueError(
                f"decode expected x of length {len(self.specs)}, got {x.shape[0]}"
            )

        config: Dict[str, Any] = dict(self.fixed)

        for i, s in enumerate(self.specs):
            v = float(np.clip(x[i], 0.0, 1.0))

            if s["type"] == "categorical":
                choices = s["choices"]
                idx = int(round(v * (len(choices) - 1)))
                idx = int(np.clip(idx, 0, len(choices) - 1))
                config[s["name"]] = choices[idx]

            elif s["type"] == "int":
                low = float(s["low"])
                high = float(s["high"])
                config[s["name"]] = int(round(low + v * (high - low)))

            else:  # float
                low = float(s["low"])
                high = float(s["high"])
                if s.get("log", False):
                    low_log = np.log(low)
                    high_log = np.log(high)
                    config[s["name"]] = float(
                        np.exp(low_log + v * (high_log - low_log))
                    )
                else:
                    config[s["name"]] = float(low + v * (high - low))

        return config

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
        TypeError
            If config is not a dict.
        KeyError
            If required parameter is missing.
        ValueError
            If categorical value is not in choices.
        """
        if not isinstance(config, dict):
            raise TypeError("encode expects a dict config")

        x = np.zeros(len(self.specs), dtype=float)

        for i, s in enumerate(self.specs):
            name = s["name"]
            if name not in config:
                raise KeyError(f"Missing '{name}' in config")

            val = config[name]

            if s["type"] == "categorical":
                choices = s["choices"]
                try:
                    idx = choices.index(val)
                except ValueError:
                    raise ValueError(
                        f"Invalid value for '{name}': {val} (choices={choices})"
                    )
                x[i] = idx / (len(choices) - 1) if len(choices) > 1 else 0.5

            elif s["type"] == "int":
                low = float(s["low"])
                high = float(s["high"])
                v = float(int(val))
                x[i] = (v - low) / (high - low)

            else:  # float
                low = float(s["low"])
                high = float(s["high"])
                v = float(val)
                if s.get("log", False):
                    low_log = np.log(low)
                    high_log = np.log(high)
                    x[i] = (np.log(v) - low_log) / (high_log - low_log)
                else:
                    x[i] = (v - low) / (high - low)

        return np.clip(x, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Return bounds for internal optimization (all [0, 1])."""
        return [(0.0, 1.0)] * self.dim

    def get_parameter_names(self) -> List[str]:
        """Return names of optimizable parameters in order."""
        return [s["name"] for s in self.specs]

    def get_all_parameter_names(self) -> List[str]:
        """Return names of all parameters (optimizable + fixed) in order."""
        return list(self.param_order)

    def sample_random(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample a random point in the normalized space.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Random point in [0, 1]^dim.
        """
        return rng.random(self.dim)

    def sample_random_config(self, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Sample a random configuration.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        Dict[str, Any]
            Random configuration dictionary.
        """
        return self.decode(self.sample_random(rng))

    def __repr__(self) -> str:
        parts = []
        for s in self.specs:
            if s["type"] == "categorical":
                parts.append(f"{s['name']}: {s['choices']}")
            elif s["type"] == "int":
                parts.append(f"{s['name']}: [{s['low']}, {s['high']}] (int)")
            elif s.get("log", False):
                parts.append(f"{s['name']}: [{s['low']}, {s['high']}] (log)")
            else:
                parts.append(f"{s['name']}: [{s['low']}, {s['high']}]")
        for name, val in self.fixed.items():
            parts.append(f"{name}: {val} (fixed)")
        return f"ParamSpaceHandler(dim={self.dim}, params=[{', '.join(parts)}])"
