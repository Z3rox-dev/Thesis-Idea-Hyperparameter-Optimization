from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from hpo_curvature import QuadHPO
from hpo_adaptive import QuadHPO as QuadHPO_Adaptive

try:
    import optuna

    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.datasets import make_classification
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def format_hparams(fun_name: str, x_norm: np.ndarray) -> str:
    """Return a concise, human-readable string with x_norm and mapped hyperparams."""
    x_norm = np.asarray(x_norm, dtype=float)
    try:
        x_str = np.array2string(x_norm, precision=3, separator=",")
    except Exception:
        x_str = str(list(map(float, x_norm)))

    if fun_name == "xgboost_tabular":
        if len(x_norm) >= 20:
            # Use the shared mapping to avoid duplication.
            cfg = build_xgb_config(x_norm=x_norm, use_gpu=False, seed=0)
            return (
                f"x={x_str} | scaler_code={cfg.scaler_code:.3f}, pca={cfg.pca_apply}, "
                f"pca_ratio={cfg.pca_ratio:.2f}, feat_sel_code={cfg.feat_sel_code:.3f}, "
                f"feat_sel_ratio={cfg.feat_sel_ratio:.2f} | "
                f"n_estimators={cfg.n_estimators}, max_depth={cfg.max_depth}, lr={cfg.learning_rate:.3e}, "
                f"subsample={cfg.subsample:.2f}, colsample_bytree={cfg.colsample_bytree:.2f}, "
                f"min_child_weight={cfg.min_child_weight:.2f}, gamma={cfg.gamma:.2f}, "
                f"alpha={cfg.reg_alpha:.2e}, lambda={cfg.reg_lambda:.2e}, "
                f"max_delta_step={cfg.max_delta_step:.2f}, max_bin={cfg.max_bin}"
            )

    return f"x={x_str}"


# ----------------------------- tabular dataset ------------------------------

_TABULAR_CACHE: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {
    "train": None,
    "val": None,
    "test": None,
}


def get_tabular_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_val, y_val, X_test, y_test), creating once."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for the XGBoost tabular benchmark.")

    if _TABULAR_CACHE["train"] is None:
        # High-dimensional but still reasonably fast synthetic classification task.
        X, y = make_classification(
            n_samples=10000,
            n_features=100,
            n_informative=40,
            n_redundant=20,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=2,
            weights=None,
            flip_y=0.01,
            class_sep=1.0,
            random_state=42,
        )

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.25, random_state=321, stratify=y_trainval
        )

        _TABULAR_CACHE["train"] = (X_train, y_train)
        _TABULAR_CACHE["val"] = (X_val, y_val)
        _TABULAR_CACHE["test"] = (X_test, y_test)

    X_train, y_train = _TABULAR_CACHE["train"]  # type: ignore[assignment]
    X_val, y_val = _TABULAR_CACHE["val"]  # type: ignore[assignment]
    X_test, y_test = _TABULAR_CACHE["test"]  # type: ignore[assignment]
    return X_train, y_train, X_val, y_val, X_test, y_test


@dataclass
class XGBConfig:
    scaler_code: float
    pca_apply: bool
    pca_ratio: float
    pca_whiten: bool
    feat_sel_code: float
    feat_sel_ratio: float
    var_threshold: float
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    colsample_bylevel: float
    colsample_bynode: float
    min_child_weight: float
    gamma: float
    reg_alpha: float
    reg_lambda: float
    max_delta_step: float
    max_bin: int
    tree_method: str
    device: Optional[str]
    seed: int


def build_xgb_config(x_norm: np.ndarray, use_gpu: bool, seed: int) -> XGBConfig:
    """Map normalized vector x_norm to a concrete XGBConfig."""
    x_norm = np.asarray(x_norm, dtype=float)
    if x_norm.shape[0] < 20:
        raise ValueError(f"XGBoost config expects at least 20 dimensions, got {x_norm.shape[0]}")

    scaler_code = float(x_norm[0])
    pca_apply = bool(x_norm[1] > 0.5)
    pca_ratio = 0.1 + 0.8 * float(x_norm[2])
    pca_whiten = bool(x_norm[3] > 0.5)

    feat_sel_code = float(x_norm[4])
    feat_sel_ratio = 0.1 + 0.9 * float(x_norm[5])
    var_threshold = 0.0 + 0.2 * float(x_norm[6])

    n_estimators = int(round(100 + float(x_norm[7]) * 700))
    max_depth = int(round(3 + float(x_norm[8]) * 9))

    lr_log_min, lr_log_max = np.log10(0.01), np.log10(0.3)
    lr_log = lr_log_min + float(x_norm[9]) * (lr_log_max - lr_log_min)
    learning_rate = float(10**lr_log)

    subsample = float(0.5 + 0.5 * float(x_norm[10]))
    colsample_bytree = float(0.5 + 0.5 * float(x_norm[11]))
    colsample_bylevel = float(0.5 + 0.5 * float(x_norm[12]))
    colsample_bynode = float(0.5 + 0.5 * float(x_norm[13]))

    min_child_weight = float(1.0 + 19.0 * float(x_norm[14]))
    gamma = float(10.0 * float(x_norm[15]))

    reg_alpha = float(10 ** (-8.0 + float(x_norm[16]) * 10.0))
    reg_lambda = float(10 ** (-8.0 + float(x_norm[17]) * 10.0))
    max_delta_step = float(10.0 * float(x_norm[18]))
    max_bin = int(round(128 + float(x_norm[19]) * 384))

    # tree_method handling adapted to XGBoost>=3.1.1 in this env:
    # - this build does not accept 'gpu_hist' as tree_method
    # - GPU is enabled instead via device='cuda' when available
    # We keep 'hist' which is valid both for CPU and GPU devices.
    tree_method = "hist"
    device: Optional[str] = "cuda" if use_gpu else None

    return XGBConfig(
        scaler_code=scaler_code,
        pca_apply=pca_apply,
        pca_ratio=pca_ratio,
        pca_whiten=pca_whiten,
        feat_sel_code=feat_sel_code,
        feat_sel_ratio=feat_sel_ratio,
        var_threshold=var_threshold,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        colsample_bynode=colsample_bynode,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        max_delta_step=max_delta_step,
        max_bin=max_bin,
        tree_method=tree_method,
        device=device,
        seed=seed,
    )


def build_xgb_pipeline(config: XGBConfig, n_features: int) -> Pipeline:
    """Build the sklearn Pipeline (preprocessing + XGBClassifier) from a config."""
    steps: List[Tuple[str, Any]] = []

    # Scaling
    if config.scaler_code < 1.0 / 3.0:
        scaler = None
    elif config.scaler_code < 2.0 / 3.0:
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    if scaler is not None:
        steps.append(("scaler", scaler))

    # PCA
    if config.pca_apply:
        n_components = max(2, int(round(config.pca_ratio * n_features)))
        pca = PCA(n_components=n_components, whiten=config.pca_whiten, random_state=config.seed)
        steps.append(("pca", pca))
        n_after_pca = n_components
    else:
        n_after_pca = n_features

    # Feature selection
    if config.feat_sel_code < 1.0 / 3.0:
        selector = None
    elif config.feat_sel_code < 2.0 / 3.0:
        k = max(2, int(round(config.feat_sel_ratio * n_after_pca)))
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = VarianceThreshold(threshold=config.var_threshold)

    if selector is not None:
        steps.append(("feature_sel", selector))

    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        colsample_bylevel=config.colsample_bylevel,
        colsample_bynode=config.colsample_bynode,
        min_child_weight=config.min_child_weight,
        gamma=config.gamma,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        max_delta_step=config.max_delta_step,
        max_bin=config.max_bin,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=config.tree_method,
        n_jobs=4,
        random_state=config.seed,
        verbosity=0,
    )

    steps.append(("xgb", model))
    return Pipeline(steps)


# ------------------------- XGBoost tabular benchmark ------------------------

def xgboost_tabular(x_norm: np.ndarray, use_gpu: bool, trial_seed: int = 97) -> Dict[str, float]:
    """High-dimensional XGBoost benchmark on tabular data with ~20 hyperparameters.

    Hyperparameters include both model and preprocessing:
    - Scaling: none / StandardScaler / RobustScaler
    - PCA: on/off, n_components ratio, whiten
    - Feature selection: none / SelectKBest / VarianceThreshold
    - XGBoost core hparams: n_estimators, max_depth, learning_rate, subsample,
      colsample_bytree / bylevel / bynode, min_child_weight, gamma, reg_alpha,
      reg_lambda, max_delta_step, max_bin.
    """
    if not (XGBOOST_AVAILABLE and SKLEARN_AVAILABLE):
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    X_train, y_train, X_val, y_val, _, _ = get_tabular_data()
    n_features = X_train.shape[1]
    cfg = build_xgb_config(x_norm=x_norm, use_gpu=use_gpu, seed=trial_seed)
    pipeline = build_xgb_pipeline(cfg, n_features=n_features)

    # ------------------------------ training eval -----------------------------
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    elapsed = time.time() - t0

    accuracy = float(accuracy_score(y_val, y_pred))
    precision = float(precision_score(y_val, y_pred, average="binary", zero_division=0))
    recall = float(recall_score(y_val, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_val, y_pred, average="binary", zero_division=0))

    _ = elapsed  # kept to mirror other benchmarks; not returned but useful when logging.

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


ML_FUNS: Dict[str, Tuple[Callable[[np.ndarray, bool, int], Dict[str, float]], int]] = {
    "xgboost_tabular": (xgboost_tabular, 20),
}


def evaluate_on_test_set(x_norm: np.ndarray, use_gpu: bool, seed: int = 99999) -> Dict[str, float]:
    """Evaluate the given hyperparameters on the held-out test set."""
    if not (XGBOOST_AVAILABLE and SKLEARN_AVAILABLE):
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    X_train, y_train, X_val, y_val, X_test, y_test = get_tabular_data()
    n_features = X_train.shape[1]
    cfg = build_xgb_config(x_norm=x_norm, use_gpu=use_gpu, seed=seed)
    pipeline = build_xgb_pipeline(cfg, n_features=n_features)

    # Retrain on train+val and evaluate on test
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    pipeline.fit(X_train_full, y_train_full)
    y_pred = pipeline.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="binary", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="binary", zero_division=0))

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# -------------------------- debugging utilities -----------------------------


def debug_max_depth_effect(logger: Optional[Callable[[str], None]] = None) -> None:
    """Debug helper to inspect how max_depth affects the trained model."""
    def log(msg: str) -> None:
        if logger is not None:
            logger(msg)
        else:
            print(msg)
    
    if not (XGBOOST_AVAILABLE and SKLEARN_AVAILABLE):
        log("XGBoost and scikit-learn are required for --debug-max-depth.")
        return

    X_train, y_train, X_val, y_val, _, _ = get_tabular_data()
    n_features = X_train.shape[1]

    # Fixed normalized vector used as a base point in hyperparameter space.
    x_norm = np.full(20, 0.5, dtype=float)

    log("=" * 78)
    log("DEBUG: Effect of max_depth on XGBoost trees (train on train, eval on val)")
    log("=" * 78)

    for depth in [1, 2, 3, 4, 5]:
        cfg = build_xgb_config(x_norm=x_norm, use_gpu=False, seed=0)
        cfg.max_depth = depth

        pipeline = build_xgb_pipeline(cfg, n_features=n_features)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        val_acc = float(accuracy_score(y_val, y_pred))

        model = pipeline.named_steps["xgb"]
        booster = model.get_booster()
        dump = booster.get_dump()
        n_trees = len(dump)

        if dump:
            first_tree_lines = "\n".join(dump[0].splitlines()[:10])
        else:
            first_tree_lines = "<empty booster dump>"

        log(f"max_depth={depth}, val_acc={val_acc:.4f}, n_trees={n_trees}")
        log("booster_dump_first_tree:")
        log(first_tree_lines)
        log("-" * 78)


# ------------------------------ optimization --------------------------------


def run_optimizer(
    optimizer: str,
    fun_name: str,
    seed: int,
    budget: int,
    use_gpu: bool,
    verbose: bool = False,
    trial_logger: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    func, dim = ML_FUNS[fun_name]
    bounds = [(0.0, 1.0)] * dim
    maximize = True

    def objective_wrapper(x_norm: np.ndarray) -> Dict[str, float]:
        # Use a fixed seed for all trials for full determinism
        return func(x_norm, use_gpu, seed)

    if optimizer == "curvature":
        hpo = QuadHPO(bounds=bounds, maximize=maximize, rng_seed=seed)
        # Track best metrics
        best_metrics = {"accuracy": -np.inf, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        best_x_norm: Optional[np.ndarray] = None

        def wrapped_objective(x_norm: np.ndarray, epochs: int = 1) -> float:
            """Objective wrapper called by QuadHPO for each trial."""

            nonlocal best_metrics, best_x_norm
            t0 = time.time()
            metrics = objective_wrapper(x_norm)
            elapsed = time.time() - t0
            # Update best
            if metrics["accuracy"] > best_metrics["accuracy"]:
                best_metrics = metrics.copy()
                best_x_norm = np.array(x_norm, dtype=float)

            try:
                trial_idx = int(hpo.trial_id)
            except Exception:
                trial_idx = -1

            if verbose:
                hp_desc = format_hparams(fun_name, x_norm)
                msg = (
                    f"{fun_name:20s} | seed {seed:2d} | curv trial {trial_idx:3d} | "
                    f"acc: {metrics['accuracy']:.4f} | prec: {metrics['precision']:.4f} | "
                    f"rec: {metrics['recall']:.4f} | f1: {metrics['f1']:.4f} | t: {elapsed:.1f}s | {hp_desc}"
                )
                if trial_logger is not None:
                    trial_logger(msg)
                else:
                    print(msg)
            return metrics["accuracy"]

        hpo.optimize(wrapped_objective, budget=budget)
        return best_metrics, best_x_norm

    elif optimizer == "adaptive":
        hpo = QuadHPO_Adaptive(bounds=bounds, maximize=maximize, rng_seed=seed)
        # Track best metrics
        best_metrics = {"accuracy": -np.inf, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        best_x_norm: Optional[np.ndarray] = None

        def wrapped_objective(x_norm: np.ndarray, epochs: int = 1) -> float:
            """Objective wrapper called by QuadHPO for each trial."""

            nonlocal best_metrics, best_x_norm
            t0 = time.time()
            metrics = objective_wrapper(x_norm)
            elapsed = time.time() - t0
            # Update best
            if metrics["accuracy"] > best_metrics["accuracy"]:
                best_metrics = metrics.copy()
                best_x_norm = np.array(x_norm, dtype=float)

            try:
                trial_idx = int(hpo.trial_id)
            except Exception:
                trial_idx = -1

            if verbose:
                hp_desc = format_hparams(fun_name, x_norm)
                msg = (
                    f"{fun_name:20s} | seed {seed:2d} | adapt trial {trial_idx:3d} | "
                    f"acc: {metrics['accuracy']:.4f} | prec: {metrics['precision']:.4f} | "
                    f"rec: {metrics['recall']:.4f} | f1: {metrics['f1']:.4f} | t: {elapsed:.1f}s | {hp_desc}"
                )
                if trial_logger is not None:
                    trial_logger(msg)
                else:
                    print(msg)
            return metrics["accuracy"]

        hpo.optimize(wrapped_objective, budget=budget)
        return best_metrics, best_x_norm

    elif optimizer == "optuna":
        if not OPTUNA_AVAILABLE:
            return (
                {
                    "accuracy": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                },
                None,
            )

        best_metrics = {"accuracy": -np.inf, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        best_x_norm: Optional[np.ndarray] = None

        def objective_optuna(trial: optuna.trial.Trial) -> float:
            nonlocal best_metrics, best_x_norm
            x_norm = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)])
            t0 = time.time()
            metrics = objective_wrapper(x_norm)
            elapsed = time.time() - t0
            if metrics["accuracy"] > best_metrics["accuracy"]:
                best_metrics = metrics.copy()
                best_x_norm = np.array(x_norm, dtype=float)
            if verbose:
                hp_desc = format_hparams(fun_name, x_norm)
                msg = (
                    f"{fun_name:20s} | seed {seed:2d} | optuna trial {trial.number:3d} | "
                    f"acc: {metrics['accuracy']:.4f} | prec: {metrics['precision']:.4f} | "
                    f"rec: {metrics['recall']:.4f} | f1: {metrics['f1']:.4f} | t: {elapsed:.1f}s | {hp_desc}"
                )
                if trial_logger is not None:
                    trial_logger(msg)
                else:
                    print(msg)
            return metrics["accuracy"]

        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="maximize" if maximize else "minimize", sampler=sampler)
        study.optimize(objective_optuna, n_trials=budget, show_progress_bar=False)
        return best_metrics, best_x_norm

    elif optimizer == "random":
        rng = np.random.default_rng(seed)
        best_metrics = {"accuracy": -np.inf, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        best_x_norm: Optional[np.ndarray] = None

        for i in range(budget):
            x_norm = rng.random(dim)
            t0 = time.time()
            metrics = objective_wrapper(x_norm)
            elapsed = time.time() - t0
            if metrics["accuracy"] > best_metrics["accuracy"]:
                best_metrics = metrics.copy()
                best_x_norm = np.array(x_norm, dtype=float)
            if verbose:
                hp_desc = format_hparams(fun_name, x_norm)
                msg = (
                    f"{fun_name:20s} | seed {seed:2d} | rand trial {i+1:3d} | "
                    f"acc: {metrics['accuracy']:.4f} | prec: {metrics['precision']:.4f} | "
                    f"rec: {metrics['recall']:.4f} | f1: {metrics['f1']:.4f} | t: {elapsed:.1f}s | {hp_desc}"
                )
            else:
                msg = ""
            if verbose:
                if trial_logger is not None:
                    trial_logger(msg)
                else:
                    print(msg)

        return best_metrics, best_x_norm

    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


# ----------------------------------- main -----------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost Tabular Benchmark: Curvature vs Optuna vs Random"
    )
    parser.add_argument("--budget", type=int, default=20, help="Trials per method per seed")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated list of seeds")
    parser.add_argument(
        "--functions",
        type=str,
        default="xgboost_tabular",
        help="Comma-separated function names",
    )
    parser.add_argument(
        "--methods", type=str, default="curvature,adaptive,optuna,random", help="Which methods to run"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU-backed XGBoost if available")
    parser.add_argument("--verbose", action="store_true", help="Print per-seed details")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: tests/benchmark_results_TIMESTAMP.txt)",
    )
    parser.add_argument(
        "--test-topk",
        type=int,
        default=10,
        help="Evaluate top-K validation winners on the true test set",
    )
    parser.add_argument(
        "--debug-max-depth",
        action="store_true",
        help="Run a small diagnostic to see how max_depth changes trees",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    names = [n for n in args.functions.split(",") if n in ML_FUNS]
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    if not names:
        raise ValueError(f"No valid function names in --functions={args.functions}")

    # Setup output file
    if args.output is None:
        os.makedirs("tests", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"tests/benchmark_results_{timestamp}.txt"
    else:
        output_file = args.output
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)


    def print_and_log(msg: str, file_handle):
        print(msg)
        file_handle.write(msg + "\n")
        file_handle.flush()

    print("=" * 78)
    print("XGBOOST TABULAR BENCHMARK: Curvature vs Optuna vs Random")
    print("=" * 78)
    print(f"Seeds: {seeds}")
    print(f"Budget per method: {args.budget}")
    print(f"GPU requested: {args.gpu}")
    print(f"Functions: {', '.join(names)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Output file: {output_file}")
    print("=" * 78)

    # Open output file and write header
    with open(output_file, "w", newline="") as f:
        # Handle debug mode inside the with block so output goes to file
        if args.debug_max_depth:
            debug_max_depth_effect(logger=lambda msg: print_and_log(msg, f))
            return
        
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "fun",
                "method",
                "seed",
                "best_acc",
                "best_prec",
                "best_rec",
                "best_f1",
                "best_x_norm",
                "test_acc",
                "test_prec",
                "test_rec",
                "test_f1",
            ]
        )

        def trial_logger(msg: str) -> None:
            print_and_log(msg, f)

        # Collect per-(fun, method) summaries in memory
        summary_rows: List[Tuple[str, str, float, float, float, float, float, float, float, float]] = []

        for fun_name in names:
            for method in methods:
                for seed in seeds:
                    print_and_log(
                        f"Running {fun_name} with {method} (seed={seed})", f
                    )
                    best_metrics, best_x_norm = run_optimizer(
                        optimizer=method,
                        fun_name=fun_name,
                        seed=seed,
                        budget=args.budget,
                        use_gpu=args.gpu,
                        verbose=args.verbose,
                        trial_logger=trial_logger,
                    )

                    if best_x_norm is not None:
                        test_metrics = evaluate_on_test_set(
                            best_x_norm, args.gpu, seed=seed
                        )
                    else:
                        test_metrics = {
                            "accuracy": float("nan"),
                            "precision": float("nan"),
                            "recall": float("nan"),
                            "f1": float("nan"),
                        }

                    writer.writerow(
                        [
                            fun_name,
                            method,
                            seed,
                            best_metrics["accuracy"],
                            best_metrics["precision"],
                            best_metrics["recall"],
                            best_metrics["f1"],
                            None if best_x_norm is None else best_x_norm.tolist(),
                            test_metrics["accuracy"],
                            test_metrics["precision"],
                            test_metrics["recall"],
                            test_metrics["f1"],
                        ]
                    )

                    summary_rows.append(
                        (
                            fun_name,
                            method,
                            float(best_metrics["accuracy"]),
                            float(best_metrics["precision"]),
                            float(best_metrics["recall"]),
                            float(best_metrics["f1"]),
                            float(test_metrics["accuracy"]),
                            float(test_metrics["precision"]),
                            float(test_metrics["recall"]),
                            float(test_metrics["f1"]),
                        )
                    )

        # Write a formatted summary at the end of the file
        print_and_log("", f)
        print_and_log("=" * 120, f)
        print_and_log("BENCHMARK SUMMARY (aggregated over seeds)", f)
        print_and_log("=" * 120, f)
        
        # Aggregate by (fun, method)
        summary_dict: Dict[Tuple[str, str], List[Tuple[float, float, float, float, float, float, float, float]]] = {}
        for row in summary_rows:
            key = (row[0], row[1])
            vals = tuple(row[2:])  # type: ignore[assignment]
            summary_dict.setdefault(key, []).append(vals)  # type: ignore[arg-type]

        # Print header
        header = (
            f"{'Function':<20} | {'Method':<12} | "
            f"{'Best Acc':>8} {'Best Prec':>9} {'Best Rec':>8} {'Best F1':>8} | "
            f"{'Test Acc':>8} {'Test Prec':>9} {'Test Rec':>8} {'Test F1':>8}"
        )
        print_and_log(header, f)
        print_and_log("-" * 120, f)
        
        for (fun_name, method), vals_list in summary_dict.items():
            arr = np.array(vals_list, dtype=float)  # shape (n_seeds, 8)
            means = arr.mean(axis=0)
            row_str = (
                f"{fun_name:<20} | {method:<12} | "
                f"{means[0]:>8.4f} {means[1]:>9.4f} {means[2]:>8.4f} {means[3]:>8.4f} | "
                f"{means[4]:>8.4f} {means[5]:>9.4f} {means[6]:>8.4f} {means[7]:>8.4f}"
            )
            print_and_log(row_str, f)
        
        print_and_log("=" * 120, f)


if __name__ == "__main__":
    main()
