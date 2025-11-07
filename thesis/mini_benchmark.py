from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Optional dependencies
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
    XGB_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    XGB_AVAILABLE = False


# ------------------------------- utilities ---------------------------------

def map_to_domain(x_norm: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    x_norm = np.asarray(x_norm, dtype=float)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + x_norm * (hi - lo)


# --------------------------- synthetic functions ---------------------------

def sphere5(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def rastrigin5(x: np.ndarray) -> float:
    d = x.size
    return float(10.0 * d + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


SYN_FUNS: Dict[str, Tuple[Callable[[np.ndarray], float], List[Tuple[float, float]]]] = {
    'sphere5': (sphere5, [(-5.0, 5.0)] * 5),
    'rastrigin5': (rastrigin5, [(-5.12, 5.12)] * 5),
}


# ------------------------------ ML benchmark -------------------------------

@dataclass
class MLConfig:
    test_size: float = 0.2
    random_state: int = 42


def xgb_iris(x_norm: np.ndarray, use_gpu: bool, cfg: Optional[MLConfig] = None) -> float:
    """Small XGBoost benchmark on Iris (multiclass). Returns accuracy (maximize)."""
    if not (SKLEARN_AVAILABLE and XGB_AVAILABLE):
        return 0.0
    cfg = cfg or MLConfig()

    bounds = [
        (2, 8),        # max_depth
        (0.01, 0.3),   # learning_rate
        (20, 150),     # n_estimators
        (0.6, 1.0),    # subsample
    ]
    hp = map_to_domain(x_norm, bounds)

    data = load_iris()
    Xtr, Xte, ytr, yte = train_test_split(
        data.data, data.target, test_size=cfg.test_size, random_state=cfg.random_state
    )

    gpu_kwargs: Dict[str, Any] = {}
    if use_gpu:
        # XGBoost >= 2.0 uses device='cuda'. If not supported, it will raise TypeError
        gpu_kwargs = {'device': 'cuda'}

    try:
        model = xgb.XGBClassifier(
            max_depth=int(hp[0]),
            learning_rate=float(hp[1]),
            n_estimators=int(hp[2]),
            subsample=float(hp[3]),
            eval_metric='mlogloss',
            verbosity=0,
            use_label_encoder=False,
            random_state=cfg.random_state,
            **gpu_kwargs,
        )
    except TypeError:
        # Fallback: CPU hist
        model = xgb.XGBClassifier(
            max_depth=int(hp[0]),
            learning_rate=float(hp[1]),
            n_estimators=int(hp[2]),
            subsample=float(hp[3]),
            eval_metric='mlogloss',
            verbosity=0,
            use_label_encoder=False,
            random_state=cfg.random_state,
            tree_method='hist',
        )

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return float(accuracy_score(yte, pred))


ML_FUNS: Dict[str, Tuple[Callable[[np.ndarray, bool], float], int]] = {
    'xgb_iris': (lambda x, g: xgb_iris(x, g), 4),
}


# ------------------------------ optimization --------------------------------

def run_random(fun_name: str, seed: int, budget: int, use_gpu: bool) -> float:
    rng = np.random.default_rng(seed)
    if fun_name in ML_FUNS:
        func, dim = ML_FUNS[fun_name]
        d = dim
        maximize = True
    else:
        func, bounds = SYN_FUNS[fun_name]
        d = len(bounds)
        maximize = False

    best = -np.inf if maximize else np.inf
    for _ in range(budget):
        x_norm = rng.random(d)
        if fun_name in ML_FUNS:
            val = float(func(x_norm, use_gpu))
        else:
            x = map_to_domain(x_norm, bounds)
            val = float(func(x))
        if maximize:
            best = max(best, val)
        else:
            best = min(best, val)
    return float(best)


def run_optuna(fun_name: str, seed: int, budget: int, use_gpu: bool) -> float:
    if not OPTUNA_AVAILABLE:
        return float('nan')
    if fun_name in ML_FUNS:
        func, dim = ML_FUNS[fun_name]
        d = dim
        maximize = True
    else:
        func, bounds = SYN_FUNS[fun_name]
        d = len(bounds)
        maximize = False

    def objective(trial: optuna.trial.Trial) -> float:  # type: ignore
        x_norm = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(d)], dtype=float)
        if fun_name in ML_FUNS:
            return func(x_norm, use_gpu)
        else:
            x = map_to_domain(x_norm, bounds)
            return func(x)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize' if maximize else 'minimize', sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return float(study.best_value)


def run_curvature(fun_name: str, seed: int, budget: int, use_gpu: bool) -> float:
    if fun_name in ML_FUNS:
        func, dim = ML_FUNS[fun_name]
        bounds = [(0.0, 1.0)] * dim
        maximize = True
    else:
        func, bounds = SYN_FUNS[fun_name]
        maximize = False

    hpo = QuadHPO(bounds=[(0.0, 1.0)] * len(bounds), maximize=maximize, rng_seed=seed)

    def objective(x_norm: np.ndarray, epochs: int = 1) -> float:
        if fun_name in ML_FUNS:
            return float(func(x_norm, use_gpu))
        else:
            x = map_to_domain(x_norm, bounds)
            return float(func(x))

    hpo.optimize(objective, budget=budget)
    return float(hpo.sign * hpo.best_score_global)


# ----------------------------------- main -----------------------------------

def main():
    parser = argparse.ArgumentParser(description='Mini benchmark: Curvature vs Optuna vs Random')
    parser.add_argument('--budget', type=int, default=40, help='Trials per method per seed (small & quick)')
    parser.add_argument('--seeds', type=str, default='0,1', help='Comma-separated list of seeds')
    parser.add_argument('--functions', type=str, default='sphere5,rastrigin5,xgb_iris', help='Comma-separated function names')
    parser.add_argument('--methods', type=str, default='curv,optuna,random', help='Which methods to run')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for ML (XGBoost) if available')
    parser.add_argument('--verbose', action='store_true', help='Print per-seed details')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    names = [n for n in args.functions.split(',') if n in SYN_FUNS or n in ML_FUNS]
    methods = [m.strip().lower() for m in args.methods.split(',') if m.strip()]
    do_curv = 'curv' in methods or 'curvature' in methods
    do_opt = 'optuna' in methods or 'opt' in methods
    do_rand = 'random' in methods or 'rand' in methods

    # Header
    print('=' * 78)
    print('MINI BENCHMARK: Curvature vs Optuna vs Random (synthetic + small ML)')
    print('=' * 78)
    print(f'Seeds: {seeds}')
    print(f'Budget per method: {args.budget}')
    print(f'GPU requested: {args.gpu}')
    print(f'Functions: {", ".join(names)}')
    print(f'Methods: {", ".join(methods)}')
    print('=' * 78)

    if do_opt and not OPTUNA_AVAILABLE:
        print('ERROR: Optuna requested but not available. Install with: pip install optuna')
        return
    if any(n in ML_FUNS for n in names) and not (SKLEARN_AVAILABLE and XGB_AVAILABLE):
        print('ERROR: ML functions requested but scikit-learn/xgboost not available.')
        return

    results: Dict[str, Dict[str, List[float]]] = {}

    # Table header
    headers = ['Function']
    widths = [max(max((len(n) for n in names), default=8), len('Function'))]

    def add_cols(pfx: str):
        headers.extend([f'{pfx}_mean', f'{pfx}_std'])
        widths.extend([14, 12])

    if do_curv:
        add_cols('Curv')
    if do_opt:
        add_cols('Optuna')
    if do_rand:
        add_cols('Random')
    headers.append('W/T')  # Wins (strict) / Ties against best baseline
    widths.append(10)

    line = [f"{headers[0]:<{widths[0]}}"]
    for i in range(1, len(headers) - 1):
        line.append(f"{headers[i]:>{widths[i]}}")
    line.append(f"{headers[-1]:>{widths[-1]}}")
    print(' | '.join(line))
    print('-' * (sum(widths) + 3 * (len(widths) - 1)))

    for name in names:
        results[name] = {'curv': [], 'optuna': [], 'random': []}
        maximize = name in ML_FUNS
        for seed in seeds:
            t0 = time.time()
            if do_curv:
                v = run_curvature(name, seed, args.budget, args.gpu)
                results[name]['curv'].append(v)
            if do_opt:
                v = run_optuna(name, seed, args.budget, args.gpu)
                results[name]['optuna'].append(v)
            if do_rand:
                v = run_random(name, seed, args.budget, args.gpu)
                results[name]['random'].append(v)
            if args.verbose:
                parts = [f"{name:18s}", f"seed {seed:2d}"]
                if do_curv:
                    parts.append(f"Curv: {results[name]['curv'][-1]:.6f}")
                if do_opt:
                    parts.append(f"Optuna: {results[name]['optuna'][-1]:.6f}")
                if do_rand:
                    parts.append(f"Random: {results[name]['random'][-1]:.6f}")
                parts.append(f"(t={time.time()-t0:.1f}s)")
                print(' | '.join(parts))

        # Stats per function
        cm = np.array(results[name]['curv'], dtype=float)
        om = np.array(results[name]['optuna'], dtype=float) if do_opt else np.array([])
        rm = np.array(results[name]['random'], dtype=float) if do_rand else np.array([])

        wins = 0
        ties = 0
        for i in range(len(seeds)):
            c = cm[i] if i < cm.size else np.nan
            baselines = []
            if do_opt and i < om.size:
                baselines.append(om[i])
            if do_rand and i < rm.size:
                baselines.append(rm[i])
            if not baselines or np.isnan(c):
                continue
            best_base = max(baselines) if maximize else min(baselines)
            # Strict win excludes ties; count ties separately for transparency.
            if maximize:
                if c > best_base:
                    wins += 1
                elif c == best_base:
                    ties += 1
            else:
                if c < best_base:
                    wins += 1
                elif c == best_base:
                    ties += 1

        # Print row
        cells = [f"{name:<{widths[0]}}"]
        idx = 1
        if do_curv:
            cells.append(f"{(cm.mean() if cm.size else np.nan):>{widths[idx]}.8g}"); idx += 1
            cells.append(f"{(cm.std() if cm.size else np.nan):>{widths[idx]}.8g}"); idx += 1
        if do_opt:
            cells.append(f"{(om.mean() if om.size else np.nan):>{widths[idx]}.8g}"); idx += 1
            cells.append(f"{(om.std() if om.size else np.nan):>{widths[idx]}.8g}"); idx += 1
        if do_rand:
            cells.append(f"{(rm.mean() if rm.size else np.nan):>{widths[idx]}.8g}"); idx += 1
            cells.append(f"{(rm.std() if rm.size else np.nan):>{widths[idx]}.8g}"); idx += 1
        cells.append(f"{wins}/{ties}")
        print(' | '.join(cells))

    print('=' * 78)


if __name__ == '__main__':
    main()
