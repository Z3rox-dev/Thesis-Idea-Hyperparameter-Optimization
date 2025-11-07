from __future__ import annotations

import sys
import math
import time
from typing import List, Tuple, Dict, Callable
import argparse

import numpy as np

# Local imports
sys.path.insert(0, '/root/thesis')

# Try to import optuna from local repo if not installed
try:
    import optuna  # type: ignore
except Exception:
    sys.path.insert(0, '/root/optuna')
    import optuna  # type: ignore


def map_to_domain(x_norm: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """Map [0,1]^d to real bounds linearly."""
    x_norm = np.asarray(x_norm, dtype=float)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo + x_norm * (hi - lo)


def rosenbrock(x: np.ndarray) -> float:
    s = 0.0
    for i in range(x.size - 1):
        s += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return float(s)


def rastrigin(x: np.ndarray) -> float:
    d = x.size
    return float(10.0 * d + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def griewank(x: np.ndarray) -> float:
    s = float(np.sum(x * x) / 4000.0)
    p = 1.0
    for i in range(x.size):
        p *= math.cos(x[i] / math.sqrt(i + 1))
    return float(1.0 + s - p)


def schwefel(x: np.ndarray) -> float:
    d = x.size
    return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def levy(x: np.ndarray) -> float:
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    term2 = 0.0
    for i in range(x.size - 1):
        wi = w[i]
        term2 += (wi - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * wi + 1.0) ** 2)
    return float(term1 + term2 + term3)


def ackley(x: np.ndarray) -> float:
    d = x.size
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi
    sum_sq = np.sum(x * x)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return float(a + np.e + term1 + term2)


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def zakharov(x: np.ndarray) -> float:
    i = np.arange(1, x.size + 1, dtype=float)
    term1 = np.sum(x * x)
    term2 = (0.5 * np.sum(i * x)) ** 2
    term3 = (0.5 * np.sum(i * x)) ** 4
    return float(term1 + term2 + term3)


def sum_squares(x: np.ndarray) -> float:
    i = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(i * (x ** 2)))


def ellipsoid(x: np.ndarray) -> float:
    # Weighted sphere (aka SumSquares variant)
    i = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum((10.0 ** ((i - 1) / (x.size - 1 + 1e-12))) * (x ** 2)))


def dixon_price(x: np.ndarray) -> float:
    term1 = (x[0] - 1.0) ** 2
    s = 0.0
    for i in range(1, x.size):
        s += (i + 1) * (2.0 * x[i] ** 2 - x[i - 1]) ** 2
    return float(term1 + s)


def styblinski_tang(x: np.ndarray) -> float:
    return float(0.5 * np.sum(x ** 4 - 16.0 * x ** 2 + 5.0 * x))


def alpine1(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x * np.sin(x) + 0.1 * x)))


def michalewicz(x: np.ndarray, m: float = 10.0) -> float:
    i = np.arange(1, x.size + 1, dtype=float)
    return float(-np.sum(np.sin(x) * (np.sin(i * x ** 2 / np.pi) ** (2.0 * m))))


def schwefel_222(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))


def schwefel_12(x: np.ndarray) -> float:
    s = 0.0
    cumsum = 0.0
    for i in range(x.size):
        cumsum += x[i]
        s += cumsum ** 2
    return float(s)


def salomon(x: np.ndarray) -> float:
    sqrt_sum = np.sqrt(np.sum(x ** 2))
    return float(1.0 - np.cos(2.0 * np.pi * sqrt_sum) + 0.1 * sqrt_sum)


def bent_cigar(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2))


def bohachevsky1(x: np.ndarray) -> float:
    s = 0.0
    for i in range(x.size - 1):
        xi, xj = x[i], x[i + 1]
        s += xi ** 2 + 2.0 * xj ** 2 - 0.3 * np.cos(3.0 * np.pi * xi) - 0.4 * np.cos(4.0 * np.pi * xj) + 0.7
    return float(s)


def bohachevsky2(x: np.ndarray) -> float:
    s = 0.0
    for i in range(x.size - 1):
        xi, xj = x[i], x[i + 1]
        s += xi ** 2 + 2.0 * xj ** 2 - 0.3 * np.cos(3.0 * np.pi * xi) * np.cos(4.0 * np.pi * xj) + 0.3
    return float(s)


def bohachevsky3(x: np.ndarray) -> float:
    s = 0.0
    for i in range(x.size - 1):
        xi, xj = x[i], x[i + 1]
        s += xi ** 2 + 2.0 * xj ** 2 - 0.3 * np.cos(3.0 * np.pi * xi + 4.0 * np.pi * xj) + 0.3
    return float(s)


def discus(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(1e6 * x[0] ** 2 + np.sum(x[1:] ** 2))


def quartic(x: np.ndarray) -> float:
    i = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(i * (x ** 4)))


def step(x: np.ndarray) -> float:
    return float(np.sum(np.floor(x + 0.5) ** 2))


def sum_diff_powers(x: np.ndarray) -> float:
    exps = np.arange(2, x.size + 2, dtype=float)
    return float(np.sum(np.abs(x) ** exps))


def perm_function(x: np.ndarray, beta: float = 10.0) -> float:
    d = x.size
    s = 0.0
    i = np.arange(1, d + 1, dtype=float)
    for k in range(1, d + 1):
        inner = np.sum((i ** k + beta) * (x ** k - 1.0))
        s += inner ** 2
    return float(s)


def trid(x: np.ndarray) -> float:
    s1 = np.sum((x - 1.0) ** 2)
    s2 = np.sum(x[1:] * x[:-1]) if x.size > 1 else 0.0
    return float(s1 - s2)


def powell_singular(x: np.ndarray) -> float:
    # Requires dimension multiple of 4; if not, pad with zeros for last group
    n = x.size
    m = (n + 3) // 4
    total = 0.0
    for g in range(m):
        i0 = 4 * g
        x1 = x[i0] if i0 < n else 0.0
        x2 = x[i0 + 1] if i0 + 1 < n else 0.0
        x3 = x[i0 + 2] if i0 + 2 < n else 0.0
        x4 = x[i0 + 3] if i0 + 3 < n else 0.0
        term1 = (x1 + 10.0 * x2) ** 2
        term2 = 5.0 * (x3 - x4) ** 2
        term3 = (x2 - 2.0 * x3) ** 4
        term4 = 10.0 * (x1 - x4) ** 4
        total += term1 + term2 + term3 + term4
    return float(total)


def _u_penalty(x: np.ndarray, a: float, k: float, m: float) -> float:
    xa = x - a
    xb = -x - a
    pos = np.where(x > a, k * (xa ** m), 0.0)
    neg = np.where(x < -a, k * (xb ** m), 0.0)
    return float(np.sum(pos + neg))


def penalized1(x: np.ndarray) -> float:
    d = x.size
    y = 1.0 + (x + 1.0) / 4.0
    term = 10.0 * np.sin(np.pi * y[0]) ** 2
    term += np.sum((y[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * y[1:]) ** 2))
    term += (y[-1] - 1.0) ** 2
    term *= (np.pi / d)
    pen = _u_penalty(x, a=10.0, k=100.0, m=4.0)
    return float(term + pen)


def penalized2(x: np.ndarray) -> float:
    d = x.size
    term = np.sin(3.0 * np.pi * x[0]) ** 2
    term += np.sum((x[:-1] - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x[1:]) ** 2))
    term += (x[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x[-1]) ** 2)
    term *= 0.1
    pen = _u_penalty(x, a=5.0, k=100.0, m=4.0)
    return float(term + pen)


def weierstrass(x: np.ndarray) -> float:
    a = 0.5
    b = 3.0
    kmax = 20
    ks = np.arange(0, kmax + 1, dtype=float)
    a_k = a ** ks
    b_k = b ** ks
    # term1: sum_i sum_k a^k cos(2π b^k (x_i + 0.5))
    term1 = np.sum([np.sum(a_k * np.cos(2.0 * np.pi * b_k * (xi + 0.5))) for xi in x])
    # term2: d * sum_k a^k cos(π b^k)
    term2 = x.size * np.sum(a_k * np.cos(np.pi * b_k))
    return float(term1 - term2)


def katsuura(x: np.ndarray) -> float:
    d = x.size
    prod = 1.0
    for i in range(d):
        xi = x[i]
        s = 0.0
        for j in range(1, 33):
            p = 2.0 ** j * xi
            s += abs(p - np.round(p)) / (2.0 ** j)
        prod *= (1.0 + (i + 1) * s) ** (10.0 / d)
    return float(prod - 1.0)


def expanded_schaffer(x: np.ndarray) -> float:
    d = x.size
    s = 0.0
    for i in range(d):
        j = (i + 1) % d
        xi2 = x[i] ** 2
        xj2 = x[j] ** 2
        r = np.sqrt(xi2 + xj2)
        s += 0.5 + (np.sin(r) ** 2 - 0.5) / (1.0 + 0.001 * (xi2 + xj2)) ** 2
    return float(s)


def egg_crate(x: np.ndarray) -> float:
    return float(np.sum(x ** 2 + 25.0 * (np.sin(x) ** 2)))


def qing(x: np.ndarray) -> float:
    i = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum((x ** 2 - i) ** 2))


def chung_reynolds(x: np.ndarray) -> float:
    s = np.sum(x ** 2)
    return float(s ** 2)


def schwefel_221(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def schumer_steiglitz(x: np.ndarray) -> float:
    return float(np.sum(x ** 4))


def alpine2(x: np.ndarray) -> float:
    # domain requires x >= 0
    return float(np.sum(np.sqrt(np.abs(x)) * np.sin(x)))


FUNS: Dict[str, Tuple[Callable[[np.ndarray], float], List[Tuple[float, float]]]] = {
    # name: (function, domain bounds)
    'rosenbrock': (rosenbrock, [(-2.0, 2.0)] * 10),
    'rastrigin': (rastrigin, [(-5.12, 5.12)] * 10),
    'griewank': (griewank, [(-600.0, 600.0)] * 10),
    'schwefel': (schwefel, [(-500.0, 500.0)] * 10),
    'levy': (levy, [(-10.0, 10.0)] * 10),
    'ackley': (ackley, [(-32.768, 32.768)] * 10),
    'sphere': (sphere, [(-5.12, 5.12)] * 10),
    'zakharov': (zakharov, [(-5.0, 10.0)] * 10),
    'sum_squares': (sum_squares, [(-10.0, 10.0)] * 10),
    'ellipsoid': (ellipsoid, [(-5.12, 5.12)] * 10),
    'dixon_price': (dixon_price, [(-10.0, 10.0)] * 10),
    'styblinski_tang': (styblinski_tang, [(-5.0, 5.0)] * 10),
    'alpine1': (alpine1, [(0.0, 10.0)] * 10),
    'michalewicz': (michalewicz, [(0.0, np.pi)] * 10),
    'schwefel_222': (schwefel_222, [(-10.0, 10.0)] * 10),
    'schwefel_12': (schwefel_12, [(-100.0, 100.0)] * 10),
    'salomon': (salomon, [(-100.0, 100.0)] * 10),
    'bent_cigar': (bent_cigar, [(-100.0, 100.0)] * 10),
    'bohachevsky1': (bohachevsky1, [(-100.0, 100.0)] * 10),
    'bohachevsky2': (bohachevsky2, [(-100.0, 100.0)] * 10),
    'bohachevsky3': (bohachevsky3, [(-100.0, 100.0)] * 10),
    'discus': (discus, [(-100.0, 100.0)] * 10),
    'quartic': (quartic, [(-1.28, 1.28)] * 10),
    'step': (step, [(-5.12, 5.12)] * 10),
    'sum_diff_powers': (sum_diff_powers, [(-1.0, 1.0)] * 10),
    'perm': (perm_function, [(-1.0, 1.0)] * 10),
    'trid': (trid, [(-100.0, 100.0)] * 10),
    'powell_singular': (powell_singular, [(-4.0, 5.0)] * 10),
    'penalized1': (penalized1, [(-50.0, 50.0)] * 10),
    'penalized2': (penalized2, [(-50.0, 50.0)] * 10),
    'weierstrass': (weierstrass, [(-0.5, 0.5)] * 10),
    'katsuura': (katsuura, [(-5.0, 5.0)] * 10),
    'expanded_schaffer': (expanded_schaffer, [(-100.0, 100.0)] * 10),
    'egg_crate': (egg_crate, [(-5.0, 5.0)] * 10),
    'qing': (qing, [(-500.0, 500.0)] * 10),
    'chung_reynolds': (chung_reynolds, [(-100.0, 100.0)] * 10),
    'schwefel_221': (schwefel_221, [(-100.0, 100.0)] * 10),
    'schumer_steiglitz': (schumer_steiglitz, [(-100.0, 100.0)] * 10),
    'alpine2': (alpine2, [(0.0, 10.0)] * 10),
}


def run_optuna(fun_name: str, seed: int, budget: int, quiet: bool) -> float:
    func, bounds = FUNS[fun_name]

    def objective(trial: optuna.trial.Trial) -> float:  # type: ignore
        d = len(bounds)
        x_norm = np.array([
            trial.suggest_float(f"x{i}", 0.0, 1.0)
            for i in range(d)
        ], dtype=float)
        x = map_to_domain(x_norm, bounds)
        return func(x)

    if quiet:
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return float(study.best_value)


def run_curvature(fun_name: str, seed: int, budget: int) -> float:
    func, bounds = FUNS[fun_name]
    d = len(bounds)

    # Curvature optimizer lives in normalized space [0,1]^d
    hpo = QuadHPO(bounds=[(0.0, 1.0)] * d, maximize=False, rng_seed=seed)

    def objective(x_norm: np.ndarray, epochs: int = 1):
        x = map_to_domain(x_norm, bounds)
        return float(func(x))

    hpo.optimize(objective, budget=budget)
    best_raw = float(hpo.sign * hpo.best_score_global)
    return best_raw


def run_curvature_vpca(fun_name: str, seed: int, budget: int) -> float:
    func, bounds = FUNS[fun_name]
    d = len(bounds)

    # Virtual-PCA curvature optimizer in normalized space [0,1]^d
    hpo = QuadHPO_vpca(bounds=[(0.0, 1.0)] * d, maximize=False, rng_seed=seed)

    def objective(x_norm: np.ndarray, epochs: int = 1):
        x = map_to_domain(x_norm, bounds)
        return float(func(x))

    hpo.optimize(objective, budget=budget)
    best_raw = float(hpo.sign * hpo.best_score_global)
    return best_raw


def run_qtree_v1(fun_name: str, seed: int, budget: int) -> float:
    """Run the legacy QuadTree v1 framework as a maximizer over -f(x). Returns best original f(x) (minimized)."""
    func, bounds = FUNS[fun_name]
    d = len(bounds)

    hpo = QuadHPO_v1(bounds=[(0.0, 1.0)] * d, maximize=True, rng_seed=seed)

    def objective(x_norm: np.ndarray, epochs: int = 1):
        x = map_to_domain(x_norm, bounds)
        return float(-func(x))  # maximize -f(x) => minimize f(x)

    hpo.optimize(objective, budget=budget)
    best_raw = float(-hpo.best_score_global)
    return best_raw


def main():
    parser = argparse.ArgumentParser(description="Curvature vs Optuna on synthetic functions")
    parser.add_argument("--budget", type=int, default=200, help="Trials per method per seed")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated list of seeds")
    parser.add_argument("--functions", type=str, default=",".join(FUNS.keys()), help="Comma-separated function names to run")
    parser.add_argument("--verbose", action="store_true", help="Print per-seed results as well")
    parser.add_argument("--methods", type=str, default="curv,optuna,qt1", help="Comma-separated methods: curv,curv_vpca,optuna,qt1")
    args = parser.parse_args()

    budget = int(args.budget)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    names = [n for n in args.functions.split(",") if n in FUNS]
    quiet = not args.verbose
    methods = [m.strip().lower() for m in args.methods.split(',') if m.strip()]
    do_curv = 'curv' in methods
    do_curv_vpca = 'curv_vpca' in methods
    do_opt = 'optuna' in methods
    do_qt1 = 'qt1' in methods

    if args.verbose:
        print("Benchmark: Curvature vs Optuna (synthetic only)")
        print(f"seeds={seeds} budget={budget}")
        print("functions:", ", ".join(names))
        print()
    results: Dict[str, Dict[str, List[float]]] = {}
    t0 = time.time()
    # Prepare table formatting
    name_w = max(max((len(n) for n in names), default=0), len("Function"))
    # Build dynamic headers/widths
    headers = ["Function"]
    widths = [name_w]
    def add_cols(prefix: str):
        headers.extend([f"{prefix}_mean", f"{prefix}_std"])
        widths.extend([12, 10])
    if do_curv:
        add_cols("Curv")
    if do_curv_vpca:
        add_cols("CurvVPCA")
    if do_opt:
        add_cols("Opt")
    if do_qt1:
        add_cols("QT1")
    headers.append("Wins")
    widths.append(7)
    # Compose header/separator lines
    parts = [f"{headers[0]:<{widths[0]}}"]
    for i in range(1, len(headers)-1):
        parts.append(f"{headers[i]:>{widths[i]}}")
    parts.append(f"{headers[-1]:>{widths[-1]}}")
    header_line = " | ".join(parts)
    sep_chunks = ["-"*widths[0]] + ["-"*w for w in widths[1:]]
    sep_line = "-+-".join(sep_chunks)
    print(header_line)
    print(sep_line)
    table_rows: List[str] = []
    for name in names:
        results[name] = {"curv": [], "curv_vpca": [], "optuna": [], "qtree1": []}
        for seed in seeds:
            t_fun0 = time.time()
            if do_curv:
                b_curv = run_curvature(name, seed, budget)
                results[name].setdefault("curv", []).append(b_curv)
            if do_curv_vpca:
                b_curv_v = run_curvature_vpca(name, seed, budget)
                results[name].setdefault("curv_vpca", []).append(b_curv_v)
            if do_opt:
                b_opt = run_optuna(name, seed, budget, quiet=quiet)
                results[name].setdefault("optuna", []).append(b_opt)
            if do_qt1:
                b_qt1 = run_qtree_v1(name, seed, budget)
                results[name].setdefault("qtree1", []).append(b_qt1)
            if args.verbose:
                dt = time.time() - t_fun0
                details = [f"{name:12s}", f"seed {seed:2d}"]
                if do_curv:
                    details.append(f"Curv: {b_curv:.6f}")
                if do_curv_vpca:
                    details.append(f"CurvVPCA: {b_curv_v:.6f}")
                if do_opt:
                    details.append(f"Optuna: {b_opt:.6f}")
                if do_qt1:
                    details.append(f"QT1: {b_qt1:.6f}")
                details.append(f"(t={dt:.1f}s)")
                print(" | ".join(details))
        curv_arr = np.array(results[name].get("curv", []), dtype=float)
        curv_v_arr = np.array(results[name].get("curv_vpca", []), dtype=float)
        opt_arr = np.array(results[name].get("optuna", []), dtype=float)
        qt1_arr = np.array(results[name].get("qtree1", []), dtype=float)
        # Wins: Curv vs Opt if available else Curv vs QT1
        if do_curv and do_opt:
            win = int(np.sum(curv_arr <= opt_arr))
        elif do_curv and do_qt1:
            win = int(np.sum(curv_arr <= qt1_arr))
        else:
            win = 0
        # compact general formatting: build row dynamically
        cells = [f"{name:<{widths[0]}}"]
        col_idx = 1
        if do_curv:
            cm, cs = float(curv_arr.mean()), float(curv_arr.std())
            cells.append(f"{cm:>{widths[col_idx]}.6g}"); col_idx += 1
            cells.append(f"{cs:>{widths[col_idx]}.6g}"); col_idx += 1
        if do_curv_vpca:
            vm = float(curv_v_arr.mean()) if curv_v_arr.size > 0 else float('nan')
            vs = float(curv_v_arr.std()) if curv_v_arr.size > 0 else float('nan')
            cells.append(f"{vm:>{widths[col_idx]}.6g}"); col_idx += 1
            cells.append(f"{vs:>{widths[col_idx]}.6g}"); col_idx += 1
        if do_opt:
            om, os = float(opt_arr.mean()), float(opt_arr.std())
            cells.append(f"{om:>{widths[col_idx]}.6g}"); col_idx += 1
            cells.append(f"{os:>{widths[col_idx]}.6g}"); col_idx += 1
        if do_qt1:
            qm, qs = float(qt1_arr.mean()), float(qt1_arr.std())
            cells.append(f"{qm:>{widths[col_idx]}.6g}"); col_idx += 1
            cells.append(f"{qs:>{widths[col_idx]}.6g}"); col_idx += 1
        cells.append(f"{win}/{len(seeds):<{widths[-1]-2}}")
        row = " | ".join(cells)
        print(row)
        table_rows.append(row)
    dt_all = time.time() - t0
    print(f"Total time: {dt_all:.1f}s")



if __name__ == "__main__":
    main()
