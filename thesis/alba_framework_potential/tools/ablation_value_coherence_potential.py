#!/usr/bin/env python3
"""
Value-added ablation: coherence vs potential field
==================================================

This tool answers a simple question empirically:
  "Do coherence gating and the potential field add value to ALBA, or are they fluff?"

We do not modify the core framework code. We run multiple seeds on a small suite
of synthetic functions and compare ablations:

  - none:       no coherence gating, no potential field
  - coherence:  coherence gating only
  - potential:  potential field only
  - both:       coherence gating + potential field

Metrics
-------
- final_best: best objective value after the full budget (lower is better).
- auc_best:   mean of the best-so-far curve over iterations (lower is better).

Usage
-----
From repo root:
  python3 thesis/alba_framework_potential/tools/ablation_value_coherence_potential.py

Optional:
  python3 thesis/alba_framework_potential/tools/ablation_value_coherence_potential.py --seeds 10 --outdir /tmp/alba_ablation
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# Make `alba_framework_potential` importable when executed from repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_THESIS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _THESIS_DIR not in sys.path:
    sys.path.insert(0, _THESIS_DIR)

from alba_framework_potential.optimizer import ALBA  # noqa: E402


def sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    return float(10.0 * d + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


@dataclass(frozen=True)
class Task:
    name: str
    dim: int
    bounds: List[Tuple[float, float]]
    budget: int
    optimum: float = 0.0
    fn: Callable[[np.ndarray], float] = sphere


def _make_tasks() -> List[Task]:
    # Default: focus on high-D where global signals should matter most.
    return _make_tasks_preset("highd")


def _make_tasks_preset(preset: str) -> List[Task]:
    preset = preset.lower().strip()
    if preset == "highd":
        return [
            Task(name="Sphere", dim=20, bounds=[(-5.0, 5.0)] * 20, budget=500, optimum=0.0, fn=sphere),
            Task(name="Rastrigin", dim=20, bounds=[(-5.12, 5.12)] * 20, budget=500, optimum=0.0, fn=rastrigin),
            Task(name="Rosenbrock", dim=20, bounds=[(-5.0, 10.0)] * 20, budget=500, optimum=0.0, fn=rosenbrock),
        ]
    if preset == "lowd":
        return [
            Task(name="Sphere", dim=2, bounds=[(-5.0, 5.0)] * 2, budget=200, optimum=0.0, fn=sphere),
            Task(name="Rastrigin", dim=2, bounds=[(-5.12, 5.12)] * 2, budget=200, optimum=0.0, fn=rastrigin),
            Task(name="Rosenbrock", dim=2, bounds=[(-5.0, 10.0)] * 2, budget=200, optimum=0.0, fn=rosenbrock),
        ]
    if preset == "mixed":
        return _make_tasks_preset("lowd") + _make_tasks_preset("highd")
    raise ValueError(f"Unknown preset: {preset}")


def run_one(
    task: Task,
    *,
    seed: int,
    use_coherence_gating: bool,
    use_potential_field: bool,
) -> Dict[str, float]:
    opt = ALBA(
        bounds=task.bounds,
        total_budget=task.budget,
        maximize=False,
        seed=seed,
        use_coherence_gating=use_coherence_gating,
        use_potential_field=use_potential_field,
        # Keep the rest fixed (defaults) to isolate the ablation.
    )

    best_hist = np.empty(task.budget, dtype=float)
    t0 = time.perf_counter()
    for t in range(task.budget):
        x = opt.ask()
        y = task.fn(x)
        opt.tell(x, y)
        best_hist[t] = opt.best_y
    dt = time.perf_counter() - t0

    final_best = float(best_hist[-1])
    auc_best = float(np.mean(best_hist))
    regret = final_best - float(task.optimum)

    # Diagnostics: capture whether the global signals were actually "active".
    n_leaves = float(len(opt.leaves))
    global_coh = float("nan")
    pot_std = float("nan")
    coh_std = float("nan")
    potential_scale = float("nan")
    if getattr(opt, "_coherence_tracker", None) is not None:
        tracker = opt._coherence_tracker
        # Force a final recomputation for reporting purposes only.
        tracker.update(opt.leaves, task.budget, force=True)
        global_coh = float(tracker.global_coherence)
        potentials = np.array(list(tracker._cache.potentials.values()), dtype=float)
        coherences = np.array(list(tracker._cache.scores.values()), dtype=float)
        if potentials.size > 0 and np.all(np.isfinite(potentials)):
            pot_std = float(np.std(potentials))
        if coherences.size > 0 and np.all(np.isfinite(coherences)):
            coh_std = float(np.std(coherences))
        # Same scaling used in the sampler: trust potential only when global_coh > 0.5.
        potential_scale = float(max(0.0, min(1.0, (global_coh - 0.5) * 3.33)))

    return {
        "final_best": final_best,
        "auc_best": auc_best,
        "regret": regret,
        "runtime_s": float(dt),
        "n_leaves": n_leaves,
        "global_coherence": global_coh,
        "potential_std": pot_std,
        "coherence_std": coh_std,
        "potential_scale": potential_scale,
    }


def summarize(df: pd.DataFrame, baseline_variant: str = "none") -> str:
    lines: List[str] = []
    lines.append("# Coherence + Potential Field — Value-Added Ablation\n")
    lines.append("Lower is better for `final_best` and `auc_best`.\n")

    for (task, dim, budget), g in df.groupby(["task", "dim", "budget"], sort=False):
        lines.append(f"## {task} ({dim}D, budget={budget})\n")

        # Aggregate table
        agg = (
            g.groupby("variant")[["final_best", "auc_best", "runtime_s"]]
            .agg(["mean", "std", "median"])
            .sort_index()
        )
        # Flatten columns
        agg.columns = ["_".join(col).strip() for col in agg.columns.to_flat_index()]
        agg = agg.reset_index()

        # Win rates vs baseline
        base = g[g["variant"] == baseline_variant].set_index("seed")
        lines.append(f"Baseline for win-rate: `{baseline_variant}`.\n")
        win_rows = []
        for v in sorted(g["variant"].unique()):
            if v == baseline_variant:
                continue
            cur = g[g["variant"] == v].set_index("seed")
            joined = base[["final_best"]].join(cur[["final_best"]], lsuffix="_base", rsuffix="_cur", how="inner")
            if len(joined) == 0:
                continue
            win_rate = float(np.mean(joined["final_best_cur"] < joined["final_best_base"]))
            median_delta = float(np.median(joined["final_best_cur"] - joined["final_best_base"]))
            win_rows.append((v, win_rate, median_delta))

        # Markdown table
        lines.append("| variant | final_best (mean±std) | final_best (median) | auc_best (mean±std) | time s (mean±std) |\n")
        lines.append("|---|---:|---:|---:|---:|\n")
        for _, r in agg.iterrows():
            lines.append(
                "| {variant} | {fbm:.4g}±{fbs:.3g} | {fbmed:.4g} | {aum:.4g}±{aus:.3g} | {tm:.3g}±{ts:.2g} |".format(
                    variant=r["variant"],
                    fbm=r["final_best_mean"],
                    fbs=r["final_best_std"],
                    fbmed=r["final_best_median"],
                    aum=r["auc_best_mean"],
                    aus=r["auc_best_std"],
                    tm=r["runtime_s_mean"],
                    ts=r["runtime_s_std"],
                )
            )
        lines.append("")

        # Signal activity snapshot (median over seeds).
        diag = (
            g.groupby("variant")[["n_leaves", "global_coherence", "potential_std", "potential_scale"]]
            .median(numeric_only=True)
            .reset_index()
            .sort_values("variant")
        )
        lines.append("| variant | n_leaves (median) | global_coherence (median) | potential_std (median) | potential_scale (median) |\n")
        lines.append("|---|---:|---:|---:|---:|\n")
        for _, r in diag.iterrows():
            lines.append(
                "| {variant} | {nl:.0f} | {gc:.3f} | {ps:.3f} | {sc:.3f} |".format(
                    variant=r["variant"],
                    nl=r["n_leaves"],
                    gc=r["global_coherence"] if np.isfinite(r["global_coherence"]) else float("nan"),
                    ps=r["potential_std"] if np.isfinite(r["potential_std"]) else float("nan"),
                    sc=r["potential_scale"] if np.isfinite(r["potential_scale"]) else float("nan"),
                )
            )
        lines.append("")

        if win_rows:
            lines.append("| vs baseline | win-rate | median(Δ final_best) |\n")
            lines.append("|---|---:|---:|\n")
            for v, wr, md in win_rows:
                lines.append(f"| {v} | {wr:.0%} | {md:+.4g} |")
            lines.append("")

    lines.append("## Notes / Caveats\n")
    lines.append(
        "- This is a synthetic suite: conclusions should be confirmed on your target benchmarks.\n"
        "- `auc_best` captures sample-efficiency (how quickly good solutions appear), not just the final result.\n"
        "- If you see mixed results (wins on some functions, losses on others), that is expected: these signals\n"
        "  are most useful when local gradients are informative but not globally consistent.\n"
    )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8, help="Number of seeds (0..seeds-1)")
    parser.add_argument("--outdir", type=str, default=os.path.join(_THIS_DIR), help="Output directory")
    parser.add_argument("--baseline", type=str, default="none", choices=["none", "coherence", "potential", "both"])
    parser.add_argument("--preset", type=str, default="highd", choices=["highd", "lowd", "mixed"])
    parser.add_argument("--tag", type=str, default=None, help="Filename tag (default: preset)")
    args = parser.parse_args()

    tasks = _make_tasks_preset(args.preset)
    seeds = list(range(int(args.seeds)))
    tag = args.tag or args.preset

    variants = {
        "none": dict(use_coherence_gating=False, use_potential_field=False),
        "coherence": dict(use_coherence_gating=True, use_potential_field=False),
        "potential": dict(use_coherence_gating=False, use_potential_field=True),
        "both": dict(use_coherence_gating=True, use_potential_field=True),
    }

    rows: List[Dict[str, object]] = []
    for task in tasks:
        for seed in seeds:
            for variant, cfg in variants.items():
                metrics = run_one(task, seed=seed, **cfg)
                rows.append(
                    dict(
                        task=task.name,
                        dim=task.dim,
                        budget=task.budget,
                        seed=seed,
                        variant=variant,
                        **metrics,
                    )
                )

    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"ABLATION_COHERENCE_POTENTIAL_RESULTS_{tag}.csv")
    md_path = os.path.join(args.outdir, f"ABLATION_COHERENCE_POTENTIAL_REPORT_{tag}.md")

    df.to_csv(csv_path, index=False)
    report = summarize(df, baseline_variant=args.baseline)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")

    # Return code: 0 always; interpretation is in the report.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
