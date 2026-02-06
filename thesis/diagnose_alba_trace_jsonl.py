#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _as_floats(xs: Iterable[Any]) -> np.ndarray:
    out: List[float] = []
    for x in xs:
        try:
            if x is None:
                continue
            v = float(x)
            if np.isfinite(v):
                out.append(v)
        except Exception:
            continue
    return np.asarray(out, dtype=float)


def _pct(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    p50, p90, p99 = np.percentile(x, [50, 90, 99])
    return {"n": int(x.size), "mean": float(np.mean(x)), "p50": float(p50), "p90": float(p90), "p99": float(p99)}

def _summarize_dict(path: Path, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    ask = [e for e in events if str(e.get("event", "ask")) == "ask"]
    tell = [e for e in events if str(e.get("event", "")) == "tell"]

    # --- Taylor / trust-region diagnostics (ask-side) ---
    taylor_misfit = np.asarray([bool(_get(e, "taylor.misfit", False)) for e in ask], dtype=bool)
    misfit_rate = float(np.mean(taylor_misfit)) if taylor_misfit.size else 0.0
    tr_scale = _as_floats(_get(e, "taylor.tr_scale") for e in ask)
    rel_mse = _as_floats(_get(e, "taylor.rel_mse") for e in ask)

    # --- Ask-side metrics ---
    sigma_ratio_chosen = _as_floats(_get(e, "chosen_pred.sigma_ratio") for e in ask)
    sigma_ratio_batch_p50 = _as_floats(_get(e, "batch_stats.sigma_ratio.p50") for e in ask)
    sigma_ratio_batch_p99 = _as_floats(_get(e, "batch_stats.sigma_ratio.p99") for e in ask)

    frac_cells_with_resid = _as_floats(_get(e, "heatmap_index.frac_cells_with_resid") for e in ask)
    frac_candidates_n_r_gt0 = _as_floats(_get(e, "heatmap_usage.frac_candidates_n_r_gt0") for e in ask)
    frac_candidates_n_r_ge3 = _as_floats(_get(e, "heatmap_usage.frac_candidates_n_r_ge3") for e in ask)
    unique_cell_rate = _as_floats(_get(e, "heatmap_usage.unique_cell_rate_overall") for e in ask)

    chosen_penalty = _as_floats(_get(e, "chosen_pred.penalty") for e in ask)
    chosen_novelty = _as_floats(_get(e, "chosen_pred.novelty_bonus") for e in ask)
    chosen_ucb = _as_floats(_get(e, "chosen_pred.ucb") for e in ask)
    chosen_score = _as_floats(_get(e, "chosen_pred.score") for e in ask)
    chosen_visits = _as_floats(_get(e, "chosen_pred.visits") for e in ask)
    penalty_to_abs_ucb = chosen_penalty / np.maximum(np.abs(chosen_ucb), 1e-12) if chosen_penalty.size else np.zeros(0)
    novelty_to_abs_ucb = chosen_novelty / np.maximum(np.abs(chosen_ucb), 1e-12) if chosen_novelty.size else np.zeros(0)

    key_mismatch = np.asarray([bool(_get(e, "categorical.key_mismatch", False)) for e in ask], dtype=bool)
    key_mismatch_rate = float(np.mean(key_mismatch)) if key_mismatch.size else 0.0
    delta_score_after_cat = _as_floats(_get(e, "categorical.delta_score_after_cat") for e in ask)

    heatmap_gate_enabled = np.asarray([bool(_get(e, "heatmap_gate.enabled", False)) for e in ask], dtype=bool)
    heatmap_gate_enabled_rate = float(np.mean(heatmap_gate_enabled)) if heatmap_gate_enabled.size else 0.0
    heatmap_tau_eff = _as_floats(_get(e, "heatmap_gate.tau_eff") for e in ask)

    sigma_calib_scale_ask = _as_floats(_get(e, "surrogate.sigma_calib_scale") for e in ask)

    lgs_feature_kind: List[str] = []
    lgs_selected_kind: List[str] = []
    for e in ask:
        lgs = _get(e, "lgs", {})
        if isinstance(lgs, dict):
            fk = lgs.get("feature_kind")
            sk = lgs.get("selected_kind")
            if fk is not None:
                lgs_feature_kind.append(str(fk))
            if sk is not None:
                lgs_selected_kind.append(str(sk))
    lgs_feature_kind_counts: Dict[str, int] = {}
    for s in lgs_feature_kind:
        lgs_feature_kind_counts[s] = lgs_feature_kind_counts.get(s, 0) + 1
    lgs_selected_kind_counts: Dict[str, int] = {}
    for s in lgs_selected_kind:
        lgs_selected_kind_counts[s] = lgs_selected_kind_counts.get(s, 0) + 1

    lgs_gcv = _as_floats(_get(e, "lgs.gcv") for e in ask)
    lgs_df = _as_floats(_get(e, "lgs.df") for e in ask)
    lgs_n_feat = _as_floats(_get(e, "lgs.n_feat") for e in ask)
    lgs_noise_var = _as_floats(_get(e, "lgs.noise_var") for e in ask)
    lgs_sigma_scale = _as_floats(_get(e, "lgs.sigma_scale") for e in ask)
    lgs_rel_mse = _as_floats(_get(e, "lgs.rel_mse") for e in ask)
    lgs_loo_regret = _as_floats(_get(e, "lgs.loo_regret") for e in ask)
    lgs_loo_topk_overlap = _as_floats(_get(e, "lgs.loo_topk_overlap") for e in ask)
    lgs_loo_spearman = _as_floats(_get(e, "lgs.loo_spearman") for e in ask)

    # --- Internal consistency checks (cheap invariants) ---
    # New scheme: novelty_bonus ~= score - ucb, sigma_tot^2 ~= sigma_adj^2 + sigma_nov^2, sigma_tot >= sigma_adj.
    score_minus_ucb_minus_bonus: List[float] = []
    sigma_tot_sq_err: List[float] = []
    n_bonus_negative = 0
    n_sigma_tot_lt_sigma = 0
    n_sigma_nov_negative = 0
    n_checked = 0

    score_minus_ucb_minus_bonus_after: List[float] = []
    sigma_tot_sq_err_after: List[float] = []
    n_bonus_negative_after = 0
    n_sigma_tot_lt_sigma_after = 0
    n_sigma_nov_negative_after = 0
    n_checked_after = 0

    for e in ask:
        cp = _get(e, "chosen_pred", {})
        if isinstance(cp, dict) and ("novelty_bonus" in cp):
            try:
                ucb = float(cp.get("ucb", 0.0))
                score = float(cp.get("score", 0.0))
                bonus = float(cp.get("novelty_bonus", 0.0))
                sigma = float(cp.get("sigma", 0.0))
                sigma_nov = float(cp.get("sigma_nov", 0.0))
                sigma_tot = float(cp.get("sigma_tot", 0.0))
                if np.isfinite(ucb) and np.isfinite(score) and np.isfinite(bonus):
                    score_minus_ucb_minus_bonus.append(float(score - ucb - bonus))
                    if bonus < -1e-9:
                        n_bonus_negative += 1
                if np.isfinite(sigma) and np.isfinite(sigma_nov) and np.isfinite(sigma_tot):
                    sigma_tot_sq_err.append(float(sigma_tot * sigma_tot - (sigma * sigma + sigma_nov * sigma_nov)))
                    if sigma_tot + 1e-12 < sigma:
                        n_sigma_tot_lt_sigma += 1
                    if sigma_nov < -1e-12:
                        n_sigma_nov_negative += 1
                n_checked += 1
            except Exception:
                pass

        cp2 = _get(e, "chosen_pred_after_cat", {})
        if isinstance(cp2, dict) and ("novelty_bonus" in cp2):
            try:
                ucb = float(cp2.get("ucb", 0.0))
                score = float(cp2.get("score", 0.0))
                bonus = float(cp2.get("novelty_bonus", 0.0))
                sigma = float(cp2.get("sigma", 0.0))
                sigma_nov = float(cp2.get("sigma_nov", 0.0))
                sigma_tot = float(cp2.get("sigma_tot", 0.0))
                if np.isfinite(ucb) and np.isfinite(score) and np.isfinite(bonus):
                    score_minus_ucb_minus_bonus_after.append(float(score - ucb - bonus))
                    if bonus < -1e-9:
                        n_bonus_negative_after += 1
                if np.isfinite(sigma) and np.isfinite(sigma_nov) and np.isfinite(sigma_tot):
                    sigma_tot_sq_err_after.append(float(sigma_tot * sigma_tot - (sigma * sigma + sigma_nov * sigma_nov)))
                    if sigma_tot + 1e-12 < sigma:
                        n_sigma_tot_lt_sigma_after += 1
                    if sigma_nov < -1e-12:
                        n_sigma_nov_negative_after += 1
                n_checked_after += 1
            except Exception:
                pass

    # --- Tell-side residual diagnostics ---
    resid_raw = _as_floats(_get(e, "resid_raw") for e in tell)
    resid_clip = _as_floats(_get(e, "resid_for_grid") for e in tell)
    clipped_flags = np.asarray([bool(_get(e, "resid_clipped", False)) for e in tell], dtype=bool)
    clip_rate = float(np.mean(clipped_flags)) if clipped_flags.size else 0.0

    z = _as_floats(_get(e, "z") for e in tell)
    if z.size == 0 and tell:
        # Backward/forward compatibility: if z is missing, compute it when possible.
        z_fallback: List[float] = []
        for e in tell:
            zr = _get(e, "resid_raw")
            zs = _get(e, "sigma_pred_base")
            if zr is None or zs is None:
                continue
            try:
                zr_f = float(zr)
                zs_f = float(zs)
                if np.isfinite(zr_f) and np.isfinite(zs_f) and zs_f > 0:
                    z_fallback.append(zr_f / max(zs_f, 1e-12))
            except Exception:
                continue
        z = _as_floats(z_fallback)
    z_abs = np.abs(z)
    p_z_lt_1 = float(np.mean(z_abs < 1.0)) if z_abs.size else 0.0
    p_z_lt_2 = float(np.mean(z_abs < 2.0)) if z_abs.size else 0.0

    sigma_calib_scale_tell = _as_floats(_get(e, "sigma_calib_scale") for e in tell)
    sigma_pred_base_raw = _as_floats(_get(e, "sigma_pred_base_raw") for e in tell)
    sigma_pred_base = _as_floats(_get(e, "sigma_pred_base") for e in tell)

    # --- Objective progress (internal, always higher better) ---
    y_internal = _as_floats(_get(e, "y_internal") for e in tell)
    best_internal = float(np.max(y_internal)) if y_internal.size else float("nan")

    # Progress in original objective units, but tracked via internal scores so it
    # works for both maximize and minimize objectives:
    # - y_internal is always "higher is better"
    # - best_y_raw is the y_raw of the best-internal-so-far point.
    y_raw = _as_floats(_get(e, "y_raw") for e in tell)
    best_raw = np.zeros(0, dtype=float)
    n_improvements = 0
    last_improve_iter: Optional[int] = None
    if y_internal.size and y_raw.size and y_internal.size == y_raw.size:
        best_raw = np.empty_like(y_raw, dtype=float)
        best_int = -np.inf
        best_raw_so_far = float("nan")
        for i in range(int(y_internal.size)):
            yi = float(y_internal[i])
            yr = float(y_raw[i])
            if np.isfinite(yi) and (yi > best_int + 1e-12 or not np.isfinite(best_int)):
                best_int = yi
                best_raw_so_far = yr
                n_improvements += 1
                last_improve_iter = i + 1
            best_raw[i] = best_raw_so_far

    # Categorical exploration signal: how often we revisit the same full categorical key.
    key_chosen = []
    stage_eff = []
    for e in ask:
        cat = _get(e, "categorical", {})
        if isinstance(cat, dict):
            k = cat.get("key_chosen")
            if isinstance(k, list):
                try:
                    key_chosen.append(tuple(int(x) for x in k))
                except Exception:
                    pass
            se = cat.get("stage_effective")
            if se is not None:
                stage_eff.append(str(se))
    unique_key_count = len(set(key_chosen)) if key_chosen else 0
    stage_eff_counts: Dict[str, int] = {}
    if stage_eff:
        for s in stage_eff:
            stage_eff_counts[s] = stage_eff_counts.get(s, 0) + 1

    # Build a machine-readable summary. Keep the structure stable and close to what we print.
    out: Dict[str, Any] = {
        "trace": str(path),
        "events": {"ask": int(len(ask)), "tell": int(len(tell))},
        "best_y_internal": float(best_internal) if np.isfinite(best_internal) else None,
        "taylor": {
            "misfit_rate": float(misfit_rate),
            "tr_scale": _pct(tr_scale),
            "rel_mse": _pct(rel_mse),
        },
        "heatmap": {
            "gate_enabled_rate": float(heatmap_gate_enabled_rate),
            "tau_eff": _pct(heatmap_tau_eff),
            "frac_cells_with_resid": _pct(frac_cells_with_resid),
            "frac_candidates_n_r_gt0": _pct(frac_candidates_n_r_gt0),
            "frac_candidates_n_r_ge3": _pct(frac_candidates_n_r_ge3),
            "unique_cell_rate": _pct(unique_cell_rate),
            "sigma_ratio_chosen": _pct(sigma_ratio_chosen),
            "sigma_ratio_batch_p50": _pct(sigma_ratio_batch_p50),
            "sigma_ratio_batch_p99": _pct(sigma_ratio_batch_p99),
        },
        "novelty_or_penalty": {
            "mode": ("novelty" if chosen_novelty.size else "penalty"),
            "chosen_bonus_or_penalty": (_pct(chosen_novelty) if chosen_novelty.size else _pct(chosen_penalty)),
            "chosen_visits": _pct(chosen_visits),
            "bonus_or_penalty_over_abs_ucb": (_pct(novelty_to_abs_ucb) if chosen_novelty.size else _pct(penalty_to_abs_ucb)),
        },
        "surrogate": {
            "sigma_calib_scale_ask": _pct(sigma_calib_scale_ask),
            "sigma_calib_scale_tell": _pct(sigma_calib_scale_tell),
        },
        "lgs": {
            "feature_kind_counts": lgs_feature_kind_counts,
            "selected_kind_counts": lgs_selected_kind_counts,
            "gcv": _pct(lgs_gcv),
            "df": _pct(lgs_df),
            "n_feat": _pct(lgs_n_feat),
            "noise_var": _pct(lgs_noise_var),
            "sigma_scale": _pct(lgs_sigma_scale),
            "rel_mse": _pct(lgs_rel_mse),
            "loo_regret": _pct(lgs_loo_regret),
            "loo_topk_overlap": _pct(lgs_loo_topk_overlap),
            "loo_spearman": _pct(lgs_loo_spearman),
        },
        "categorical": {
            "key_mismatch_rate": float(key_mismatch_rate),
            "delta_score_after_cat": _pct(delta_score_after_cat),
            "stage_effective_counts": stage_eff_counts,
            "unique_key_chosen": int(unique_key_count),
            "n_key_chosen": int(len(key_chosen)),
        },
        "tell": {
            "clip_rate": float(clip_rate),
            "resid_raw": _pct(resid_raw),
            "resid_for_grid": _pct(resid_clip),
            "sigma_pred_base_raw": _pct(sigma_pred_base_raw),
            "sigma_pred_base": _pct(sigma_pred_base),
            "z_abs_lt_1": float(p_z_lt_1),
            "z_abs_lt_2": float(p_z_lt_2),
            "z": _pct(z),
        },
        "invariants": {
            "n_checked": int(n_checked),
            "score_minus_ucb_minus_bonus_max_abs": float(np.max(np.abs(np.asarray(score_minus_ucb_minus_bonus, dtype=float))))
            if score_minus_ucb_minus_bonus
            else 0.0,
            "sigma_tot_sq_err_max_abs": float(np.max(np.abs(np.asarray(sigma_tot_sq_err, dtype=float))))
            if sigma_tot_sq_err
            else 0.0,
            "bonus_negative": int(n_bonus_negative),
            "sigma_tot_lt_sigma": int(n_sigma_tot_lt_sigma),
            "sigma_nov_negative": int(n_sigma_nov_negative),
        },
        "invariants_after_cat": {
            "n_checked": int(n_checked_after),
            "score_minus_ucb_minus_bonus_max_abs": float(
                np.max(np.abs(np.asarray(score_minus_ucb_minus_bonus_after, dtype=float)))
            )
            if score_minus_ucb_minus_bonus_after
            else 0.0,
            "sigma_tot_sq_err_max_abs": float(np.max(np.abs(np.asarray(sigma_tot_sq_err_after, dtype=float))))
            if sigma_tot_sq_err_after
            else 0.0,
            "bonus_negative": int(n_bonus_negative_after),
            "sigma_tot_lt_sigma": int(n_sigma_tot_lt_sigma_after),
            "sigma_nov_negative": int(n_sigma_nov_negative_after),
        },
        "progress": {
            "n_improvements": int(n_improvements),
            "last_improvement_iter": int(last_improve_iter) if last_improve_iter is not None else None,
            "best_y_raw_at_iters": {},
        },
    }

    if best_raw.size:
        cps = [50, 100, 200, 300, int(best_raw.size)]
        for cp in cps:
            if 1 <= cp <= int(best_raw.size):
                out["progress"]["best_y_raw_at_iters"][str(cp)] = float(best_raw[cp - 1])

    return out


def _format_summary(summary: Dict[str, Any]) -> str:
    path = Path(str(summary.get("trace", "")))
    events = summary.get("events", {})
    lines: List[str] = []
    lines.append(f"{path.name}")
    lines.append(
        f"  events: ask={events.get('ask', 0)} tell={events.get('tell', 0)} best_y_internal={summary.get('best_y_internal')}"
    )
    lines.append(f"  taylor: misfit_rate={summary['taylor']['misfit_rate']:.3f} tr_scale={summary['taylor']['tr_scale']} rel_mse={summary['taylor']['rel_mse']}")
    lines.append(f"  heatmap_gate: enabled_rate={summary['heatmap']['gate_enabled_rate']:.3f} tau_eff={summary['heatmap']['tau_eff']}")
    lines.append(f"  heatmap: frac_cells_with_resid={summary['heatmap']['frac_cells_with_resid']}")
    lines.append(
        "  heatmap: frac_candidates_n_r_gt0="
        f"{summary['heatmap']['frac_candidates_n_r_gt0']} frac_candidates_n_r_ge3={summary['heatmap']['frac_candidates_n_r_ge3']}"
    )
    lines.append(f"  heatmap: unique_cell_rate={summary['heatmap']['unique_cell_rate']}")
    lines.append(
        "  sigma_ratio: chosen="
        f"{summary['heatmap']['sigma_ratio_chosen']} batch_p50={summary['heatmap']['sigma_ratio_batch_p50']} batch_p99={summary['heatmap']['sigma_ratio_batch_p99']}"
    )

    np_mode = summary.get("novelty_or_penalty", {}).get("mode")
    if np_mode == "novelty":
        lines.append(
            "  novelty: chosen_bonus="
            f"{summary['novelty_or_penalty']['chosen_bonus_or_penalty']} chosen_visits={summary['novelty_or_penalty']['chosen_visits']} bonus/|ucb|={summary['novelty_or_penalty']['bonus_or_penalty_over_abs_ucb']}"
        )
    else:
        lines.append(
            "  penalty: chosen_penalty="
            f"{summary['novelty_or_penalty']['chosen_bonus_or_penalty']} chosen_visits={summary['novelty_or_penalty']['chosen_visits']} penalty/|ucb|={summary['novelty_or_penalty']['bonus_or_penalty_over_abs_ucb']}"
        )

    lines.append(
        "  sigma_calib: ask_scale="
        f"{summary['surrogate']['sigma_calib_scale_ask']} tell_scale={summary['surrogate']['sigma_calib_scale_tell']}"
    )
    lines.append(
        "  lgs: selected_kind="
        f"{summary['lgs']['selected_kind_counts']} rel_mse={summary['lgs']['rel_mse']} gcv={summary['lgs']['gcv']}"
    )
    lines.append(
        f"  categorical: key_mismatch_rate={summary['categorical']['key_mismatch_rate']:.3f} delta_score_after_cat={summary['categorical']['delta_score_after_cat']}"
    )
    if summary["categorical"].get("stage_effective_counts"):
        lines.append(
            "  categorical: stage_effective="
            f"{summary['categorical']['stage_effective_counts']} unique_key_chosen={summary['categorical']['unique_key_chosen']}/{summary['categorical']['n_key_chosen']}"
        )
    lines.append(
        "  resid_clip: clip_rate="
        f"{summary['tell']['clip_rate']:.3f} resid_raw={summary['tell']['resid_raw']} resid_for_grid={summary['tell']['resid_for_grid']} "
        f"P(|z|<1)={summary['tell']['z_abs_lt_1']:.3f} P(|z|<2)={summary['tell']['z_abs_lt_2']:.3f}"
    )
    if summary.get("progress", {}).get("best_y_raw_at_iters"):
        lines.append(f"  progress: best_y_raw@iters={summary['progress']['best_y_raw_at_iters']}")
        lines.append(
            f"  progress: n_improvements={summary['progress']['n_improvements']} last_improvement_iter={summary['progress']['last_improvement_iter']}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze ALBA JSONL traces (ask + optional tell events).")
    p.add_argument("paths", nargs="+", help="One or more .jsonl files")
    p.add_argument("--out", help="Optional JSON output path for machine-readable summaries.")
    args = p.parse_args()

    paths = [Path(x) for x in args.paths]
    results: List[Dict[str, Any]] = []
    for path in paths:
        events = _load_jsonl(path)
        summary = _summarize_dict(path, events)
        results.append(summary)
        print(_format_summary(summary))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
