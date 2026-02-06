from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any, Dict, Tuple as TypingTuple

import json
import csv
import os
import numpy as np
from utils_functions.param_space import ParamSpace


@dataclass
class Cube:
    """Rappresenta una regione (cubo) nello spazio degli iperparametri normalizzato in [0,1]^d."""

    # limiti [min, max] per ogni dimensione
    bounds: List[Tuple[float, float]]

    # collegamenti nella gerarchia dei cubi
    parent: Optional["Cube"] = None
    children: List["Cube"] = field(default_factory=list)

    # statistiche
    n_trials: int = 0
    scores: List[float] = field(default_factory=list)            # punteggi finali
    scores_early: List[float] = field(default_factory=list)      # punteggi early
    best_score: float = -np.inf
    mean_score: float = 0.0
    var_score: float = 0.0

    # parametri regionali
    prior_var: float = 1.0  # varianza prior da combinare con var_score
    q_threshold: float = 0.0  # quantile per early stopping
    early_quantile_p: float = 0.7
    adaptive_early_quantile: bool = False
    # nuovo: timeout pruning
    stale_steps: int = 0

    def sample_uniform(self) -> np.ndarray:
        """Campiona un punto uniformemente all’interno del cubo."""
        dims = len(self.bounds)
        point = np.zeros(dims)
        for i, (low, high) in enumerate(self.bounds):
            point[i] = np.random.uniform(low, high)
        return point

    def update_early(self, early_score: float) -> None:
        """Aggiorna le statistiche con un punteggio early con fallback al padre."""
        self.scores_early.append(early_score)
        n = len(self.scores_early)
        if self.adaptive_early_quantile:
            # quantile p cresce con i dati: da 0.6 a 0.8
            p = 0.6 + 0.2 * min(1.0, n / 20.0)
        else:
            p = self.early_quantile_p
        if n >= 3:
            self.q_threshold = float(np.quantile(self.scores_early, p))
        else:
            if self.parent is not None and self.parent.q_threshold > 0:
                self.q_threshold = 0.5 * float(self.parent.q_threshold) + 0.5 * float(self.q_threshold)
        # aggiorna prior_var ereditando dal padre (se esiste)
        if self.parent is not None:
            self.prior_var = 0.5 * float(self.parent.prior_var) + 0.5 * float(self.prior_var)

    def update_final(self, final_score: float) -> None:
        """Aggiorna le statistiche con un punteggio finale."""
        self.n_trials += 1
        self.scores.append(float(final_score))
        self.mean_score = float(np.mean(self.scores))
        self.var_score = float(np.var(self.scores)) if len(self.scores) > 1 else 0.0
        if final_score > self.best_score:
            self.best_score = float(final_score)

    def ucb(self, beta: float = 1.0, eps: float = 1e-8) -> float:
        """UCB che usa anche gli early e un prior regionale dal padre; niente inf."""
        n_e = len(self.scores_early)
        n_eff = self.n_trials + 0.25 * n_e
        if self.n_trials > 0:
            mu = float(self.mean_score)
            if self.n_trials > 1:
                var = float(self.var_score)
            else:
                var = float(np.var(self.scores_early)) if n_e > 1 else float(self.prior_var)
        else:
            mu = float(np.mean(self.scores_early)) if n_e > 0 else 0.0
            var = float(np.var(self.scores_early)) if n_e > 1 else float(self.prior_var)
        if self.parent is not None and self.parent.n_trials > 0:
            mu = 0.5 * float(self.parent.mean_score) + 0.5 * mu
            pv = float(self.parent.var_score) if self.parent.var_score > 0 else float(self.parent.prior_var)
            var = 0.5 * pv + 0.5 * var
        if n_eff <= 0:
            return float(mu + beta * np.sqrt(var + self.prior_var))
        return float(mu + beta * np.sqrt(var / (n_eff + eps) + self.prior_var))

    def should_split(self, min_trials: int = 10, gamma: float = 0.1, min_points: int = 12) -> bool:
        """Decide se il cubo deve essere suddiviso, considerando anche i punti osservati."""
        # richiede un minimo di finali OPPURE un minimo di punti osservati
        if self.n_trials < min_trials and len(self._points_history) < min_points:
            return False
        heterogeneity = self.var_score
        # se non hai var dei finali, usa var degli early come proxy
        if heterogeneity == 0.0 and len(self.scores_early) > 1:
            heterogeneity = float(np.var(self.scores_early))
        return bool(heterogeneity > gamma)

    def split(self) -> List["Cube"]:
        """Divide in base alla varianza dei punti osservati, smistando la history."""
        points = np.array(self._points_history)
        if points.size == 0:
            return []
        points = points.reshape((-1, len(self.bounds)))
        variances = points.var(axis=0)
        axis = int(np.argmax(variances))
        low, high = self.bounds[axis]
        med = float(np.median(points[:, axis]))
        if not (low < med < high):
            med = (low + high) / 2.0
        b1, b2 = list(self.bounds), list(self.bounds)
        b1[axis], b2[axis] = (low, med), (med, high)
        c1 = Cube(bounds=b1, parent=self,
                  early_quantile_p=self.early_quantile_p,
                  adaptive_early_quantile=self.adaptive_early_quantile)
        c2 = Cube(bounds=b2, parent=self,
                  early_quantile_p=self.early_quantile_p,
                  adaptive_early_quantile=self.adaptive_early_quantile)
        c1.prior_var = float(self.prior_var)
        c2.prior_var = float(self.prior_var)
        c1.q_threshold = float(self.q_threshold)
        c2.q_threshold = float(self.q_threshold)
        # smista i punti
        c1._tested_points = []
        c2._tested_points = []
        for p in points:
            (c1._tested_points if p[axis] < med else c2._tested_points).append(np.array(p, dtype=float))
        self.children = [c1, c2]
        return self.children

    @property
    def _points_history(self) -> List[np.ndarray]:
        """Restituisce la cronologia dei punti testati nel cubo."""
        # Se non registriamo i punti, restituiamo una lista vuota. In una
        # implementazione completa, questa lista andrebbe mantenuta in update.
        return getattr(self, "_tested_points", [])

    def add_tested_point(self, point: np.ndarray) -> None:
        """Aggiunge il punto ai punti testati (per il calcolo della varianza)."""
        if not hasattr(self, "_tested_points"):
            self._tested_points: List[np.ndarray] = []
        # assicurati di salvare una copia float
        self._tested_points.append(np.array(point, dtype=float))


class CubeHPO:
    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        min_trials: int = 10,
        gamma: float = 0.1,
        beta: float = 1.0,
        early_epochs: int = 5,
        full_epochs: int = 50,
        rng_seed: Optional[int] = None,
        param_space: Optional[ParamSpace] = None,
        objective_in_normalized_space: bool = True,
        log_path: Optional[str] = None,
        early_quantile_p: float = 0.7,
        adaptive_early_quantile: bool = False,
        on_best: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.root = Cube(list(bounds))
        self.leaf_cubes: List[Cube] = [self.root]
        self.min_trials = int(min_trials)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.early_epochs = int(early_epochs)
        self.full_epochs = int(full_epochs)
        self.best_score_global = -np.inf
        self.best_x_norm: Optional[List[float]] = None
        self.best_x_real: Optional[List[Any]] = None
        self.trial_id: int = 0
        self.param_space = param_space
        self.objective_in_normalized_space = bool(objective_in_normalized_space)
        self.log_path = log_path
        self.on_best = on_best
        # configura quantile nei cubi
        self.root.early_quantile_p = float(early_quantile_p)
        self.root.adaptive_early_quantile = bool(adaptive_early_quantile)
        if rng_seed is not None:
            np.random.seed(int(rng_seed))
        # diagnostica
        self.total_trials: int = 0
        self.early_stops: int = 0
        self.s_early_all: List[float] = []
        self.s_final_all: List[float] = []
        self.s_early_pass: List[float] = []
        self.splits_count: int = 0
        self.prunes_count: int = 0
        # prepara logging con header
        if self.log_path:
            if not os.path.exists(self.log_path):
                with open(self.log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'trial', 'cube_id', 'x_norm', 'x_real', 's_early', 'q_threshold',
                        'early_stop', 's_final', 'best_score_global', 'n_leaves'
                    ])

    def _log(self, row: List[Any]) -> None:
        if self.log_path:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

    def _safe_json(self, obj: Any) -> str:
        try:
            return json.dumps(obj)
        except Exception:
            return str(obj)

    # ------------------------------- Neighbor prior -------------------------------
    def _cube_center(self, cube: Cube) -> np.ndarray:
        ctr = []
        for low, high in cube.bounds:
            ctr.append((low + high) / 2.0)
        return np.array(ctr, dtype=float)

    def _adjacent(self, a: Cube, b: Cube, tol: float = 1e-12) -> bool:
        dims = len(a.bounds)
        touch = 0
        overlap = 0
        for i in range(dims):
            a_low, a_high = a.bounds[i]
            b_low, b_high = b.bounds[i]
            # verifica contatto su faccia
            if abs(a_high - b_low) <= tol or abs(b_high - a_low) <= tol:
                touch += 1
            # verifica overlap interno
            if min(a_high, b_high) - max(a_low, b_low) > 0:
                overlap += 1
        # vicini se sovrappongono in almeno d-1 dimensioni e toccano in >=1
        return overlap >= dims - 1 and touch >= 1

    def _neighbor_variance(self, cube: Cube) -> Optional[float]:
        if len(self.leaf_cubes) <= 1:
            return None
        center_c = self._cube_center(cube)
        weights = []
        vars_ = []
        for other in self.leaf_cubes:
            if other is cube:
                continue
            if not self._adjacent(cube, other):
                continue
            if other.n_trials == 0:
                continue
            center_o = self._cube_center(other)
            dist = float(np.linalg.norm(center_c - center_o))
            w = 1.0 / (dist + 1e-6)
            weights.append(w)
            vars_.append(max(other.var_score, 1e-9))
        if not weights:
            return None
        weights = np.array(weights, dtype=float)
        vars_ = np.array(vars_, dtype=float)
        return float(np.sum(weights * vars_) / np.sum(weights))

    def _update_prior_from_neighbors(self, cube: Cube) -> None:
        nv = self._neighbor_variance(cube)
        if nv is not None:
            cube.prior_var = 0.5 * float(cube.prior_var) + 0.5 * float(nv)

    # ----------------------------- Selection and trials ----------------------------
    def select_cube(self) -> Cube:
        # epsilon-exploration per evitare lock-in
        if len(self.leaf_cubes) > 1 and np.random.rand() < 0.10:
            return np.random.choice(self.leaf_cubes)
        # seleziona il cubo con UCB massimo
        ucb_values = [c.ucb(beta=self.beta) for c in self.leaf_cubes]
        idx = int(np.argmax(ucb_values))
        return self.leaf_cubes[idx]

    def _maybe_denormalize(self, x_norm: np.ndarray) -> Optional[List[Any]]:
        if self.param_space is None:
            return None
        return self.param_space.denormalize(x_norm)

    def _call_objective(self, objective_fn: Callable[[np.ndarray, int], Any], x: np.ndarray, epochs: int) -> TypingTuple[float, Optional[Any]]:
        res = objective_fn(x, epochs=epochs)
        if isinstance(res, tuple):
            score = float(res[0])
            artifact = res[1] if len(res) > 1 else None
        else:
            score = float(res)
            artifact = None
        return score, artifact

    def _cube_depth(self, cube: Cube) -> int:
        d = 0
        cur = cube
        while cur.parent is not None:
            d += 1
            cur = cur.parent
        return d

    def leaf_volumes(self) -> List[float]:
        vols: List[float] = []
        for c in self.leaf_cubes:
            v = 1.0
            for (lo, hi) in c.bounds:
                v *= float(hi - lo)
            vols.append(v)
        return vols

    def diagnostics(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        total = max(1, self.total_trials)
        res['early_stop_rate'] = float(self.early_stops) / float(total)
        # correlazione solo su coppie (early che hanno passato, final)
        corr = None
        if len(self.s_early_pass) > 1 and len(self.s_final_all) > 1 and len(self.s_early_pass) == len(self.s_final_all):
            try:
                corr = float(np.corrcoef(np.array(self.s_early_pass), np.array(self.s_final_all))[0, 1])
            except Exception:
                corr = None
        res['corr_early_final'] = corr
        # FN proxy: early-stopped con s_early >= mediana degli early che hanno passato
        fn_proxy = None
        if self.s_early_pass:
            thr_pass = float(np.median(self.s_early_pass))
            early_stopped = [e for e in self.s_early_all if e not in self.s_early_pass]
            if early_stopped:
                fn = sum(1 for e in early_stopped if e >= thr_pass)
                fn_proxy = fn / float(len(early_stopped))
        res['fn_proxy_rate'] = fn_proxy
        res['splits'] = self.splits_count
        res['prunes'] = self.prunes_count
        # profondità massima
        if self.leaf_cubes:
            res['max_depth'] = max(self._cube_depth(c) for c in self.leaf_cubes)
            res['leaf_volumes'] = self.leaf_volumes()
        else:
            res['max_depth'] = 0
            res['leaf_volumes'] = []
        return res

    def run_trial(self, cube: Cube, objective_fn: Callable[[np.ndarray, int], Any]) -> None:
        self.trial_id += 1
        self.total_trials += 1
        # campiona un punto, registra il punto nel cubo
        x_norm = cube.sample_uniform()
        cube.add_tested_point(x_norm)
        x_real = self._maybe_denormalize(x_norm)
        # calcola punteggio early
        x_for_obj = x_norm if self.objective_in_normalized_space or x_real is None else np.array(x_real, dtype=float)
        s_early, _ = self._call_objective(objective_fn, x_for_obj, epochs=self.early_epochs)
        self.s_early_all.append(s_early)
        cube.update_early(s_early)
        # aggiorna prior con info dai vicini
        self._update_prior_from_neighbors(cube)
        early_stop = s_early < cube.q_threshold
        if early_stop:
            self.early_stops += 1
            # log trial early-stopped
            self._log([
                self.trial_id, id(cube), json.dumps(x_norm.tolist()),
                self._safe_json(x_real) if x_real is not None else '',
                s_early, cube.q_threshold, True, '', self.best_score_global, len(self.leaf_cubes)
            ])
            # marca staleness anche per il cubo corrente e per gli altri
            cube.stale_steps += 1
            for c in self.leaf_cubes:
                if c is not cube:
                    c.stale_steps += 1
            return
        # calcola punteggio finale
        s_final, artifact = self._call_objective(objective_fn, x_for_obj, epochs=self.full_epochs)
        cube.update_final(s_final)
        self.s_final_all.append(s_final)
        self.s_early_pass.append(s_early)
        # aggiorna prior con info dai vicini dopo il final
        self._update_prior_from_neighbors(cube)
        # aggiorna best globale e best config
        if s_final > self.best_score_global:
            self.best_score_global = s_final
            self.best_x_norm = x_norm.tolist()
            self.best_x_real = x_real
            if self.on_best is not None:
                try:
                    self.on_best({
                        'trial': self.trial_id,
                        'score': s_final,
                        'x_norm': self.best_x_norm,
                        'x_real': self.best_x_real,
                        'artifact': artifact,
                    })
                except Exception:
                    pass
        # reset/increment staleness
        cube.stale_steps = 0
        for c in self.leaf_cubes:
            if c is not cube:
                c.stale_steps += 1
        # verifica split
        if cube.should_split(min_trials=self.min_trials, gamma=self.gamma):
            children = cube.split()
            if children:
                self.splits_count += 1
                if cube in self.leaf_cubes:
                    self.leaf_cubes.remove(cube)
                for child in children:
                    self.leaf_cubes.append(child)
        # prune cubi poco promettenti
        prev = len(self.leaf_cubes)
        self.prune_cubes()
        removed = max(0, prev - len(self.leaf_cubes))
        self.prunes_count += removed
        # log trial completato
        self._log([
            self.trial_id, id(cube), json.dumps(x_norm.tolist()),
            self._safe_json(x_real) if x_real is not None else '',
            s_early, cube.q_threshold, False, s_final, self.best_score_global, len(self.leaf_cubes)
        ])

    def prune_cubes(self) -> None:
        # elimina cubi con UCB molto inferiore al best globale e staleness elevata
        delta = 0.01  # margine di tolleranza (1%)
        lcb_best = float(self.best_score_global - delta)
        keep: List[Cube] = []
        for c in self.leaf_cubes:
            if c.ucb(beta=self.beta) >= lcb_best and c.stale_steps < 50:
                keep.append(c)
        self.leaf_cubes = keep

    def optimize(self, objective_fn: Callable[[np.ndarray, int], Any], budget: int) -> None:
        for _ in range(int(budget)):
            cube = self.select_cube()
            self.run_trial(cube, objective_fn)
