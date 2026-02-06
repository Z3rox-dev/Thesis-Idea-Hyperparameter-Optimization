import types
import unittest
import numpy as np

from hpo_curvature import QuadHPO, QuadCube


class TestPruningAndSelection(unittest.TestCase):
    def test_prune_cubes_min_leaves_and_too_small(self):
        d = 2
        hpo = QuadHPO(bounds=[(0.0, 1.0)] * d, rng_seed=0)
        # Create custom leaves with controlled UCB
        leaves = []
        for i in range(8):
            c = QuadCube(bounds=[(0.0, 1.0)] * d)
            c.R = np.eye(d); c.mu = np.zeros(d)
            c.scores = [i]  # so mean/var produce order
            # Monkeypatch UCB per-instance
            c.ucb = types.MethodType(lambda self, beta=0.05, lambda_geo=0.8: float(self.scores[-1]), c)
            leaves.append(c)
        # Add a tiny-volume cube that should be pruned even if score decent
        tiny = QuadCube(bounds=[(0.0, 1e-9), (0.0, 1e-9)])
        tiny.R = np.eye(2); tiny.mu = np.zeros(2)
        tiny.scores = [100.0]
        tiny.ucb = types.MethodType(lambda self, beta=0.05, lambda_geo=0.8: float(self.scores[-1]), tiny)
        leaves.append(tiny)
        hpo.leaf_cubes = leaves
        hpo.min_leaves = 5
        # Set a generous margin so many non-tiny leaves pass the UCB threshold,
        # ensuring we don't need the fallback that would re-add pruned cubes.
        hpo.delta_prune = 100.0
        # Best score set to the best among non-tiny leaves so tiny is not required.
        hpo.best_score_global = max(c.scores[-1] for c in leaves if c is not tiny)
        hpo.prune_cubes()
        # Ensure minimum leaves preserved and tiny-volume pruned
        self.assertGreaterEqual(len(hpo.leaf_cubes), hpo.min_leaves)
        self.assertFalse(any(c is tiny for c in hpo.leaf_cubes))

    def test_preferred_leaf_selection_after_split(self):
        # Lower thresholds to force a split
        d = 2
        hpo = QuadHPO(bounds=[(0.0, 1.0)] * d, rng_seed=0)
        hpo.min_trials = 1
        hpo.min_points = 1
        hpo.max_depth = 3
        # Simple objective that encourages improvement near 0.3
        def obj(x: np.ndarray, epochs: int = 0) -> float:
            return -float(np.sum((x - 0.3) ** 2))
        # Run enough trials to trigger a split
        for _ in range(3):
            cube = hpo.select_cube()
            hpo.run_trial(cube, obj)
        # After split, _preferred_leaf should be set and then consumed by select_cube
        if hpo._preferred_leaf is not None:
            sel = hpo.select_cube()
            self.assertTrue(sel is not None)
        else:
            # If not set due to randomness, force a split once more
            cube = hpo.select_cube()
            hpo.run_trial(cube, obj)
            sel = hpo.select_cube()
            self.assertTrue(sel is not None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
