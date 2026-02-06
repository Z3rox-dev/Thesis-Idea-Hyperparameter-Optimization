import unittest
import numpy as np

from hpo_curvature import QuadCube


class TestMoreSplitsAndUCB(unittest.TestCase):
    def test_split4_curvature_axes_selection(self):
        # 3D cube; force curvature to select axes 0 and 1 via surrogate_2d
        cube = QuadCube(bounds=[(-2.0, 2.0), (-4.0, 4.0), (-8.0, 8.0)])
        cube.R = np.eye(3)
        cube.mu = np.zeros(3)
        cube.depth = 2
        # Seed minimal history to allow path, though curvature path doesn't require points
        cube._tested_points = [np.zeros(3)] * 12
        cube._tested_pairs = [(np.zeros(3), 0.0)] * 12
        # Surrogate with stronger curvature on PC1 then PC2, identity frame
        cube.surrogate_2d = {
            'w': np.zeros(6),
            'mu': np.zeros(3),
            'R': np.eye(3),
            'A_inv': np.eye(6),
            'sigma2': 0.1,
            'n': 12,
            'r2': 0.9,
            'pca_ok': True,
            'df_eff': 6.0,
            'H': np.array([[5.0, 0.0],[0.0, 2.0]]),
            'lambda': np.array([5.0, 2.0]),
        }
        children = cube.split4()
        self.assertEqual(len(children), 4)
        # Parent widths in prime coords
        pw = np.array([abs(hi - lo) for (lo, hi) in cube.bounds], dtype=float)
        # Check each child has dims 0 and 1 halved and dim 2 unchanged
        for ch in children:
            cw = np.array([abs(hi - lo) for (lo, hi) in ch.bounds], dtype=float)
            ratios = np.divide(cw, pw, out=np.ones_like(cw), where=pw > 0)
            # Dims 0 and 1 halved, dim 2 unchanged (1.0)
            self.assertTrue(np.allclose(ratios[:2], [0.5, 0.5], atol=1e-8))
            self.assertTrue(np.isclose(ratios[2], 1.0, atol=1e-8))

    def test_split4_widest_two_when_pca_not_ok(self):
        # Force ok=False and choose top-2 widths by bounds
        # Bounds widths: [1, 5, 3] -> expect axes 1 and 2
        cube = QuadCube(bounds=[(0.0, 1.0), (-2.5, 2.5), (-1.5, 1.5)])
        cube.R = np.eye(3)
        cube.mu = np.zeros(3)
        cube.depth = 0  # below depth_min -> ok False
        cube._tested_points = []
        cube._tested_pairs = []
        children = cube.split4()
        pw = np.array([abs(hi - lo) for (lo, hi) in cube.bounds], dtype=float)
        for ch in children:
            cw = np.array([abs(hi - lo) for (lo, hi) in ch.bounds], dtype=float)
            ratios = np.divide(cw, pw, out=np.ones_like(cw), where=pw > 0)
            # axis 0 unchanged, axes 1 and 2 halved
            self.assertTrue(np.isclose(ratios[0], 1.0, atol=1e-8))
            self.assertTrue(np.allclose(ratios[1:], [0.5, 0.5], atol=1e-8))

    def test_ucb_with_geometric_bonus(self):
        c = QuadCube(bounds=[(-1.0, 2.0), (0.0, 3.0)])  # widths 3 and 3 -> vol = 9
        c.R = np.eye(2); c.mu = np.zeros(2)
        c.prior_var = 0.0
        c.scores = [2.0, 4.0]
        c.n_trials = len(c.scores)
        c.mean_score = float(np.mean(c.scores))
        c.var_score = float(np.var(c.scores))
        beta = 0.0  # isolate geometric term
        base = c.ucb(beta=beta, lambda_geo=0.0)
        # Manual volume
        vol = 1.0
        for lo, hi in c.bounds:
            vol *= max(hi - lo, 0.0)
        bonus_expected = 1.23 * vol / np.sqrt(c.n_trials + 1.0)
        u = c.ucb(beta=beta, lambda_geo=1.23)
        self.assertAlmostEqual(u - base, bonus_expected, places=7)

    def test_should_split_min_width_gate(self):
        cube = QuadCube(bounds=[(-1e-6, 1e-6), (-1e-7, 1e-7)])
        cube.R = np.eye(2)
        cube.mu = np.zeros(2)
        cube.n_trials = 100
        cube._tested_points = [np.zeros(2) for _ in range(20)]
        cube._tested_pairs = [(np.zeros(2), 0.0) for _ in range(20)]
        # min_width larger than both widths -> no split
        dec = cube.should_split(min_trials=5, min_points=10, max_depth=4, min_width=1e-5, gamma=0.02)
        self.assertEqual(dec, 'none')

    def test_predict_surrogate_no_model(self):
        cube = QuadCube(bounds=[(-1.0, 1.0), (-1.0, 1.0)])
        m, s = cube.predict_surrogate(np.array([0.0, 0.0]))
        self.assertEqual(m, 0.0)
        self.assertEqual(s, 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
