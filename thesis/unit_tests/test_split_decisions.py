import types
import unittest
import numpy as np

from hpo_curvature import QuadCube


class TestSplitDecisions(unittest.TestCase):
    def test_should_split_curvature_flat_region(self):
        cube = QuadCube(bounds=[(-1.0, 1.0), (-1.0, 1.0)])
        cube.R = np.eye(2)
        cube.mu = np.zeros(2)
        cube.n_trials = 20
        cube._tested_points = [np.zeros(2) for _ in range(20)]
        cube._tested_pairs = [(np.zeros(2), 0.0) for _ in range(20)]
        # Force curvature to be very small
        cube._curvature_scores = types.MethodType(lambda self: np.array([1e-9, 5e-9]), cube)
        # With enough trials/points but flat curvature, should be 'none'
        dec = cube.should_split(min_trials=5, min_points=10, max_depth=4, min_width=1e-6, gamma=0.02)
        self.assertEqual(dec, 'none')

    def test_should_split_info_gain_gamma_gate(self):
        cube = QuadCube(bounds=[(-1.0, 1.0), (-1.0, 1.0)])
        cube.R = np.eye(2)
        cube.mu = np.zeros(2)
        cube.n_trials = 20
        # Fill with arbitrary pairs
        rng = np.random.default_rng(0)
        cube._tested_points = [rng.uniform(-1, 1, size=2) for _ in range(20)]
        cube._tested_pairs = list(zip([p.copy() for p in cube._tested_points], rng.normal(0, 1, size=20)))
        # Pretend surrogate exists with certain residual variance and n
        cube.surrogate_2d = {
            'sigma2': 1.0,
            'n': 20,
            'lambda': np.array([0.1, 0.05]),
            'R': np.eye(2),
            'mu': np.zeros(2),
        }
        # Patch simulate splits to return weak vs strong improvements
        def sim_vars(low_var):
            v = 0.9 if low_var else 0.99
            # Two children with nearly same counts
            return [{'n': 10, 'var': v}, {'n': 10, 'var': v}]
        cube._simulate_split2 = types.MethodType(lambda self: sim_vars(True), cube)
        cube._simulate_split4 = types.MethodType(lambda self: sim_vars(True), cube)
        # Low gamma -> allow split
        dec1 = cube.should_split(min_trials=5, min_points=10, gamma=0.05)
        self.assertIn(dec1, ('binary', 'quad'))
        # High gamma -> block split
        cube._simulate_split2 = types.MethodType(lambda self: sim_vars(False), cube)
        cube._simulate_split4 = types.MethodType(lambda self: sim_vars(False), cube)
        dec2 = cube.should_split(min_trials=5, min_points=10, gamma=0.2)
        self.assertEqual(dec2, 'none')


if __name__ == '__main__':
    unittest.main(verbosity=2)
