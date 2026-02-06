import unittest
import numpy as np

from hpo_curvature import QuadHPO


class TestQuadHPOEndToEnd(unittest.TestCase):
    def test_optimize_sphere_like(self):
        # Maximize -||x-0.3||^2 on [0,1]^d
        def obj(x: np.ndarray, epochs: int = 0) -> float:
            return -float(np.sum((x - 0.3) ** 2))

        d = 4
        bounds = [(0.0, 1.0) for _ in range(d)]
        hpo = QuadHPO(bounds=bounds, beta=0.05, lambda_geo=0.2, full_epochs=1, maximize=True, rng_seed=0)
        hpo.optimize(obj, budget=40)
        # Sanity assertions
        self.assertIsNotNone(hpo.best_x_norm)
        self.assertEqual(len(hpo.best_x_norm), d)
        self.assertGreater(len(hpo.leaf_cubes), 0)
        diag = hpo.diagnostics()
        self.assertIn('max_depth', diag)
        self.assertLessEqual(diag['max_depth'], hpo.max_depth)
        # Best should be reasonably close to 0.3 (not exact, but better than random 0.5 expectation)
        best = np.array(hpo.best_x_norm)
        self.assertLess(np.linalg.norm(best - 0.3), np.sqrt(d) * 0.35)


if __name__ == '__main__':
    unittest.main(verbosity=2)
