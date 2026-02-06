import unittest
import numpy as np

from hpo_curvature import QuadHPO


class TestMinimizeMode(unittest.TestCase):
    def test_minimize_sphere(self):
        # Minimize sum((x-0.1)^2) on [0,1]^d
        def obj(x: np.ndarray, epochs: int = 0) -> float:
            return float(np.sum((x - 0.1) ** 2))

        d = 3
        bounds = [(0.0, 1.0) for _ in range(d)]
        hpo = QuadHPO(bounds=bounds, maximize=False, rng_seed=0, full_epochs=1)
        hpo.optimize(obj, budget=50)
        self.assertIsNotNone(hpo.best_x_norm)
        best = np.array(hpo.best_x_norm)
        # Should be reasonably close to 0.1 in L2
        self.assertLess(np.linalg.norm(best - 0.1), np.sqrt(d) * 0.4)
        # Best raw score near 0
        best_raw = float(hpo.sign * hpo.best_score_global)  # convert back to raw
        self.assertGreaterEqual(best_raw, -0.05)


if __name__ == '__main__':
    unittest.main(verbosity=2)
