import unittest
import numpy as np

import synthetic_functions as sf


class TestSyntheticFunctions(unittest.TestCase):
    def test_zero_minima(self):
        x0 = np.zeros(10, dtype=float)
        self.assertAlmostEqual(sf.sphere(x0), 0.0, places=12)
        self.assertAlmostEqual(sf.rastrigin(x0), 0.0, places=12)
        self.assertAlmostEqual(sf.ackley(x0), 0.0, places=8)
        self.assertAlmostEqual(sf.griewank(x0), 0.0, places=12)

    def test_random_finite(self):
        rng = np.random.default_rng(0)
        funcs = [sf.sphere, sf.rastrigin, sf.griewank, sf.ackley, sf.rosenbrock]
        for f in funcs:
            x = rng.standard_normal(10)
            v = f(x)
            self.assertTrue(np.isfinite(v), f"Function {f.__name__} returned non-finite value {v}")

    def test_funs_registry(self):
        self.assertIn('sphere', sf.FUNS)
        self.assertIn('rastrigin', sf.FUNS)
        func, bounds = sf.FUNS['sphere']
        self.assertTrue(callable(func))
        self.assertEqual(len(bounds), 10)
        self.assertTrue(all(len(b) == 2 for b in bounds))


if __name__ == '__main__':
    unittest.main(verbosity=2)
