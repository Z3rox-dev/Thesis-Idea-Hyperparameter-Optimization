import numpy as np
import types
import unittest

from hpo_curvature import QuadCube

class TestSplit4AxisExtended(unittest.TestCase):
    def cube3(self):
        c = QuadCube(bounds=[(-2,2),(-2,2),(-2,2)])
        c.R = np.eye(3)
        c.mu = np.zeros(3)
        c.depth = 2
        c._tested_points = [np.zeros(3) for _ in range(12)]
        c._tested_pairs = [(np.zeros(3),0.0) for _ in range(12)]
        return c

    def quadrant_counts(self, parent, children):
        pw = np.array([abs(hi-lo) for (lo,hi) in parent.bounds])
        cw = [np.array([abs(hi-lo) for (lo,hi) in ch.bounds]) for ch in children]
        # Ensure each chosen axis halved once among children
        halves = [np.where(np.isclose(c/pw,0.5,atol=1e-6))[0] for c in cw]
        return halves

    def test_curvature_picks_top_two(self):
        cube = self.cube3()
        cube._curvature_scores = types.MethodType(lambda self: np.array([5.0, 3.0]), cube)
        cube._principal_axes = types.MethodType(lambda self: (np.eye(3), np.zeros(3), np.array([5.0,3.0,1.0]), True), cube)
        children = cube.split4()
        # Expect split along PC1 and PC2, so both axes 0 and 1 each halved in children set
        pw = np.array([abs(hi-lo) for (lo,hi) in cube.bounds])
        halved_axes = set()
        for ch in children:
            cw = np.array([abs(hi-lo) for (lo,hi) in ch.bounds])
            for ax in range(3):
                if np.isclose(cw[ax]/pw[ax],0.5,atol=1e-6):
                    halved_axes.add(ax)
        self.assertEqual(halved_axes, {0,1}, f"Expected axes 0 and 1 halved, got {halved_axes}")

    def test_fallback_uses_widest_two_when_pca_not_ok(self):
        cube = self.cube3()
        cube._curvature_scores = types.MethodType(lambda self: None, cube)
        cube._principal_axes = types.MethodType(lambda self: (np.eye(3), np.zeros(3), np.ones(3), False), cube)
        # Modify bounds widths to make axis 2 narrower
        cube.bounds = [(-2,2), (-2,2), (-1,1)]
        children = cube.split4()
        pw = np.array([abs(hi-lo) for (lo,hi) in cube.bounds])
        halved_axes = set()
        for ch in children:
            cw = np.array([abs(hi-lo) for (lo,hi) in ch.bounds])
            for ax in range(3):
                if np.isclose(cw[ax]/pw[ax],0.5,atol=1e-6):
                    halved_axes.add(ax)
        self.assertEqual(halved_axes, {0,1}, f"Expected widest two raw axes 0,1 halved; got {halved_axes}")

if __name__ == '__main__':
    unittest.main()
