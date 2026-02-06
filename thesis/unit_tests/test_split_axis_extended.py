import numpy as np
import types
import unittest

from hpo_curvature import QuadCube

class TestSplitAxisExtended(unittest.TestCase):
    def make_cube(self, bounds, depth=2):
        c = QuadCube(bounds=list(bounds))
        d = len(bounds)
        c.R = np.eye(d)
        c.mu = np.zeros(d)
        c.depth = depth
        c._tested_points = []
        c._tested_pairs = []
        return c

    def half_axis(self, parent, children):
        pw = np.array([abs(hi-lo) for (lo,hi) in parent.bounds])
        cw = np.array([abs(hi-lo) for (lo,hi) in children[0].bounds])
        ratios = np.divide(cw, pw, out=np.ones_like(cw), where=pw>0)
        diffs = np.abs(ratios - 0.5)
        ax = int(np.argmin(diffs))
        self.assertTrue(np.isclose(ratios[ax],0.5,atol=1e-6))
        return ax

    def test_projected_span_axis_choice_differs_from_raw_width(self):
        # Construct a case where raw widest axis != projected widest axis
        # Raw widths: x:4, y:2, z:2. Rotation swaps x<->y so projected widths reorder.
        cube = self.make_cube([(-2,2),(-1,1),(-1,1)])
        # Force PCA reliable returning identity (so we only test branch with ok=True but R different)
        cube.R = np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=float)  # permutation
        cube._principal_axes = types.MethodType(lambda self: (np.eye(3), np.zeros(3), np.ones(3), True), cube)
        cube._curvature_scores = types.MethodType(lambda self: None, cube)
        children = cube.split2()
        chosen = self.half_axis(cube, children)
        # Projected width order: |I^T R| @ [4,2,2] = |R|@[4,2,2] = [2,4,2] so expect axis 1
        self.assertEqual(chosen,1, f"Expected axis 1 (projected widest), got {chosen}")

    def test_curvature_overrides_projected_span(self):
        cube = self.make_cube([(-1,1),(-1,1),(-1,1)])
        cube._principal_axes = types.MethodType(lambda self: (np.eye(3), np.zeros(3), np.ones(3), True), cube)
        cube._curvature_scores = types.MethodType(lambda self: np.array([0.5, 10.0]), cube)
        children = cube.split2()
        chosen = self.half_axis(cube, children)
        self.assertEqual(chosen,1, f"Curvature should force axis 1, got {chosen}")

    def test_curvature_fallback_no_pca(self):
        cube = self.make_cube([(-1,1),(-3,3),(-1,1)])
        # Force PCA not ok
        cube._principal_axes = types.MethodType(lambda self: (np.eye(3), np.zeros(3), np.ones(3), False), cube)
        cube._curvature_scores = types.MethodType(lambda self: None, cube)
        children = cube.split2()
        chosen = self.half_axis(cube, children)
        self.assertEqual(chosen,1, f"Fallback should choose widest raw axis 1, got {chosen}")

if __name__ == '__main__':
    unittest.main()
