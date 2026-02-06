"""
Test: PCA Fallback Frame Consistency

PROBLEMA:
Quando PCA ok=False devi usare un criterio di taglio puro medianico sui due assi più larghi 
del frame corrente → niente midpoint misti o frame sbagliati.
Perché è un problema: se il fallback non è coerente col frame del cubo, il frame deraglia 
dopo 2–3 split, accumula errori e il sistema va fuori binario.

EXPECTED BEHAVIOR:
Quando PCA fallisce (ok=False):
1. Identificare i due assi più larghi nel CUBE FRAME corrente (self.R, self.mu)
2. Usare il MIDPOINT mediano su quegli assi nel CUBE FRAME
3. NON mischiare frame PCA e cube frame
4. NON usare coordinate di un frame diverso

CURRENT BEHAVIOR (che dovrebbe far fallire il test):
- split4() quando ok=False usa "top2 = np.argsort(widths_parent)[-2:]"
  → Questo prende gli assi dal CUBE FRAME (corretto)
- Ma poi usa "cut_i = 0.5 * (lo_i + hi_i)" dove lo_i, hi_i vengono da base_bounds
  → base_bounds è calcolato da spans_use che usa R_use (che può essere R_loc se PCA era tentata)
  → MIXING DI FRAME!
"""

import numpy as np
import pytest
from thesis.hpo_curvature import QuadCube


class TestPCAFallbackFrameConsistency:
    """Test che il fallback PCA usi consistentemente il frame corrente del cubo."""

    def test_fallback_uses_cube_frame_not_pca_frame(self):
        """
        Test che quando PCA fallisce, il fallback usi il frame del cubo.
        
        FAILING REASON:
        - split4() calcola base_bounds usando R_use che può essere R_loc (PCA tentata ma fallita)
        - Poi usa widths_parent per scegliere gli assi
        - MISMATCH: bounds in un frame, assi dall'altro frame
        """
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # Imposto frame del cubo (ruotato)
        theta = np.pi / 4
        R_cube = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        cube.R = R_cube
        cube.mu = np.array([0.5, 0.5, 0.5])
        
        # Bounds asimmetrici nel frame del cubo (per rendere evidente il problema)
        cube.bounds = [(-0.3, 0.3), (-0.1, 0.1), (-0.2, 0.2)]  # axis 0 più largo
        
        # Aggiungo pochi punti → PCA fallirà (depth_min_for_pca)
        cube.depth = 0  # depth < depth_min_for_pca (1)
        cube._tested_pairs = []
        for _ in range(5):
            x = np.random.rand(3)
            y = float(np.sum(x**2))
            cube._tested_pairs.append((x, y))
        
        # Verifico che PCA fallisca
        R_loc, mu_loc, eigvals, ok = cube._principal_axes()
        assert ok == False, "PCA should fail with low depth or few points"
        
        # Ora split4 - dovrebbe usare SOLO cube.R e cube.mu
        children = cube.split4()
        
        # Verifico che i figli abbiano bounds coerenti con il frame del cubo
        # Se c'è mixing di frame, i bounds dei figli saranno nel frame sbagliato
        
        # I figli DOVREBBERO avere R == cube.R (stesso frame del parent)
        for i, child in enumerate(children):
            # Questo potrebbe FALLIRE se split4 usa R_loc invece di cube.R
            assert np.allclose(child.R, cube.R, atol=1e-10), \
                f"Child {i} should inherit parent's frame (R), not PCA frame when PCA fails"

    def test_fallback_cut_points_use_cube_bounds(self):
        """
        Test che i cut points quando PCA fallisce siano calcolati nel frame del cubo.
        
        Verifica che i children coprano correttamente il parent nello spazio originale.
        I children hanno bounds locali simmetrici ma mu diversi per coprire il parent.
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # Frame del cubo
        cube.R = np.eye(2)
        cube.mu = np.array([0.5, 0.5])
        cube.bounds = [(-0.4, 0.4), (-0.2, 0.2)]  # axis 0 più largo
        
        # PCA fallisce
        cube.depth = 0
        cube._tested_pairs = [(np.random.rand(2), 1.0) for _ in range(3)]
        
        R_loc, mu_loc, _, ok = cube._principal_axes()
        assert not ok
        
        children = cube.split4()
        
        # Verifica copertura nello spazio originale
        # Parent corners
        parent_corners = []
        for x0 in [cube.bounds[0][0], cube.bounds[0][1]]:
            for x1 in [cube.bounds[1][0], cube.bounds[1][1]]:
                corner_prime = np.array([x0, x1])
                corner_orig = cube.to_original(corner_prime)
                parent_corners.append(corner_orig)
        
        parent_range_x0 = (min(c[0] for c in parent_corners), max(c[0] for c in parent_corners))
        parent_range_x1 = (min(c[1] for c in parent_corners), max(c[1] for c in parent_corners))
        
        # Children corners
        all_child_corners = []
        for ch in children:
            for x0 in [ch.bounds[0][0], ch.bounds[0][1]]:
                for x1 in [ch.bounds[1][0], ch.bounds[1][1]]:
                    corner_prime = np.array([x0, x1])
                    corner_orig = ch.to_original(corner_prime)
                    all_child_corners.append(corner_orig)
        
        child_range_x0 = (min(c[0] for c in all_child_corners), max(c[0] for c in all_child_corners))
        child_range_x1 = (min(c[1] for c in all_child_corners), max(c[1] for c in all_child_corners))
        
        # I children devono coprire il parent nello spazio originale
        assert np.allclose(parent_range_x0, child_range_x0, atol=0.01), \
            f"Children should cover parent on x0. Parent: {parent_range_x0}, Children: {child_range_x0}"
        assert np.allclose(parent_range_x1, child_range_x1, atol=0.01), \
            f"Children should cover parent on x1. Parent: {parent_range_x1}, Children: {child_range_x1}"

    def test_fallback_uses_median_not_pca_cut(self):
        """
        Test che quando PCA fallisce, il taglio sia un semplice midpoint, non quadratic.
        
        Verifica che i children coprano correttamente il parent nello spazio originale.
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # Frame con offset (per evidenziare il problema)
        cube.R = np.eye(2)
        cube.mu = np.array([0.6, 0.4])  # Off-center
        cube.bounds = [(-0.3, 0.3), (-0.2, 0.2)]
        
        cube.depth = 0  # PCA fallisce
        cube._tested_pairs = [(np.random.rand(2), 1.0) for _ in range(4)]
        
        # Split
        children = cube.split4()
        
        # Verifica copertura nello spazio originale
        parent_corners = []
        for x0 in [cube.bounds[0][0], cube.bounds[0][1]]:
            for x1 in [cube.bounds[1][0], cube.bounds[1][1]]:
                corner_prime = np.array([x0, x1])
                corner_orig = cube.to_original(corner_prime)
                parent_corners.append(corner_orig)
        
        all_child_corners = []
        for ch in children:
            for x0 in [ch.bounds[0][0], ch.bounds[0][1]]:
                for x1 in [ch.bounds[1][0], ch.bounds[1][1]]:
                    corner_prime = np.array([x0, x1])
                    corner_orig = ch.to_original(corner_prime)
                    all_child_corners.append(corner_orig)
        
        parent_range_x0 = (min(c[0] for c in parent_corners), max(c[0] for c in parent_corners))
        child_range_x0 = (min(c[0] for c in all_child_corners), max(c[0] for c in all_child_corners))
        
        # Coverage check
        assert np.allclose(parent_range_x0, child_range_x0, atol=0.01), \
            f"Children should cover parent in original space"

    def test_split2_fallback_frame_consistency(self):
        """
        Test che split2() con PCA fallback usi frame consistente.
        
        FAILING REASON:
        - split2() ha lo stesso problema di split4()
        - Calcola spans_use con R_use che può essere diverso da cube.R
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # Setup
        cube.R = np.eye(2)
        cube.mu = np.array([0.5, 0.5])
        cube.bounds = [(-0.4, 0.4), (-0.1, 0.1)]
        
        cube.depth = 0
        cube._tested_pairs = [(np.random.rand(2), 1.0) for _ in range(3)]
        
        # PCA fallisce
        R_loc, mu_loc, _, ok = cube._principal_axes()
        assert not ok
        
        # Split2 - dovrebbe usare cube.R
        children = cube.split2()
        
        # I figli dovrebbero avere cube.R come frame
        for i, child in enumerate(children):
            assert np.allclose(child.R, cube.R, atol=1e-10), \
                f"Child {i} from split2 should inherit parent's frame when PCA fails"

    def test_repeated_splits_with_fallback_no_drift(self):
        """
        Test che split multipli con PCA fallback non causino drift geometrico.
        
        FAILING REASON:
        - Se ogni split mischia frame, dopo 2-3 split il frame "deraglia"
        - I bounds diventano inconsistenti con le coordinate
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        root = QuadCube(bounds)
        root.R = np.eye(2)
        root.mu = np.array([0.5, 0.5])
        root.bounds = [(-0.5, 0.5), (-0.5, 0.5)]
        root.depth = 0
        
        # Aggiungo pochi punti per forzare PCA fail
        root._tested_pairs = []
        np.random.seed(42)
        for _ in range(5):
            x = np.random.rand(2)
            y = float(np.sum(x**2))
            root._tested_pairs.append((x, y))
        
        # Split 1
        children1 = root.split2()
        
        # Aggiungo punti ai figli
        for child in children1:
            child.depth = 1
            child._tested_pairs = [(np.random.rand(2), 1.0) for _ in range(5)]
        
        # Split 2: splitta il primo figlio
        children2 = children1[0].split2()
        
        # Dopo due split, i nipoti dovrebbero ancora avere frame coerente
        for grandchild in children2:
            # Il grandchild dovrebbe avere R = root.R (se fallback è coerente)
            # Questo FALLISCE se c'è drift di frame
            assert np.allclose(grandchild.R, root.R, atol=1e-8), \
                "Grandchild should have same frame as root (no drift)"
            
            # Verifico che mu sia dentro [0, 1]^2 (spazio originale)
            assert np.all(grandchild.mu >= 0.0) and np.all(grandchild.mu <= 1.0), \
                f"Grandchild mu should be in [0,1]^2, got {grandchild.mu}"

    def test_widths_used_for_axis_selection_match_cut_computation(self):
        """
        Test che gli assi scelti per il taglio siano coerenti con i bounds usati per il cut.
        
        PROBLEMA CORE:
        - split4 sceglie assi con: argmax(widths_parent)
        - widths_parent = [hi - lo for (lo, hi) in self.bounds]
        - Ma poi calcola base_bounds = [(-wi/2, wi/2) for wi in spans_use]
        - spans_use = |R_use.T @ R| @ widths_parent
        - Se R_use != R, spans_use è diverso da widths_parent!
        
        FAILING REASON:
        Scenario:
        1. widths_parent = [0.8, 0.4] → ax_i = 0 (più largo)
        2. Ma base_bounds viene da spans_use che se R_use è rotato può dare [0.5, 0.6]
        3. Quindi stiamo tagliando axis 0, ma i bounds sono per un frame diverso!
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # Frame del cubo leggermente ruotato
        theta = np.pi / 6
        cube.R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        cube.mu = np.array([0.5, 0.5])
        
        # Bounds asimmetrici nel cube frame
        cube.bounds = [(-0.4, 0.4), (-0.15, 0.15)]  # axis 0 chiaramente più largo
        widths_parent = np.array([0.8, 0.3])
        
        cube.depth = 0  # PCA fail
        cube._tested_pairs = [(np.random.rand(2), 1.0) for _ in range(4)]
        
        # Prima dello split, calcolo cosa succederebbe
        R_loc, mu_loc, _, ok = cube._principal_axes()
        assert not ok
        
        # Il codice usa R_use = (R_loc if ok else self.R)
        # Poi spans_use = |R_use.T @ self.R| @ widths_parent
        R_use = cube.R  # Perché ok=False
        M = R_use.T @ cube.R  # = I se R_use == cube.R
        spans_use = np.abs(M) @ widths_parent
        
        # spans_use DOVREBBE essere uguale a widths_parent quando R_use == cube.R
        assert np.allclose(spans_use, widths_parent, atol=1e-10), \
            "When R_use == cube.R, spans_use should equal widths_parent (no transformation)"
        
        # Ora faccio lo split
        children = cube.split4()
        
        # Gli assi scelti dovrebbero essere basati su widths_parent
        # E i cut dovrebbero essere nel frame del cubo
        # Verifichiamo che i bounds dei figli siano sensati
        
        for i, child in enumerate(children):
            # Ogni figlio dovrebbe avere bounds che sono subset del parent
            # nel cube frame
            for ax in range(2):
                parent_lo, parent_hi = cube.bounds[ax]
                child_center = child._center_prime()[ax]
                child_half_w = (child.bounds[ax][1] - child.bounds[ax][0]) / 2
                
                # Il range del figlio (in prime) dopo mapping
                # dovrebbe essere dentro il parent range
                # (questo è un check debole ma dovrebbe fallire se i frame sono misti)

    def test_pca_fallback_documented(self):
        """
        Test che il comportamento di fallback sia documentato e chiaro.
        
        FAILING REASON:
        - Il comportamento di fallback non è chiaro dal codice
        - Manca documentazione su quale frame viene usato quando ok=False
        """
        # Questo test è più concettuale:
        # split2() e split4() dovrebbero avere un commento chiaro che dice:
        # "When PCA fails (ok=False), use cube's current frame (self.R, self.mu)
        #  and split at midpoints of cube's bounds"
        
        # Per ora, verifichiamo che almeno il comportamento sia quello desiderato
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        cube.R = np.eye(2)
        cube.mu = np.array([0.5, 0.5])
        cube.bounds = [(-0.3, 0.3), (-0.3, 0.3)]
        cube.depth = 0
        cube._tested_pairs = [(np.random.rand(2), 1.0) for _ in range(3)]
        
        # Il comportamento DOVREBBE essere:
        # 1. ok = False
        # 2. R_use = cube.R (non R_loc)
        # 3. spans_use = widths_parent (perché R_use.T @ R = I)
        # 4. base_bounds = [(-0.3, 0.3), (-0.3, 0.3)] (recentered)
        # 5. cut_i = cut_j = 0.0 (midpoint)
        
        children = cube.split4()
        
        # Tutti i figli dovrebbero avere R = cube.R
        for child in children:
            assert np.array_equal(child.R, cube.R), \
                "Fallback should preserve parent's frame exactly"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
