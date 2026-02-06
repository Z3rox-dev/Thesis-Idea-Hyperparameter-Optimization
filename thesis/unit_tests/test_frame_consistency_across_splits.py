"""
Test: Frame Consistency Across Splits

PROBLEMA:
"La logica per i tagli c'è già, ma va mantenuta coerente con il frame corrente 
per evitare drift dopo più split"
→ il punto cruciale: devi usare sempre lo stesso frame quando scegli l'asse, 
quando applichi il taglio e quando costruisci i figli.
Perché è un problema: se mescoli PCA-frame e cube-frame, ottieni drift numerico e geometrico.

EXPECTED BEHAVIOR:
1. Scegli UN frame (PCA o cube)
2. USA QUEL frame per:
   - Identificare gli assi più larghi
   - Calcolare i cut points
   - Costruire i bounds dei figli
   - Impostare R e mu dei figli
3. NON mischiare frame in nessun passo

CURRENT BEHAVIOR (che dovrebbe far fallire il test):
In split4():
- Calcola R_loc, mu_loc, ok = _principal_axes()
- R_use = R_loc if ok else self.R  ✓ Scelta del frame
- M = R_use.T @ self.R  ← MIXING! Se R_use è PCA, moltiplichi PCA con cube
- spans_use = |M| @ widths_parent  ← Widths in frame misto
- base_bounds = [(-wi/2, wi/2) for wi in spans_use]  ← Bounds in frame misto
- Poi sceglie assi con argmax(widths_parent)  ← Assi nel cube frame
- Ma usa base_bounds[ax_i] per i cut  ← Bounds in frame diverso!
→ INCOERENZA TOTALE
"""

import numpy as np
import pytest
from thesis.hpo_curvature import QuadCube


class TestFrameConsistencyAcrossSplits:
    """Test che il frame usato sia consistente in tutti i passi dello split."""

    def test_split4_uses_single_frame_throughout(self):
        """
        Test che split4() usi UN solo frame per tutto il processo.
        
        FAILING REASON:
        - R_use viene scelto (PCA o cube)
        - Ma poi M = R_use.T @ self.R mescola i frame
        - spans_use è nel frame misto
        - Gli assi vengono scelti con widths_parent (cube frame)
        - I cut vengono fatti con base_bounds (frame misto)
        → DRIFT GEOMETRICO
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # Setup frame del cubo ruotato
        theta = np.pi / 4
        R_cube = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        cube.R = R_cube
        cube.mu = np.array([0.5, 0.5])
        cube.bounds = [(-0.4, 0.4), (-0.2, 0.2)]
        
        # Aggiungo punti per far succedere PCA (ma con frame diverso)
        cube.depth = 2
        cube._tested_pairs = []
        np.random.seed(42)
        
        # Punti in una direzione diversa dalla rotazione del cubo
        # (per far sì che R_PCA != R_cube)
        for i in range(15):
            # Creo punti lungo una diagonale diversa
            t = np.random.rand()
            x = np.array([0.3 + 0.3 * t, 0.2 + 0.6 * t])
            y = float(1.0 - t)
            cube._tested_pairs.append((x, y))
        
        # PCA dovrebbe dare un frame diverso da R_cube
        R_pca, mu_pca, eigvals, ok = cube._principal_axes()
        
        if ok:
            # Se PCA è ok, R_pca dovrebbe essere diverso da R_cube
            # (a meno di simmetrie accidentali)
            
            # Ora split4 - DOVREBBE usare R_pca consistentemente
            children = cube.split4()
            
            # Tutti i figli DOVREBBERO avere R = R_pca
            for i, child in enumerate(children):
                # Questo FALLISCE perché split4 mescola frame
                # I figli potrebbero avere R diverso o bounds in frame sbagliato
                assert np.allclose(child.R, R_pca, atol=1e-8) or np.allclose(child.R, R_cube, atol=1e-8), \
                    f"Child {i} should have consistent frame (either all PCA or all cube)"
                
                # Più importante: tutti i figli dovrebbero avere LO STESSO R
                if i > 0:
                    assert np.allclose(child.R, children[0].R, atol=1e-10), \
                        f"All children should have the same R (frame consistency)"

    def test_axis_selection_matches_cut_frame(self):
        """
        Test che gli assi scelti per split siano nello stesso frame dei cut.
        
        Verifica che i children coprano correttamente il parent nello spazio originale.
        """
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # Setup con bounds fortemente asimmetrici
        cube.R = np.eye(3)
        cube.mu = np.array([0.5, 0.5, 0.5])
        cube.bounds = [(-0.5, 0.5), (-0.1, 0.1), (-0.2, 0.2)]
        # widths: [1.0, 0.2, 0.4] → asse 0 più largo, poi 2, poi 1
        
        cube.depth = 0  # PCA fallisce
        cube._tested_pairs = [(np.random.rand(3), 1.0) for _ in range(3)]
        
        # Split
        children = cube.split4()
        
        # Verifica copertura nello spazio originale
        parent_corners = []
        for x0 in [cube.bounds[0][0], cube.bounds[0][1]]:
            for x1 in [cube.bounds[1][0], cube.bounds[1][1]]:
                for x2 in [cube.bounds[2][0], cube.bounds[2][1]]:
                    corner_prime = np.array([x0, x1, x2])
                    corner_orig = cube.to_original(corner_prime)
                    parent_corners.append(corner_orig)
        
        all_child_corners = []
        for ch in children:
            for x0 in [ch.bounds[0][0], ch.bounds[0][1]]:
                for x1 in [ch.bounds[1][0], ch.bounds[1][1]]:
                    for x2 in [ch.bounds[2][0], ch.bounds[2][1]]:
                        corner_prime = np.array([x0, x1, x2])
                        corner_orig = ch.to_original(corner_prime)
                        all_child_corners.append(corner_orig)
        
        # Check coverage per ogni dimensione
        for dim in range(3):
            parent_range = (min(c[dim] for c in parent_corners), max(c[dim] for c in parent_corners))
            child_range = (min(c[dim] for c in all_child_corners), max(c[dim] for c in all_child_corners))
            
            assert np.allclose(parent_range, child_range, atol=0.05), \
                f"Dimension {dim}: children should cover parent. Parent: {parent_range}, Children: {child_range}"

    def test_cut_points_in_correct_frame(self):
        """
        Test che i cut points siano calcolati nel frame usato per i bounds.
        
        FAILING REASON:
        - _quad_cut_along_axis() viene chiamato con (axis_idx, R, mu)
        - Ma axis_idx può essere nel cube frame, mentre R è PCA frame
        - Il cut risultante è in un frame, ma i bounds sono in un altro
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        cube.R = np.eye(2)
        cube.mu = np.array([0.5, 0.5])
        cube.bounds = [(-0.3, 0.3), (-0.3, 0.3)]
        
        # Setup per PCA success con frame diverso
        cube.depth = 2
        cube._tested_pairs = []
        np.random.seed(42)
        for i in range(12):
            # Punti lungo una linea a 30 gradi
            t = np.random.randn() * 0.2
            angle = np.pi / 6
            x = np.array([0.5 + t * np.cos(angle), 0.5 + t * np.sin(angle)])
            x = np.clip(x, 0.0, 1.0)
            y = float(t ** 2)
            cube._tested_pairs.append((x, y))
        
        R_pca, mu_pca, eigvals, ok = cube._principal_axes()
        
        if ok:
            # Il cut dovrebbe essere nel frame PCA
            # axis_idx = 0 (PC1)
            cut_0 = cube._quad_cut_along_axis(0, R_pca, mu_pca)
            
            # Il cut è in PCA prime coordinates
            # Ma se axis_idx viene scelto in cube frame → PROBLEMA
            
            # Verifichiamo che il cut sia sensato rispetto ai bounds nel PCA frame
            # Proietto i bounds del cube nel PCA frame per vedere il range
            
            # Parent box corners nel cube frame
            corners_prime = []
            for s0 in [-0.3, 0.3]:
                for s1 in [-0.3, 0.3]:
                    corners_prime.append(np.array([s0, s1]))
            
            # Map to original space
            corners_orig = [cube.mu + cube.R @ cp for cp in corners_prime]
            
            # Map to PCA frame
            corners_pca_prime = [(R_pca.T @ (co - mu_pca)) for co in corners_orig]
            
            # Range sull'asse 0 nel PCA frame
            pca_ax0_vals = [cp[0] for cp in corners_pca_prime]
            pca_range = (min(pca_ax0_vals), max(pca_ax0_vals))
            
            # Il cut DOVREBBE essere dentro questo range
            assert pca_range[0] <= cut_0 <= pca_range[1], \
                f"Cut should be within PCA frame range {pca_range}, got {cut_0}"

    def test_children_bounds_in_parent_frame(self):
        """
        Test che i children coprano correttamente il parent nello spazio originale.
        
        I children possono ereditare un frame diverso (es. PCA) dal parent,
        ma devono coprire correttamente il parent nello spazio originale.
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        parent = QuadCube(bounds)
        parent.R = np.eye(2)
        parent.mu = np.array([0.5, 0.5])
        parent.bounds = [(-0.4, 0.4), (-0.4, 0.4)]
        parent.depth = 2
        
        # Setup per far succedere PCA
        parent._tested_pairs = []
        np.random.seed(42)
        for i in range(15):
            x = np.random.rand(2)
            y = float(np.sum(x ** 2))
            parent._tested_pairs.append((x, y))
        
        children = parent.split4()
        
        # Verifica copertura nello spazio originale
        parent_corners = []
        for x0 in [parent.bounds[0][0], parent.bounds[0][1]]:
            for x1 in [parent.bounds[1][0], parent.bounds[1][1]]:
                corner_prime = np.array([x0, x1])
                corner_orig = parent.to_original(corner_prime)
                parent_corners.append(corner_orig)
        
        all_child_corners = []
        for ch in children:
            for x0 in [ch.bounds[0][0], ch.bounds[0][1]]:
                for x1 in [ch.bounds[1][0], ch.bounds[1][1]]:
                    corner_prime = np.array([x0, x1])
                    corner_orig = ch.to_original(corner_prime)
                    all_child_corners.append(corner_orig)
        
        # Check coverage per ogni dimensione
        # Children may cover MORE than parent due to frame rotation, but should at least cover parent
        for dim in range(2):
            parent_range = (min(c[dim] for c in parent_corners), max(c[dim] for c in parent_corners))
            child_range = (min(c[dim] for c in all_child_corners), max(c[dim] for c in all_child_corners))
            
            # Check parent is contained within child range (with tolerance)
            assert child_range[0] - 0.1 <= parent_range[0] <= child_range[1] + 0.1, \
                f"Dimension {dim}: parent min should be covered. Parent: {parent_range}, Children: {child_range}"
            assert child_range[0] - 0.1 <= parent_range[1] <= child_range[1] + 0.1, \
                f"Dimension {dim}: parent max should be covered. Parent: {parent_range}, Children: {child_range}"

    def test_no_frame_drift_after_deep_splits(self):
        """
        Test che dopo molti split, non ci sia drift geometrico.
        
        FAILING REASON:
        - Ogni split può introdurre un piccolo errore di frame mixing
        - Dopo molti split, questi errori si accumulano
        - I bounds diventano inconsistenti con le coordinate originali
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        root = QuadCube(bounds)
        root.R = np.eye(2)
        root.mu = np.array([0.5, 0.5])
        root.bounds = [(-0.5, 0.5), (-0.5, 0.5)]
        root.depth = 0
        
        # Simulate multiple generations of splits
        current_gen = [root]
        
        np.random.seed(42)
        for gen in range(3):  # 3 generazioni
            next_gen = []
            for cube in current_gen:
                cube.depth = gen
                cube._tested_pairs = []
                for _ in range(15):
                    x = np.random.rand(2)
                    y = float(np.sum(x ** 2))
                    cube._tested_pairs.append((x, y))
                
                children = cube.split2()
                next_gen.extend(children)
            
            current_gen = next_gen
        
        # Dopo 3 generazioni, i leaf dovrebbero ancora avere frame coerente
        for leaf in current_gen:
            # Tutti dovrebbero avere frame consistente con root
            # (potrebbero essere ruotati da PCA, ma consistentemente)
            
            # Il centro del leaf dovrebbe essere dentro [0, 1]^2
            assert np.all(leaf.mu >= -0.1) and np.all(leaf.mu <= 1.1), \
                f"Leaf mu should be roughly in [0,1]^2, got {leaf.mu} (drift detected)"
            
            # Sample un punto random nel leaf e verifica sia in [0,1]^2
            x_prime = leaf.sample_uniform_prime()
            x_orig = leaf.to_original(x_prime)
            
            assert np.all(x_orig >= -0.01) and np.all(x_orig <= 1.01), \
                f"Sampled point should be in [0,1]^2, got {x_orig} (geometric drift)"

    def test_transform_chain_original_prime_original(self):
        """
        Test che la catena di trasformazioni sia consistente dopo split.
        
        Se mischio frame, to_original(to_prime(x)) != x
        
        FAILING REASON:
        - Se child.R e child.mu sono in un frame diverso dai bounds,
          la trasformazione round-trip fallisce
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        parent = QuadCube(bounds)
        parent.R = np.eye(2)
        parent.mu = np.array([0.5, 0.5])
        parent.bounds = [(-0.4, 0.4), (-0.4, 0.4)]
        parent.depth = 2
        
        parent._tested_pairs = []
        np.random.seed(42)
        for _ in range(15):
            x = np.random.rand(2)
            y = float(np.sum(x ** 2))
            parent._tested_pairs.append((x, y))
        
        children = parent.split4()
        
        for i, child in enumerate(children):
            # Prendo un punto nel child (original space)
            x_orig_expected = np.random.rand(2)
            
            # Transform to child prime
            x_prime = child.to_prime(x_orig_expected)
            
            # Transform back
            x_orig_recovered = child.to_original(x_prime)
            
            # Dovrebbero essere uguali (entro tolleranza numerica)
            # Questo FALLISCE se R e mu non sono coerenti
            assert np.allclose(x_orig_recovered, x_orig_expected, atol=1e-8), \
                f"Child {i}: to_original(to_prime(x)) != x. Expected {x_orig_expected}, got {x_orig_recovered}"

    def test_split2_frame_consistency(self):
        """
        Test che split2() abbia la stessa consistenza di frame di split4().
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        cube.R = np.eye(2)
        cube.mu = np.array([0.5, 0.5])
        cube.bounds = [(-0.5, 0.5), (-0.5, 0.5)]
        cube.depth = 2
        
        cube._tested_pairs = []
        np.random.seed(42)
        for _ in range(15):
            x = np.random.rand(2)
            y = float(np.sum(x ** 2))
            cube._tested_pairs.append((x, y))
        
        children = cube.split2()
        
        # Gli stessi test di frame consistency dovrebbero valere
        for i, child in enumerate(children):
            # I figli dovrebbero avere lo stesso R (consistente)
            if i > 0:
                assert np.allclose(child.R, children[0].R, atol=1e-10), \
                    "split2: all children should have same R"
            
            # Round-trip transform
            x_test = np.random.rand(2)
            x_prime = child.to_prime(x_test)
            x_back = child.to_original(x_prime)
            assert np.allclose(x_back, x_test, atol=1e-8), \
                f"split2 child {i}: transform round-trip failed"

    def test_spans_use_calculation_is_correct(self):
        """
        Test che il calcolo di spans_use sia matematicamente corretto.
        
        PROBLEMA:
        spans_use = |R_use.T @ self.R| @ widths_parent
        
        Questo calcola quanto sono larghi i bounds del parent quando visti nel frame R_use.
        Ma il codice poi usa spans_use per creare base_bounds e scegliere assi.
        
        DOVREBBE:
        - Se R_use == self.R → spans_use == widths_parent (no transform)
        - Se R_use != self.R → spans_use è widths nel nuovo frame (OK)
        - MA gli assi dovrebbero essere scelti in R_use frame, non self.R frame!
        
        FAILING REASON:
        - Quando ok=False, split4 usa "top2 = argsort(widths_parent)"
        - widths_parent è nel self.R frame
        - Ma base_bounds è nel R_use frame
        - MISMATCH!
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # Frame ruotato
        theta = np.pi / 3
        cube.R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        cube.mu = np.array([0.5, 0.5])
        cube.bounds = [(-0.4, 0.4), (-0.2, 0.2)]
        widths_parent = np.array([0.8, 0.4])
        
        # Simulo PCA che da un frame diverso
        theta2 = np.pi / 4
        R_pca = np.array([
            [np.cos(theta2), -np.sin(theta2)],
            [np.sin(theta2), np.cos(theta2)]
        ])
        
        # Calcolo spans_use come fa il codice
        R_use = R_pca
        M = R_use.T @ cube.R
        spans_use = np.abs(M) @ widths_parent
        
        # spans_use è quanto sono larghi i bounds nel frame PCA
        # Ora, se scelgo gli assi con argmax(widths_parent), sto scegliendo nel frame sbagliato!
        ax_widths = int(np.argmax(widths_parent))  # Asse nel cube frame
        ax_spans = int(np.argmax(spans_use))        # Asse nel PCA frame
        
        # Questi DOVREBBERO essere diversi se i frame sono diversi
        # Ma il codice usa ax_widths per indicizzare base_bounds che è in spans_use frame!
        
        # Questo test documenta il problema:
        if not np.allclose(cube.R, R_pca, atol=1e-6):
            # Se i frame sono diversi, gli assi più larghi potrebbero essere diversi
            if ax_widths != ax_spans:
                # PROBLEMA: il codice usa ax_widths ma base_bounds è in spans frame!
                pytest.fail(
                    f"Frame mixing detected: widest axis in cube frame ({ax_widths}) "
                    f"differs from widest axis in PCA frame ({ax_spans}), "
                    f"but code uses cube frame index on PCA frame bounds!"
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
