"""
Unit test: verificare che cubes con 0 osservazioni non vengano mai potati.
"""
import pytest
import numpy as np
from hpo_curvature import QuadHPO, QuadCube


class TestNoPruneUntestedCubes:
    """Test che verifica che cubes non testati non vengano mai pruned."""

    def test_untested_cubes_never_pruned(self):
        """
        Dopo uno split, alcuni child potrebbero non essere mai selezionati.
        Questi cubes con n_trials=0 NON devono essere pruned.
        """
        d = 5
        bounds = [(0.0, 1.0)] * d
        
        # Crea root e forza uno split
        root = QuadCube(bounds=bounds)
        np.random.seed(42)
        for _ in range(20):
            x = np.random.rand(d)
            root.add_tested_point(x)
            root.update_final(np.random.rand())
        
        children = root.split4()
        assert len(children) == 4
        
        # Simula scenario: solo 1 child viene testato, gli altri 3 no
        tested_child = children[0]
        for _ in range(5):
            x = np.random.rand(d)
            tested_child.add_tested_point(x)
            tested_child.update_final(np.random.rand())
        
        # Gli altri 3 hanno n_trials = 0
        for child in children[1:]:
            assert child.n_trials == 0, f"Child should have 0 trials, got {child.n_trials}"
        
        # Ora simula pruning con QuadHPO
        hpo = QuadHPO(
            bounds=bounds,
            beta=0.05,
            lambda_geo=0.8
        )
        # Override hardcoded values for testing
        hpo.max_depth = 3
        hpo.min_leaves = 2
        hpo.delta_prune = 0.5
        
        # Sostituisci leaf_cubes con i nostri children
        hpo.leaf_cubes = list(children)
        hpo.trial_id = 30  # Età sufficiente per superare grace period
        
        # Simula best_score_global alto per triggerare pruning
        hpo.best_score_global = 0.9
        
        # Set birth_trial per i cubes (prima del grace period)
        for child in children:
            child.birth_trial = 10  # age = 30 - 10 = 20 >> grace_period
        
        # Esegui pruning
        hpo.prune_cubes()
        
        # CRITICAL: I 3 cubes non testati devono essere ancora nelle foglie
        remaining_untested = [c for c in hpo.leaf_cubes if c.n_trials == 0]
        assert len(remaining_untested) == 3, (
            f"Expected 3 untested cubes to remain, but got {len(remaining_untested)}. "
            f"Remaining leaves: {len(hpo.leaf_cubes)}"
        )
        
        # Il cube testato potrebbe essere pruned o no, dipende dal suo UCB
        # Ma i non testati devono essere SEMPRE preservati

    def test_untested_cubes_preserved_with_min_leaves_fallback(self):
        """
        Anche quando min_leaves forza il mantenimento di cubes,
        quelli non testati devono essere inclusi automaticamente.
        """
        d = 10
        bounds = [(0.0, 1.0)] * d
        
        hpo = QuadHPO(
            bounds=bounds,
            beta=0.05,
            lambda_geo=0.8
        )
        # Override hardcoded values for testing
        hpo.max_depth = 3
        hpo.min_leaves = 5
        
        # Crea 10 cubes: 2 testati, 8 non testati
        cubes = []
        for i in range(10):
            cube = QuadCube(bounds=bounds)
            cube.birth_trial = 0
            if i < 2:
                # Testa solo i primi 2
                for _ in range(5):
                    x = np.random.rand(d)
                    cube.add_tested_point(x)
                    cube.update_final(np.random.rand())
            cubes.append(cube)
        
        hpo.leaf_cubes = cubes
        hpo.trial_id = 50
        hpo.best_score_global = 1.0  # High value per triggerare pruning
        
        # Esegui pruning
        hpo.prune_cubes()
        
        # Deve mantenere almeno min_leaves=5
        assert len(hpo.leaf_cubes) >= 5
        
        # TUTTI gli 8 non testati devono essere presenti
        remaining_untested = [c for c in hpo.leaf_cubes if c.n_trials == 0]
        assert len(remaining_untested) == 8, (
            f"Expected all 8 untested cubes preserved, got {len(remaining_untested)}"
        )

    def test_newly_created_cubes_zero_trials(self):
        """
        Verifica che cubes appena creati abbiano n_trials=0.
        """
        d = 5
        bounds = [(0.0, 1.0)] * d
        cube = QuadCube(bounds=bounds)
        
        assert cube.n_trials == 0, f"New cube should have n_trials=0, got {cube.n_trials}"
        assert len(cube.scores) == 0, f"New cube should have empty scores, got {len(cube.scores)}"

    def test_split_children_start_with_zero_trials(self):
        """
        Dopo split2() o split4(), tutti i children devono avere n_trials=0.
        """
        d = 10
        bounds = [(0.0, 1.0)] * d
        parent = QuadCube(bounds=bounds)
        
        # Aggiungi punti al parent
        np.random.seed(42)
        for _ in range(20):
            x = np.random.rand(d)
            parent.add_tested_point(x)
            parent.update_final(np.random.rand())
        
        # Test split2
        children2 = parent.split2()
        for child in children2:
            assert child.n_trials == 0, f"split2 child should have n_trials=0, got {child.n_trials}"
        
        # Test split4
        children4 = parent.split4()
        for child in children4:
            assert child.n_trials == 0, f"split4 child should have n_trials=0, got {child.n_trials}"
