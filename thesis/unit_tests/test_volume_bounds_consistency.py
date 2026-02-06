"""
Unit tests per verificare la consistenza dei bounds e del volume dopo split.

Bug corretto: le widths dei child esplodevano invece di dimezzarsi,
causando overflow del termine geometrico UCB.

Test strategy:
1. Verifica che child widths ≤ parent widths (sempre!)
2. Verifica che volume child < volume parent
3. Verifica che UCB rimanga in range ragionevole
4. Verifica per dimensionalità diverse (5, 10, 20, 50)
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hpo_curvature import QuadCube, QuadHPO


class TestVolumeBoundsConsistency:
    """Test che i bounds e il volume rimangano consistenti dopo split."""
    
    @pytest.mark.parametrize("d", [5, 10, 20, 50])
    def test_child_widths_never_exceed_parent_split2(self, d):
        """Child widths devono essere ≤ parent widths dopo split2."""
        bounds = [(0.0, 1.0)] * d
        parent = QuadCube(bounds=bounds)
        
        # Aggiungi alcuni punti per permettere split
        np.random.seed(42)
        for _ in range(10):
            x = np.random.rand(d)
            parent.add_tested_point(x)
            parent.update_final(np.random.rand())
        
        # Esegui split2
        children = parent.split2()
        
        # Verifica che ogni child abbia widths ≤ parent
        parent_widths = np.array([hi - lo for lo, hi in parent.bounds])
        for i, child in enumerate(children):
            child_widths = np.array([hi - lo for lo, hi in child.bounds])
            
            # CRITICAL: child widths devono essere ≤ parent widths
            assert np.all(child_widths <= parent_widths + 1e-10), (
                f"Child {i} has widths > parent! "
                f"parent_widths={parent_widths[:3]}..., "
                f"child_widths={child_widths[:3]}..."
            )
    
    @pytest.mark.parametrize("d", [5, 10, 20, 50])
    def test_child_widths_never_exceed_parent_split4(self, d):
        """Child widths devono essere ≤ parent widths dopo split4."""
        if d < 2:
            pytest.skip("split4 richiede almeno 2 dimensioni")
        
        bounds = [(0.0, 1.0)] * d
        parent = QuadCube(bounds=bounds)
        
        # Aggiungi alcuni punti per permettere split
        np.random.seed(42)
        for _ in range(15):
            x = np.random.rand(d)
            parent.add_tested_point(x)
            parent.update_final(np.random.rand())
        
        # Esegui split4
        children = parent.split4()
        
        # Verifica che ogni child abbia widths ≤ parent
        parent_widths = np.array([hi - lo for lo, hi in parent.bounds])
        for i, child in enumerate(children):
            child_widths = np.array([hi - lo for lo, hi in child.bounds])
            
            # CRITICAL: child widths devono essere ≤ parent widths
            assert np.all(child_widths <= parent_widths + 1e-10), (
                f"Child {i} has widths > parent! "
                f"d={d}, parent_widths={parent_widths[:5]}, "
                f"child_widths={child_widths[:5]}"
            )
    
    @pytest.mark.parametrize("d", [5, 10, 20])
    def test_volume_decreases_after_split(self, d):
        """Volume deve SEMPRE decrescere dopo split."""
        bounds = [(0.0, 1.0)] * d
        parent = QuadCube(bounds=bounds)
        
        # Calcola volume parent
        parent_widths = np.array([hi - lo for lo, hi in parent.bounds])
        vol_parent = np.prod(parent_widths)
        
        # Aggiungi punti
        np.random.seed(42)
        for _ in range(15):
            x = np.random.rand(d)
            parent.add_tested_point(x)
            parent.update_final(np.random.rand())
        
        # Test split4
        children = parent.split4()
        
        for i, child in enumerate(children):
            child_widths = np.array([hi - lo for lo, hi in child.bounds])
            vol_child = np.prod(child_widths)

            # CRITICAL: volume child deve essere < volume parent
            assert vol_child < vol_parent + 1e-10, (
                f"Child {i} volume {vol_child} >= parent volume {vol_parent}!"
            )

            # Volume non negativo
            assert vol_child >= 0, f"Child {i} has negative volume {vol_child}"

    @pytest.mark.parametrize("d", [10, 20, 50])
    def test_multiple_splits_volume_exponential_decay(self, d):
        """Dopo N splits, volume deve decrescere esponenzialmente."""
        bounds = [(0.0, 1.0)] * d
        root = QuadCube(bounds=bounds)
        
        # Aggiungi punti al root
        np.random.seed(42)
        for _ in range(20):
            x = np.random.rand(d)
            root.add_tested_point(x)
            root.update_final(np.random.rand())
        
        # Primo split
        children1 = root.split4()
        assert len(children1) == 4
        
        # Prendi un child e splitta di nuovo
        child1 = children1[0]
        for _ in range(10):
            x = np.random.rand(d)
            child1.add_tested_point(x)
            child1.update_final(np.random.rand())
        
        children2 = child1.split4()
        assert len(children2) == 4
        
        # Calcola volumi
        vol_root = np.prod([hi - lo for lo, hi in root.bounds])
        vol_child1 = np.prod([hi - lo for lo, hi in children1[0].bounds])
        vol_child2 = np.prod([hi - lo for lo, hi in children2[0].bounds])

        # Verifica decrescita (o almeno non crescita se bounds degenerano)
        assert vol_child1 <= vol_root + 1e-10, "Volume cresce al primo split!"
        assert vol_child2 <= vol_child1 + 1e-10, "Volume cresce al secondo split!"
        
        # Se i volumi sono positivi, devono decrescere strettamente
        if vol_root > 1e-10 and vol_child1 > 1e-10:
            assert vol_child1 < vol_root, "Volume positivo non decresce al primo split!"
        if vol_child1 > 1e-10 and vol_child2 > 1e-10:
            assert vol_child2 < vol_child1, "Volume positivo non decresce al secondo split!"
        assert vol_child2 < vol_root * 0.25, "Volume non decresce esponenzialmente!"
    
    def test_ucb_stays_reasonable_after_splits(self):
        """UCB deve rimanere in range ragionevole anche dopo molti splits."""
        d = 20
        bounds = [(0.0, 1.0)] * d
        root = QuadCube(bounds=bounds)
        
        # Parametri UCB
        beta = 0.05
        lambda_geo = 0.8
        
        # Aggiungi punti al root
        np.random.seed(42)
        for _ in range(20):
            x = np.random.rand(d)
            root.add_tested_point(x)
            root.update_final(0.5 + np.random.rand() * 0.3)  # score in [0.5, 0.8]
        
        # UCB del root
        ucb_root = root.ucb(beta=beta, lambda_geo=lambda_geo)
        
        # Verifica UCB root ragionevole (con volume=1, n_trials~20)
        # UCB = base + 0.8 * 1.0 / sqrt(20+1) ≈ base + 0.17
        # base dovrebbe essere ~ 0.5-0.8, quindi UCB ~ 0.7-1.0
        assert 0.0 < ucb_root < 10.0, f"UCB root fuori range: {ucb_root}"
        
        # Primo split
        children1 = root.split4()
        
        for child in children1:
            ucb_child = child.ucb(beta=beta, lambda_geo=lambda_geo)
            
            # CRITICAL: UCB child deve essere ragionevole
            # Anche con heritage (n_trials ereditato), il volume è ~0.25 del root
            # Quindi bonus ~ 0.8 * 0.25 / sqrt(...) che è MINORE del root
            assert not np.isnan(ucb_child), "UCB child is NaN!"
            assert not np.isinf(ucb_child), "UCB child is infinite!"
            assert abs(ucb_child) < 1e10, f"UCB child overflow: {ucb_child}"
            
            # UCB dovrebbe essere simile o leggermente maggiore del root (per exploration)
            # ma MAI ordini di grandezza più grande
            assert ucb_child < ucb_root * 100, (
                f"UCB child {ucb_child} è 100x più grande del root {ucb_root}!"
            )
    
    def test_no_width_explosion_with_pca_rotation(self):
        """
        Regression test per il bug specifico:
        Composizione di rotazioni PCA causava widths > 1.0
        """
        d = 20
        bounds = [(0.0, 1.0)] * d
        root = QuadCube(bounds=bounds)
        
        # Aggiungi punti con pattern che causa PCA rotation
        np.random.seed(42)
        for _ in range(30):
            # Punti correlati per triggerare PCA
            base = np.random.rand(d) * 0.5
            noise = np.random.randn(d) * 0.1
            x = np.clip(base + noise, 0.0, 1.0)
            root.add_tested_point(x)
            root.update_final(np.random.rand())
        
        # Split con PCA attiva
        children = root.split4()
        
        for i, child in enumerate(children):
            child_widths = np.array([hi - lo for lo, hi in child.bounds])

            # CRITICAL: nessuna width può essere > 1.0 (lo spazio originale è [0,1])
            max_width = np.max(child_widths)
            assert max_width <= 1.0 + 1e-10, (
                f"Child {i} has width > 1.0: max_width={max_width}, "
                f"widths[:5]={child_widths[:5]}"
            )

    def test_volume_bug_regression(self):
        """
        Regression test specifico per il bug osservato:
        Volume esplodeva a 10^20 invece di decrescere.
        """
        d = 20
        bounds = [(0.0, 1.0)] * d
        
        # Simula la sequenza che causava il bug
        root = QuadCube(bounds=bounds)
        
        np.random.seed(42)
        for _ in range(20):
            x = np.random.rand(d)
            root.add_tested_point(x)
            root.update_final(np.random.rand())
        
        # Split sequence: root → children1 → children2
        children1 = root.split4()
        
        # Aggiungi punti ai children
        for child in children1:
            for _ in range(10):
                x = np.random.rand(d)
                child.add_tested_point(x)
                child.update_final(np.random.rand())
        
        # Split un child
        children2 = children1[0].split4()
        
        # Aggiungi punti e splitta ancora
        for child in children2:
            for _ in range(10):
                x = np.random.rand(d)
                child.add_tested_point(x)
                child.update_final(np.random.rand())
        
        children3 = children2[0].split4()
        
        # Verifica che il volume NON esploda
        for depth, cubes in [(1, children1), (2, children2), (3, children3)]:
            for i, cube in enumerate(cubes):
                widths = np.array([hi - lo for lo, hi in cube.bounds])
                vol = np.prod(widths)
                
                # CRITICAL: volume deve rimanere < 1.0 (volume dello spazio originale)
                assert vol < 1.0 + 1e-10, (
                    f"Depth {depth}, cube {i}: volume {vol} > 1.0! BUG REGRESSION!"
                )
                
                # Con 3 livelli di split4 (6 splits totali, ~12 assi splittati su 20),
                # volume dovrebbe essere ~ 0.5^12 = 0.0002
                # In pratica è più grande perché non tutti split toccano assi nuovi,
                # ma deve essere << 1.0
                if depth == 3:
                    assert vol < 0.1, (
                        f"Depth 3 volume {vol} troppo grande! "
                        f"Dovrebbe essere molto < 1.0"
                    )
    
    def test_ucb_overflow_regression(self):
        """
        Regression test per UCB overflow osservato (10^19 - 10^40).
        """
        d = 20
        bounds = [(0.0, 1.0)] * d
        
        beta = 0.05
        lambda_geo = 0.8
        
        # Crea una sequenza di split profonda
        root = QuadCube(bounds=bounds)
        
        np.random.seed(42)
        for _ in range(20):
            x = np.random.rand(d)
            root.add_tested_point(x)
            root.update_final(0.5 + np.random.rand() * 0.3)
        
        current_cubes = [root]
        
        # Fai 4 livelli di split
        for depth in range(4):
            next_cubes = []
            for cube in current_cubes:
                # Aggiungi punti
                for _ in range(15):
                    x = np.random.rand(d)
                    cube.add_tested_point(x)
                    cube.update_final(0.5 + np.random.rand() * 0.3)
                
                # Split
                if len(cube.bounds) >= 2:
                    children = cube.split4()
                    next_cubes.extend(children)
            
            current_cubes = next_cubes if next_cubes else current_cubes
            
            # Verifica UCB per tutti i cubi a questo depth
            for i, cube in enumerate(current_cubes):
                ucb_val = cube.ucb(beta=beta, lambda_geo=lambda_geo)
                
                # CRITICAL: UCB non deve andare in overflow
                assert not np.isnan(ucb_val), f"Depth {depth}, cube {i}: UCB is NaN!"
                assert not np.isinf(ucb_val), f"Depth {depth}, cube {i}: UCB is inf!"
                assert abs(ucb_val) < 1e10, (
                    f"Depth {depth}, cube {i}: UCB overflow {ucb_val}! "
                    f"BUG REGRESSION!"
                )
                
                # UCB dovrebbe essere ragionevole: base ~ [0.5, 0.9], bonus < 1.0
                # Quindi UCB < 2.0 in condizioni normali
                assert ucb_val < 100.0, (
                    f"Depth {depth}, cube {i}: UCB {ucb_val} troppo grande!"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
