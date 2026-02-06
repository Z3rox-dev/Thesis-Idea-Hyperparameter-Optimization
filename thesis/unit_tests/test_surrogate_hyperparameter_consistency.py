"""
Test: Surrogate Hyperparameter Consistency

PROBLEMA:
Inizializzi sempre gli iperparametri del surrogato nella __init__
→ vuole dire: centralizza gli iperparametri, non ridefinirli in giro.
Perché è un problema: il surrogato non è stabile → comportamento incoerente dopo gli split.

EXPECTED BEHAVIOR:
- ridge_alpha e altri iperparametri del surrogato dovrebbero essere definiti una volta
  nella __init__ o in una configurazione centrale
- Non dovrebbero essere ridefiniti localmente in fit_surrogate, _simulate_split2, _simulate_split4

CURRENT BEHAVIOR (che dovrebbe far fallire il test):
- ridge_alpha = 1e-3 è hardcoded in fit_surrogate()
- ridge_alpha = 1e-3 è ridefinito in _simulate_split2()
- ridge_alpha = 1e-3 è ridefinito in _simulate_split4()
- ridge_alpha = 1e-3 è ridefinito in _quad_cut_along_axis()
→ Se vuoi cambiare ridge_alpha, devi modificare 4 posti diversi!
"""

import numpy as np
import pytest
from thesis.hpo_curvature import QuadCube, QuadHPO


class TestSurrogateHyperparameterConsistency:
    """Test che il surrogato usi iperparametri consistenti e centralizzati."""

    def test_ridge_alpha_should_be_centralized(self):
        """
        Test che ridge_alpha sia definito in un posto solo e usato ovunque.
        
        FAILING REASON:
        - ridge_alpha è hardcoded in 4 metodi diversi
        - Non c'è un attributo cube.ridge_alpha o simile
        - Non c'è modo di cambiare ridge_alpha senza modificare il codice in 4 posti
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # DOVREBBE esistere un attributo centralizzato per ridge_alpha
        # Questo test FALLISCE perché l'attributo non esiste
        assert hasattr(cube, 'ridge_alpha'), \
            "QuadCube should have a centralized ridge_alpha attribute"
        
        # DOVREBBE essere configurabile
        expected_alpha = 1e-3
        assert cube.ridge_alpha == expected_alpha, \
            f"Default ridge_alpha should be {expected_alpha}"

    def test_ridge_alpha_used_in_fit_surrogate(self):
        """
        Test che fit_surrogate usi il ridge_alpha centralizzato invece di hardcoded.
        
        FAILING REASON:
        - fit_surrogate() ha `ridge_alpha = 1e-3` hardcoded alla riga ~86
        - Non usa self.ridge_alpha
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        cube.ridge_alpha = 2e-3  # Provo a cambiarlo
        
        # Aggiungo dati per fitting
        cube._tested_pairs = []
        np.random.seed(42)
        for _ in range(12):
            x = np.random.rand(2)
            y = float(np.sum(x**2))  # funzione semplice
            cube._tested_pairs.append((x, y))
        
        # Fitto il surrogato
        cube.fit_surrogate(min_points=8)
        
        # Se il codice fosse corretto, dovrebbe usare ridge_alpha = 2e-3
        # Ma ora usa il valore hardcoded 1e-3, quindi questo test FALLISCE
        # (Non possiamo verificare direttamente ridge_alpha usato, ma possiamo
        # verificare che cambiare ridge_alpha ha effetto)
        
        # Salvo il surrogato con ridge_alpha modificato
        surrogate_modified = cube.surrogate_2d.copy()
        
        # Reset e rifitto con valore diverso
        cube.ridge_alpha = 1e-5
        cube.fit_surrogate(min_points=8)
        surrogate_different = cube.surrogate_2d
        
        # I pesi DOVREBBERO essere diversi se ridge_alpha cambia
        w_modified = surrogate_modified['w']
        w_different = surrogate_different['w']
        
        # Questo FALLISCE perché fit_surrogate ignora self.ridge_alpha
        assert not np.allclose(w_modified, w_different, atol=1e-6), \
            "Surrogate weights should differ when ridge_alpha changes, but they don't (hardcoded value used)"

    def test_ridge_alpha_used_in_simulate_split(self):
        """
        Test che _simulate_split2 e _simulate_split4 usino ridge_alpha centralizzato.
        
        FAILING REASON:
        - _simulate_split2() ha `ridge_alpha = 1e-3` hardcoded alla riga ~370
        - _simulate_split4() ha `ridge_alpha = 1e-3` hardcoded alla riga ~430
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # Setup frame
        cube.R = np.eye(2)
        cube.mu = np.array([0.5, 0.5])
        
        # DOVREBBE usare self.ridge_alpha nelle simulazioni
        # Ma il codice ha valori hardcoded, quindi non possiamo controllare
        assert hasattr(cube, 'ridge_alpha'), \
            "ridge_alpha should be a cube attribute used by simulate methods"
        
        # Aggiungo punti per split simulation
        cube._tested_pairs = []
        np.random.seed(42)
        for _ in range(20):
            x = np.random.rand(2)
            y = float(np.sum(x**2))
            cube._tested_pairs.append((x, y))
        
        cube.ridge_alpha = 5e-3
        
        # Simulo split - DOVREBBE usare ridge_alpha = 5e-3
        children = cube._simulate_split2()
        
        # Se il codice fosse corretto, cambiere ridge_alpha dovrebbe
        # cambiare i risultati della simulazione
        # Ma ora usa valore hardcoded, test FALLISCE
        cube.ridge_alpha = 1e-5
        children_different = cube._simulate_split2()
        
        # Le varianze dei figli DOVREBBERO essere diverse
        if children and children_different:
            var1 = [ch['var'] for ch in children]
            var2 = [ch['var'] for ch in children_different]
            
            assert not np.allclose(var1, var2, rtol=0.01), \
                "Simulated child variances should differ when ridge_alpha changes"

    def test_surrogate_hyperparams_inheritance_to_children(self):
        """
        Test che i figli ereditino gli iperparametri del surrogato dal genitore.
        
        FAILING REASON:
        - split2() e split4() non propagano ridge_alpha ai figli
        - Ogni figlio userebbe il default hardcoded, non il valore del genitore
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        cube.R = np.eye(2)
        cube.mu = np.array([0.5, 0.5])
        cube.ridge_alpha = 3e-3  # Valore custom
        
        # Aggiungo punti per permettere split
        cube._tested_pairs = []
        cube.n_trials = 15
        np.random.seed(42)
        for _ in range(20):
            x = np.random.rand(2)
            y = float(np.sum(x**2))
            cube._tested_pairs.append((x, y))
            cube.scores.append(y)
        
        # Split
        children = cube.split2()
        
        # I figli DOVREBBERO ereditare ridge_alpha
        for i, child in enumerate(children):
            assert hasattr(child, 'ridge_alpha'), \
                f"Child {i} should have ridge_alpha attribute"
            assert child.ridge_alpha == cube.ridge_alpha, \
                f"Child {i} should inherit parent's ridge_alpha={cube.ridge_alpha}, got {getattr(child, 'ridge_alpha', 'MISSING')}"

    def test_multiple_surrogate_hyperparameters_centralized(self):
        """
        Test che TUTTI gli iperparametri del surrogato siano centralizzati.
        
        EXPECTED:
        - min_points per fit_surrogate
        - ridge_alpha per ridge regression
        - Eventuali altri parametri (sigma2_threshold, etc.)
        
        FAILING REASON:
        - min_points è passato come parametro invece di essere attributo
        - Altri parametri non sono configurabili
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # DOVREBBERO esistere attributi configurabili
        expected_attrs = [
            'ridge_alpha',
            'surrogate_min_points',  # invece di passare min_points=8
            # Altri eventuali parametri del surrogato
        ]
        
        for attr in expected_attrs:
            assert hasattr(cube, attr), \
                f"QuadCube should have centralized attribute: {attr}"

    def test_surrogate_consistency_after_multiple_fits(self):
        """
        Test che fit_surrogate sia deterministico con stessi dati e hyperparams.
        
        FAILING REASON:
        - Se gli hyperparams sono hardcoded in modo diverso in vari punti,
          potrebbero esserci inconsistenze
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        
        # Setup identico per due cube
        np.random.seed(42)
        points = [np.random.rand(2) for _ in range(12)]
        scores = [float(np.sum(x**2)) for x in points]
        
        cube1 = QuadCube(bounds)
        cube1._tested_pairs = [(x, y) for x, y in zip(points, scores)]
        
        cube2 = QuadCube(bounds)
        cube2._tested_pairs = [(x, y) for x, y in zip(points, scores)]
        
        # Se ridge_alpha fosse centralizzato, potremmo impostarlo
        if hasattr(cube1, 'ridge_alpha'):
            cube1.ridge_alpha = 1e-3
            cube2.ridge_alpha = 1e-3
        
        # Fit
        cube1.fit_surrogate(min_points=8)
        cube2.fit_surrogate(min_points=8)
        
        # I surrogati DOVREBBERO essere identici
        assert cube1.surrogate_2d is not None
        assert cube2.surrogate_2d is not None
        
        w1 = cube1.surrogate_2d['w']
        w2 = cube2.surrogate_2d['w']
        
        assert np.allclose(w1, w2, atol=1e-10), \
            "Identical data should produce identical surrogate with same hyperparams"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
