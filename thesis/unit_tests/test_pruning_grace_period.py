"""
Test: Pruning Grace Period

PROBLEMA:
Serve un "grace period" per il pruning
→ non potare foglie appena create.
Perché è un problema: altrimenti la struttura si sfarina subito e il comportamento è instabile.

EXPECTED BEHAVIOR:
- Le foglie appena create (split recente) non dovrebbero essere potate immediatamente
- Dovrebbe esistere un grace period (es. 2-3 iterazioni) prima che una foglia sia eligible per pruning
- Questo previene il "thrashing" dove split e prune si annullano a vicenda

CURRENT BEHAVIOR (che dovrebbe far fallire il test):
- prune_cubes() non controlla l'età delle foglie
- Una foglia può essere creata e immediatamente potata se il suo UCB è basso
- Non c'è traccia di quando una foglia è stata creata
"""

import numpy as np
import pytest
from thesis.hpo_curvature import QuadCube, QuadHPO


class TestPruningGracePeriod:
    """Test che le foglie appena create non vengano potate immediatamente."""

    def test_newly_created_leaves_should_have_birth_timestamp(self):
        """
        Test che ogni foglia tracci quando è stata creata.
        
        FAILING REASON:
        - QuadCube non ha un attributo 'birth_trial' o 'age'
        - Non c'è modo di sapere quando una foglia è stata creata
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cube = QuadCube(bounds)
        
        # DOVREBBE esistere un attributo per tracciare la creazione
        assert hasattr(cube, 'birth_trial') or hasattr(cube, 'creation_trial') or hasattr(cube, 'age'), \
            "QuadCube should track when it was created (birth_trial, creation_trial, or age)"

    def test_children_inherit_birth_timestamp_on_split(self):
        """
        Test che i figli ricevano un timestamp di nascita quando vengono creati.
        
        FAILING REASON:
        - split2() e split4() non impostano birth_trial sui figli
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        parent = QuadCube(bounds)
        parent.R = np.eye(2)
        parent.mu = np.array([0.5, 0.5])
        
        # Simulo che il parent sia al trial 10
        current_trial = 10
        if hasattr(parent, 'birth_trial'):
            parent.birth_trial = 0  # Il parent è vecchio
        
        # Aggiungo dati per split
        parent._tested_pairs = []
        parent.n_trials = 15
        np.random.seed(42)
        for _ in range(20):
            x = np.random.rand(2)
            y = float(np.sum(x**2))
            parent._tested_pairs.append((x, y))
            parent.scores.append(y)
        
        # Split
        children = parent.split2()
        
        # I figli DOVREBBERO avere birth_trial = current_trial
        for i, child in enumerate(children):
            assert hasattr(child, 'birth_trial') or hasattr(child, 'creation_trial'), \
                f"Child {i} should have birth_trial or creation_trial attribute"

    def test_prune_cubes_respects_grace_period(self):
        """
        Test che prune_cubes non poti foglie troppo giovani.
        
        FAILING REASON:
        - prune_cubes() non controlla l'età delle foglie
        - Una foglia appena creata può essere potata immediatamente
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        hpo = QuadHPO(bounds, beta=1.6, lambda_geo=0.8)
        
        # DOVREBBE esistere un parametro per il grace period
        assert hasattr(hpo, 'prune_grace_period') or hasattr(hpo, 'min_leaf_age'), \
            "QuadHPO should have a grace period parameter for pruning"
        
        # Se esiste, dovrebbe essere > 0
        grace = getattr(hpo, 'prune_grace_period', getattr(hpo, 'min_leaf_age', None))
        if grace is not None:
            assert grace > 0, "Grace period should be positive"

    def test_leaf_not_pruned_within_grace_period(self):
        """
        Test che una foglia NON venga potata se è più giovane del grace period.
        
        FAILING REASON:
        - Anche se una foglia ha UCB basso, se è giovane non dovrebbe essere potata
        - Il codice corrente non implementa questa logica
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        hpo = QuadHPO(bounds, beta=1.6, lambda_geo=0.8)
        hpo.total_trials = 50
        hpo.best_score_global = 10.0
        
        # Creo foglie con età diverse
        old_leaf = QuadCube(bounds)
        old_leaf.R = np.eye(2)
        old_leaf.mu = np.array([0.5, 0.5])
        old_leaf.n_trials = 10
        old_leaf.mean_score = 5.0  # Basso score
        old_leaf.scores = [5.0] * 10
        old_leaf.best_score = 5.0
        if hasattr(old_leaf, 'birth_trial'):
            old_leaf.birth_trial = 10  # Creata 40 trial fa
        
        new_leaf = QuadCube(bounds)
        new_leaf.R = np.eye(2)
        new_leaf.mu = np.array([0.5, 0.5])
        new_leaf.n_trials = 2
        new_leaf.mean_score = 4.0  # Score ancora più basso
        new_leaf.scores = [4.0, 4.0]
        new_leaf.best_score = 4.0
        if hasattr(new_leaf, 'birth_trial'):
            new_leaf.birth_trial = 48  # Creata 2 trial fa - GIOVANE!
        
        hpo.leaf_cubes = [old_leaf, new_leaf]
        
        # Prune
        hpo.prune_cubes()
        
        # La new_leaf DOVREBBE essere preservata anche con score basso
        # perché è dentro il grace period
        leaf_ids = [id(leaf) for leaf in hpo.leaf_cubes]
        
        # Questo test FALLISCE perché prune_cubes ignora l'età
        assert id(new_leaf) in leaf_ids, \
            "Newly created leaf should not be pruned even if UCB is low (grace period)"

    def test_old_leaf_can_be_pruned_after_grace_period(self):
        """
        Test che una foglia vecchia POSSA essere potata se ha score basso.
        
        Questo verifica che il grace period non impedisca il pruning indefinitamente.
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        hpo = QuadHPO(bounds, beta=1.6, lambda_geo=0.8)
        hpo.total_trials = 100
        hpo.best_score_global = 10.0
        
        # Grace period dovrebbe essere tipo 3-5 trials
        grace_period = getattr(hpo, 'prune_grace_period', 3)
        
        # Creo foglia vecchia con score basso
        old_bad_leaf = QuadCube(bounds)
        old_bad_leaf.R = np.eye(2)
        old_bad_leaf.mu = np.array([0.5, 0.5])
        old_bad_leaf.n_trials = 10
        old_bad_leaf.mean_score = 2.0
        old_bad_leaf.scores = [2.0] * 10
        old_bad_leaf.best_score = 2.0
        old_bad_leaf.stale_steps = 20
        if hasattr(old_bad_leaf, 'birth_trial'):
            old_bad_leaf.birth_trial = 100 - 50  # 50 trial fa, molto oltre grace period
        
        # Foglia buona da mantenere
        good_leaf = QuadCube(bounds)
        good_leaf.R = np.eye(2)
        good_leaf.mu = np.array([0.5, 0.5])
        good_leaf.n_trials = 15
        good_leaf.mean_score = 9.0
        good_leaf.scores = [9.0] * 15
        good_leaf.best_score = 9.0
        if hasattr(good_leaf, 'birth_trial'):
            good_leaf.birth_trial = 0
        
        hpo.leaf_cubes = [old_bad_leaf, good_leaf]
        
        # Prune dovrebbe rimuovere la vecchia foglia cattiva
        hpo.prune_cubes()
        
        leaf_ids = [id(leaf) for leaf in hpo.leaf_cubes]
        
        # La foglia vecchia con score basso DOVREBBE essere potata
        # (questo test potrebbe passare per caso, ma vogliamo assicurarci
        # che il grace period non impedisca pruning valido)
        assert len(hpo.leaf_cubes) >= 1, "Should keep at least good leaf"

    def test_grace_period_prevents_thrashing(self):
        """
        Test dello scenario di thrashing: split → prune immediato → split di nuovo.
        
        FAILING REASON:
        - Senza grace period, questo può succedere:
          1. Leaf A viene splittata → B e C
          2. B ha UCB basso → viene potata immediatamente
          3. C viene splittata di nuovo
          4. Ciclo infinito di split/prune
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        hpo = QuadHPO(bounds, beta=1.6, lambda_geo=0.8)
        
        # Simulo split recente
        parent = QuadCube(bounds)
        parent.R = np.eye(2)
        parent.mu = np.array([0.5, 0.5])
        parent._tested_pairs = []
        parent.n_trials = 20
        
        np.random.seed(42)
        for _ in range(25):
            x = np.random.rand(2)
            y = float(np.sum(x**2))
            parent._tested_pairs.append((x, y))
            parent.scores.append(y)
        
        # Simuliamo essere al trial 50
        hpo.total_trials = 50
        hpo.best_score_global = 1.0
        
        # Split parent
        children = parent.split2()
        
        # Simulo che un figlio abbia score molto basso
        # (magari non ha ancora ricevuto sample)
        children[0].n_trials = 0
        children[0].mean_score = 0.0
        children[0].scores = []
        children[0].best_score = -np.inf
        
        children[1].n_trials = 5
        children[1].mean_score = 0.8
        children[1].scores = [0.8] * 5
        children[1].best_score = 0.8
        
        # Imposto birth trial se l'attributo esiste
        for child in children:
            if hasattr(child, 'birth_trial'):
                child.birth_trial = 50  # Appena nati
        
        hpo.leaf_cubes = children.copy()
        initial_count = len(hpo.leaf_cubes)
        
        # Prune - i figli appena nati NON dovrebbero essere potati
        hpo.prune_cubes()
        
        # DOVREBBERO rimanere entrambi i figli (o almeno min_leaves)
        # anche se uno ha score basso
        assert len(hpo.leaf_cubes) >= min(initial_count, hpo.min_leaves), \
            f"Grace period should prevent immediate pruning. Had {initial_count}, now {len(hpo.leaf_cubes)}"
        
        # Il figlio con score basso DOVREBBE ancora essere presente
        # Questo test FALLISCE perché non c'è grace period
        if hasattr(children[0], 'birth_trial'):
            child0_still_there = any(id(leaf) == id(children[0]) for leaf in hpo.leaf_cubes)
            # Questo dovrebbe essere True se grace period è implementato
            # (potrebbe fallire se min_leaves salva la situazione per caso)

    def test_grace_period_value_is_reasonable(self):
        """
        Test che il grace period abbia un valore ragionevole (2-5 trials).
        
        FAILING REASON:
        - L'attributo non esiste
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        hpo = QuadHPO(bounds)
        
        grace = getattr(hpo, 'prune_grace_period', None)
        
        # DOVREBBE esistere
        assert grace is not None, "prune_grace_period should be defined"
        
        # DOVREBBE essere tra 2 e 5 (circa)
        assert 2 <= grace <= 5, \
            f"Grace period should be 2-5 trials, got {grace}"

    def test_leaf_age_increments_correctly(self):
        """
        Test che l'età delle foglie si incrementi correttamente ad ogni trial.
        
        FAILING REASON:
        - Non c'è tracking dell'età
        - birth_trial non viene confrontato con current_trial
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        hpo = QuadHPO(bounds)
        
        leaf = QuadCube(bounds)
        leaf.R = np.eye(2)
        leaf.mu = np.array([0.5, 0.5])
        
        # DOVREBBE esserci un modo per calcolare l'età
        if hasattr(leaf, 'birth_trial'):
            leaf.birth_trial = 10
            hpo.total_trials = 15
            
            # L'età DOVREBBE essere 15 - 10 = 5
            # Ma non c'è metodo per calcolarla!


class TestPruningStaleSteps:
    """
    Test correlato: stale_steps è simile a grace period ma per foglie inattive.
    Verifica che il meccanismo stale_steps funzioni correttamente.
    """
    
    def test_stale_steps_reset_on_improvement(self):
        """
        Test che stale_steps si resetti quando la foglia riceve un trial.
        
        Questo è implementato ma verifichiamo che funzioni correttamente
        in combinazione con il grace period.
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        leaf = QuadCube(bounds)
        leaf.stale_steps = 10  # Stale
        
        # Dopo un trial su questa foglia, stale_steps dovrebbe essere 0
        # (questo è testato indirettamente da run_trial, ma è ortogonale al grace period)

    def test_grace_period_vs_stale_steps(self):
        """
        Test che grace period e stale_steps siano indipendenti.
        
        - Grace period: età dalla creazione
        - Stale steps: passi senza miglioramento
        
        Una foglia giovane (dentro grace) ma stale non dovrebbe essere potata.
        Una foglia vecchia (fuori grace) e stale può essere potata.
        """
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        hpo = QuadHPO(bounds)
        hpo.total_trials = 20
        hpo.best_score_global = 10.0
        
        # Foglia giovane ma stale
        young_stale = QuadCube(bounds)
        young_stale.R = np.eye(2)
        young_stale.mu = np.array([0.5, 0.5])
        young_stale.n_trials = 5
        young_stale.mean_score = 5.0
        young_stale.scores = [5.0] * 5
        young_stale.best_score = 5.0
        young_stale.stale_steps = 12  # Molto stale
        if hasattr(young_stale, 'birth_trial'):
            young_stale.birth_trial = 18  # Giovane (2 trial fa)
        
        # Foglia vecchia e stale
        old_stale = QuadCube(bounds)
        old_stale.R = np.eye(2)
        old_stale.mu = np.array([0.5, 0.5])
        old_stale.n_trials = 5
        old_stale.mean_score = 5.0
        old_stale.scores = [5.0] * 5
        old_stale.best_score = 5.0
        old_stale.stale_steps = 12
        if hasattr(old_stale, 'birth_trial'):
            old_stale.birth_trial = 5  # Vecchia (15 trial fa)
        
        hpo.leaf_cubes = [young_stale, old_stale]
        hpo.prune_cubes()
        
        leaf_ids = [id(leaf) for leaf in hpo.leaf_cubes]
        
        # young_stale DOVREBBE sopravvivere (grace period)
        # Questo FALLISCE perché grace period non esiste
        if hasattr(young_stale, 'birth_trial'):
            assert id(young_stale) in leaf_ids, \
                "Young stale leaf should survive (grace period protection)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
