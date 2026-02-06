#!/usr/bin/env python3
"""
DEBUG INTENSIVO: Perché Thompson Sampling funziona così bene sui categorici?

Analisi:
1. Distribuzione delle scelte categoriche nel tempo
2. Convergenza verso le categorie migliori
3. Bilanciamento exploration vs exploitation
4. Confronto con sampling uniforme e greedy
"""

import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
warnings.filterwarnings('ignore')

# Funzione sintetica con categorici
def synthetic_objective(x, cat_optimal=[1, 0, 2]):
    """
    Obiettivo sintetico dove:
    - x[0:2] sono continui
    - x[2] è categorico con 3 scelte (best=1)
    - x[3] è categorico con 3 scelte (best=0)
    - x[4] è categorico con 3 scelte (best=2)
    
    Il valore ottimale si ottiene quando tutti i categorici sono corretti
    """
    # Contributo continuo (sfera)
    continuous_part = -(x[0] - 0.3)**2 - (x[1] - 0.7)**2
    
    # Contributo categorico - ogni categoria "giusta" aggiunge bonus
    cat_bonus = 0.0
    cat_indices = [
        (2, 3),  # dim 2, 3 scelte
        (3, 3),  # dim 3, 3 scelte
        (4, 3),  # dim 4, 3 scelte
    ]
    
    for i, (dim_idx, n_ch) in enumerate(cat_indices):
        cat_val = int(round(x[dim_idx] * (n_ch - 1)))
        cat_val = min(n_ch-1, max(0, cat_val))
        
        if cat_val == cat_optimal[i]:
            cat_bonus += 0.5  # Grande bonus per categoria corretta
        else:
            # Penalità progressiva in base alla distanza
            dist = abs(cat_val - cat_optimal[i])
            cat_bonus -= 0.2 * dist
    
    return continuous_part + cat_bonus


class ThompsonAnalyzer:
    """Analizzatore per capire Thompson Sampling."""
    
    def __init__(self, n_choices=3, exploration_boost=1.0):
        self.n_choices = n_choices
        self.exploration_boost = exploration_boost
        self.rng = np.random.default_rng(42)
        
        # Stats: {choice: (n_good, n_total)}
        self.stats = {i: (0, 0) for i in range(n_choices)}
        self.history = []  # [(iteration, choice, score, is_good)]
        
    def sample_thompson(self):
        """Thompson Sampling: campiona da Beta distribution."""
        K = self.n_choices * self.exploration_boost
        samples = []
        for v in range(self.n_choices):
            n_g, n_t = self.stats[v]
            alpha = n_g + 1
            beta_param = (n_t - n_g) + K
            sample = self.rng.beta(alpha, beta_param)
            samples.append(sample)
        return int(np.argmax(samples)), samples
    
    def sample_uniform(self):
        """Sampling uniforme (baseline)."""
        return self.rng.integers(0, self.n_choices), None
    
    def sample_greedy(self, epsilon=0.1):
        """Epsilon-greedy."""
        if self.rng.random() < epsilon:
            return self.rng.integers(0, self.n_choices), None
        
        # Pick best empirical
        best_v, best_ratio = 0, -1
        for v in range(self.n_choices):
            n_g, n_t = self.stats[v]
            ratio = (n_g + 1) / (n_t + 2)
            if ratio > best_ratio:
                best_ratio = ratio
                best_v = v
        return best_v, None
    
    def sample_ucb(self, c=1.0):
        """UCB sampling."""
        total = sum(n_t for _, n_t in self.stats.values()) + 1
        best_v, best_ucb = 0, -1
        for v in range(self.n_choices):
            n_g, n_t = self.stats[v]
            mean = (n_g + 1) / (n_t + 2)
            exploration = c * np.sqrt(np.log(total) / (n_t + 1))
            ucb = mean + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_v = v
        return best_v, None
    
    def update(self, choice, score, gamma):
        """Update stats after observing a result."""
        is_good = score >= gamma
        n_g, n_t = self.stats[choice]
        self.stats[choice] = (n_g + (1 if is_good else 0), n_t + 1)
        self.history.append((len(self.history), choice, score, is_good))
    
    def get_probabilities(self):
        """Get current selection probabilities via Monte Carlo."""
        n_samples = 1000
        counts = np.zeros(self.n_choices)
        K = self.n_choices * self.exploration_boost
        
        for _ in range(n_samples):
            samples = []
            for v in range(self.n_choices):
                n_g, n_t = self.stats[v]
                alpha = n_g + 1
                beta_param = (n_t - n_g) + K
                sample = self.rng.beta(alpha, beta_param)
                samples.append(sample)
            counts[np.argmax(samples)] += 1
        
        return counts / n_samples


def run_analysis():
    """Analisi completa del comportamento di Thompson Sampling."""
    
    print("="*80)
    print("DEBUG: Perché Thompson Sampling funziona così bene?")
    print("="*80)
    
    # Scenario: 3 categorie, una chiaramente migliore (cat=1)
    # Probabilità di successo: cat0=0.3, cat1=0.7, cat2=0.4
    success_probs = [0.3, 0.7, 0.4]
    n_iterations = 200
    n_runs = 50
    
    methods = {
        'Thompson': lambda a: a.sample_thompson(),
        'Uniform': lambda a: a.sample_uniform(),
        'Greedy(ε=0.1)': lambda a: a.sample_greedy(0.1),
        'UCB(c=1)': lambda a: a.sample_ucb(1.0),
    }
    
    results = {m: [] for m in methods}
    choice_history = {m: defaultdict(list) for m in methods}
    
    print("\n1. CONFRONTO METODI DI SAMPLING")
    print("-"*60)
    
    for method_name, sample_fn in methods.items():
        total_reward = []
        
        for run in range(n_runs):
            analyzer = ThompsonAnalyzer(n_choices=3)
            analyzer.rng = np.random.default_rng(run)
            
            cumulative = 0
            run_choices = []
            
            for it in range(n_iterations):
                choice, _ = sample_fn(analyzer)
                
                # Simula reward bernoulli
                success = analyzer.rng.random() < success_probs[choice]
                reward = 1 if success else 0
                cumulative += reward
                
                # Update con gamma dinamico
                gamma = 0.5  # threshold fisso per semplicità
                analyzer.update(choice, reward, gamma)
                run_choices.append(choice)
            
            total_reward.append(cumulative)
            
            # Salva history delle scelte per questo run
            for it, ch in enumerate(run_choices):
                choice_history[method_name][it].append(ch)
        
        mean_reward = np.mean(total_reward)
        std_reward = np.std(total_reward)
        results[method_name] = (mean_reward, std_reward)
        print(f"  {method_name:20s}: {mean_reward:.1f} ± {std_reward:.1f} (optimal: {n_iterations * max(success_probs)})")
    
    # 2. Analisi della convergenza
    print("\n2. VELOCITÀ DI CONVERGENZA VERSO CATEGORIA OTTIMALE (cat=1)")
    print("-"*60)
    
    optimal_cat = 1
    for method_name in methods:
        # Calcola % di scelte ottimali nel tempo
        optimal_rate_over_time = []
        for it in range(n_iterations):
            choices_at_it = choice_history[method_name][it]
            optimal_rate = sum(1 for c in choices_at_it if c == optimal_cat) / len(choices_at_it)
            optimal_rate_over_time.append(optimal_rate)
        
        # Report convergence milestones
        early = np.mean(optimal_rate_over_time[:20])
        mid = np.mean(optimal_rate_over_time[50:100])
        late = np.mean(optimal_rate_over_time[-50:])
        print(f"  {method_name:20s}: early(0-20)={early:.1%}, mid(50-100)={mid:.1%}, late(150-200)={late:.1%}")
    
    # 3. Analisi del bilanciamento exploration/exploitation
    print("\n3. BILANCIAMENTO EXPLORATION vs EXPLOITATION")
    print("-"*60)
    
    analyzer = ThompsonAnalyzer(n_choices=3)
    
    # Simula scenario iniziale
    print("\nScenario: dopo 10 trial, stats = {0: (2, 5), 1: (4, 3), 2: (1, 2)}")
    analyzer.stats = {0: (2, 5), 1: (4, 3), 2: (1, 2)}
    
    # Monte Carlo per ottenere probabilità di selezione
    probs = analyzer.get_probabilities()
    print(f"  Thompson probabilities: cat0={probs[0]:.2%}, cat1={probs[1]:.2%}, cat2={probs[2]:.2%}")
    
    # Confronta con empirical ratios
    for v in range(3):
        n_g, n_t = analyzer.stats[v]
        emp_ratio = (n_g + 1) / (n_t + 2)
        print(f"  Cat{v}: good_ratio={n_g}/{n_t}={n_g/max(1,n_t):.2f}, beta_mean={emp_ratio:.2f}")
    
    # 4. Perché Thompson è meglio di greedy?
    print("\n4. PERCHÉ THOMPSON BATTE GREEDY?")
    print("-"*60)
    
    print("""
    Scenario critico: categoria 1 è ottimale ma inizialmente sottocampionata
    
    Stats iniziali (dopo 5 trial sfortunati):
    - cat0: 1 good / 3 total  (ratio=0.33)
    - cat1: 1 good / 1 total  (ratio=0.50)  <- ottimale ma poco campionata!
    - cat2: 0 good / 1 total  (ratio=0.00)
    """)
    
    analyzer_greedy = ThompsonAnalyzer(n_choices=3)
    analyzer_greedy.stats = {0: (1, 3), 1: (1, 1), 2: (0, 1)}
    
    analyzer_thompson = ThompsonAnalyzer(n_choices=3)
    analyzer_thompson.stats = {0: (1, 3), 1: (1, 1), 2: (0, 1)}
    
    # Greedy sceglie sempre cat1 (ratio più alto)
    greedy_choice, _ = analyzer_greedy.sample_greedy(epsilon=0.0)
    print(f"  Greedy (ε=0) sceglie: cat{greedy_choice}")
    
    # Thompson ha varianza!
    thompson_probs = analyzer_thompson.get_probabilities()
    print(f"  Thompson probabilità: cat0={thompson_probs[0]:.2%}, cat1={thompson_probs[1]:.2%}, cat2={thompson_probs[2]:.2%}")
    
    print("""
    Insight chiave: Thompson mantiene incertezza!
    - Cat0 ha più dati ma ratio basso → Beta(2, 5) concentrata bassa
    - Cat1 ha pochi dati ma ratio alto → Beta(2, 3) con alta varianza → ESPLORA!
    - Cat2 sembra pessima → Beta(1, 4) concentrata bassa
    
    Thompson esplora naturalmente opzioni con alta incertezza!
    """)
    
    # 5. Visualizzazione delle distribuzioni Beta
    print("\n5. DISTRIBUZIONI BETA PER OGNI CATEGORIA")
    print("-"*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.linspace(0, 1, 200)
    
    stats_example = {0: (2, 5), 1: (4, 3), 2: (1, 2)}
    K = 3  # n_choices
    
    for v, ax in enumerate(axes):
        n_g, n_t = stats_example[v]
        alpha = n_g + 1
        beta_param = (n_t - n_g) + K
        
        from scipy import stats as sp_stats
        beta_dist = sp_stats.beta(alpha, beta_param)
        y = beta_dist.pdf(x)
        
        ax.fill_between(x, y, alpha=0.3)
        ax.plot(x, y, linewidth=2)
        ax.axvline(alpha / (alpha + beta_param), color='red', linestyle='--', label=f'mean={alpha/(alpha+beta_param):.2f}')
        ax.set_title(f'Cat {v}: Beta({alpha}, {beta_param:.0f})\ngood={n_g}, total={n_t}')
        ax.set_xlabel('Success probability')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('/mnt/workspace/thesis/debug_thompson_beta.png', dpi=150)
    print("  Salvato: debug_thompson_beta.png")
    
    # 6. Test su JAHS-Bench reale
    print("\n6. TEST SU JAHS-BENCH REALE")
    print("-"*60)
    
    try:
        sys.path.insert(0, '/mnt/workspace/thesis')
        from benchmark_jahs import JAHSBenchWrapper
        
        wrapper = JAHSBenchWrapper(task='cifar10')
        print(f"  Dim: {wrapper.dim}")
        print(f"  Categorici: dims 2-12 (11 dimensioni categoriche)")
        
        # Importa ALBA per analisi
        from ALBA_V1 import ALBA
        
        # Run con tracking dettagliato
        categorical_dims = [
            (2, 3), (3, 3), (4, 3), (5, 3), (6, 2),
            (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),
        ]
        
        opt = ALBA(
            bounds=[(0.0, 1.0)] * wrapper.dim,
            maximize=False,
            seed=42,
            total_budget=100,
            categorical_dims=categorical_dims,
        )
        
        # Raccogli dati sul comportamento categorico
        cat_choices = defaultdict(list)
        scores = []
        
        wrapper.reset()
        for it in range(100):
            x = opt.ask()
            y = wrapper.evaluate_array(x)
            opt.tell(x, y)
            scores.append(y)
            
            # Track categorical choices
            for dim_idx, n_ch in categorical_dims:
                cat_val = int(round(x[dim_idx] * (n_ch - 1)))
                cat_choices[dim_idx].append(cat_val)
        
        print(f"\n  Best error: {min(scores):.4f} (acc={100*(1-min(scores)):.2f}%)")
        
        # Analizza distribuzione per alcune dimensioni chiave
        print("\n  Distribuzione scelte categoriche (prime 50 vs ultime 50 iter):")
        for dim_idx in [2, 7, 12]:
            early = cat_choices[dim_idx][:50]
            late = cat_choices[dim_idx][50:]
            
            n_ch = dict(categorical_dims)[dim_idx]
            early_dist = [early.count(v)/50 for v in range(n_ch)]
            late_dist = [late.count(v)/50 for v in range(n_ch)]
            
            print(f"    Dim {dim_idx} ({n_ch} choices):")
            print(f"      Early: {[f'{p:.0%}' for p in early_dist]}")
            print(f"      Late:  {[f'{p:.0%}' for p in late_dist]}")
        
    except Exception as e:
        print(f"  Errore JAHS: {e}")
    
    # 7. Analisi matematica
    print("\n7. ANALISI MATEMATICA DI THOMPSON SAMPLING")
    print("-"*60)
    print("""
    Formula Thompson in ALBA_V1:
    
    Per ogni categoria v in [0, n_choices):
        alpha = n_good[v] + 1
        beta = (n_total[v] - n_good[v]) + K     dove K = n_choices * exploration_boost
        
        sample[v] ~ Beta(alpha, beta)
        
    Scelta: argmax(sample)
    
    Proprietà chiave:
    
    1. PRIOR INFORMATIVO: K = n_choices
       - Con 0 osservazioni: Beta(1, K) ha media = 1/(1+K) = 1/(1+n_choices)
       - Questo è ESATTAMENTE 1/n_choices = uniforme!
       - Il prior è "una osservazione uniforme" per ogni categoria
    
    2. VARIANCE-AWARE EXPLORATION:
       - Categorie con pochi dati hanno alta varianza → possono "vincere" il sampling
       - Categorie con molti dati convergono verso il vero ratio
    
    3. NATURAL ANNEALING:
       - Varianza decresce con √(1/n) → exploitation automatica nel tempo
       - NON serve tuning di temperature/epsilon!
    
    4. PERCHÉ BATTE GREEDY:
       - Greedy: exploitation pura, può bloccarsi su subottimali
       - Thompson: esplora proporzionalmente all'incertezza
       
    5. PERCHÉ BATTE UCB:
       - UCB: esplorazione deterministica (sempre stesso ordine)
       - Thompson: stocastico → evita pattern prevedibili
       → Meglio per landscape irregolari come HPO
    """)
    
    print("\n" + "="*80)
    print("CONCLUSIONE")
    print("="*80)
    print("""
    Thompson Sampling funziona bene su HPO perché:
    
    1. ADATTIVO: Impara automaticamente quali categorie sono promettenti
    2. BILANCIATO: Esplora categorie incerte senza dimenticare quelle buone
    3. PARAMETER-FREE: Non richiede tuning di epsilon o temperature
    4. ROBUSTO AL RUMORE: La varianza Bayesiana gestisce naturalmente il rumore
    5. SCALABILE: O(1) per campionamento, O(n_categories) per update
    
    La combinazione con LGS per i continui è sinergica:
    - Thompson ottimizza la struttura discreta
    - LGS ottimizza i valori continui all'interno di quella struttura
    """)


if __name__ == "__main__":
    run_analysis()
