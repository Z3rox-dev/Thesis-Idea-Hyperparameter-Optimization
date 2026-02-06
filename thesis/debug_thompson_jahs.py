#!/usr/bin/env python3
"""
DEBUG INTENSIVO su JAHS-Bench: Tracking dettagliato del comportamento categorico.
"""

import sys
import warnings
import numpy as np
from collections import defaultdict
warnings.filterwarnings('ignore')

sys.path.insert(0, '/mnt/workspace/thesis')
from benchmark_jahs import JAHSBenchWrapper

# Import diretto del codice Thompson per analisi
class ThompsonDebugger:
    """Replica semplificata di Thompson per debug."""
    
    def __init__(self, categorical_dims, seed=42):
        self.categorical_dims = categorical_dims
        self.rng = np.random.default_rng(seed)
        
        # Stats per cube (global per semplicità)
        # {dim_idx: {val_idx: (n_good, n_total)}}
        self.stats = {}
        for dim_idx, n_ch in categorical_dims:
            self.stats[dim_idx] = {v: (0, 0) for v in range(n_ch)}
        
        self.gamma = 0.0  # threshold
        self.scores = []
        
        # Tracking
        self.choice_history = defaultdict(list)  # dim -> list of choices
        self.probs_history = defaultdict(list)   # dim -> list of probability dicts
        
    def discretize(self, x_val, n_choices):
        return min(int(round(x_val * (n_choices - 1))), n_choices - 1)
    
    def to_continuous(self, val_idx, n_choices):
        return val_idx / (n_choices - 1) if n_choices > 1 else 0.5
    
    def sample_categorical(self, exploration_boost=1.0):
        """Sample categorical dims using Thompson Sampling."""
        result = {}
        probs_snapshot = {}
        
        for dim_idx, n_choices in self.categorical_dims:
            stats = self.stats[dim_idx]
            K = n_choices * exploration_boost
            
            # Thompson: sample from Beta for each category
            samples = []
            beta_params = []
            for v in range(n_choices):
                n_g, n_t = stats[v]
                alpha = n_g + 1
                beta_param = (n_t - n_g) + K
                sample = self.rng.beta(alpha, beta_param)
                samples.append(sample)
                beta_params.append((alpha, beta_param))
            
            chosen = int(np.argmax(samples))
            result[dim_idx] = chosen
            
            # Monte Carlo per prob reali
            probs = self._mc_probabilities(stats, K, n_choices)
            probs_snapshot[dim_idx] = probs
            
            self.choice_history[dim_idx].append(chosen)
            self.probs_history[dim_idx].append(probs)
        
        return result, probs_snapshot
    
    def _mc_probabilities(self, stats, K, n_choices, n_samples=500):
        """Monte Carlo estimate of selection probabilities."""
        counts = np.zeros(n_choices)
        for _ in range(n_samples):
            samples = []
            for v in range(n_choices):
                n_g, n_t = stats[v]
                alpha = n_g + 1
                beta_param = (n_t - n_g) + K
                samples.append(self.rng.beta(alpha, beta_param))
            counts[np.argmax(samples)] += 1
        return counts / n_samples
    
    def update(self, x, score):
        """Update stats after evaluation."""
        self.scores.append(score)
        
        # Update gamma (percentile-based)
        if len(self.scores) >= 10:
            self.gamma = np.percentile(self.scores, 80)  # top 20%
        
        is_good = score >= self.gamma
        
        for dim_idx, n_choices in self.categorical_dims:
            val_idx = self.discretize(x[dim_idx], n_choices)
            n_g, n_t = self.stats[dim_idx][val_idx]
            self.stats[dim_idx][val_idx] = (n_g + (1 if is_good else 0), n_t + 1)


def analyze_jahs():
    """Analisi dettagliata su JAHS."""
    
    print("="*80)
    print("DEBUG JAHS: Comportamento Thompson Sampling")
    print("="*80)
    
    # JAHS categorical structure
    categorical_dims = [
        (2, 3), (3, 3), (4, 3), (5, 3), (6, 2),  # Architettura
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),  # N x L
    ]
    
    wrapper = JAHSBenchWrapper(task='cifar10')
    dim = wrapper.dim
    
    print(f"\nDimensionalità: {dim}")
    print(f"Categorici: {categorical_dims}")
    print(f"  Dims 2-5: Op1-Op4 (3 scelte: none, skip, conv)")
    print(f"  Dim 6: TrivialAugment (2 scelte)")
    print(f"  Dims 7-12: N1-N6 (5 scelte: 1,2,4,8,16)")
    
    # Run con Thompson tracker
    debugger = ThompsonDebugger(categorical_dims, seed=42)
    
    n_evals = 200
    best_error = float('inf')
    best_x = None
    
    wrapper.reset()
    print(f"\nEseguo {n_evals} valutazioni...")
    
    for it in range(n_evals):
        # Sample categorici con Thompson
        cat_choices, probs = debugger.sample_categorical()
        
        # Sample continui uniformemente (per semplicità)
        x = np.array([debugger.rng.uniform(0, 1) for _ in range(dim)])
        
        # Applica scelte categoriche
        for dim_idx, val_idx in cat_choices.items():
            n_ch = dict(categorical_dims)[dim_idx]
            x[dim_idx] = debugger.to_continuous(val_idx, n_ch)
        
        # Valuta
        error = wrapper.evaluate_array(x)
        score = -error  # Higher is better internally
        
        debugger.update(x, score)
        
        if error < best_error:
            best_error = error
            best_x = x.copy()
    
    print(f"\nBest error: {best_error:.4f} (acc={100*(1-best_error):.2f}%)")
    
    # Analisi delle scelte
    print("\n" + "="*80)
    print("ANALISI SCELTE CATEGORICHE")
    print("="*80)
    
    # Estrai configurazione migliore
    print("\nConfigurazione migliore trovata:")
    for dim_idx, n_ch in categorical_dims:
        val_idx = debugger.discretize(best_x[dim_idx], n_ch)
        print(f"  Dim {dim_idx} ({n_ch} choices): {val_idx}")
    
    # Distribuzione scelte nel tempo
    print("\nDistribuzione scelte (early vs mid vs late):")
    for dim_idx, n_ch in categorical_dims[:3]:  # Prime 3 dimensioni
        early = debugger.choice_history[dim_idx][:50]
        mid = debugger.choice_history[dim_idx][50:100]
        late = debugger.choice_history[dim_idx][100:]
        
        early_dist = [early.count(v)/len(early) for v in range(n_ch)]
        mid_dist = [mid.count(v)/len(mid) for v in range(n_ch)]
        late_dist = [late.count(v)/len(late) for v in range(n_ch)] if late else [0]*n_ch
        
        print(f"\n  Dim {dim_idx} ({n_ch} choices):")
        print(f"    Early (0-50):  {[f'{p:.0%}' for p in early_dist]}")
        print(f"    Mid (50-100):  {[f'{p:.0%}' for p in mid_dist]}")
        print(f"    Late (100+):   {[f'{p:.0%}' for p in late_dist]}")
        
        # Mostra stats finali
        print(f"    Final stats:")
        for v in range(n_ch):
            n_g, n_t = debugger.stats[dim_idx][v]
            ratio = n_g / max(1, n_t)
            print(f"      Val {v}: {n_g}/{n_t} good ({ratio:.1%})")
    
    # Probabilità Thompson nel tempo
    print("\n" + "="*80)
    print("EVOLUZIONE PROBABILITÀ THOMPSON")
    print("="*80)
    
    dim_to_analyze = 2  # Prima dimensione categorica
    n_ch = 3
    
    print(f"\nDim {dim_to_analyze} (3 choices: 0=none, 1=skip, 2=conv):")
    print("\nIter |  P(0)  |  P(1)  |  P(2)  | Choice | Stats")
    print("-"*65)
    
    for it in [0, 5, 10, 20, 50, 100, 150, 199]:
        if it < len(debugger.probs_history[dim_to_analyze]):
            probs = debugger.probs_history[dim_to_analyze][it]
            choice = debugger.choice_history[dim_to_analyze][it]
            
            # Stats a quel punto (approssimativo)
            stats_str = ""
            print(f"{it:4d} | {probs[0]:5.1%} | {probs[1]:5.1%} | {probs[2]:5.1%} |   {choice}    |")
    
    # Analisi convergenza
    print("\n" + "="*80)
    print("CONVERGENZA: Quanto velocemente trova le categorie migliori?")
    print("="*80)
    
    # Trova quale categoria ha vinto per ogni dimensione
    print("\nCategoria finale preferita (>50% scelte nelle ultime 50 iter):")
    for dim_idx, n_ch in categorical_dims:
        late = debugger.choice_history[dim_idx][-50:]
        most_common = max(set(late), key=late.count)
        freq = late.count(most_common) / len(late)
        print(f"  Dim {dim_idx}: categoria {most_common} ({freq:.0%})")
    
    # Correlazione tra categorie
    print("\n" + "="*80)
    print("COMBINAZIONI CATEGORICHE PIÙ FREQUENTI (top 10)")
    print("="*80)
    
    combo_counts = defaultdict(int)
    for it in range(n_evals):
        combo = tuple(debugger.choice_history[dim_idx][it] 
                      for dim_idx, _ in categorical_dims)
        combo_counts[combo] += 1
    
    top_combos = sorted(combo_counts.items(), key=lambda x: -x[1])[:10]
    for combo, count in top_combos:
        print(f"  {combo}: {count} volte ({count/n_evals:.1%})")
    
    # Insight finale
    print("\n" + "="*80)
    print("INSIGHTS CHIAVE")
    print("="*80)
    
    # Calcola quanto tempo ci vuole per convergere
    for dim_idx, n_ch in categorical_dims[:3]:
        choices = debugger.choice_history[dim_idx]
        final_choice = max(set(choices[-20:]), key=choices[-20:].count)
        
        # Trova prima iterazione dove questa scelta diventa dominante
        window = 10
        for start in range(len(choices) - window):
            window_choices = choices[start:start+window]
            if window_choices.count(final_choice) >= window * 0.6:
                print(f"  Dim {dim_idx}: converge a val={final_choice} all'iter ~{start}")
                break
    
    print("""
    
    Osservazioni:
    1. Thompson bilancia esplorazione iniziale con sfruttamento finale
    2. Le categorie poco esplorate mantengono probabilità non-zero
    3. La convergenza è graduale, non brusca
    4. Le combinazioni categoriche buone emergono naturalmente
    """)


if __name__ == "__main__":
    analyze_jahs()
