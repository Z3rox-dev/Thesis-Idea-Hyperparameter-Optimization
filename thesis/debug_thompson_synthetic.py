#!/usr/bin/env python3
"""
DEBUG INTENSIVO: Simulazione JAHS-like per capire Thompson Sampling
"""

import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def jahs_like_objective(x, optimal_cats):
    """
    Simula un obiettivo JAHS-like:
    - x[0:2] continui (learning rate, weight decay) - contributo gaussiano
    - x[2:6] categorici per architettura (4 ops x 3 choices)
    - x[6] categorico per augment (2 choices)
    - x[7:13] categorici per width (6 layers x 5 choices)
    
    Obiettivo: minimizzare error (o massimizzare -error)
    """
    dim = len(x)
    
    # Contributo continuo: ottimo vicino a (0.3, 0.7)
    cont_loss = (x[0] - 0.3)**2 + (x[1] - 0.7)**2
    
    # Contributo categorico: ogni categoria giusta riduce l'errore
    cat_dims = [
        (2, 3), (3, 3), (4, 3), (5, 3),  # ops
        (6, 2),                            # augment
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),  # widths
    ]
    
    cat_bonus = 0.0
    for i, (dim_idx, n_ch) in enumerate(cat_dims):
        cat_val = int(round(x[dim_idx] * (n_ch - 1)))
        cat_val = min(n_ch-1, max(0, cat_val))
        
        if cat_val == optimal_cats[i]:
            cat_bonus -= 0.05  # Reduce error
        else:
            dist = abs(cat_val - optimal_cats[i])
            cat_bonus += 0.02 * dist  # Increase error
    
    # Add noise
    noise = np.random.normal(0, 0.01)
    
    # Final error (lower is better)
    error = cont_loss + cat_bonus + noise + 0.1  # Base error ~0.1
    return max(0.01, min(0.5, error))


class ThompsonDebugger:
    """Tracker per Thompson Sampling con analisi dettagliata."""
    
    def __init__(self, categorical_dims, seed=42):
        self.categorical_dims = categorical_dims
        self.rng = np.random.default_rng(seed)
        self.dim = max(d[0] for d in categorical_dims) + 1
        
        # Stats: {dim_idx: {val_idx: (n_good, n_total)}}
        self.stats = {}
        for dim_idx, n_ch in categorical_dims:
            self.stats[dim_idx] = {v: (0, 0) for v in range(n_ch)}
        
        self.gamma = -np.inf
        self.scores = []  # Internal scores (higher is better)
        
        # Tracking
        self.choice_history = defaultdict(list)
        self.thompson_samples_history = defaultdict(list)  # Raw samples before argmax
        self.beta_params_history = defaultdict(list)  # Alpha, Beta params
        
    def discretize(self, x_val, n_choices):
        return min(int(round(x_val * (n_choices - 1))), n_choices - 1)
    
    def to_continuous(self, val_idx, n_choices):
        return val_idx / (n_choices - 1) if n_choices > 1 else 0.5
    
    def sample(self, exploration_boost=1.0):
        """Generate next point using Thompson Sampling for categoricals."""
        x = np.zeros(self.dim)
        
        # Continuous dims: uniform (simplified)
        x[0] = self.rng.uniform(0, 1)
        x[1] = self.rng.uniform(0, 1)
        
        # Categorical dims: Thompson Sampling
        for dim_idx, n_ch in self.categorical_dims:
            stats = self.stats[dim_idx]
            K = n_ch * exploration_boost
            
            samples = []
            beta_params = []
            for v in range(n_ch):
                n_g, n_t = stats[v]
                alpha = n_g + 1
                beta_param = (n_t - n_g) + K
                sample = self.rng.beta(alpha, beta_param)
                samples.append(sample)
                beta_params.append((alpha, beta_param))
            
            chosen = int(np.argmax(samples))
            x[dim_idx] = self.to_continuous(chosen, n_ch)
            
            self.choice_history[dim_idx].append(chosen)
            self.thompson_samples_history[dim_idx].append(samples)
            self.beta_params_history[dim_idx].append(beta_params)
        
        return x
    
    def update(self, x, error):
        """Update stats after evaluation."""
        score = -error  # Higher is better
        self.scores.append(score)
        
        # Update gamma (top 20%)
        if len(self.scores) >= 10:
            self.gamma = np.percentile(self.scores, 80)
        
        is_good = score >= self.gamma
        
        for dim_idx, n_ch in self.categorical_dims:
            val_idx = self.discretize(x[dim_idx], n_ch)
            n_g, n_t = self.stats[dim_idx][val_idx]
            self.stats[dim_idx][val_idx] = (n_g + (1 if is_good else 0), n_t + 1)


def run_detailed_analysis():
    """Analisi dettagliata del comportamento di Thompson."""
    
    print("="*80)
    print("DEBUG: Thompson Sampling su problema JAHS-like")
    print("="*80)
    
    # Configurazione ottimale nascosta
    optimal_cats = [1, 2, 0, 1, 1, 3, 3, 2, 4, 2, 1]  # 11 categorici
    
    categorical_dims = [
        (2, 3), (3, 3), (4, 3), (5, 3),  # ops (4 dims x 3 choices)
        (6, 2),                            # augment (1 dim x 2 choices)
        (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5),  # widths (6 dims x 5 choices)
    ]
    
    print(f"\nConfigurazione ottimale nascosta: {optimal_cats}")
    print(f"Categorici: {len(categorical_dims)} dimensioni")
    
    debugger = ThompsonDebugger(categorical_dims, seed=42)
    np.random.seed(42)  # Per noise
    
    n_evals = 300
    errors = []
    best_error = float('inf')
    
    print(f"\nEseguo {n_evals} valutazioni...\n")
    
    for it in range(n_evals):
        x = debugger.sample()
        error = jahs_like_objective(x, optimal_cats)
        errors.append(error)
        debugger.update(x, error)
        
        if error < best_error:
            best_error = error
            if it % 50 == 0 or it < 10:
                print(f"  Iter {it}: nuovo best error = {best_error:.4f}")
    
    print(f"\n{'='*80}")
    print(f"RISULTATI")
    print(f"{'='*80}")
    print(f"Best error: {best_error:.4f}")
    print(f"Mean error (last 50): {np.mean(errors[-50:]):.4f}")
    
    # 1. CONVERGENZA VERSO OTTIMALI
    print(f"\n{'='*80}")
    print("1. CONVERGENZA VERSO CATEGORIE OTTIMALI")
    print("="*80)
    
    for i, (dim_idx, n_ch) in enumerate(categorical_dims[:5]):
        optimal = optimal_cats[i]
        choices = debugger.choice_history[dim_idx]
        
        # % scelte ottimali nel tempo
        early = choices[:50]
        mid = choices[50:150]
        late = choices[150:]
        
        early_opt = sum(1 for c in early if c == optimal) / len(early)
        mid_opt = sum(1 for c in mid if c == optimal) / len(mid)
        late_opt = sum(1 for c in late if c == optimal) / len(late) if late else 0
        
        print(f"\nDim {dim_idx} (optimal={optimal}):")
        print(f"  Early (0-50):   {early_opt:.0%} scelte ottimali")
        print(f"  Mid (50-150):   {mid_opt:.0%} scelte ottimali")
        print(f"  Late (150-300): {late_opt:.0%} scelte ottimali")
        
        # Stats finali
        print(f"  Stats finali per ogni valore:")
        for v in range(n_ch):
            n_g, n_t = debugger.stats[dim_idx][v]
            ratio = n_g / max(1, n_t)
            marker = " <-- OPTIMAL" if v == optimal else ""
            print(f"    Val {v}: {n_g}/{n_t} good ({ratio:.1%}){marker}")
    
    # 2. EVOLUZIONE BETA PARAMETERS
    print(f"\n{'='*80}")
    print("2. EVOLUZIONE PARAMETRI BETA NEL TEMPO")
    print("="*80)
    
    dim_idx = 2  # Prima dimensione categorica
    n_ch = 3
    optimal = optimal_cats[0]
    
    print(f"\nDim {dim_idx} (3 choices, optimal={optimal}):")
    print("\nIter | Beta params (alpha, beta) per ogni choice | Thompson samples | Chosen")
    print("-"*90)
    
    for it in [0, 5, 10, 20, 50, 100, 150, 200, 299]:
        if it < len(debugger.beta_params_history[dim_idx]):
            beta_params = debugger.beta_params_history[dim_idx][it]
            samples = debugger.thompson_samples_history[dim_idx][it]
            chosen = debugger.choice_history[dim_idx][it]
            
            beta_str = " | ".join([f"({a:.0f},{b:.1f})" for a, b in beta_params])
            samples_str = " | ".join([f"{s:.3f}" for s in samples])
            marker = "✓" if chosen == optimal else " "
            print(f"{it:4d} | {beta_str} | {samples_str} | {chosen} {marker}")
    
    # 3. PERCHÉ ALCUNE ITERAZIONI SCELGONO SUBOTTIMALI?
    print(f"\n{'='*80}")
    print("3. ANALISI SCELTE 'SUBOTTIMALI'")
    print("="*80)
    
    print(f"\nQuando e perché Thompson sceglie categorie non-ottimali?")
    
    dim_idx = 2
    optimal = optimal_cats[0]
    suboptimal_iters = []
    
    for it in range(min(100, n_evals)):
        choice = debugger.choice_history[dim_idx][it]
        if choice != optimal:
            beta_params = debugger.beta_params_history[dim_idx][it]
            samples = debugger.thompson_samples_history[dim_idx][it]
            suboptimal_iters.append((it, choice, beta_params, samples))
    
    print(f"\nPrime 10 scelte subottimali per dim {dim_idx}:")
    for it, choice, beta_params, samples in suboptimal_iters[:10]:
        print(f"  Iter {it}: scelse {choice} invece di {optimal}")
        print(f"    Beta params: {[(f'{a:.0f}', f'{b:.1f}') for a, b in beta_params]}")
        print(f"    Samples:     {[f'{s:.3f}' for s in samples]}")
        print(f"    -> Sample({choice})={samples[choice]:.3f} > Sample({optimal})={samples[optimal]:.3f}")
    
    # Insight: perché queste esplorazioni aiutano?
    print(f"\n  Insight: queste esplorazioni accadono quando:")
    print(f"    - La categoria ottimale ha ancora alta varianza")
    print(f"    - Il campione stocastico favorisce alternative")
    print(f"    - Questo RIDUCE l'overfit a scelte subottimali early!")
    
    # 4. PROBABILITÀ IMPLICITE DI SELEZIONE
    print(f"\n{'='*80}")
    print("4. PROBABILITÀ IMPLICITE DI SELEZIONE")
    print("="*80)
    
    print(f"\nProbabilità di selezionare ogni categoria (Monte Carlo, 10k samples):")
    
    def compute_thompson_probs(stats, n_ch, exploration_boost=1.0, n_mc=10000):
        """Monte Carlo per probabilità di selezione."""
        rng = np.random.default_rng(0)
        K = n_ch * exploration_boost
        counts = np.zeros(n_ch)
        
        for _ in range(n_mc):
            samples = []
            for v in range(n_ch):
                n_g, n_t = stats[v]
                alpha = n_g + 1
                beta_param = (n_t - n_g) + K
                samples.append(rng.beta(alpha, beta_param))
            counts[np.argmax(samples)] += 1
        
        return counts / n_mc
    
    for i, (dim_idx, n_ch) in enumerate(categorical_dims[:3]):
        optimal = optimal_cats[i]
        stats = debugger.stats[dim_idx]
        probs = compute_thompson_probs(stats, n_ch)
        
        print(f"\n  Dim {dim_idx} (optimal={optimal}):")
        for v in range(n_ch):
            n_g, n_t = stats[v]
            marker = " <-- OPTIMAL" if v == optimal else ""
            print(f"    Val {v}: P(select)={probs[v]:.1%}, stats={n_g}/{n_t}{marker}")
    
    # 5. CONFRONTO CON ALTRI METODI
    print(f"\n{'='*80}")
    print("5. CONFRONTO: THOMPSON vs UNIFORM vs GREEDY")
    print("="*80)
    
    n_runs = 20
    n_evals_short = 200
    
    methods = {
        'Thompson': lambda: run_method('thompson', categorical_dims, optimal_cats, n_evals_short),
        'Uniform': lambda: run_method('uniform', categorical_dims, optimal_cats, n_evals_short),
        'Greedy(ε=0.1)': lambda: run_method('greedy', categorical_dims, optimal_cats, n_evals_short),
    }
    
    for method_name, method_fn in methods.items():
        final_errors = []
        for run in range(n_runs):
            np.random.seed(run + 100)
            _, best = method_fn()
            final_errors.append(best)
        
        mean = np.mean(final_errors)
        std = np.std(final_errors)
        print(f"  {method_name:20s}: {mean:.4f} ± {std:.4f}")
    
    # 6. KEY INSIGHT
    print(f"\n{'='*80}")
    print("6. KEY INSIGHTS")
    print("="*80)
    
    print("""
    PERCHÉ THOMPSON FUNZIONA SU HPO CATEGORICO:
    
    1. EXPLORATION PROPORZIONALE ALL'INCERTEZZA
       - Categorie poco testate hanno alta varianza Beta
       - Possono "vincere" il sampling anche se empiricamente peggiori
       - Questo evita di bloccarsi su minimi locali
    
    2. EXPLOITATION AUTOMATICA
       - Categorie buone accumulano n_good alto
       - Beta si concentra su valori alti → alta prob di selezione
       - NON serve temperature scheduling!
    
    3. ROBUSTO AL RUMORE
       - HPO ha rumore intrinseco (init random, etc.)
       - Beta distribution modella naturalmente l'incertezza
       - Una categoria "sembra buona" per caso? Verrà ri-esplorata!
    
    4. COMBINAZIONI CATEGORICHE
       - In JAHS ci sono 11 categorici con ~13M combinazioni
       - Thompson evita di fissarsi su combinazioni subottimali
       - Esplora alternative anche quando ha trovato "qualcosa che funziona"
    
    5. SINERGIA CON LGS (per continui)
       - Thompson sceglie la struttura categorica
       - LGS ottimizza i continui dentro quella struttura
       - Divide et impera: ogni metodo fa quello che sa fare meglio
    """)


def run_method(method, categorical_dims, optimal_cats, n_evals):
    """Helper per confronto metodi."""
    rng = np.random.default_rng()
    dim = 13
    
    # Stats
    stats = {}
    for dim_idx, n_ch in categorical_dims:
        stats[dim_idx] = {v: (0, 0) for v in range(n_ch)}
    
    gamma = -np.inf
    scores = []
    best_error = float('inf')
    
    for it in range(n_evals):
        x = np.zeros(dim)
        x[0] = rng.uniform(0, 1)
        x[1] = rng.uniform(0, 1)
        
        for dim_idx, n_ch in categorical_dims:
            if method == 'uniform':
                chosen = rng.integers(0, n_ch)
            elif method == 'greedy':
                if rng.random() < 0.1:
                    chosen = rng.integers(0, n_ch)
                else:
                    best_v, best_ratio = 0, -1
                    for v in range(n_ch):
                        n_g, n_t = stats[dim_idx][v]
                        ratio = (n_g + 1) / (n_t + 2)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_v = v
                    chosen = best_v
            else:  # thompson
                K = n_ch
                samples = []
                for v in range(n_ch):
                    n_g, n_t = stats[dim_idx][v]
                    alpha = n_g + 1
                    beta_param = (n_t - n_g) + K
                    samples.append(rng.beta(alpha, beta_param))
                chosen = int(np.argmax(samples))
            
            x[dim_idx] = chosen / (n_ch - 1) if n_ch > 1 else 0.5
        
        error = jahs_like_objective(x, optimal_cats)
        best_error = min(best_error, error)
        
        score = -error
        scores.append(score)
        if len(scores) >= 10:
            gamma = np.percentile(scores, 80)
        
        is_good = score >= gamma
        for dim_idx, n_ch in categorical_dims:
            val_idx = int(round(x[dim_idx] * (n_ch - 1)))
            n_g, n_t = stats[dim_idx].get(val_idx, (0, 0))
            stats[dim_idx][val_idx] = (n_g + (1 if is_good else 0), n_t + 1)
    
    return scores, best_error


if __name__ == "__main__":
    run_detailed_analysis()
