#!/usr/bin/env python3
"""
ANALISI: Thompson Sampling su CONTINUI vs CATEGORICI
Perché funziona diversamente?
"""

import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def analyze_thompson_continuous():
    """Analisi teorica e pratica di Thompson su variabili continue."""
    
    print("="*80)
    print("ANALISI: Thompson Sampling su Continui vs Categorici")
    print("="*80)
    
    print("""
    DIFFERENZA FONDAMENTALE:
    
    CATEGORICI (3 scelte):
    ┌─────┬─────┬─────┐
    │  0  │  1  │  2  │   ← Solo 3 opzioni discrete
    └─────┴─────┴─────┘
    
    CONTINUI discretizzati (5 bins):
    ┌─────┬─────┬─────┬─────┬─────┐
    │ 0.1 │ 0.3 │ 0.5 │ 0.7 │ 0.9 │   ← 5 "pseudo-categorie"
    └─────┴─────┴─────┴─────┴─────┘
    
    PROBLEMA: L'ottimo continuo potrebbe essere a 0.42, ma con 5 bins
    devo scegliere 0.3 o 0.5 - PERDO PRECISIONE!
    """)
    
    # Simulazione: funzione continua con ottimo a 0.42
    def f_continuous(x):
        """Funzione con ottimo a x=0.42"""
        return -(x - 0.42)**2 + np.random.normal(0, 0.01)
    
    n_evals = 200
    n_runs = 30
    
    # Test 1: Thompson con diversi numeri di bins
    print("\n" + "="*80)
    print("TEST 1: Impatto del numero di bins")
    print("="*80)
    
    for n_bins in [3, 5, 10, 20, 50]:
        results = []
        for run in range(n_runs):
            rng = np.random.default_rng(run)
            stats = {i: (0, 0) for i in range(n_bins)}
            best = -np.inf
            gamma = -np.inf
            scores = []
            
            for _ in range(n_evals):
                # Thompson sampling
                samples = []
                for v in range(n_bins):
                    n_g, n_t = stats[v]
                    alpha = n_g + 1
                    beta_param = (n_t - n_g) + n_bins
                    samples.append(rng.beta(alpha, beta_param))
                
                chosen_bin = int(np.argmax(samples))
                # Centro del bin
                x = (chosen_bin + 0.5) / n_bins
                
                score = f_continuous(x)
                scores.append(score)
                best = max(best, score)
                
                # Update
                if len(scores) >= 10:
                    gamma = np.percentile(scores, 80)
                is_good = score >= gamma
                n_g, n_t = stats[chosen_bin]
                stats[chosen_bin] = (n_g + (1 if is_good else 0), n_t + 1)
            
            results.append(best)
        
        mean = np.mean(results)
        std = np.std(results)
        # Ottimo teorico è 0 (a x=0.42)
        print(f"  {n_bins:3d} bins: best = {mean:.5f} ± {std:.5f}  (ottimo = 0.0)")
    
    # Test 2: Confronto Thompson vs LGS su continuo
    print("\n" + "="*80)
    print("TEST 2: Thompson vs Gradient-based su funzione continua")
    print("="*80)
    
    def run_thompson_continuous(n_bins, n_evals, seed):
        rng = np.random.default_rng(seed)
        stats = {i: (0, 0) for i in range(n_bins)}
        best = -np.inf
        gamma = -np.inf
        scores = []
        
        for _ in range(n_evals):
            samples = []
            for v in range(n_bins):
                n_g, n_t = stats[v]
                alpha = n_g + 1
                beta_param = (n_t - n_g) + n_bins
                samples.append(rng.beta(alpha, beta_param))
            
            chosen_bin = int(np.argmax(samples))
            x = (chosen_bin + 0.5) / n_bins
            
            score = f_continuous(x)
            scores.append(score)
            best = max(best, score)
            
            if len(scores) >= 10:
                gamma = np.percentile(scores, 80)
            is_good = score >= gamma
            n_g, n_t = stats[chosen_bin]
            stats[chosen_bin] = (n_g + (1 if is_good else 0), n_t + 1)
        
        return best
    
    def run_gradient_continuous(n_evals, seed):
        """Semplice gradient-based search."""
        rng = np.random.default_rng(seed)
        
        # Punti e score
        points = []
        scores_list = []
        best = -np.inf
        
        for it in range(n_evals):
            if it < 10:
                # Random iniziale
                x = rng.uniform(0, 1)
            else:
                # Gradient-based: usa regressione lineare per stimare gradiente
                X = np.array(points[-20:]).reshape(-1, 1)
                y = np.array(scores_list[-20:])
                
                # Fit lineare
                X_mean = X.mean()
                y_mean = y.mean()
                grad = np.sum((X.flatten() - X_mean) * (y - y_mean)) / (np.sum((X.flatten() - X_mean)**2) + 1e-9)
                
                # Trova punto migliore e muovi lungo gradiente
                best_idx = np.argmax(scores_list[-20:])
                best_x = points[-20:][best_idx]
                
                step = rng.uniform(0.05, 0.2)
                x = best_x + step * np.sign(grad) + rng.normal(0, 0.05)
                x = np.clip(x, 0, 1)
            
            score = f_continuous(x)
            points.append(x)
            scores_list.append(score)
            best = max(best, score)
        
        return best
    
    thompson_results = []
    gradient_results = []
    
    for seed in range(n_runs):
        thompson_results.append(run_thompson_continuous(10, n_evals, seed))
        gradient_results.append(run_gradient_continuous(n_evals, seed))
    
    print(f"  Thompson (10 bins): {np.mean(thompson_results):.5f} ± {np.std(thompson_results):.5f}")
    print(f"  Gradient-based:     {np.mean(gradient_results):.5f} ± {np.std(gradient_results):.5f}")
    
    # Test 3: Caso MISTO (categorico + continuo)
    print("\n" + "="*80)
    print("TEST 3: Caso MISTO - dove Thompson eccelle")
    print("="*80)
    
    def f_mixed(cat, cont):
        """
        Funzione mista: il categorico determina la "regione" e il continuo ottimizza dentro.
        cat=0: ottimo continuo a 0.2
        cat=1: ottimo continuo a 0.8 (migliore overall!)
        cat=2: ottimo continuo a 0.5
        """
        if cat == 0:
            return 0.7 - (cont - 0.2)**2 + np.random.normal(0, 0.02)
        elif cat == 1:
            return 0.9 - (cont - 0.8)**2 + np.random.normal(0, 0.02)  # BEST
        else:
            return 0.5 - (cont - 0.5)**2 + np.random.normal(0, 0.02)
    
    def run_thompson_cat_lgs_cont(n_evals, seed):
        """Thompson per cat, gradient per cont."""
        rng = np.random.default_rng(seed)
        
        # Stats per categorici
        cat_stats = {0: (0, 0), 1: (0, 0), 2: (0, 0)}
        
        # History per continui
        cont_history = defaultdict(list)  # cat -> [(x, score)]
        
        best = -np.inf
        gamma = -np.inf
        scores = []
        
        for it in range(n_evals):
            # Thompson per categorico
            samples = []
            for v in range(3):
                n_g, n_t = cat_stats[v]
                alpha = n_g + 1
                beta_param = (n_t - n_g) + 3
                samples.append(rng.beta(alpha, beta_param))
            cat = int(np.argmax(samples))
            
            # Gradient-based per continuo (dentro la categoria scelta)
            hist = cont_history[cat]
            if len(hist) < 5:
                cont = rng.uniform(0, 1)
            else:
                # Trova best e muovi
                best_idx = np.argmax([s for _, s in hist[-10:]])
                best_cont = hist[-10:][best_idx][0]
                cont = best_cont + rng.normal(0, 0.1)
                cont = np.clip(cont, 0, 1)
            
            score = f_mixed(cat, cont)
            scores.append(score)
            best = max(best, score)
            cont_history[cat].append((cont, score))
            
            # Update Thompson stats
            if len(scores) >= 10:
                gamma = np.percentile(scores, 80)
            is_good = score >= gamma
            n_g, n_t = cat_stats[cat]
            cat_stats[cat] = (n_g + (1 if is_good else 0), n_t + 1)
        
        return best
    
    def run_thompson_all(n_evals, seed):
        """Thompson per tutto (cat + cont discretizzato)."""
        rng = np.random.default_rng(seed)
        
        n_bins_cont = 10
        stats = {}
        for cat in range(3):
            for cont_bin in range(n_bins_cont):
                stats[(cat, cont_bin)] = (0, 0)
        
        best = -np.inf
        gamma = -np.inf
        scores = []
        
        for it in range(n_evals):
            # Thompson su tutto lo spazio discretizzato
            samples = {}
            for key in stats:
                n_g, n_t = stats[key]
                alpha = n_g + 1
                beta_param = (n_t - n_g) + len(stats)
                samples[key] = rng.beta(alpha, beta_param)
            
            chosen = max(samples, key=samples.get)
            cat, cont_bin = chosen
            cont = (cont_bin + 0.5) / n_bins_cont
            
            score = f_mixed(cat, cont)
            scores.append(score)
            best = max(best, score)
            
            if len(scores) >= 10:
                gamma = np.percentile(scores, 80)
            is_good = score >= gamma
            n_g, n_t = stats[chosen]
            stats[chosen] = (n_g + (1 if is_good else 0), n_t + 1)
        
        return best
    
    hybrid_results = []
    full_thompson_results = []
    
    for seed in range(n_runs):
        hybrid_results.append(run_thompson_cat_lgs_cont(n_evals, seed))
        full_thompson_results.append(run_thompson_all(n_evals, seed))
    
    print(f"  Thompson(cat) + Gradient(cont): {np.mean(hybrid_results):.5f} ± {np.std(hybrid_results):.5f}")
    print(f"  Thompson(tutto discretizzato):  {np.mean(full_thompson_results):.5f} ± {np.std(full_thompson_results):.5f}")
    
    # Analisi teorica
    print("\n" + "="*80)
    print("SPIEGAZIONE TEORICA")
    print("="*80)
    
    print("""
    PERCHÉ THOMPSON SUI CONTINUI È PROBLEMATICO:
    
    1. DISCRETIZZAZIONE = PERDITA DI PRECISIONE
       - Con 10 bins su [0,1], ogni bin copre 0.1
       - Se l'ottimo è a 0.42, devo scegliere 0.35 o 0.45
       - Errore intrinseco = 0.07 nel migliore dei casi!
    
    2. ESPLOSIONE COMBINATORIA
       - 1 continuo con 10 bins = 10 opzioni
       - 5 continui con 10 bins = 10^5 = 100,000 opzioni
       - Thompson ha bisogno di visitare ogni opzione multiple volte!
    
    3. VARIABILI CONTINUE SONO "SMOOTH"
       - f(0.41) ≈ f(0.42) ≈ f(0.43)
       - Il gradiente locale è INFORMATIVO
       - Thompson ignora questa struttura!
    
    4. VARIABILI CATEGORICHE SONO "DISCRETE"
       - f(cat=0) potrebbe essere MOLTO diverso da f(cat=1)
       - Non c'è "gradiente" tra categorie
       - Thompson è PERFETTO per questo!
    
    SOLUZIONE OTTIMALE (come in ALBA_V1):
    ┌─────────────────────────────────────────────────────────┐
    │  CATEGORICI → Thompson Sampling (Bandit)                │
    │  CONTINUI   → LGS/Gradient-based (Surrogate locale)     │
    └─────────────────────────────────────────────────────────┘
    
    Ogni metodo fa quello che sa fare meglio!
    """)
    
    # Ma perché thompson_all ha funzionato su JAHS?
    print("\n" + "="*80)
    print("MA PERCHÉ THOMPSON_ALL HA FUNZIONATO SU JAHS?")
    print("="*80)
    
    print("""
    JAHS-Bench ha una struttura particolare:
    
    1. SOLO 2 DIMENSIONI CONTINUE (learning_rate, weight_decay)
       - Non c'è esplosione combinatoria!
       - Con 5 bins ciascuna = 25 combinazioni continue
    
    2. 11 DIMENSIONI CATEGORICHE
       - Op1-Op4: 3 scelte ciascuna = 81 combinazioni
       - Augment: 2 scelte
       - N1-N6: 5 scelte ciascuna = 15,625 combinazioni
    
    3. IL LANDSCAPE È "QUASI-DISCRETO"
       - Le variabili continue hanno pochi valori "buoni"
       - learning_rate: spesso 0.1 o 0.01 funzionano
       - weight_decay: spesso 0 o 1e-4 funzionano
    
    QUINDI su JAHS:
    - Thompson_all funziona perché i continui sono "quasi-discreti"
    - Ma su problemi con continui smooth (es. Rosenbrock), fallirebbe!
    
    REGOLA PRATICA:
    ┌──────────────────────────────────────────────────────────┐
    │ Se i continui sono pochi (≤3) e il landscape è           │
    │ "quasi-discreto", Thompson_all può funzionare.           │
    │                                                          │
    │ Se i continui sono tanti (>5) o il landscape è smooth,   │
    │ usa Thompson(cat) + LGS(cont) come in ALBA_V1.           │
    └──────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    analyze_thompson_continuous()
