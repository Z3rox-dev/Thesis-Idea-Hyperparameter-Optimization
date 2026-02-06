# Metodologia di Debugging per Algoritmi di Ottimizzazione (QuadHPO)

Questo documento descrive l'approccio metodologico utilizzato per diagnosticare e risolvere problemi di instabilità numerica e convergenza nell'algoritmo `QuadHPO_Subspace`. Può essere utilizzato come guida per future sessioni di debugging.

## 1. Fase di Riproduzione e Isolamento

Il primo passo è sempre ridurre il problema alla sua forma più semplice. Eseguire l'intero benchmark richiede troppo tempo per un ciclo di debug efficace.

*   **Identificazione**: Analizzare i log del benchmark completo per trovare i "seed" o le configurazioni che falliscono (es. Seed 707 che rimaneva a Depth 0).
*   **Script Minimo**: Creare uno script dedicato (es. `tests/debug_quadhpo_subspace.py`) che:
    1.  Esegue l'algoritmo **solo** sui seed problematici (e uno funzionante per confronto).
    2.  Usa una funzione obiettivo veloce (es. surrogato tabulare o funzione sintetica) invece di addestrare modelli reali.
    3.  Imposta un budget sufficiente a manifestare il problema, ma non eccessivo.

**Obiettivo**: Avere un ciclo "Modifica -> Esegui -> Verifica" che dura meno di 30 secondi.

## 2. Strumentazione (Logging Strategico)

I semplici `print` spesso non bastano perché generano troppo rumore o mancano di contesto. È necessario esporre lo "stato mentale" dell'algoritmo.

*   **Stato Interno**: Aggiungere attributi temporanei alle classi per tracciare le decisioni.
    *   *Esempio*: Aggiunto `self.split_block_reason` alla classe `QuadCube` per memorizzare *perché* un cubo ha deciso di non dividersi (es. "insufficient_data", "min_width", "low_r2").
*   **Logging Strutturato**: Scrivere un file di log (CSV o JSON) che registri ogni evento critico:
    *   Trial ID
    *   Azione (Split, Prune, Sample)
    *   Metriche del Modello (R2, Sigma2, Curvature Scores)
    *   Motivo del blocco (se applicabile)

## 3. Analisi delle Tracce (Trace Analysis)

Invece di leggere i log a occhio nudo, creare uno script di analisi (es. `tests/analyze_debug_trace.py`) che parsa i log e risponde a domande specifiche:

*   **Analisi Topologica**: "Qual è la profondità massima raggiunta per ogni seed?"
*   **Analisi Decisionale**: "Quante volte lo split è stato bloccato per 'insufficient_data' vs 'flat_region'?"
*   **Analisi Qualitativa**: "Quando il modello ha deciso di splittare, qual era il valore di R2?"

**Esempio Pratico (Caso Seed 707):**
L'analisi ha mostrato che il Seed 707 aveva un modello surrogato con $R^2 < 0.1$ (pessimo fit), ma l'algoritmo provava comunque a usare la curvatura per decidere il taglio. Questo generava tagli casuali o bloccava l'esplorazione.

## 4. Ipotesi e Fix

Basandosi sui dati, formulare ipotesi e applicare correzioni mirate.

*   **Ipotesi**: La Ridge Regression è instabile su sottospazi piccoli o mal condizionati.
    *   *Fix*: Normalizzazione degli input (`T_scaled`) prima del fitting per avere varianza unitaria.
*   **Ipotesi**: La curvatura è rumorosa quando il modello non fitta bene i dati.
    *   *Fix*: "Quality Gate". Se $R^2 < 0.1$, ignorare la curvatura e usare un fallback geometrico (taglio a metà).

## 5. Verifica

Rilanciare lo script minimo di debug.
*   Verificare che il Seed problematico ora si comporti come quello funzionante (es. raggiunge Depth 3+).
*   Verificare che il Seed funzionante non sia peggiorato (regression testing).
*   Infine, rilanciare il benchmark completo per confermare la generalizzazione.

---

## Idee Future e Approcci Non Ancora Provati

Queste sono idee che potrebbero migliorare ulteriormente la robustezza o le performance, ma che non sono state implementate in questa sessione.

### 1. Miglioramento del Surrogato
*   **Gaussian Processes (GP)**: Sostituire la Ridge Regression quadratica con un GP (es. con kernel Matern). I GP forniscono una stima dell'incertezza ($\sigma^2$) molto più accurata, utile per l'Acquisition Function (EI/UCB).
    *   *Pro*: Migliore gestione di dati sparsi.
    *   *Contro*: Costo computazionale cubico $O(N^3)$.
*   **Ensemble Methods**: Usare un piccolo ensemble di modelli (es. 3-5 Ridge Regressors con diversi iperparametri o bootstrap) per stimare la robustezza della curvatura.

### 2. Criteri di Split Avanzati
*   **Pure Variance Reduction**: Invece di usare la curvatura, simulare lo split e calcolare esplicitamente la riduzione attesa della varianza globale (Information Gain puro). È più costoso ma teoricamente più solido.
*   **Lookahead**: Simulare 1 o 2 passi di ottimizzazione futuri per decidere se lo split vale la pena (stile Monte Carlo Tree Search).

### 3. Gestione Dinamica degli Iperparametri
*   **Adaptive `min_points`**: Invece di fissare `min_points = D + 2`, renderlo dinamico in base alla rumorosità osservata ($R^2$). Se il segnale è basso, richiedere più punti prima di splittare.
*   **Budget-Awareness**: Modificare l'aggressività dell'esplorazione (parametro $\beta$ di UCB o soglia $\gamma$ di split) in funzione del budget rimanente. Essere più esplorativi all'inizio e più "greedy" alla fine.

### 4. Robustezza Geometrica
*   **Soft Restarts**: Se un ramo dell'albero sembra stagnare (varianza bassa, nessun miglioramento), invece di "prunarlo" completamente, resettare i suoi parametri interni (es. dimenticare parzialmente la storia passata) per permettere al modello di riadattarsi a nuovi dati locali.
*   **Overlap**: Permettere ai cubi figli di sovrapporsi leggermente (soft boundaries) per evitare che l'ottimo si nasconda proprio sul bordo di taglio.
