# Copilot instructions (bench suite)

Questo repo contiene più suite di benchmark che richiedono **ambienti Python diversi** (dipendenze incompatibili tra loro, es. pandas/numpy).
Quando GitHub Copilot ti chiede di eseguire o modificare benchmark, usa questa tabella per scegliere l’environment corretto.

## Matrice ambienti

### 1) `base` (python3 di sistema / devcontainer)
Usalo per:
- YAHPO Gym (benchmark tabular)
- Script generali di analisi/parsing risultati

Comandi tipici:
- `python3 thesis/benchmark_full_battery.py`

Note:
- In questo repo YAHPO è già stato eseguito con `python3` (senza conda).

### 2) Conda `py39`
Usalo per:
- JAHS-Bench-201 (e wrapper correlati)

Comandi tipici:
- `source /mnt/workspace/miniconda3/bin/activate py39`
- `python thesis/benchmark_jahs_battery.py`

Note importanti:
- JAHS può essere sensibile a incompatibilità binarie (numpy/pandas). Se compaiono errori tipo
  `numpy.dtype size changed`, serve riallineare versioni nel solo env `py39`.

### 3) HPOBench (consigliato: env dedicato)
Usalo per:
- HPOBench NASBench-201 / NASBench-101
- (Opzionale) altri benchmark di HPOBench (ML/RL/OD)

Due modalità:

**A) “editable path” (rapida, senza install)**
Funziona se lo script fa `sys.path.insert(0, '/mnt/workspace/HPOBench')`.

**B) env dedicato (consigliato per run lunghi)**
Crea un env isolato per non rompere `py39`:
- `conda create -n hpobench python=3.10 -y`
- `source /mnt/workspace/miniconda3/bin/activate hpobench`
- `pip install -r /mnt/workspace/HPOBench/requirements.txt`
- `pip install -e /mnt/workspace/HPOBench`

## Dove sono i benchmark

- YAHPO + JAHS battery: `thesis/benchmark_full_battery.py`
- JAHS-only battery (conda py39): `thesis/benchmark_jahs_battery.py`
- HPOBench repo vendorizzato: `HPOBench/`

## Convenzioni risultati

- I benchmark salvano JSON incrementali in: `thesis/benchmark_results/`
- I log vengono scritti in: `thesis/*.log`

## Gotcha frequenti

- Se un benchmark muore “alla fine” nel summary, controlla prima che il JSON sia completo: spesso è solo un bug di stampa.
- Evita di installare dipendenze “pesanti” nell’env `py39` usato per JAHS: preferisci un env separato per HPOBench.
