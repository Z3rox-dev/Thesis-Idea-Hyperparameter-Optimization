#!/usr/bin/env bash
# ============================================================================
#  build_package.sh — crea un pacchetto distribuibile di ALBA
#
#  Uso:   bash build_package.sh
#  Output: alba_dist/dist/alba_framework-1.0.0-py3-none-any.whl
#          alba_dist/dist/alba_framework-1.0.0.tar.gz
#          alba_release.zip   (archivio pronto da inviare)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/alba_framework_potential"
DIST="$SCRIPT_DIR/alba_dist"

echo "=== 1/4  Preparazione layout in alba_dist/ ==="

# Copia il pacchetto Python nella dist directory
rm -rf "$DIST/alba_framework_potential"
mkdir -p "$DIST/alba_framework_potential"

# Copia solo i file .py del pacchetto core (no tools/, no __pycache__)
for f in "$SRC"/*.py; do
    cp "$f" "$DIST/alba_framework_potential/"
done

# Copia examples
mkdir -p "$DIST/alba_framework_potential/examples"
cp "$SRC/examples/quick_demo.py" "$DIST/alba_framework_potential/examples/"
touch "$DIST/alba_framework_potential/examples/__init__.py"

# Copia documentazione
cp "$SRC/INSTALL.md" "$DIST/"

echo "=== 2/4  Build del pacchetto ==="

# Usa un venv temporaneo per il build
TMPENV=$(mktemp -d)
python3 -m venv "$TMPENV"
source "$TMPENV/bin/activate"
pip install --upgrade pip setuptools wheel build -q

cd "$DIST"
python -m build 2>&1 | tail -5

echo ""
echo "=== 3/4  Verifica installazione ==="

TESTENV=$(mktemp -d)
python3 -m venv "$TESTENV"
source "$TESTENV/bin/activate"
pip install --upgrade pip -q

WHL=$(ls "$DIST/dist/"*.whl 2>/dev/null | head -1)
pip install "$WHL" -q

python -c "
from alba_framework_potential import ALBA
import numpy as np

opt = ALBA(bounds=[(-5,5)]*3, maximize=False, seed=42, total_budget=50)
best_x, best_y = opt.optimize(lambda x: float(np.sum(x**2)), budget=50)
print(f'  ✓ ALBA installed and working. Sphere 3D → best_y={best_y:.6f}')
"

echo ""
echo "=== 4/4  Creazione archivio tar.gz ==="

cd "$SCRIPT_DIR"
rm -f alba_release.tar.gz
tar czf alba_release.tar.gz \
    -C "$DIST" \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude="*.egg-info" \
    --exclude="build" \
    pyproject.toml README.md INSTALL.md \
    alba_framework_potential/ \
    dist/

echo ""
echo "============================================================"
echo "  ✓  DONE!"
echo ""
echo "  File da inviare al professore:"
echo ""
echo "  Opzione A (wheel, un solo file):"
echo "    $(ls "$DIST/dist/"*.whl)"
echo ""
echo "  Opzione B (archivio completo con sorgenti + docs):"
echo "    $SCRIPT_DIR/alba_release.tar.gz"
echo ""
echo "  Il professore dovrà solo fare:"
echo "    pip install alba_framework-1.0.0-py3-none-any.whl"
echo "    python -m alba_framework_potential.examples.quick_demo"
echo "============================================================"

# Pulizia
rm -rf "$TMPENV" "$TESTENV"
