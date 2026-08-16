#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# test_all_datasets.sh — Test FM-NSGA-II en todos los datasets
# Uso: ./scripts/test_all_datasets.sh  (desde la raíz del proyecto)
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "╔══════════════════════════════════════════════════════╗"
echo "║      FM-NSGA-II — Test en todos los datasets        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Compilar si no existe ──────────────────────────────────────
if [[ ! -f "$BUILD_DIR/fm_nsga2" ]]; then
    echo "⚠ Ejecutable no encontrado. Compilando..."
    cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --parallel
    echo ""
fi

# ── Generar datasets si no existen ────────────────────────────
if [[ ! -f "$BUILD_DIR/data/moons_2d.csv" ]]; then
    echo "⚠ Datasets no encontrados. Generando..."
    python3 "$BUILD_DIR/generate_datasets.py"
    echo ""
fi

cd "$BUILD_DIR"

DATASETS=(
    "data/blobs_2d.csv"
    "data/moons_2d.csv"
    "data/circles_2d.csv"
    "data/spirals_2d.csv"
    "data/chessboard_2d.csv"
    "data/adversarial_52d.csv"
    "data/iris.csv"
    "data/wine.csv"
    "data/breast_cancer.csv"
)

PASS=0
FAIL=0

for ds in "${DATASETS[@]}"; do
    if [[ ! -f "$ds" ]]; then
        echo "⚠ Skipping $ds (no encontrado)"
        ((FAIL++)) || true
        continue
    fi
    echo "══════════════════════════════════════"
    echo "▶ Dataset: $ds"
    echo "══════════════════════════════════════"
    if ./fm_nsga2 "$ds" 2>&1; then
        ((PASS++)) || true
    else
        echo "✗ FALLÓ en $ds"
        ((FAIL++)) || true
    fi
    echo ""
done

echo "══════════════════════════════════════"
echo "Resumen: $PASS OK, $FAIL fallidos"
echo "══════════════════════════════════════"
