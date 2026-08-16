#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# test_quick.sh — Test rápido: FM-NSGA-II en Two Moons 2D
# Uso: ./scripts/test_quick.sh  (desde la raíz del proyecto)
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "╔══════════════════════════════════════════════════════╗"
echo "║         FM-NSGA-II — Test Rápido (Two Moons)        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Verificar que el build existe ──────────────────────────────
if [[ ! -f "$BUILD_DIR/fm_nsga2" ]]; then
    echo "⚠ Ejecutable no encontrado. Compilando..."
    cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --parallel
    echo ""
fi

# ── Verificar / generar datasets ───────────────────────────────
if [[ ! -f "$BUILD_DIR/data/moons_2d.csv" ]]; then
    echo "⚠ Datasets no encontrados. Generando..."
    python3 "$BUILD_DIR/generate_datasets.py"
    echo ""
fi

# ── Ejecutar test ──────────────────────────────────────────────
echo "▶ Corriendo FM-NSGA-II en data/moons_2d.csv ..."
echo ""
cd "$BUILD_DIR"
./fm_nsga2 data/moons_2d.csv 2>&1

echo ""
echo "✅ Test rápido completado."
