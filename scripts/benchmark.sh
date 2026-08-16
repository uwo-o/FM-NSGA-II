#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# benchmark.sh — Benchmark completo FM-NSGA-II vs Baselines
# Genera tabla comparativa y exporta benchmark_results.csv
# Uso: ./scripts/benchmark.sh  (desde la raíz del proyecto)
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "╔══════════════════════════════════════════════════════╗"
echo "║    FM-NSGA-II — Benchmark completo multi-dataset    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Nota: este proceso puede tardar varios minutos."
echo ""

# ── Compilar si no existe ──────────────────────────────────────
if [[ ! -f "$BUILD_DIR/benchmark" ]]; then
    echo "⚠ Ejecutable benchmark no encontrado. Compilando..."
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
./benchmark 2>&1

if [[ -f "benchmark_results.csv" ]]; then
    echo ""
    echo "📄 Resultados guardados en: $BUILD_DIR/benchmark_results.csv"
fi
