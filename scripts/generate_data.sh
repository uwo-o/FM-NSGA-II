#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# generate_data.sh — Generar todos los datasets CSV
# Uso: ./scripts/generate_data.sh
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "▶ Generando datasets en $BUILD_DIR/data/ ..."
python3 "$BUILD_DIR/generate_datasets.py"

echo ""
echo "✅ Datasets generados:"
ls -lh "$BUILD_DIR/data/"
