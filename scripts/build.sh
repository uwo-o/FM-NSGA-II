#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# build.sh — Compilar el proyecto FM-NSGA-II
# Uso: ./scripts/build.sh [release|debug|asan]
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

MODE="${1:-release}"

case "$MODE" in
    release)
        echo "▶ Compilando en modo Release..."
        cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_CXX_FLAGS="-O2" 2>&1
        cmake --build "$BUILD_DIR" --parallel 2>&1
        ;;
    debug)
        echo "▶ Compilando en modo Debug..."
        cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
              -DCMAKE_BUILD_TYPE=Debug 2>&1
        cmake --build "$BUILD_DIR" --parallel 2>&1
        ;;
    asan)
        echo "▶ Compilando con AddressSanitizer..."
        cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
              -DCMAKE_BUILD_TYPE=Debug \
              -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
              -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" 2>&1
        cmake --build "$BUILD_DIR" --target fm_nsga2_asan --parallel 2>&1
        ;;
    *)
        echo "Uso: $0 [release|debug|asan]"
        exit 1
        ;;
esac

echo ""
echo "✅ Build OK — ejecutables en $BUILD_DIR/"
ls -lh "$BUILD_DIR/fm_nsga2" "$BUILD_DIR/benchmark" 2>/dev/null || true
