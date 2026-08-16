#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# make_report.sh — Orquestador para generar el reporte en PDF
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "╔══════════════════════════════════════════════════════╗"
echo "║       Generador de Reporte RM-NSGA-II (LaTeX)       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# 1. Asegurar que tenemos benchmark_results.csv
if [[ ! -f "$BUILD_DIR/benchmark_results.csv" ]]; then
    echo "▶ Ejecutando benchmark completo para generar datos..."
    "$SCRIPT_DIR/benchmark.sh"
else
    echo "▶ Usando benchmark_results.csv existente."
fi

# 2. Generar gráficos (sin UI)
echo "▶ Generando visualizaciones (plots)..."
cd "$BUILD_DIR"
python3 visualize.py --no-show

# 3. Generar LaTeX
echo "▶ Generando código fuente LaTeX (report.tex)..."
python3 "$SCRIPT_DIR/generate_tex.py" --csv benchmark_results.csv --out report.tex

# 4. Compilar PDF
echo "▶ Compilando PDF (requiere pdflatex)..."
if command -v pdflatex &> /dev/null; then
    # Primera pasada (archivos aux)
    pdflatex -interaction=nonstopmode report.tex > /dev/null || true
    # Segunda pasada (referencias)
    pdflatex -interaction=nonstopmode report.tex > /dev/null || true
    
    # Limpieza
    rm -f report.aux report.log report.out
    
    if [[ -f "report.pdf" ]]; then
        echo ""
        echo "✅ ¡Éxito! Reporte generado en: $BUILD_DIR/report.pdf"
    else
        echo "⚠ Error al compilar el PDF. Revisa el log de pdflatex."
    fi
else
    echo "⚠ Error: 'pdflatex' no encontrado. Por favor instala una distribución TeX (texlive, miktex, etc)."
    echo "   El archivo 'report.tex' ha sido generado exitosamente de todas formas en $BUILD_DIR/report.tex"
fi
