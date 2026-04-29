#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Generating paper artifacts ==="
python ../scripts/make_response_law_artifacts.py

echo "=== Compiling paper ==="
pdflatex -interaction=nonstopmode paper.tex
bibtex paper || true
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex

echo "=== Done: paper/paper.pdf ==="
