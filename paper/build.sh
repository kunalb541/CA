#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Generating paper artifacts ==="
python ../scripts/make_response_law_artifacts.py

echo "=== Compiling lean journal version (paper.tex) ==="
pdflatex -interaction=nonstopmode paper.tex
bibtex paper || true
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex

echo "=== Compiling full preprint (paper_full_preprint.tex) ==="
pdflatex -interaction=nonstopmode paper_full_preprint.tex
bibtex paper_full_preprint || true
pdflatex -interaction=nonstopmode paper_full_preprint.tex
pdflatex -interaction=nonstopmode paper_full_preprint.tex

echo "=== Done: paper/paper.pdf  paper/paper_full_preprint.pdf ==="
