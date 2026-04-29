#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/paper"
bash build.sh
cp paper.pdf "$SCRIPT_DIR/paper.pdf"
cp paper_full_preprint.pdf "$SCRIPT_DIR/paper_full_preprint.pdf"
echo "Root paper.pdf updated."
echo "Root paper_full_preprint.pdf updated."
