#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/paper"
bash build.sh
cp paper.pdf "$SCRIPT_DIR/paper.pdf"
echo "Root paper.pdf updated."
