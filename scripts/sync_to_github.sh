#!/usr/bin/env bash
set -euo pipefail

# Run this script from the repository root after copying the organized files.
# Example:
#   cd /mnt/e/Research/sxSNF/sxSNF
#   bash scripts/sync_to_github.sh "Reorganize sxSNF into package and update PyDoc"

COMMIT_MSG="${1:-Reorganize sxSNF package and update documentation}"

echo "[1/5] Current repository:"
pwd
git remote -v

echo "[2/5] Pull latest main..."
git pull origin main

echo "[3/5] Generate PyDoc..."
python scripts/generate_pydoc.py

echo "[4/5] Commit changes..."
git status
git add .
git commit -m "$COMMIT_MSG"

echo "[5/5] Push to GitHub..."
git push origin main

echo "[done] GitHub updated."
