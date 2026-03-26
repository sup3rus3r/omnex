#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Omnex — Linux Installer
# Tested on Ubuntu 22.04+ and Debian 12+
# Usage: bash install.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

OMNEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_header() {
  echo ""
  echo "============================================================"
  echo "  Omnex — Install"
  echo "  Everything, indexed. Nothing lost."
  echo "============================================================"
  echo ""
}

check_command() {
  if ! command -v "$1" &>/dev/null; then
    echo "  ✗ $1 not found. $2"
    exit 1
  fi
  echo "  ✓ $1 found"
}

install_uv() {
  if command -v uv &>/dev/null; then
    echo "  ✓ uv found"
    return
  fi
  echo "  Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
  echo "  ✓ uv installed"
}

print_header

echo "Checking prerequisites..."
check_command python3 "Install Python 3.11+: https://python.org"
check_command docker "Install Docker: https://docs.docker.com/engine/install/"
install_uv
echo ""

echo "Starting Docker services..."
cd "$OMNEX_DIR"
docker compose up -d
echo "  ✓ Ollama service started"
echo ""

echo "Setting up Python environment with uv..."
uv venv .venv
uv pip install -r requirements.txt
echo "  ✓ Python environment ready"
echo ""

echo "Setting up environment config..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  ✓ .env created from .env.example"
  echo "  ⚠  Edit .env to set OMNEX_SOURCE_PATH and OMNEX_DATA_PATH"
else
  echo "  ✓ .env already exists — skipping"
fi
echo ""

echo "Downloading ML models (~4 GB, this runs once)..."
uv run python models/download.py
echo ""

echo "============================================================"
echo "  Omnex installed successfully."
echo ""
echo "  Next steps:"
echo "  1. Edit .env — set OMNEX_SOURCE_PATH to the drive to index"
echo "  2. Start API:     uv run uvicorn api.main:app --host 127.0.0.1 --port 8000"
echo "  3. Start UI:      cd interface && npm install && npm run dev"
echo "  4. Open browser:  http://localhost:3000"
echo "============================================================"
echo ""
