#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Omnex — Linux Installer
# Tested on Ubuntu 22.04+ and Debian 12+
# Usage: bash install.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

OMNEX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_MIN="3.11"

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

check_python_version() {
  local version
  version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  local major minor required_major required_minor
  major=$(echo "$version" | cut -d. -f1)
  minor=$(echo "$version" | cut -d. -f2)
  required_major=$(echo "$PYTHON_MIN" | cut -d. -f1)
  required_minor=$(echo "$PYTHON_MIN" | cut -d. -f2)

  if [ "$major" -lt "$required_major" ] || { [ "$major" -eq "$required_major" ] && [ "$minor" -lt "$required_minor" ]; }; then
    echo "  ✗ Python $PYTHON_MIN+ required, found $version"
    exit 1
  fi
  echo "  ✓ Python $version"
}

print_header

echo "Checking prerequisites..."
check_command python3 "Install Python 3.11+: https://python.org"
check_python_version
check_command docker "Install Docker: https://docs.docker.com/engine/install/"
check_command "docker" "Docker Compose required"
echo ""

echo "Starting Docker services..."
cd "$OMNEX_DIR"
docker compose up -d
echo "  ✓ MongoDB and Ollama services started"
echo ""

echo "Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
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
python models/download.py
echo ""

echo "============================================================"
echo "  Omnex installed successfully."
echo ""
echo "  Next steps:"
echo "  1. Edit .env — set OMNEX_SOURCE_PATH to the drive to index"
echo "  2. Activate env:  source .venv/bin/activate"
echo "  3. Start API:     uvicorn api.main:app --host 127.0.0.1 --port 8000"
echo "  4. Start UI:      cd interface && npm install && npm run dev"
echo "  5. Open browser:  http://localhost:3000"
echo "============================================================"
echo ""
