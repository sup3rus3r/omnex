# ─────────────────────────────────────────────────────────────────────────────
# Omnex — Windows Installer
# Requires: Python 3.11+, Docker Desktop, WinFsp (for FUSE layer)
# Usage: Right-click → Run with PowerShell  OR  .\install.ps1
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$OmnexDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Print-Header {
  Write-Host ""
  Write-Host "============================================================"
  Write-Host "  Omnex — Install (Windows)"
  Write-Host "  Everything, indexed. Nothing lost."
  Write-Host "============================================================"
  Write-Host ""
}

function Check-Command {
  param($cmd, $hint)
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
    Write-Host "  x $cmd not found. $hint"
    exit 1
  }
  Write-Host "  + $cmd found"
}

function Check-PythonVersion {
  $version = python --version 2>&1
  if ($version -match "Python (\d+)\.(\d+)") {
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
      Write-Host "  x Python 3.11+ required, found $version"
      exit 1
    }
    Write-Host "  + $version"
  } else {
    Write-Host "  x Could not determine Python version"
    exit 1
  }
}

Print-Header

Write-Host "Checking prerequisites..."
Check-Command python "Install Python 3.11+: https://python.org/downloads"
Check-PythonVersion
Check-Command docker "Install Docker Desktop: https://docs.docker.com/desktop/windows/install/"
Write-Host ""

Write-Host "Starting Docker services..."
Set-Location $OmnexDir
docker compose up -d
Write-Host "  + MongoDB and Ollama services started"
Write-Host ""

Write-Host "Setting up Python environment..."
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
Write-Host "  + Python environment ready"
Write-Host ""

Write-Host "Setting up environment config..."
if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Host "  + .env created from .env.example"
  Write-Host "  ! Edit .env to set OMNEX_SOURCE_PATH and OMNEX_DATA_PATH"
} else {
  Write-Host "  + .env already exists — skipping"
}
Write-Host ""

Write-Host "Downloading ML models (~4 GB, this runs once)..."
python models\download.py
Write-Host ""

Write-Host "============================================================"
Write-Host "  Omnex installed successfully."
Write-Host ""
Write-Host "  Next steps:"
Write-Host "  1. Edit .env — set OMNEX_SOURCE_PATH to the drive to index"
Write-Host "  2. Activate env:  .\.venv\Scripts\Activate.ps1"
Write-Host "  3. Start API:     uvicorn api.main:app --host 127.0.0.1 --port 8000"
Write-Host "  4. Start UI:      cd interface; npm install; npm run dev"
Write-Host "  5. Open browser:  http://localhost:3000"
Write-Host ""
Write-Host "  For FUSE virtual filesystem support:"
Write-Host "  Install WinFsp: https://winfsp.dev/rel/"
Write-Host "============================================================"
Write-Host ""
