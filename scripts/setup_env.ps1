# One-shot environment setup for the YOLO benchmark project on Windows.
# Creates a virtual environment, installs dependencies, and verifies which
# OpenVINO devices (CPU/GPU/NPU) this machine actually exposes.

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python not found on PATH. Install Python 3.10-3.12 from https://www.python.org/downloads/ and re-run this script."
}

$versionOutput = & python --version
Write-Host "[setup] using $versionOutput"
if ($versionOutput -match "Python 3\.(\d+)") {
    $minor = [int]$Matches[1]
    if ($minor -ge 13) {
        Write-Warning "Python 3.$minor detected. OpenVINO/NNCF wheels can lag the newest CPython release. If 'pip install' below fails, install Python 3.10-3.12 instead and re-run this script with that interpreter on PATH."
    }
}

if (-not (Test-Path ".venv")) {
    Write-Host "[setup] creating virtual environment (.venv)"
    python -m venv .venv
} else {
    Write-Host "[setup] .venv already exists, reusing it"
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

Write-Host "[setup] upgrading pip"
& $venvPython -m pip install --upgrade pip

Write-Host "[setup] installing requirements.txt"
& $venvPython -m pip install -r requirements.txt

Write-Host "[setup] verifying OpenVINO device visibility"
& $venvPython -c "from openvino import Core; c = Core(); print('Available OpenVINO devices:', c.available_devices)"

Write-Host ""
Write-Host "[setup] done. Activate the environment in new shells with:"
Write-Host "  .venv\Scripts\Activate.ps1"
