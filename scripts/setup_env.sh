#!/usr/bin/env bash
# One-shot environment setup for the YOLO benchmark project on Ubuntu 24.04.
# Creates a virtual environment, installs dependencies, and verifies which
# OpenVINO devices (CPU/GPU/NPU) this machine actually exposes.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "[setup] $python_bin not found. Install it with: sudo apt install python3 python3-venv" >&2
    exit 1
fi

version_output="$("$python_bin" --version 2>&1)"
echo "[setup] using $version_output"
minor="$("$python_bin" -c 'import sys; print(sys.version_info.minor)')"
if [ "$minor" -ge 13 ]; then
    echo "[setup] WARNING: Python 3.$minor detected. OpenVINO/NNCF wheels can lag the newest CPython release." >&2
    echo "[setup] If 'pip install' below fails, re-run with a 3.10-3.12 interpreter: PYTHON=python3.12 scripts/setup_env.sh" >&2
fi

if [ ! -d ".venv" ]; then
    echo "[setup] creating virtual environment (.venv)"
    if ! "$python_bin" -m venv .venv; then
        echo "[setup] venv creation failed. Install the venv module with: sudo apt install python3-venv" >&2
        exit 1
    fi
else
    echo "[setup] .venv already exists, reusing it"
fi

venv_python="$project_root/.venv/bin/python"

echo "[setup] upgrading pip"
"$venv_python" -m pip install --upgrade pip

echo "[setup] installing requirements.txt"
"$venv_python" -m pip install -r requirements.txt

echo "[setup] installing yolobench (editable) so 'python -m yolobench.benchmark' works"
"$venv_python" -m pip install -e .

if ! id -nG | grep -qw render; then
    echo "[setup] WARNING: user '$USER' is not in the 'render' group, so GPU/NPU may not be visible." >&2
    echo "[setup] Fix with: sudo usermod -aG render \$USER  (then log out and back in)" >&2
fi

echo "[setup] verifying OpenVINO device visibility"
"$venv_python" -c "from openvino import Core; c = Core(); print('Available OpenVINO devices:', c.available_devices)"

echo ""
echo "[setup] done. Activate the environment in new shells with:"
echo "  source .venv/bin/activate"
