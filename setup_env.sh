#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PARENT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_PARENT}/.venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

if command -v module >/dev/null 2>&1; then
    module load Python/3.11.5
fi

if [ ! -d "${VENV_DIR}" ]; then
    python -m venv "${VENV_DIR}"
    echo "Created venv: ${VENV_DIR}"
else
    echo "Using existing venv: ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${REQUIREMENTS_FILE}"

echo "Environment is up to date: ${VENV_DIR}"
