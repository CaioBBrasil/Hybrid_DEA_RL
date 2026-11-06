#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rpi.txt
python scripts/verify_bundle.py
echo "Ambiente pronto. Execute: source .venv/bin/activate && ./run_all.sh"
