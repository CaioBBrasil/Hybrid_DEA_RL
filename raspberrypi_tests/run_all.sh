#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python scripts/verify_bundle.py
python scripts/system_info.py --output results/system.json
python scripts/evaluate.py --all --threads 4
for repeat in 1 2 3; do
  python scripts/benchmark.py --all --threads 4 --warmup 20 --runs 100 --repeat "$repeat"
done
echo "Concluído. Devolva a pasta results/ completa."
