#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="$(pwd)/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo 'ERRO: .venv/bin/python ausente.' >&2; exit 1; }
"$PYTHON" -c 'import psutil, serial, numpy, onnxruntime, tflite_runtime'
"$PYTHON" scripts/verify_bundle.py
PORTA="${PORTA:-/dev/ttyACM0}"
mkdir -p results/monitorado
MODELOS=(ssd_mobilenet_v2 efficientdet_d0 nanodet_plus yolov5nu yolov8n yolo11n)
for repeat in 7 8 9; do
  for modelo in "${MODELOS[@]}"; do
    case "$modelo" in
      ssd_mobilenet_v2|efficientdet_d0) formato=tflite;;
      nanodet_plus|yolov5nu|yolov8n|yolo11n) formato=onnx;;
      *) echo "Modelo desconhecido: $modelo" >&2; exit 1;;
    esac
    prefixo="results/monitorado/${modelo}_run${repeat}"
    json="results/${modelo}_${formato}_4threads_run${repeat}.json"
    for caminho in "${prefixo}.csv" "${prefixo}_energia.csv" "${prefixo}_meta.json" "$json"; do
      if [[ -e "$caminho" ]]; then
        echo "ERRO: já existe $caminho; não sobrescrevendo." >&2; exit 1
      fi
    done
    echo "=== Modelo=$modelo | repetição=$repeat | Arduino=$PORTA ==="
    "$PYTHON" monitor.py \
      --modelo "$modelo" --repeticao "$repeat" --intervalo 0.5 \
      --porta "$PORTA" --csv "${prefixo}.csv" \
      --energia-csv "${prefixo}_energia.csv" \
      --resumo "${prefixo}_meta.json" \
      --comando "$PYTHON" scripts/benchmark.py \
      --model "$modelo" --threads 4 --warmup 20 --runs 100 --repeat "$repeat"
    [[ -s "$json" ]] || { echo "ERRO: resultado ausente $json" >&2; exit 1; }
  done
done
echo 'Concluídas as 18 execuções monitoradas.'
