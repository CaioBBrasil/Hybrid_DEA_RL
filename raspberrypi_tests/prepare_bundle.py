"""Monta o diretório autocontido que será enviado ao operador do Raspberry Pi."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "raspberrypi_tests"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--test-images", type=int, default=0,
                    help="Quantidade de imagens; 0 inclui todo o split de teste.")
parser.add_argument("--output", type=Path, default=HERE / "bundle")
args = parser.parse_args()
artifacts = HERE / "artifacts"
models = [
    {"name": "ssd_mobilenet_v2", "format": "tflite", "score_threshold": 0.325,
     "threshold_status": "calibrated_on_validation", "source": artifacts / "ssd_mobilenet_v2/ssd_mobilenet_v2_fp32.tflite"},
    {"name": "efficientdet_d0", "format": "tflite", "score_threshold": 0.06,
     "threshold_status": "calibrated_on_validation", "source": artifacts / "efficientdet_d0/efficientdet_d0_fp32.tflite"},
    {"name": "nanodet_plus", "format": "onnx", "score_threshold": 0.40,
     "threshold_status": "calibrated_on_validation", "source": artifacts / "nanodet_plus/nanodet_plus.onnx"},
    {"name": "yolov5nu", "format": "onnx", "score_threshold": 0.25,
     "threshold_status": "provisional_not_calibrated",
     "source": HERE / "contributions/raspberry_pi4_yolo/YOLOv5nu.onnx"},
    {"name": "yolov8n", "format": "onnx", "score_threshold": 0.25,
     "threshold_status": "provisional_not_calibrated",
     "source": HERE / "contributions/raspberry_pi4_yolo/YOLOv8n.onnx"},
    {"name": "yolo11n", "format": "onnx", "score_threshold": 0.25,
     "threshold_status": "provisional_not_calibrated",
     "source": HERE / "contributions/raspberry_pi4_yolo/YOLO11n.onnx"},
]
missing = [str(m["source"]) for m in models if not m["source"].is_file()]
if missing:
    raise SystemExit("Artefatos ausentes; execute export_artifacts.py após os treinos:\n" + "\n".join(missing))
if args.output.exists():
    raise SystemExit(f"Destino já existe: {args.output}")
for directory in ("models", "test_images", "test_labels", "scripts", "results"):
    (args.output / directory).mkdir(parents=True, exist_ok=True)
for filename in ("README.md", "BENCHMARK_PROTOCOL.md", "REFERENCE_RESULTS.json", "requirements-rpi.txt", "energy_measurements.csv", "setup.sh", "run_all.sh"):
    shutil.copy2(HERE / filename, args.output / filename)
for script in (HERE / "scripts").glob("*.py"):
    shutil.copy2(script, args.output / "scripts" / script.name)
manifest_models = []
for model in models:
    target = args.output / "models" / model["source"].name
    shutil.copy2(model["source"], target)
    manifest_models.append({"name": model["name"], "format": model["format"],
                            "path": target.relative_to(args.output).as_posix(), "sha256": sha256(target),
                            "input_size": [640, 640], "class_names": ["acai"],
                            "score_threshold": model["score_threshold"],
                            "threshold_status": model["threshold_status"]})
images = sorted((ROOT / "dataset_merged/test/images").iterdir())
if args.test_images > 0:
    images = images[:args.test_images]
rows = []
for image in images:
    if not image.is_file():
        continue
    label = ROOT / "dataset_merged/test/labels" / f"{image.stem}.txt"
    image_target = args.output / "test_images" / image.name
    label_target = args.output / "test_labels" / label.name
    shutil.copy2(image, image_target)
    shutil.copy2(label, label_target)
    rows.append({"image": image.name, "image_sha256": sha256(image_target), "label": label.name,
                 "label_sha256": sha256(label_target), "annotations": len([x for x in label.read_text().splitlines() if x.strip()])})
with (args.output / "test_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
(args.output / "manifest.json").write_text(json.dumps({"models": manifest_models, "test_images": len(rows),
    "score_thresholds": "per-model", "iou_threshold": 0.50}, indent=2) + "\n", encoding="utf-8")
print("Bundle criado em", args.output)
