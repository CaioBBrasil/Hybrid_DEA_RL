#!/usr/bin/env python3
"""Gera predições por imagem dos seis artefatos reais usados no Raspberry Pi 4."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from rl.common.config import ROOT
from rl.common.map_generator import IMAGE_SUFFIXES


MODEL_SPECS = {
    "ssd_mobilenet_v2": ("tflite", "raspberrypi_tests/artifacts/ssd_mobilenet_v2/ssd_mobilenet_v2_fp32.tflite"),
    "efficientdet_d0": ("tflite", "raspberrypi_tests/artifacts/efficientdet_d0/efficientdet_d0_fp32.tflite"),
    "nanodet_plus": ("onnx", "raspberrypi_tests/artifacts/nanodet_plus/nanodet_plus.onnx"),
    "yolov5nu": ("onnx", "raspberrypi_tests/contributions/raspberry_pi4_yolo/YOLOv5nu.onnx"),
    "yolov8n": ("onnx", "raspberrypi_tests/contributions/raspberry_pi4_yolo/YOLOv8n.onnx"),
    "yolo11n": ("onnx", "raspberrypi_tests/contributions/raspberry_pi4_yolo/YOLO11n.onnx"),
}


def inference_module():
    path = ROOT / "raspberrypi_tests/scripts/inference.py"
    spec = importlib.util.spec_from_file_location("raspberrypi_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não foi possível importar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset_merged"))
    parser.add_argument("--output", type=Path,
                        default=Path("rl/plan_a/data/predictions.raspberry_pi4_real.json"))
    parser.add_argument("--minimum-score", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model", action="append", choices=sorted(MODEL_SPECS))
    args = parser.parse_args()
    if not 0 <= args.minimum_score <= 1 or args.threads <= 0:
        parser.error("minimum-score e threads devem ser válidos")
    dataset = (ROOT / args.dataset).resolve()
    output = (ROOT / args.output).resolve()
    module = inference_module()
    data = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    metadata = data.setdefault("_metadata", {})
    metadata.update({"schema_version": 1, "minimum_score": args.minimum_score,
                     "bbox_format": "normalized_xyxy", "quality_source": "real_deployment_artifacts"})
    completed = metadata.setdefault("completed", [])
    selected = args.model or list(MODEL_SPECS)
    for model in selected:
        format_name, relative_path = MODEL_SPECS[model]
        artifact = (ROOT / relative_path).resolve()
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        model_spec = {"name": model, "format": format_name, "path": str(artifact)}
        runtime = module.detector(model_spec, Path("/"), args.threads)
        model_data = data.setdefault(model, {})
        for split in ("train", "valid", "test"):
            marker = f"{model}:{split}"
            if marker in completed:
                print("Ignorando", marker, flush=True)
                continue
            images = sorted(path for path in (dataset / split / "images").iterdir()
                            if path.suffix.lower() in IMAGE_SUFFIXES)
            for index, image in enumerate(images, 1):
                prepared = (runtime.prepare_path(image) if hasattr(runtime, "prepare_path")
                            else runtime.prepare(module.load_rgb(image)))
                boxes, scores = runtime.detect(prepared, args.minimum_score)
                model_data[image.stem] = [
                    {"bbox": [float(value) for value in box], "score": float(score), "class_id": 0}
                    for box, score in zip(boxes, scores)
                ]
                if index % 100 == 0 or index == len(images):
                    print(f"{marker}: {index}/{len(images)}", flush=True)
            completed.append(marker)
            save(output, data)
    print("Predições salvas em", output)


if __name__ == "__main__":
    main()
