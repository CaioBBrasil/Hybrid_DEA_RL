from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from inference import detector, load_rgb


ROOT = Path(__file__).resolve().parents[1]


def iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = np.maximum(first[0], second[:, 0])
    top = np.maximum(first[1], second[:, 1])
    right = np.minimum(first[2], second[:, 2])
    bottom = np.minimum(first[3], second[:, 3])
    intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = np.maximum(0, second[:, 2] - second[:, 0]) * np.maximum(0, second[:, 3] - second[:, 1])
    return intersection / np.maximum(first_area + second_area - intersection, 1e-12)


def labels(path: Path) -> np.ndarray:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _class_id, xc, yc, width, height = map(float, line.split())
        boxes.append((xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def evaluate(spec: dict, threads: int, iou_threshold: float, override_threshold=None) -> dict:
    runtime = detector(spec, ROOT, threads)
    threshold = float(override_threshold if override_threshold is not None else spec["score_threshold"])
    tp = fp = fn = 0
    images = sorted(path for path in (ROOT / "test_images").iterdir() if path.is_file())
    for index, image in enumerate(images, 1):
        prepared = runtime.prepare_path(image) if hasattr(runtime, "prepare_path") else runtime.prepare(load_rgb(image))
        boxes, scores = runtime.detect(prepared, threshold)
        expected = labels(ROOT / "test_labels" / f"{image.stem}.txt")
        matched: set[int] = set()
        for box in boxes[np.argsort(scores)[::-1]]:
            available = [item for item in range(len(expected)) if item not in matched]
            if not available:
                fp += 1
                continue
            overlaps = iou(box, expected[available])
            best_position = int(np.argmax(overlaps))
            if overlaps[best_position] >= iou_threshold:
                tp += 1
                matched.add(available[best_position])
            else:
                fp += 1
        fn += len(expected) - len(matched)
        if index % 25 == 0 or index == len(images):
            print(f"{spec['name']}: {index}/{len(images)}", flush=True)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "model": spec["name"], "format": spec["format"],
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "images": len(images), "score_threshold": threshold, "iou_threshold": iou_threshold,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


parser = argparse.ArgumentParser(description="Valida qualidade dos artefatos convertidos.")
parser.add_argument("--all", action="store_true")
parser.add_argument("--model", action="append")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--iou-threshold", type=float, default=0.5)
parser.add_argument("--score-threshold", type=float, default=None,
                    help="Limiar PROVISÓRIO para teste; não substitui calibração em validação")
args = parser.parse_args()
if args.score_threshold is not None and not (0 <= args.score_threshold <= 1):
    parser.error('--score-threshold deve estar entre 0 e 1')
manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
selected = manifest["models"] if args.all else [m for m in manifest["models"] if m["name"] in (args.model or [])]
if not selected:
    raise SystemExit("Selecione --all ou ao menos um --model.")
(ROOT / "results").mkdir(exist_ok=True)
for specification in selected:
    result = evaluate(specification, args.threads, args.iou_threshold, args.score_threshold)
    if args.score_threshold is not None:
        result["threshold_status"] = "provisorio_nao_calibrado_em_validacao"
        target = ROOT / "results" / f"{specification['name']}_quality_provisorio_thr{args.score_threshold:.3f}.json"
    else:
        target = ROOT / "results" / f"{specification['name']}_quality.json"
    if target.exists():
        raise SystemExit(f'Arquivo já existe; não sobrescrevendo: {target}')
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
