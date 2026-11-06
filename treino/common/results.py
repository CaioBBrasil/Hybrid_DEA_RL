from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


Box = Tuple[float, float, float, float]


def _iou(first: Box, second: Box) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def count_detections(
    ground_truth: Dict[str, List[Box]],
    predictions: Dict[str, List[Tuple[Box, float]]],
    score_threshold: float,
    iou_threshold: float,
) -> Dict[str, float]:
    true_positive = false_positive = false_negative = 0
    for image_id in sorted(set(ground_truth) | set(predictions)):
        expected = ground_truth.get(image_id, [])
        matched = set()
        detections = sorted(
            (item for item in predictions.get(image_id, []) if item[1] >= score_threshold),
            key=lambda item: item[1], reverse=True,
        )
        for box, _score in detections:
            candidates = [
                (_iou(box, target), index)
                for index, target in enumerate(expected) if index not in matched
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= iou_threshold:
                true_positive += 1
                matched.add(best_index)
            else:
                false_positive += 1
        false_negative += len(expected) - len(matched)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def read_yolo_ground_truth(labels: Path) -> Dict[str, List[Box]]:
    result: Dict[str, List[Box]] = {}
    for label in sorted(labels.glob("*.txt")):
        boxes: List[Box] = []
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, xc, yc, width, height = map(float, line.split())
            if int(class_id) != 0:
                raise ValueError(f"Classe inesperada em {label}")
            boxes.append((xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2))
        result[label.stem] = boxes
    return result


def extract_tensorboard_metrics(events_dir: Path) -> Dict[str, float]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util import tensor_util

    metrics: Dict[str, float] = {}
    for event_file in events_dir.rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(event_file.parent))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values = accumulator.Scalars(tag)
            if values:
                metrics[tag] = float(values[-1].value)
        for tag in accumulator.Tags().get("tensors", []):
            values = accumulator.Tensors(tag)
            if values:
                array = tensor_util.make_ndarray(values[-1].tensor_proto)
                if array.size == 1:
                    metrics[tag] = float(array.reshape(-1)[0])
    return metrics


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_tensorflow_saved_model(
    saved_model: Path,
    images: Path,
    labels: Path,
    score_threshold: float,
    iou_threshold: float,
) -> Dict[str, float]:
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    infer = tf.saved_model.load(str(saved_model)).signatures["serving_default"]
    predictions: Dict[str, List[Tuple[Box, float]]] = {}
    for image_path in sorted(path for path in images.iterdir() if path.is_file()):
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        outputs = infer(tf.convert_to_tensor(array[None, ...]))
        count = int(outputs["num_detections"][0])
        boxes = outputs["detection_boxes"][0][:count].numpy()
        scores = outputs["detection_scores"][0][:count].numpy()
        detections: List[Tuple[Box, float]] = []
        for box, score in zip(boxes, scores):
            ymin, xmin, ymax, xmax = map(float, box)
            detections.append(((xmin, ymin, xmax, ymax), float(score)))
        predictions[image_path.stem] = detections
    ground_truth = read_yolo_ground_truth(labels)
    return count_detections(ground_truth, predictions, score_threshold, iou_threshold)


def evaluate_nanodet_results(
    results_json: Path,
    yolo_directory: Path,
    score_threshold: float,
    iou_threshold: float,
) -> Dict[str, float]:
    from PIL import Image

    # NanoDet atribui image_id na ordem devolvida por os.walk, não em ordem
    # lexicográfica. Preserve a ordem do diretório para reconstruir o mesmo mapa.
    labels = list(yolo_directory.glob("*.txt"))
    id_to_stem = {index + 1: label.stem for index, label in enumerate(labels)}
    sizes: Dict[str, Tuple[int, int]] = {}
    predictions: Dict[str, List[Tuple[Box, float]]] = {}
    for item in json.loads(results_json.read_text(encoding="utf-8")):
        stem = id_to_stem[int(item["image_id"])]
        if stem not in sizes:
            matches = [p for p in yolo_directory.glob(f"{stem}.*") if p.suffix.lower() != ".txt"]
            if not matches:
                raise FileNotFoundError(f"Imagem do resultado NanoDet ausente: {stem}")
            with Image.open(matches[0]) as image:
                sizes[stem] = image.size
        width, height = sizes[stem]
        x, y, box_width, box_height = map(float, item["bbox"])
        box = (x / width, y / height, (x + box_width) / width, (y + box_height) / height)
        predictions.setdefault(stem, []).append((box, float(item["score"])))
    ground_truth = read_yolo_ground_truth(yolo_directory)
    return count_detections(ground_truth, predictions, score_threshold, iou_threshold)


def parse_nanodet_metrics(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics
