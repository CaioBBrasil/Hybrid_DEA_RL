from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .map_generator import load_cells
from .quality import JsonQualityProvider


@dataclass(frozen=True)
class DetectionMetrics:
    threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    ap50: float
    negative_images: int
    negative_false_positives: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_model_split(
    predictions: Path,
    dataset: Path,
    split: str,
    model: str,
    threshold: float,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    provider = JsonQualityProvider(predictions, {model: threshold}, iou_threshold)
    tp = fp = fn = negative_images = negative_false_positives = 0
    for cell in load_cells(dataset.resolve(), split):
        counts = provider.evaluate(model, cell)
        tp += counts.tp
        fp += counts.fp
        fn += counts.fn
        if cell.ground_truth_count == 0:
            negative_images += 1
            negative_false_positives += counts.fp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return DetectionMetrics(
        threshold=float(threshold), tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        ap50=float(provider.aggregate_map() or 0.0),
        negative_images=negative_images,
        negative_false_positives=negative_false_positives,
    )


def prediction_scores(predictions_data: dict[str, Any], model: str, image_ids: set[str]) -> list[float]:
    return sorted({
        float(prediction.get("score", 0.0))
        for image_id, predictions in predictions_data.get(model, {}).items()
        if image_id in image_ids
        for prediction in predictions
        if int(prediction.get("class_id", 0)) == 0
    })


def calibrate_f1_threshold(
    predictions: Path,
    predictions_data: dict[str, Any],
    dataset: Path,
    split: str,
    model: str,
    iou_threshold: float = 0.5,
    threshold_candidates: list[float] | None = None,
) -> DetectionMetrics:
    cells = load_cells(dataset.resolve(), split)
    if threshold_candidates is None:
        scores = prediction_scores(predictions_data, model, {cell.image_id for cell in cells})
        candidates = sorted({0.0, 1.0, *scores})
    else:
        candidates = sorted({float(value) for value in threshold_candidates})
        if not candidates or candidates[0] < 0.0 or candidates[-1] > 1.0:
            raise ValueError("Threshold candidates must be within [0, 1].")
    evaluations = [
        evaluate_model_split(predictions, dataset, split, model, threshold, iou_threshold)
        for threshold in candidates
    ]
    # Deterministic tie break: F1, then precision, then the higher threshold.
    return max(evaluations, key=lambda item: (item.f1, item.precision, item.threshold))
