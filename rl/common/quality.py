from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .map_generator import Cell


@dataclass(frozen=True)
class DetectionCounts:
    tp: int
    fp: int
    fn: int


class QualityProvider(ABC):
    synthetic = False

    @abstractmethod
    def evaluate(self, model: str, cell: Cell) -> DetectionCounts:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def aggregate_map(self) -> float | None:
        return None


class SyntheticQualityProvider(QualityProvider):
    """Protótipo determinístico; não representa resultados experimentais."""
    synthetic = True

    def __init__(self, profiles: dict[str, dict[str, Any]], seed: int):
        self.profiles, self.seed = profiles, seed

    def evaluate(self, model: str, cell: Cell) -> DetectionCounts:
        profile = self.profiles[model]
        token = f"{self.seed}:{model}:{cell.image_id}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "big"))
        gt = cell.ground_truth_count
        recall = float(profile["synthetic_recall"])
        precision = float(profile["synthetic_precision"])
        tp = int(rng.binomial(gt, recall)) if gt else 0
        expected_fp = tp * (1 - precision) / max(precision, 1e-9)
        if gt == 0:
            expected_fp = float(profile.get("synthetic_negative_fp_rate", 0.1))
        fp = int(rng.poisson(expected_fp))
        return DetectionCounts(tp=tp, fp=fp, fn=gt - tp)


def _iou(a: list[float], b: list[float]) -> float:
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right-left) * max(0.0, bottom-top)
    union = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1]) - intersection
    return intersection / union if union else 0.0


def _ground_truth(label_path: str) -> list[list[float]]:
    boxes = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, xc, yc, width, height = map(float, line.split())
        if int(cls) == 0:
            boxes.append([xc-width/2, yc-height/2, xc+width/2, yc+height/2])
    return boxes


class JsonQualityProvider(QualityProvider):
    """Lê `{modelo: {image_id: [{bbox, score, class_id}]}}`."""

    def __init__(self, path: Path, confidence: float | dict[str, float], iou: float):
        self.data = _load_predictions(path.resolve())
        self.confidence, self.iou = confidence, iou
        self.reset()

    def reset(self) -> None:
        self.scored_matches: list[tuple[float, int]] = []
        self.total_ground_truth = 0

    def evaluate(self, model: str, cell: Cell) -> DetectionCounts:
        gt, matched = _ground_truth(cell.label_path), set()
        self.total_ground_truth += len(gt)
        confidence = (float(self.confidence[model]) if isinstance(self.confidence, dict)
                      else float(self.confidence))
        predictions = sorted(self.data.get(model, {}).get(cell.image_id, []),
                             key=lambda x: float(x.get("score", 0)), reverse=True)
        tp = fp = 0
        for pred in predictions:
            if float(pred.get("score", 0)) < confidence or int(pred.get("class_id", 0)) != 0:
                continue
            candidates = [(_iou(list(map(float, pred["bbox"])), box), i) for i, box in enumerate(gt) if i not in matched]
            score, index = max(candidates, default=(0.0, -1))
            if score >= self.iou:
                tp += 1; matched.add(index); self.scored_matches.append((float(pred["score"]), 1))
            else:
                fp += 1; self.scored_matches.append((float(pred["score"]), 0))
        return DetectionCounts(tp, fp, len(gt)-len(matched))

    def aggregate_map(self) -> float | None:
        """AP de uma classe no IoU configurado (portanto mAP=AP neste dataset)."""
        if self.total_ground_truth == 0:
            return 0.0
        ranked = sorted(self.scored_matches, reverse=True)
        if not ranked:
            return 0.0
        matches = np.asarray([match for _, match in ranked], dtype=float)
        tp = np.cumsum(matches); fp = np.cumsum(1.0 - matches)
        recall = tp / self.total_ground_truth
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = np.r_[0.0, recall, 1.0]; precision = np.r_[1.0, precision, 0.0]
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        changes = np.where(recall[1:] != recall[:-1])[0]
        return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


@lru_cache(maxsize=None)
def _load_predictions(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Compartilha predições imutáveis entre ambientes criados no mesmo processo."""
    return json.loads(path.read_text(encoding="utf-8"))
