from __future__ import annotations

from typing import Any

import numpy as np

from .base_policy import StaticModelPolicy, require_criteria
from rl.common.energy import embedded_energy_j


def topsis_rank(profiles: dict[str, dict[str, Any]], models: list[str], weights: list[float]) -> list[tuple[str, float]]:
    require_criteria(profiles, models)
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.shape != (3,) or np.any(weights_array < 0) or weights_array.sum() <= 0:
        raise ValueError("TOPSIS requer três pesos não negativos com soma positiva.")
    weights_array /= weights_array.sum()
    matrix = np.asarray([[profiles[m]["benchmark_map"], profiles[m]["latency_s"],
                          embedded_energy_j(profiles[m])] for m in models], dtype=float)
    denominators = np.linalg.norm(matrix, axis=0)
    if np.any(denominators == 0):
        raise ValueError("TOPSIS não aceita uma coluna de critérios inteiramente nula.")
    weighted = matrix / denominators * weights_array
    ideal_positive = np.array([weighted[:, 0].max(), weighted[:, 1].min(), weighted[:, 2].min()])
    ideal_negative = np.array([weighted[:, 0].min(), weighted[:, 1].max(), weighted[:, 2].max()])
    d_positive = np.linalg.norm(weighted - ideal_positive, axis=1)
    d_negative = np.linalg.norm(weighted - ideal_negative, axis=1)
    scores = d_negative / np.maximum(d_positive + d_negative, np.finfo(float).eps)
    return sorted(zip(models, scores.tolist()), key=lambda item: (-item[1], models.index(item[0])))


class TopsisStaticPolicy(StaticModelPolicy):
    name = "topsis_static"

    def __init__(self, profiles: dict[str, dict[str, Any]], models: list[str], weights: list[float]):
        self.ranking = topsis_rank(profiles, models, weights)
        self.model = self.ranking[0][0]
