from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog


def dea_bcc_input_scores(
    profiles: dict[str, dict[str, Any]],
    models: list[str],
    input_fields: tuple[str, ...] = ("latency_s", "incremental_energy_j"),
    output_field: str = "benchmark_map",
) -> dict[str, float]:
    """DEA-BCC orientado a inputs com campos configuráveis."""
    x = np.asarray([[profiles[m][field] for field in input_fields] for m in models], float)
    y = np.asarray([profiles[m][output_field] for m in models], float)
    if np.any(x <= 0) or np.any(y < 0):
        raise ValueError("DEA requer inputs positivos e outputs não negativos.")
    n = len(models); scores: dict[str, float] = {}
    for target, model in enumerate(models):
        if y[target] == 0:
            scores[model] = 0.0
            continue
        # variáveis: theta, lambda_1 ... lambda_n
        objective = np.r_[1.0, np.zeros(n)]
        a_ub, b_ub = [], []
        for criterion in range(x.shape[1]):
            row = np.r_[-x[target, criterion], x[:, criterion]]
            a_ub.append(row); b_ub.append(0.0)
        a_ub.append(np.r_[0.0, -y]); b_ub.append(-y[target])
        result = linprog(objective, A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub),
                         A_eq=np.asarray([np.r_[0.0, np.ones(n)]]), b_eq=[1.0],
                         bounds=[(0.0, None)] * (n + 1), method="highs")
        if not result.success:
            raise RuntimeError(f"DEA falhou para {model}: {result.message}")
        scores[model] = min(1.0, max(0.0, float(result.x[0])))
    return scores


def select_dea_models(
    profiles: dict[str, dict[str, Any]],
    models: list[str],
    threshold: float = 0.999,
    input_fields: tuple[str, ...] = ("latency_s", "incremental_energy_j"),
    output_field: str = "benchmark_map",
) -> tuple[list[str], dict[str, float]]:
    scores = dea_bcc_input_scores(profiles, models, input_fields, output_field)
    return [model for model in models if scores[model] >= threshold], scores
