from __future__ import annotations

from typing import Any


def pareto_frontier(
    profiles: dict[str, dict[str, Any]],
    models: list[str],
    minimize: tuple[str, ...],
    maximize: tuple[str, ...],
) -> tuple[list[str], dict[str, list[str]]]:
    """Return non-dominated models and the models that dominate each candidate.

    Dominance is weak in every criterion and strict in at least one criterion.
    """
    dominators: dict[str, list[str]] = {model: [] for model in models}
    for target in models:
        for candidate in models:
            if candidate == target:
                continue
            weakly_better = all(
                float(profiles[candidate][field]) <= float(profiles[target][field])
                for field in minimize
            ) and all(
                float(profiles[candidate][field]) >= float(profiles[target][field])
                for field in maximize
            )
            strictly_better = any(
                float(profiles[candidate][field]) < float(profiles[target][field])
                for field in minimize
            ) or any(
                float(profiles[candidate][field]) > float(profiles[target][field])
                for field in maximize
            )
            if weakly_better and strictly_better:
                dominators[target].append(candidate)
    return [model for model in models if not dominators[model]], dominators
