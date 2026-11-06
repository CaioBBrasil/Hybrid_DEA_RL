from __future__ import annotations

from typing import Any

import numpy as np

from rl.common.environment import UAVModelSelectionEnv
from rl.common.energy import embedded_energy_j

from .base_policy import SelectionPolicy, require_criteria


def _minmax(values: np.ndarray) -> np.ndarray:
    span = values.max() - values.min()
    return np.ones_like(values) if span == 0 else (values - values.min()) / span


class EnergyAwarePolicy(SelectionPolicy):
    """Heurística determinística inspirada em seleção adaptativa por energia."""
    name = "energy_aware"

    def __init__(self, profiles: dict[str, dict[str, Any]], models: list[str], parameters: dict[str, float]):
        require_criteria(profiles, models)
        self.models, self.parameters = list(models), parameters
        self.accuracy = _minmax(np.array([profiles[m]["benchmark_map"] for m in models], float))
        self.energy = _minmax(np.array([embedded_energy_j(profiles[m]) for m in models], float))
        self.latency = _minmax(np.array([profiles[m]["latency_s"] for m in models], float))

    def select(self, observation: np.ndarray, env: UAVModelSelectionEnv) -> int:
        del observation
        viable = [i for i, model in enumerate(self.models) if env.can_process_next(model)]
        if not viable:
            # The agent only selects detectors. Any detector action lets the
            # shared safety shield terminate the mission with a safe return.
            return (env.return_action if getattr(env, "agent_return_enabled", True)
                    else env.action_for_model(self.models[0]))
        return_energy = env.energy.return_energy(env.processed * env.energy.cell_distance_m)
        margin = max(0.0, env.battery_j - return_energy - env.energy.reserve_j)
        pressure = 1.0 - min(1.0, margin / env.energy.usable_energy_j)
        p = self.parameters
        energy_weight = float(p["base_energy_weight"]) + pressure * float(p["energy_pressure_weight"])
        latency_weight = float(p["base_latency_weight"]) + pressure * float(p["latency_pressure_weight"])
        scores = float(p["accuracy_weight"]) * self.accuracy - energy_weight * self.energy - latency_weight * self.latency
        best = max(viable, key=lambda i: (float(scores[i]), -i))
        return env.action_for_model(self.models[best])
