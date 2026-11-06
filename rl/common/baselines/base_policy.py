from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from rl.common.environment import UAVModelSelectionEnv
from rl.common.energy import embedded_energy_j


class SelectionPolicy(ABC):
    name: str

    @abstractmethod
    def select(self, observation: np.ndarray, env: UAVModelSelectionEnv) -> int:
        """Seleciona uma ação usando somente o estado atual e perfis públicos."""


def require_criteria(profiles: dict[str, dict[str, Any]], models: list[str]) -> None:
    required = {"benchmark_map", "latency_s"}
    for model in models:
        missing = required - set(profiles[model])
        if missing:
            raise ValueError(f"Critérios ausentes para {model}: {sorted(missing)}")
        embedded_energy_j(profiles[model])


class StaticModelPolicy(SelectionPolicy):
    model: str

    def select(self, observation: np.ndarray, env: UAVModelSelectionEnv) -> int:
        del observation
        if env.can_process_next(self.model) or not getattr(env, "agent_return_enabled", True):
            return env.action_for_model(self.model)
        return env.return_action
