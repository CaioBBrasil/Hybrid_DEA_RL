from __future__ import annotations

from typing import Any

from .base_policy import StaticModelPolicy, require_criteria
from rl.common.energy import embedded_energy_j


class LowestEnergyPolicy(StaticModelPolicy):
    name = "lowest_energy"

    def __init__(self, profiles: dict[str, dict[str, Any]], models: list[str]):
        require_criteria(profiles, models)
        self.model = min(models, key=lambda model: (embedded_energy_j(profiles[model]), models.index(model)))
