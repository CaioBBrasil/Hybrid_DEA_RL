from __future__ import annotations

from typing import Any

from .base_policy import StaticModelPolicy, require_criteria


class HighestMapPolicy(StaticModelPolicy):
    name = "highest_map"

    def __init__(self, profiles: dict[str, dict[str, Any]], models: list[str]):
        require_criteria(profiles, models)
        self.model = max(models, key=lambda model: (float(profiles[model]["benchmark_map"]), -models.index(model)))
