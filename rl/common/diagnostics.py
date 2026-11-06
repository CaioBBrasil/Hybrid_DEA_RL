from __future__ import annotations

from typing import Any

from .config import resolve_path
from .energy import EnergyModel, embedded_energy_j
from .map_generator import generate_map, generate_synthetic_map


def feasibility_report(config: dict[str, Any], split: str, models: list[str]) -> dict[str, Any]:
    energy = EnergyModel.from_config(config["physical"])
    dataset = config["dataset"]
    if dataset.get("type", "images") == "synthetic_grid":
        cells = generate_synthetic_map(int(dataset["cells_per_episode"]), split, 0)
    else:
        cells = generate_map(
            resolve_path(dataset["root"], config), split, 0,
            dataset.get("cells_per_episode"),
        )
    n = len(cells)
    movement_j = energy.movement(2 * n * energy.cell_distance_m)[1]
    per_model = {}
    for name in models:
        p = config["models"][name]
        analysis = sum(energy.analysis(float(p["latency_s"]), embedded_energy_j(p)))
        total = movement_j + n * analysis + energy.reserve_j
        round_trip_cell_j = 2 * energy.movement(energy.cell_distance_m)[1] + analysis
        max_safe_cells = min(n, int(energy.usable_energy_j // round_trip_cell_j))
        per_model[name] = {
            "full_route_energy_j": total,
            "full_route_feasible": total <= energy.battery_capacity_j,
            "max_safe_cells_estimate": max_safe_cells,
            "max_safe_coverage_estimate": max_safe_cells / n,
            "first_cell_feasible": max_safe_cells >= 1,
        }
    return {
        "split": split, "cells": n, "battery_capacity_j": energy.battery_capacity_j,
        "round_trip_movement_energy_j": movement_j, "models": per_model,
        "at_least_one_full_route_feasible": any(x["full_route_feasible"] for x in per_model.values()),
        "at_least_one_action_feasible": any(x["first_cell_feasible"] for x in per_model.values()),
    }
