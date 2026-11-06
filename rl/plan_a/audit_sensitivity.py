#!/usr/bin/env python3
"""Generate physical and energy-model sensitivity tables for revised Plan A."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

from rl.common.dea import select_dea_models
from rl.common.config import load_config
from rl.common.model_selection import pareto_frontier


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-matrix", type=Path,
                        default=Path("rl/plan_a/data/raspberry_pi4_selection_validation.yaml"))
    parser.add_argument("--hardware-matrix", type=Path,
                        default=Path("rl/plan_a/data/raspberry_pi4_six_models.yaml"))
    parser.add_argument("--config", type=Path,
                        default=Path("rl/plan_a/configs/raspberry_pi4_validation.yaml"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("rl/plan_a/results/revised_validation/sensitivity"))
    args = parser.parse_args()
    selection = yaml.safe_load(args.selection_matrix.read_text(encoding="utf-8"))
    hardware = yaml.safe_load(args.hardware_matrix.read_text(encoding="utf-8"))
    config = load_config(args.config)
    models = list(selection["models"])
    physical = config["physical"]
    cells = int(config["dataset"]["cells_per_episode"])

    physical_rows = []
    for distance in (25.0, 50.0, 75.0, 100.0, 125.0):
        for battery_factor in (0.75, 1.0, 1.25):
            battery = float(physical["battery_capacity_j"]) * battery_factor
            one_way_move = float(physical["movement_power_w"]) * distance / float(physical["speed_m_s"])
            for model in models:
                profile = selection["models"][model]
                hover = float(physical["hover_power_w"]) * float(profile["latency_s"])
                embedded = float(profile["embedded_energy_j"])
                per_cell_round_trip = 2 * one_way_move + hover + embedded
                max_cells = min(cells, math.floor(battery / per_cell_round_trip))
                physical_rows.append({
                    "model": model, "cell_distance_m": distance,
                    "battery_factor": battery_factor, "battery_j": battery,
                    "movement_j_per_outbound_cell": one_way_move,
                    "hover_j_per_inference": hover,
                    "embedded_j_per_inference": embedded,
                    "max_safe_cells": max_cells,
                    "max_safe_coverage": max_cells / cells,
                })
    write_csv(args.output_dir / "physical_sensitivity.csv", physical_rows)

    energy_rows = []
    selection_sets = {}
    for energy_variant, source_field in (
        ("gross", "gross_energy_j_per_inference"),
        ("idle_subtracted_estimate", "incremental_energy_j_per_inference_estimate"),
    ):
        profiles = {
            model: {
                "validation_f1": float(selection["models"][model]["validation_f1"]),
                "latency_s": float(selection["models"][model]["latency_s"]),
                "energy_j": float(hardware["models"][model][source_field]),
            }
            for model in models
        }
        dea_models, scores = select_dea_models(
            profiles, models, 0.999,
            input_fields=("latency_s", "energy_j"), output_field="validation_f1",
        )
        pareto_models, dominators = pareto_frontier(
            profiles, models, minimize=("latency_s", "energy_j"),
            maximize=("validation_f1",),
        )
        selection_sets[energy_variant] = {"dea_bcc": dea_models, "pareto": pareto_models}
        for model in models:
            energy_rows.append({
                "energy_variant": energy_variant, "model": model,
                "validation_f1": profiles[model]["validation_f1"],
                "latency_s": profiles[model]["latency_s"],
                "energy_j": profiles[model]["energy_j"],
                "dea_bcc_score": scores[model], "dea_selected": model in dea_models,
                "pareto_selected": model in pareto_models,
                "pareto_dominators": ";".join(dominators[model]),
            })
    write_csv(args.output_dir / "energy_definition_sensitivity.csv", energy_rows)

    latency = np.asarray([selection["models"][model]["latency_s"] for model in models], float)
    gross = np.asarray([hardware["models"][model]["gross_energy_j_per_inference"] for model in models], float)
    incremental = np.asarray([
        hardware["models"][model]["incremental_energy_j_per_inference_estimate"] for model in models
    ], float)
    report = {
        "cells_per_episode": cells,
        "pearson_latency_gross_energy": float(np.corrcoef(latency, gross)[0, 1]),
        "pearson_latency_idle_subtracted_energy": float(np.corrcoef(latency, incremental)[0, 1]),
        "selection_sets": selection_sets,
        "interpretation": (
            "The idle-subtracted values are exploratory because the physical campaign did not record long idle windows or real voltage."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
