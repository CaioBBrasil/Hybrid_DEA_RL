#!/usr/bin/env python3
"""Rebuild the Plan A selection matrix using validation quality only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml
import numpy as np

from rl.common.dea import select_dea_models
from rl.common.model_selection import pareto_frontier
from rl.common.quality_metrics import calibrate_f1_threshold, evaluate_model_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset_merged"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("rl/plan_a/data/predictions.raspberry_pi4_real.json"))
    parser.add_argument("--hardware-matrix", type=Path,
                        default=Path("rl/plan_a/data/raspberry_pi4_six_models.yaml"))
    parser.add_argument("--output-yaml", type=Path,
                        default=Path("rl/plan_a/data/raspberry_pi4_selection_validation.yaml"))
    parser.add_argument("--output-csv", type=Path,
                        default=Path("rl/plan_a/data/raspberry_pi4_selection_validation.csv"))
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    args = parser.parse_args()

    source = yaml.safe_load(args.hardware_matrix.read_text(encoding="utf-8"))
    if not 0 < args.threshold_step <= 1:
        parser.error("--threshold-step must be in (0, 1].")
    threshold_candidates = np.arange(0.0, 1.0 + args.threshold_step / 2,
                                     args.threshold_step).clip(0.0, 1.0).tolist()
    prediction_data = json.loads(args.predictions.read_text(encoding="utf-8"))
    models = list(source["models"])
    profiles: dict[str, dict[str, float]] = {}
    rows = []
    for model in models:
        calibrated = calibrate_f1_threshold(
            args.predictions, prediction_data, args.dataset, "valid", model, 0.5,
            threshold_candidates=threshold_candidates,
        )
        # The test result is reported for traceability but never used in selection.
        test = evaluate_model_split(
            args.predictions, args.dataset, "test", model, calibrated.threshold, 0.5
        )
        hardware = source["models"][model]
        profiles[model] = {
            "benchmark_map": calibrated.f1,
            "validation_f1": calibrated.f1,
            "validation_ap50": calibrated.ap50,
            "latency_s": float(hardware["latency_s"]),
            "embedded_energy_j": float(hardware["gross_energy_j_per_inference"]),
            # Compatibility alias. New code should use embedded_energy_j.
            "incremental_energy_j": float(hardware["gross_energy_j_per_inference"]),
            "score_threshold": calibrated.threshold,
        }
        rows.append({
            "model": model,
            "selection_split": "valid",
            "score_threshold": calibrated.threshold,
            "validation_precision": calibrated.precision,
            "validation_recall": calibrated.recall,
            "validation_f1": calibrated.f1,
            "validation_ap50": calibrated.ap50,
            "validation_negative_fp": calibrated.negative_false_positives,
            "test_precision_at_frozen_threshold": test.precision,
            "test_recall_at_frozen_threshold": test.recall,
            "test_f1_at_frozen_threshold": test.f1,
            "test_ap50_at_frozen_threshold": test.ap50,
            "test_negative_fp_at_frozen_threshold": test.negative_false_positives,
            "latency_ms": profiles[model]["latency_s"] * 1000,
            "gross_embedded_energy_j": profiles[model]["embedded_energy_j"],
        })

    dea_selected, dea_scores = select_dea_models(
        profiles, models, args.threshold,
        input_fields=("latency_s", "embedded_energy_j"), output_field="validation_f1",
    )
    pareto_selected, dominators = pareto_frontier(
        profiles, models,
        minimize=("latency_s", "embedded_energy_j"), maximize=("validation_f1",),
    )
    for row in rows:
        model = str(row["model"])
        row.update({
            "dea_bcc_input_score": dea_scores[model],
            "dea_selected": model in dea_selected,
            "pareto_selected": model in pareto_selected,
            "pareto_dominators": ";".join(dominators[model]),
        })

    document = {
        "matrix": {
            "name": "raspberry_pi4_selection_validation",
            "selection_split": "valid",
            "final_evaluation_split": "test",
            "quality_output": "f1",
            "quality_iou_threshold": 0.5,
            "threshold_calibration": "maximize_validation_f1",
            "threshold_grid_step": args.threshold_step,
            "dea_orientation": "input",
            "dea_returns_to_scale": "variable",
            "dea_inputs": ["latency_s", "embedded_energy_j"],
            "dea_outputs": ["validation_f1"],
            "dea_efficiency_threshold": args.threshold,
            "dea_selected": dea_selected,
            "pareto_selected": pareto_selected,
            "test_independence_warning": (
                "The existing test split was consulted during prior development and is not an untouched holdout."
            ),
        },
        "models": profiles,
    }
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("DEA-BCC:", ", ".join(dea_selected))
    print("Pareto:", ", ".join(pareto_selected))
    print("YAML:", args.output_yaml)
    print("CSV:", args.output_csv)


if __name__ == "__main__":
    main()
