#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXCLUDED = {"episode", "seed", "image_order", "model_counts", "split", "strategy",
            "dea_models", "dea_scores", "map_aggregated"}


def numeric_value(value: str) -> float:
    if value == "True": return 1.0
    if value == "False": return 0.0
    return float(value)


def summarize(input_path: Path) -> list[dict[str, object]]:
    with input_path.open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    strategies = sorted({row["strategy"] for row in source}); result = []
    for strategy in strategies:
        group = [row for row in source if row["strategy"] == strategy]
        numeric = sorted(set(group[0]) - EXCLUDED)
        output: dict[str, object] = {"strategy": strategy, "episodes": len(group)}
        for key in numeric:
            try: values = np.asarray([numeric_value(row[key]) for row in group], float)
            except (ValueError, TypeError): continue
            output[f"{key}_mean"] = float(values.mean())
            output[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        model_names = sorted({name for row in group for name in json.loads(row.get("model_counts", "{}"))})
        for model in model_names:
            values = np.asarray([json.loads(row.get("model_counts", "{}")).get(model, 0) for row in group], float)
            output[f"uses_{model}_mean"] = float(values.mean())
            output[f"uses_{model}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result.append(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument(
        "--input", default="rl/plan_a/results/revised_validation/all_methods.csv")
    parser.add_argument(
        "--output", default="rl/plan_a/results/revised_validation/all_methods_summary.csv")
    args = parser.parse_args(); rows = summarize(Path(args.input)); columns = sorted({k for r in rows for k in r})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    print(f"Resumo salvo em {output}")


if __name__ == "__main__": main()
