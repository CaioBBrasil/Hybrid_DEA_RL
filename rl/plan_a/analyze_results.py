#!/usr/bin/env python3
"""Consolidate final Plan A results without episode-level pseudoreplication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


METRICS = [
    "coverage", "precision", "recall", "f1", "negative_fp",
    "move_energy_j", "hover_energy_j", "embedded_energy_j",
    "processing_energy_j", "energy_used_j", "total_reward", "safe_return",
]


def paired_test(differences: np.ndarray) -> dict[str, float]:
    try:
        result = wilcoxon(differences, alternative="two-sided")
        return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}
    except ValueError:
        return {"statistic": float("nan"), "pvalue": float("nan")}


def paired_effect(differences: np.ndarray, seed: int = 20261030) -> dict[str, float]:
    """Seed-level paired effect with a percentile bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
    nonzero = differences[differences != 0]
    if len(nonzero):
        ranks = rankdata(np.abs(nonzero), method="average")
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        rank_biserial = (positive - negative) / (positive + negative)
    else:
        rank_biserial = 0.0
    return {
        "difference_dea_minus_rl_mean": float(differences.mean()),
        "difference_dea_minus_rl_median": float(np.median(differences)),
        "bootstrap_mean_ci95_low": float(np.quantile(samples, .025)),
        "bootstrap_mean_ci95_high": float(np.quantile(samples, .975)),
        "matched_rank_biserial": rank_biserial,
        "n_independent_training_seeds": int(len(differences)),
        "bootstrap_unit": "paired_training_seed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cada checkpoint representa uma unidade experimental; os 30 episódios são
    # primeiro resumidos dentro da seed para evitar inflar artificialmente n.
    agents = data[data["strategy"].isin(["rl_all", "dea_rl"])]
    run_means = agents.groupby(["strategy", "policy_run"], as_index=False)[METRICS].mean()
    run_means.to_csv(args.output_dir / "agent_seed_means.csv", index=False)

    agent_summary = run_means.groupby("strategy")[METRICS].agg(["mean", "std"])
    agent_summary.columns = [f"{metric}_{stat}" for metric, stat in agent_summary.columns]
    baselines = data[~data["strategy"].isin(["rl_all", "dea_rl"])]
    baseline_summary = baselines.groupby("strategy")[METRICS].agg(["mean", "std"])
    baseline_summary.columns = [f"{metric}_{stat}" for metric, stat in baseline_summary.columns]
    summary = pd.concat([agent_summary, baseline_summary]).reset_index()
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)

    rl = run_means[run_means.strategy == "rl_all"].sort_values("policy_run")
    dea = run_means[run_means.strategy == "dea_rl"].sort_values("policy_run")
    paired = {}
    for metric in METRICS:
        differences = dea[metric].to_numpy() - rl[metric].to_numpy()
        paired[metric] = {
            **paired_effect(differences),
            **paired_test(differences),
        }
    (args.output_dir / "paired_agent_tests.json").write_text(
        json.dumps(paired, indent=2, allow_nan=True), encoding="utf-8"
    )

    order = ["rl_all", "dea_rl", "highest_map", "energy_aware", "topsis_static", "lowest_energy"]
    available = [name for name in order if name in set(summary.strategy)]
    table = summary.set_index("strategy").loc[available]
    plots = [
        ("total_reward", "Total reward"),
        ("precision", "Precisão"),
        ("recall", "Recall"),
        ("processing_energy_j", "Analysis energy (J)"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, (metric, label) in zip(axes.flat, plots):
        means = table[f"{metric}_mean"].to_numpy()
        stds = table[f"{metric}_std"].fillna(0).to_numpy()
        axis.bar(np.arange(len(table)), means, yerr=stds, capsize=4)
        axis.set_xticks(np.arange(len(table)), table.index, rotation=25, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.25)
    figure.suptitle("Plan A — Raspberry Pi 4 comparison (test set)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "all_methods_comparison.png", dpi=180)
    print(f"Analysis saved to {args.output_dir}")


if __name__ == "__main__":
    main()
