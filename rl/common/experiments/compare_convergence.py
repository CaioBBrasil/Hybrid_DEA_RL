#!/usr/bin/env python3
"""Aggregate periodic DQN evaluations and compare convergence curves."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_runs(root: Path, scenarios: list[str], seeds: list[int]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for scenario in scenarios:
        for seed in seeds:
            source = root / scenario / f"seed_{seed}" / "evaluations.npz"
            if not source.exists():
                print(f"Warning: missing evaluation, skipping {source}")
                continue
            data = np.load(source)
            timesteps = data["timesteps"]
            rewards = data["results"]
            lengths = data["ep_lengths"]
            for index, step in enumerate(timesteps):
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "timestep": int(step),
                    "mean_reward": float(np.mean(rewards[index])),
                    "mean_episode_length": float(np.mean(lengths[index])),
                })
    return rows


def aggregate(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scenario"]), int(row["timestep"])), []).append(row)
    result: list[dict[str, float | int | str]] = []
    for (scenario, timestep), group in sorted(grouped.items()):
        rewards = np.asarray([float(row["mean_reward"]) for row in group])
        lengths = np.asarray([float(row["mean_episode_length"]) for row in group])
        result.append({
            "scenario": scenario,
            "timestep": timestep,
            "n_seeds": len(group),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(ddof=1)) if len(group) > 1 else 0.0,
            "episode_length_mean": float(lengths.mean()),
            "episode_length_std": float(lengths.std(ddof=1)) if len(group) > 1 else 0.0,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("rl/plan_a/outputs/raspberry_pi4_validation_dqn_5seeds"))
    parser.add_argument("--scenarios", nargs="+", default=["rl_all", "dea_rl"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("rl/plan_a/results/revised_validation/convergence"))
    parser.add_argument("--reward-threshold", type=float, default=27.0)
    args = parser.parse_args()

    rows = load_runs(args.root, args.scenarios, args.seeds)
    if not rows:
        raise SystemExit("No evaluations found. Wait for the first training evaluation cycle.")
    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, records in (("runs.csv", rows), ("summary.csv", summary)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    # Compare every run over the same horizon; longer early-stopping tails do
    # not receive an artificial AUC advantage.
    final_steps = {
        (str(row["scenario"]), int(row["seed"])): int(row["timestep"])
        for row in rows
    }
    common_horizon = min(final_steps.values())
    learning_rows = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            selected = sorted(
                (row for row in rows if row["scenario"] == scenario and row["seed"] == seed
                 and int(row["timestep"]) <= common_horizon),
                key=lambda row: int(row["timestep"]),
            )
            if not selected:
                continue
            x = np.asarray([int(row["timestep"]) for row in selected], float)
            y = np.asarray([float(row["mean_reward"]) for row in selected], float)
            reached = x[y >= args.reward_threshold]
            learning_rows.append({
                "scenario": scenario, "seed": seed,
                "common_horizon": common_horizon,
                "normalized_auc": float(np.trapz(y, x) / max(1.0, x[-1] - x[0])),
                "reward_threshold": args.reward_threshold,
                "first_step_at_threshold": int(reached[0]) if len(reached) else "",
                "final_observed_step": int(final_steps[(scenario, seed)]),
            })
    with (args.output_dir / "learning_efficiency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(learning_rows[0]))
        writer.writeheader()
        writer.writerows(learning_rows)

    figure, axis = plt.subplots(figsize=(9, 5))
    for scenario in args.scenarios:
        selected = [row for row in summary if row["scenario"] == scenario]
        if not selected:
            continue
        x = np.asarray([int(row["timestep"]) for row in selected])
        y = np.asarray([float(row["reward_mean"]) for row in selected])
        deviation = np.asarray([float(row["reward_std"]) for row in selected])
        axis.plot(x, y, marker="o", label=scenario)
        axis.fill_between(x, y - deviation, y + deviation, alpha=0.2)
    axis.set(xlabel="Training steps", ylabel="Mean validation reward",
             title="DQN convergence: DEA + RL vs. RL-All")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "convergence.png", dpi=160)

    # Individual curves make early stopping auditable. The aggregate curve can
    # hide stopped runs because the number of active seeds decreases.
    detailed, detailed_axis = plt.subplots(figsize=(10, 6))
    colors = {"rl_all": "tab:blue", "dea_rl": "tab:orange"}
    for scenario in args.scenarios:
        color = colors.get(scenario)
        for seed in args.seeds:
            selected = [row for row in rows
                        if row["scenario"] == scenario and row["seed"] == seed]
            if not selected:
                continue
            x = np.asarray([int(row["timestep"]) for row in selected])
            y = np.asarray([float(row["mean_reward"]) for row in selected])
            label = f"{scenario} — seed {seed}"
            detailed_axis.plot(x, y, color=color, alpha=.38, linewidth=1, label=label)
            run_path = args.root / scenario / f"seed_{seed}" / "run.json"
            if run_path.exists():
                stop = int(json.loads(run_path.read_text(encoding="utf-8"))["actual_timesteps"])
                index = int(np.argmin(np.abs(x - stop)))
                detailed_axis.scatter(x[index], y[index], marker="X", s=55,
                                      color=color, edgecolor="black", linewidth=.4, zorder=3)
    detailed_axis.axvline(300000, color="black", linestyle="--", alpha=.55,
                          label="start of early-stopping window")
    detailed_axis.set(
        xlabel="Training steps",
        ylabel="Mean validation reward",
        title="Convergence by seed (X = early stopping)",
    )
    detailed_axis.grid(alpha=.25)
    detailed_axis.legend(fontsize=7, ncol=2)
    detailed.tight_layout()
    detailed.savefig(args.output_dir / "convergence_by_seed.png", dpi=180)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
