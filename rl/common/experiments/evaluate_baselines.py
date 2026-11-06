#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import DQN, PPO

from rl.common.baselines import EnergyAwarePolicy, HighestMapPolicy, LowestEnergyPolicy, TopsisStaticPolicy
from rl.common.config import load_config
from rl.common.dea import select_dea_models
from rl.common.environment import UAVModelSelectionEnv
from rl.common.evaluation import run_episodes, run_policy_episodes, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia baselines nos mesmos mapas e sementes.")
    parser.add_argument("--config", default="rl/plan_a/configs/raspberry_pi4_validation.yaml")
    parser.add_argument("--baseline-config", default="rl/common/configs/baselines.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--episodes", type=int, default=30); parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--rl-all-model", action="append", default=[])
    parser.add_argument("--dea-rl-model", action="append", default=[])
    parser.add_argument("--output", default="rl/plan_a/results/revised_validation/all_methods.csv")
    args = parser.parse_args(); cfg = load_config(args.config)
    algorithm_class = {"dqn": DQN, "ppo": PPO}[args.algorithm]
    baseline_cfg = yaml.safe_load(Path(args.baseline_config).read_text(encoding="utf-8"))
    all_models = cfg["scenarios"]["rl_all"]; profiles = cfg["models"]
    policies = [
        HighestMapPolicy(profiles, all_models), LowestEnergyPolicy(profiles, all_models),
        TopsisStaticPolicy(profiles, all_models, baseline_cfg["topsis"]["weights"]),
        EnergyAwarePolicy(profiles, all_models, baseline_cfg["energy_aware"]),
    ]
    factory = lambda seed, models=all_models: UAVModelSelectionEnv(cfg, args.split, models, seed)
    rows = []
    for policy in policies:
        rows.extend(run_policy_episodes(factory, policy, args.episodes, args.seed))

    for run_index, model_path in enumerate(args.rl_all_model):
        model = algorithm_class.load(model_path, device="cpu")
        result = run_episodes(factory, lambda obs: model.predict(obs, deterministic=True)[0],
                              args.episodes, args.seed)
        for row in result:
            row["strategy"] = "rl_all"; row["policy_run"] = run_index
        rows.extend(result)
    for run_index, model_path in enumerate(args.dea_rl_model):
        selection_cfg = cfg.get("selection", {})
        dea_models, scores = select_dea_models(profiles, all_models,
            float(baseline_cfg["dea_bcc"]["efficiency_threshold"]),
            input_fields=tuple(selection_cfg.get(
                "dea_input_fields", ["latency_s", "incremental_energy_j"])),
            output_field=str(selection_cfg.get("dea_output_field", "benchmark_map")))
        dea_factory = lambda seed: UAVModelSelectionEnv(cfg, args.split, dea_models, seed)
        model = algorithm_class.load(model_path, device="cpu")
        result = run_episodes(dea_factory, lambda obs: model.predict(obs, deterministic=True)[0],
                              args.episodes, args.seed)
        for row in result:
            row["strategy"] = "dea_rl"; row["policy_run"] = run_index
            row["dea_models"] = ";".join(dea_models)
            row["dea_scores"] = ";".join(f"{m}:{scores[m]:.6f}" for m in all_models)
        rows.extend(result)
    write_metrics(rows, Path(args.output)); print(f"{len(rows)} episódios salvos em {args.output}")


if __name__ == "__main__": main()
