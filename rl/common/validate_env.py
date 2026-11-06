#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from stable_baselines3.common.env_checker import check_env

from rl.common.config import load_config
from rl.common.diagnostics import feasibility_report
from rl.common.environment import UAVModelSelectionEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rl/plan_a/configs/raspberry_pi4_validation.yaml")
    parser.add_argument("--scenario", choices=["rl_all", "dea_rl"], default="rl_all")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    models = config["scenarios"][args.scenario]
    if not models:
        raise SystemExit("The DEA+RL scenario is empty; run DEA and populate scenarios.dea_rl.")
    report = feasibility_report(config, args.split, models)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["at_least_one_action_feasible"]:
        raise SystemExit("No detector can visit even one cell and return safely.")
    check_env(UAVModelSelectionEnv(config, args.split, models, args.seed), warn=True)
    print("Ambiente Gymnasium válido.")


if __name__ == "__main__":
    main()
