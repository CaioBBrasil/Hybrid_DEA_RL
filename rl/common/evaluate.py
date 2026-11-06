#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN

from rl.common.config import load_config
from rl.common.environment import UAVModelSelectionEnv
from rl.common.evaluation import run_episodes, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="rl/plan_a/configs/raspberry_pi4_validation.yaml")
    parser.add_argument("--scenario", choices=["rl_all", "dea_rl"], default="rl_all")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--output", default="rl/plan_a/results/revised_validation/metrics.csv")
    args = parser.parse_args(); cfg = load_config(args.config)
    allowed = cfg["scenarios"][args.scenario]
    if not allowed: raise SystemExit("Cenário sem modelos configurados.")
    model = DQN.load(args.model, device="cpu")
    rows = run_episodes(lambda seed: UAVModelSelectionEnv(cfg, args.split, allowed, seed),
                        lambda obs: model.predict(obs, deterministic=True)[0], args.episodes, args.seed)
    write_metrics(rows, Path(args.output))
    print(f"Métricas salvas em {args.output}")


if __name__ == "__main__":
    main()
