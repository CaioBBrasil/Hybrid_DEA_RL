#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN

from rl.common.config import load_config
from rl.common.environment import UAVModelSelectionEnv
from rl.common.evaluation import run_episodes, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara RL-All e DEA+RL nas mesmas sementes de teste.")
    parser.add_argument("--rl-all", required=True); parser.add_argument("--dea-rl", required=True)
    parser.add_argument("--config", default="rl/plan_a/configs/raspberry_pi4_validation.yaml")
    parser.add_argument("--episodes", type=int, default=30); parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--output", default="rl/plan_a/results/revised_validation/comparison.csv")
    args = parser.parse_args(); cfg = load_config(args.config); rows = []
    for scenario, path in (("rl_all", args.rl_all), ("dea_rl", args.dea_rl)):
        allowed = cfg["scenarios"][scenario]
        if not allowed: raise SystemExit("Preencha scenarios.dea_rl com a saída real da DEA.")
        model = DQN.load(path, device="cpu")
        result = run_episodes(lambda seed, a=allowed: UAVModelSelectionEnv(cfg, "test", a, seed),
                              lambda obs, m=model: m.predict(obs, deterministic=True)[0], args.episodes, args.seed)
        for row in result: row["strategy"] = scenario
        rows.extend(result)
    write_metrics(rows, Path(args.output)); print(f"Comparação salva em {args.output}")


if __name__ == "__main__":
    main()
