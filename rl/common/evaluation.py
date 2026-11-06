from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

from .environment import UAVModelSelectionEnv


def run_episodes(
    env_factory: Callable[[int], UAVModelSelectionEnv],
    predict: Callable[[Any], int], episodes: int, seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for episode in range(episodes):
        env = env_factory(seed + episode)
        obs, reset_info = env.reset(seed=seed + episode)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action = int(predict(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        row = dict(info)
        row.update({"episode": episode, "seed": seed + episode, "total_reward": total_reward,
                    "image_order": json.dumps(reset_info["image_order"]),
                    "model_counts": json.dumps(info["model_counts"], sort_keys=True)})
        rows.append(row)
        env.close()
    return rows


def run_policy_episodes(env_factory, policy, episodes: int, seed: int) -> list[dict[str, Any]]:
    """Avalia um seletor que precisa consultar o estado físico público do ambiente."""
    rows = []
    for episode in range(episodes):
        env = env_factory(seed + episode)
        obs, reset_info = env.reset(seed=seed + episode)
        terminated = truncated = False; total_reward = 0.0
        while not (terminated or truncated):
            action = int(policy.select(obs, env))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        row = dict(info)
        row.update({"episode": episode, "seed": seed + episode, "total_reward": total_reward,
                    "image_order": json.dumps(reset_info["image_order"]),
                    "model_counts": json.dumps(info["model_counts"], sort_keys=True),
                    "strategy": policy.name})
        rows.append(row); env.close()
    return rows


def write_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
