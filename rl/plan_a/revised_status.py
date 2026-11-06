#!/usr/bin/env python3
"""Show queue and evaluation progress for revised Plan A training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("rl/plan_a/outputs/raspberry_pi4_validation_dqn_5seeds"))
    args = parser.parse_args()
    status_path = args.output / "status.json"
    if not status_path.is_file():
        raise SystemExit(f"Status not found: {status_path}")
    state = json.loads(status_path.read_text(encoding="utf-8"))
    requested = int(state["requested_timesteps"])
    print(f"state={state['status']} current={state.get('current_job')} requested={requested}")
    print(f"{'run':<20} {'status':<18} {'steps':>10} {'eval reward':>12} {'best safe':>12}")
    for job in state["jobs"]:
        scenario, seed = job["scenario"], int(job["seed"])
        run = args.output / scenario / f"seed_{seed}"
        steps = 0
        reward = None
        evaluations = run / "evaluations.npz"
        if evaluations.is_file():
            data = np.load(evaluations)
            if len(data["timesteps"]):
                steps = int(data["timesteps"][-1])
                reward = float(np.asarray(data["results"][-1]).mean())
        run_file = run / "run.json"
        if run_file.is_file():
            steps = int(json.loads(run_file.read_text(encoding="utf-8"))["actual_timesteps"])
        best_file = run / "best_safe_metrics.json"
        best = None
        if best_file.is_file():
            best = float(json.loads(best_file.read_text(encoding="utf-8"))["mean_reward"])
        reward_text = "-" if reward is None else f"{reward:.3f}"
        best_text = "-" if best is None else f"{best:.3f}"
        print(f"{scenario + '/seed_' + str(seed):<20} {job['status']:<18} "
              f"{steps:>10} {reward_text:>12} {best_text:>12}")


if __name__ == "__main__":
    main()
