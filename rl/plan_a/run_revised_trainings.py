#!/usr/bin/env python3
"""Run revised Plan A DQN trainings sequentially with resumable status."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="rl/plan_a/configs/raspberry_pi4_validation.yaml")
    parser.add_argument("--output", type=Path,
                        default=Path("rl/plan_a/outputs/raspberry_pi4_validation_dqn_5seeds"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    jobs = [
        {"scenario": scenario, "seed": seed, "status": "pending"}
        for scenario in ("dea_rl", "rl_all") for seed in args.seeds
    ]
    state = {
        "status": "running", "pid": os.getpid(), "started_at": timestamp(),
        "config": args.config, "requested_timesteps": args.timesteps, "jobs": jobs,
    }
    save(status_path, state)
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    for job in jobs:
        if interrupted:
            break
        scenario, seed = job["scenario"], int(job["seed"])
        run_dir = output / scenario / f"seed_{seed}"
        if (run_dir / "run.json").is_file() and (run_dir / "final_model.zip").is_file():
            job["status"] = "skipped_complete"
            save(status_path, state)
            continue
        job.update(status="running", started_at=timestamp())
        state["current_job"] = f"{scenario}/seed_{seed}"
        save(status_path, state)
        command = [
            sys.executable, "-m", "rl.common.train", "--config", args.config,
            "--scenario", scenario, "--timesteps", str(args.timesteps),
            "--seed", str(seed), "--output", str(output), "--device", args.device,
            "--verbose", "0",
        ]
        log_path = logs / f"{scenario}_seed_{seed}.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if interrupted:
                        process.terminate()
            return_code = int(process.returncode or 0)
        job.update(
            status="completed" if return_code == 0 else "failed",
            return_code=return_code, finished_at=timestamp(), log=str(log_path),
        )
        save(status_path, state)
        if return_code != 0:
            state.update(status="failed", finished_at=timestamp())
            save(status_path, state)
            raise SystemExit(return_code)
    state.update(
        status="interrupted" if interrupted else "completed",
        current_job=None, finished_at=timestamp(),
    )
    save(status_path, state)


if __name__ == "__main__":
    main()
