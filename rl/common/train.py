#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from rl.common.config import load_config
from rl.common.diagnostics import feasibility_report
from rl.common.dea import select_dea_models
from rl.common.environment import UAVModelSelectionEnv


class EvaluationControlCallback(BaseCallback):
    """Seleciona política segura e interrompe quando segurança/recompensa estabilizam."""

    def __init__(self, output: Path, config: dict, minimum_success_rate: float = 1.0):
        super().__init__()
        self.output = output
        self.minimum_success_rate = minimum_success_rate
        self.best_safe_reward = -np.inf
        self.best_success_rate = -np.inf
        self.best_reward_at_success = -np.inf
        self.no_improvement_evaluations = 0
        self.enabled = bool(config.get("enabled", False))
        self.minimum_timesteps = int(config.get("minimum_timesteps", 0))
        self.patience = int(config.get("patience_evaluations", 0))
        self.minimum_reward_improvement = float(config.get("minimum_reward_improvement", 0.0))

    def _on_step(self) -> bool:
        evaluator = self.parent
        successes = getattr(evaluator, "_is_success_buffer", [])
        success_rate = float(np.mean(successes)) if successes else 0.0
        mean_reward = float(getattr(evaluator, "last_mean_reward", -np.inf))
        if success_rate >= self.minimum_success_rate and mean_reward > self.best_safe_reward:
            self.model.save(self.output / "best_safe_model")
            self.best_safe_reward = mean_reward
            (self.output / "best_safe_metrics.json").write_text(json.dumps({
                "timesteps": self.num_timesteps,
                "success_rate": success_rate,
                "mean_reward": mean_reward,
            }, indent=2), encoding="utf-8")
        improved = (success_rate > self.best_success_rate + 1e-12 or
                    (abs(success_rate - self.best_success_rate) <= 1e-12 and
                     mean_reward > self.best_reward_at_success + self.minimum_reward_improvement))
        if improved:
            self.best_success_rate = success_rate
            self.best_reward_at_success = mean_reward
            self.no_improvement_evaluations = 0
        elif self.num_timesteps < self.minimum_timesteps:
            # O período de paciência começa somente após o aquecimento mínimo.
            self.no_improvement_evaluations = 0
        else:
            self.no_improvement_evaluations += 1
        should_stop = (self.enabled and self.patience > 0 and
                       self.num_timesteps >= self.minimum_timesteps and
                       self.no_improvement_evaluations >= self.patience)
        if should_stop:
            (self.output / "early_stop.json").write_text(json.dumps({
                "timesteps": self.num_timesteps,
                "best_success_rate": self.best_success_rate,
                "best_reward_at_success": self.best_reward_at_success,
                "evaluations_without_improvement": self.no_improvement_evaluations,
            }, indent=2), encoding="utf-8")
            print(f"Early stopping at {self.num_timesteps} timesteps")
            return False
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina um agente para seleção de modelo no UAV.")
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--config", default="rl/plan_a/configs/raspberry_pi4_validation.yaml")
    parser.add_argument("--scenario", choices=["rl_all", "dea_rl"], default="rl_all")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--eval-frequency", type=int)
    parser.add_argument("--checkpoint-frequency", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="rl/plan_a/outputs/raspberry_pi4_validation_dqn_5seeds")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--baseline-config", default="rl/common/configs/baselines.yaml")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()
    cfg = load_config(args.config); train_cfg = cfg["training"]
    models = cfg["scenarios"][args.scenario]
    dea_scores = None; baseline_cfg = None
    if args.scenario == "dea_rl":
        baseline_cfg = yaml.safe_load(Path(args.baseline_config).read_text(encoding="utf-8"))
        selection_cfg = cfg.get("selection", {})
        selected_models, dea_scores = select_dea_models(cfg["models"], cfg["scenarios"]["rl_all"],
            float(baseline_cfg["dea_bcc"]["efficiency_threshold"]),
            input_fields=tuple(selection_cfg.get(
                "dea_input_fields", ["latency_s", "incremental_energy_j"])),
            output_field=str(selection_cfg.get("dea_output_field", "benchmark_map")))
        if models and models != selected_models:
            raise SystemExit(
                f"scenarios.dea_rl desatualizado: configurado={models}, calculado={selected_models}"
            )
        models = selected_models
    diagnosis = feasibility_report(cfg, "train", models)
    if not diagnosis["at_least_one_action_feasible"]:
        raise SystemExit("Treino cancelado: nenhuma ação de análise seguida de retorno é viável.")
    output = Path(args.output) / args.scenario / f"seed_{args.seed}"
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "feasibility.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    (output / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if baseline_cfg is not None:
        (output / "resolved_baseline_config.json").write_text(
            json.dumps(baseline_cfg, indent=2), encoding="utf-8")
    train_env = Monitor(UAVModelSelectionEnv(cfg, "train", models, args.seed))
    valid_env = Monitor(UAVModelSelectionEnv(cfg, "valid", models, args.seed + 100000))
    eval_frequency = args.eval_frequency or int(train_cfg["eval_frequency"])
    checkpoint_frequency = (args.checkpoint_frequency
                            or int(train_cfg.get("checkpoint_frequency", eval_frequency)))
    if eval_frequency <= 0 or checkpoint_frequency <= 0:
        raise SystemExit("Frequências de avaliação e checkpoint devem ser positivas.")
    early_stopping_cfg = train_cfg.get("early_stopping", {})
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=checkpoint_frequency,
            save_path=str(output / "checkpoints"),
            name_prefix=args.algorithm,
            save_replay_buffer=args.algorithm == "dqn",
        ),
        EvalCallback(
            valid_env,
            best_model_save_path=str(output),
            log_path=str(output),
            eval_freq=eval_frequency,
            n_eval_episodes=int(train_cfg.get("eval_episodes", 5)),
            deterministic=True,
            callback_after_eval=EvaluationControlCallback(output, early_stopping_cfg),
        ),
    ])
    common = dict(policy="MlpPolicy", env=train_env, verbose=args.verbose, seed=args.seed,
                  device=args.device, tensorboard_log=str(output / "tensorboard"))
    if args.algorithm == "dqn":
        model = DQN(
            **common,
            learning_rate=float(train_cfg["learning_rate"]),
            buffer_size=int(train_cfg["buffer_size"]),
            learning_starts=int(train_cfg["learning_starts"]),
            batch_size=int(train_cfg["batch_size"]),
            gamma=float(train_cfg["gamma"]),
            target_update_interval=int(train_cfg["target_update_interval"]),
        )
    else:
        ppo_cfg = train_cfg.get("ppo", {})
        model = PPO(
            **common,
            learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
            n_steps=int(ppo_cfg.get("n_steps", 2048)),
            batch_size=int(ppo_cfg.get("batch_size", 64)),
            n_epochs=int(ppo_cfg.get("n_epochs", 10)),
            gamma=float(ppo_cfg.get("gamma", train_cfg["gamma"])),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            clip_range=float(ppo_cfg.get("clip_range", 0.2)),
            ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        )
    model.learn(total_timesteps=args.timesteps or int(train_cfg["total_timesteps"]), callback=callbacks,
                progress_bar=args.progress_bar)
    model.save(output / "final_model")
    (output / "run.json").write_text(json.dumps({"algorithm": args.algorithm,
        "scenario": args.scenario, "seed": args.seed,
        "models": models, "dea_scores": dea_scores,
        "synthetic_quality": cfg["quality_provider"]["type"] != "json",
        "requested_timesteps": args.timesteps or int(train_cfg["total_timesteps"]),
        "actual_timesteps": model.num_timesteps,
        "eval_episodes": int(train_cfg.get("eval_episodes", 5)),
        "early_stopping": early_stopping_cfg,
        "eval_frequency": eval_frequency,
        "checkpoint_frequency": checkpoint_frequency}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
