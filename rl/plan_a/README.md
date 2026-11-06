# Plan A — real Raspberry Pi 4 evidence

Plan A uses per-image predictions from six deployable artifacts and latency and
energy measured on a Raspberry Pi 4:

- SSD-MobileNetV2;
- EfficientDet-D0;
- NanoDet-Plus;
- YOLOv5nu;
- YOLOv8n;
- YOLO11n.

## Model selection evidence

Confidence thresholds are calibrated exclusively on validation. Validation F1
is the desirable output; mean latency and gross inference-loop energy are the
undesirable inputs of an input-oriented DEA-BCC model. Pareto filtering uses
the same criteria.

```bash
.env/bin/python -m rl.plan_a.rebuild_selection_matrix
```

Key files:

- `data/raspberry_pi4_selection_validation.csv`: thresholds, validation and
  frozen-threshold test metrics, DEA scores, and Pareto dominance;
- `data/raspberry_pi4_selection_validation.yaml`: experiment profiles;
- `data/raspberry_pi4_six_models.yaml`: hardware evidence and provenance;
- `../../raspberrypi_tests/analysis/pi4_six_models_summary.csv`: consolidated
  Raspberry Pi measurements.

DEA-BCC and Pareto both retain SSD-MobileNetV2, NanoDet-Plus, and YOLOv5nu.
Therefore Pareto+RL and DEA+RL define the same action space and are not trained
as duplicate policies.

## Mission scenario

`configs/raspberry_pi4_validation.yaml` uses up to 152 cells spaced 100 m
apart. Coverage is part of the optimization outcome. The common safety shield
returns the UAV before another inference would violate the return-energy
constraint.

```bash
.env/bin/python -m rl.common.validate_env \
  --config rl/plan_a/configs/raspberry_pi4_validation.yaml

.env/bin/python -m rl.plan_a.run_revised_trainings \
  --config rl/plan_a/configs/raspberry_pi4_validation.yaml \
  --output rl/plan_a/outputs/raspberry_pi4_validation_dqn_5seeds
```

Training uses seeds 41--45, a maximum of one million steps, validation-based
checkpoint selection, and early stopping.

## Evaluation

```bash
args=()
for seed in 41 42 43 44 45; do
  args+=(--dea-rl-model "rl/plan_a/outputs/raspberry_pi4_validation_dqn_5seeds/dea_rl/seed_${seed}/best_safe_model.zip")
done
for seed in 41 42 43 44 45; do
  args+=(--rl-all-model "rl/plan_a/outputs/raspberry_pi4_validation_dqn_5seeds/rl_all/seed_${seed}/best_safe_model.zip")
done

.env/bin/python -m rl.common.experiments.evaluate_baselines \
  --config rl/plan_a/configs/raspberry_pi4_validation.yaml \
  --episodes 30 --seed 9000 \
  --output rl/plan_a/results/revised_validation/all_methods.csv \
  "${args[@]}"

.env/bin/python -m rl.plan_a.analyze_results \
  --input rl/plan_a/results/revised_validation/all_methods.csv \
  --output-dir rl/plan_a/results/revised_validation/comparison

.env/bin/python -m rl.common.experiments.compare_convergence \
  --root rl/plan_a/outputs/raspberry_pi4_validation_dqn_5seeds \
  --seeds 41 42 43 44 45 \
  --output-dir rl/plan_a/results/revised_validation/convergence
```

The 30 image permutations are first summarized within each independently
trained seed. Paired statistical tests use the five training seeds as the
independent units, avoiding pseudoreplication across permutations.

Final compact outputs are written to `results/revised_validation/`. Checkpoints and the
per-image prediction JSON are ignored by Git and should be distributed as
release assets when full reruns are required.

Known limitation: the split named `test` was consulted during earlier project
development and is not a completely untouched holdout.
