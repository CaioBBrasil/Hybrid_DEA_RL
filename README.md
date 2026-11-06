# Hybrid DEA + RL for UAV detector selection

This repository contains the Plan A experimental pipeline for selecting an
embedded object detector during a simulated UAV survey. It combines
input-oriented DEA-BCC, a DQN policy, real per-image detections, and latency and
energy measurements collected on a Raspberry Pi 4.

The repository contains the experiment code, compact input matrices, Raspberry
Pi measurement summaries, and final tabular results. Large datasets, detector
weights, and DQN checkpoints are kept outside Git; the per-image predictions
needed by the mission simulator are included.

## Dataset

**Download:** [Google Drive dataset archive](https://drive.google.com/file/d/1s9PLA9p5vbnSeVyWP_5mEUx1DaNZDIP7/view)

Download and extract the archive into `dataset_merged/` at the repository root.
The expected one-class YOLO layout is:

```text
dataset_merged/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

The directory is intentionally listed in `.gitignore`. The experiment expects
YOLO label files with class identifier `0` and normalized bounding boxes in
`class x_center y_center width height` format.

## Repository layout

```text
dataset_merged/      local one-class YOLO dataset (not versioned)
raspberrypi_tests/   deployment benchmark and Raspberry Pi evidence
rl/common/           mission environment, DEA, Pareto, DQN, baselines, tests
rl/plan_a/           experiment configurations, matrices, scripts, results
treino/              training and export code for the three original detectors
scripts/             dataset preparation and training orchestration
```

## Installation

Python 3.10 is recommended. From the repository root, create the DEA/RL
environment with:

```bash
python3.10 -m venv .env
.env/bin/pip install -r rl/requirements.txt
```

Detector training requires the larger environment in `requirements.txt` and
the external repositories described in [`treino/README.md`](treino/README.md).

## Reproduction inputs

The mission experiments require both the dataset and
`rl/plan_a/data/predictions.raspberry_pi4_real.json`. The prediction file is
versioned with the repository so that the decision matrix and mission
experiments can be reproduced without rerunning all six detectors. The file
used for the reported experiments has SHA-256 checksum
`7e3c6431f2e26aa3d912d9c8baded14ca9266de2bfe5de49bd12b7d5c2c2f3ac`.

Alternatively, regenerate it from the six exported detector artifacts:

```bash
.env/bin/python -m rl.plan_a.generate_predictions \
  --dataset dataset_merged \
  --output rl/plan_a/data/predictions.raspberry_pi4_real.json
```

The required artifact paths and formats are defined in
[`rl/plan_a/generate_predictions.py`](rl/plan_a/generate_predictions.py).
Instructions for exporting and benchmarking the original three detectors are
in [`raspberrypi_tests/README.md`](raspberrypi_tests/README.md). The remaining
YOLO artifacts must be placed at the paths listed by the prediction script.

## Reproduce the experiment

Run all commands from the repository root.

### 1. Validate the installation

```bash
.env/bin/python -m unittest discover -s rl/common/tests -v
```

### 2. Rebuild the decision matrix

This recalibrates confidence thresholds on the validation split, computes the
validation quality criteria, and runs DEA-BCC and Pareto screening:

```bash
.env/bin/python -m rl.plan_a.rebuild_selection_matrix
```

The command rewrites:

- `rl/plan_a/data/raspberry_pi4_selection_validation.csv`;
- `rl/plan_a/data/raspberry_pi4_selection_validation.yaml`.

The reported mission uses the corresponding frozen profile in
`rl/plan_a/configs/models.raspberry_pi4_validation.yaml`. If different data or
predictions are supplied, update that profile and the confidence thresholds in
the mission configuration before retraining.

### 3. Validate the mission environment

```bash
.env/bin/python -m rl.common.validate_env \
  --config rl/plan_a/configs/raspberry_pi4_validation.yaml
```

### 4. Train DEA+RL and RL-All

```bash
.env/bin/python -m rl.plan_a.run_revised_trainings \
  --config rl/plan_a/configs/raspberry_pi4_validation.yaml \
  --output rl/plan_a/outputs/raspberry_pi4_validation_dqn_5seeds
```

This runs seeds 41--45 for both action spaces, with validation-based checkpoint
selection and early stopping. Training outputs are intentionally ignored by
Git. Progress can be inspected with:

```bash
.env/bin/python -m rl.plan_a.revised_status \
  --root rl/plan_a/outputs/raspberry_pi4_validation_dqn_5seeds
```

### 5. Evaluate and analyze

The exact evaluation, statistical-analysis, and convergence commands are in
[`rl/plan_a/README.md`](rl/plan_a/README.md).

Compact reported outputs are available in
`rl/plan_a/results/revised_validation/`. Raspberry Pi measurement provenance
and consolidated hardware results are documented under
`raspberrypi_tests/`.

## Reproducibility scope

- The same dataset and prediction JSON are sufficient to rebuild the decision
  matrix, retrain both DQN variants, execute all baselines, and regenerate the
  statistical results.
- The exported detector artifacts are additionally required to regenerate the
  per-image prediction JSON.
- The original training datasets, external framework repositories, and model
  weights are required to retrain the object detectors from scratch.
- The physical Raspberry Pi and external power-monitoring setup are required
  to repeat the hardware measurement campaign rather than reuse its published
  summary.
- The existing split named `test` was consulted during earlier project
  development and is therefore not claimed as a completely untouched holdout.

## Large artifacts

Datasets, virtual environments, external repositories, model binaries, and DQN
checkpoints are excluded from Git. Publish them as release assets or through
archival storage when full from-scratch reproduction is required. The per-image
prediction JSON, compact matrices, and final results are versioned in the
repository.

## Citation and license

Citation metadata and an explicit software license still need to be added
before the repository is archived or formally released. Choose the license
according to the intended reuse conditions and the licenses of redistributed
third-party assets.
