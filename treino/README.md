# Detector training

This directory contains entry points for the three locally trained Plan A
detectors on the one-class YOLO dataset in `dataset_merged/`:

- SSD-MobileNetV2;
- EfficientDet-D0;
- NanoDet-Plus.

Training uses the `train` split, checkpoint selection uses `valid`, and `test`
is reserved for final characterization. The default input size is 640×640 and
the default seed is 42.

## Environment

The full environment targets Linux x86_64, Python 3.10, and an NVIDIA GPU:

```bash
python3.10 -m venv .env
source .env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

mkdir -p external
git clone https://github.com/tensorflow/models.git external/tensorflow-models
git -C external/tensorflow-models checkout 5c6915a28dd30fba8f26343f6470499e8a4a99a2
git clone --branch v1.0.0 --depth 1 https://github.com/RangiLyu/nanodet.git external/nanodet
```

Install NanoDet in editable mode and compile/install the TensorFlow Object
Detection API from `external/tensorflow-models/research` before training.

## Commands

Run from the repository root:

```bash
python treino/ssd_mobilenet_v2/train.py \
  --models-repo external/tensorflow-models --batch-size 8 --epochs 300

python treino/efficientdet_d0/train.py \
  --models-repo external/tensorflow-models --batch-size 4 --epochs 300

python treino/nanodet_plus/train.py \
  --nanodet-repo external/nanodet --batch-size 16 --epochs 300
```

Sequential execution and status monitoring:

```bash
python scripts/run_all_trainings.py run
python scripts/run_all_trainings.py status
```

The TensorFlow wrappers derive total steps from image count, batch size, and
epochs. NanoDet receives `total_epochs: 300`. Optimizers and schedulers remain
architecture-specific; forcing identical optimization recipes is not treated
as a fair comparison.

## Evaluation and export

```bash
python treino/ssd_mobilenet_v2/test.py \
  --models-repo external/tensorflow-models \
  --checkpoint-dir treino/outputs/ssd_mobilenet_v2/checkpoint

python treino/efficientdet_d0/test.py \
  --models-repo external/tensorflow-models \
  --checkpoint-dir treino/outputs/efficientdet_d0/checkpoint

python treino/nanodet_plus/test.py \
  --nanodet-repo external/nanodet \
  --checkpoint treino/outputs/nanodet_plus/model_best/model_best.ckpt
```

Evaluation records COCO metrics and TP/FP/FN-derived precision, recall, and F1.
Do not use the test split to choose epochs, thresholds, or hyperparameters.
Workstation timing is not a substitute for Raspberry Pi latency.

Generated checkpoints, logs, normalized annotations, and exported models live
under ignored output/artifact directories. Publish them separately as release
assets when needed.
