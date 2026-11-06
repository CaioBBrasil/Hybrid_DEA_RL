from __future__ import annotations

from pathlib import Path
from typing import Optional


def write_config(
    path: Path,
    train_images: Path,
    train_labels: Path,
    eval_images: Path,
    eval_labels: Path,
    save_dir: Path,
    image_size: int,
    batch_size: int,
    epochs: int,
    workers: int,
    learning_rate: float,
    validation_interval: int,
    pretrained: Optional[Path] = None,
) -> None:
    load_model = f"  load_model: {pretrained.as_posix()}\n" if pretrained else ""
    text = f"""save_dir: {save_dir.as_posix()}
model:
  weight_averager:
    name: ExpMovingAverager
    decay: 0.9998
  arch:
    name: NanoDetPlus
    detach_epoch: 10
    backbone:
      name: ShuffleNetV2
      model_size: 1.0x
      out_stages: [2, 3, 4]
      activation: LeakyReLU
    fpn:
      name: GhostPAN
      in_channels: [116, 232, 464]
      out_channels: 96
      kernel_size: 5
      num_extra_level: 1
      use_depthwise: True
      activation: LeakyReLU
    head:
      name: NanoDetPlusHead
      num_classes: 1
      input_channel: 96
      feat_channels: 96
      stacked_convs: 2
      kernel_size: 5
      strides: [8, 16, 32, 64]
      activation: LeakyReLU
      reg_max: 7
      norm_cfg:
        type: BN
      loss:
        loss_qfl: {{name: QualityFocalLoss, use_sigmoid: True, beta: 2.0, loss_weight: 1.0}}
        loss_dfl: {{name: DistributionFocalLoss, loss_weight: 0.25}}
        loss_bbox: {{name: GIoULoss, loss_weight: 2.0}}
    aux_head:
      name: SimpleConvHead
      num_classes: 1
      input_channel: 192
      feat_channels: 192
      stacked_convs: 4
      strides: [8, 16, 32, 64]
      activation: LeakyReLU
      reg_max: 7
class_names: &class_names ['acai']
data:
  train:
    name: YoloDataset
    img_path: {train_images.as_posix()}
    ann_path: {train_labels.as_posix()}
    class_names: *class_names
    input_size: [{image_size}, {image_size}]
    keep_ratio: False
    pipeline:
      perspective: 0.0
      scale: [0.6, 1.4]
      stretch: [[0.8, 1.2], [0.8, 1.2]]
      rotation: 0
      shear: 0
      translate: 0.2
      flip: 0.5
      brightness: 0.2
      contrast: [0.6, 1.4]
      saturation: [0.5, 1.2]
      normalize: [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
  val:
    name: YoloDataset
    img_path: {eval_images.as_posix()}
    ann_path: {eval_labels.as_posix()}
    class_names: *class_names
    input_size: [{image_size}, {image_size}]
    keep_ratio: False
    pipeline:
      normalize: [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
device:
  gpu_ids: [0]
  workers_per_gpu: {workers}
  batchsize_per_gpu: {batch_size}
schedule:
{load_model}  optimizer:
    name: AdamW
    lr: {learning_rate}
    weight_decay: 0.05
  warmup: {{name: linear, steps: 500, ratio: 0.0001}}
  total_epochs: {epochs}
  lr_schedule: {{name: CosineAnnealingLR, T_max: {epochs}, eta_min: 0.00005}}
  val_intervals: {validation_interval}
grad_clip: 35
evaluator:
  name: CocoDetectionEvaluator
  save_key: mAP
log:
  interval: 50
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

