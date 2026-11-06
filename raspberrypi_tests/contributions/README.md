# External embedded-benchmark contributions

This directory isolates externally supplied artifacts from the benchmark code
maintained at the root of `raspberrypi_tests/`.

| Directory | Platform | Models | Contents |
|---|---|---|---|
| `raspberry_pi4_yolo/` | Raspberry Pi 4 | YOLOv5nu, YOLOv8n, YOLO11n | ONNX artifacts, received archives, and instructions |

External results may enter the same matrix only when they use the same device,
images, resolution, thread count, repetitions, and measurement boundaries.
Model binaries are ignored by Git and should be published as release assets.
