# Raspberry Pi 4 benchmark

This directory exports and benchmarks the six Plan A deployment artifacts on a
Raspberry Pi 4.

## Prepare the bundle on the training machine

After detector training:

```bash
python raspberrypi_tests/export_artifacts.py
python raspberrypi_tests/prepare_bundle.py
```

The first command exports SSD-MobileNetV2 and EfficientDet-D0 to FP32 TFLite
and NanoDet-Plus to ONNX. The second combines them with YOLOv5nu, YOLOv8n, and
YOLO11n, copies the 152 test images, and creates a SHA-256 manifest.

Generated model artifacts and bundles are ignored by Git and should be
distributed as release assets.

## Run on Raspberry Pi 4

Use a 64-bit Raspberry Pi OS and Python 3.10 or 3.11:

```bash
chmod +x setup.sh run_all.sh
./setup.sh
source .venv/bin/activate
./run_all.sh
```

The protocol uses 640×640 input, four CPU threads, 20 warm-ups, 100 measured
inferences, and three repetitions. Disk reads and image resizing are excluded
from the primary latency loop. Return the complete `results/` directory and
timestamped external power samples.

See [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) and the consolidated
evidence in [`analysis/`](analysis/).
