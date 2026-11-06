# README — Raspberry Pi 4 YOLO Benchmarking

## Multicriteria DEA + Reinforcement Learning Project

### Objective
Deploy and benchmark trained YOLO models on a Raspberry Pi 4 to measure inference latency, throughput, CPU usage, and memory consumption. The results will support Data Envelopment Analysis (DEA) and Reinforcement Learning (RL) for energy-aware model selection.

### Step 1: Prepare the project directory

On the Raspberry Pi 4, create the following structure:

```bash
mkdir -p ~/Multicriteria/{models,test_images,scripts,results}
```

### Step 2: Download and extract the scripts

Download `raspberry_pi_benchmark_scripts.zip` and extract its contents.

Copy the three scripts into `~/Multicriteria/scripts/`:

- `install_pi.sh` — installs the runtime and dependencies.
- `benchmark_pi.py` — benchmarks the ONNX models.
- `run_benchmark.sh` — executes the benchmark.

### Step 3: Install the runtime

On the Raspberry Pi, run:

```bash
bash ~/Multicriteria/scripts/install_pi.sh
```

This installs the required Python packages, including ONNX Runtime, NumPy, Pillow, and psutil.

### Step 4: Transfer the trained ONNX models

From the Ubuntu training computer, run:

```bash
scp /home/faveren/Desktop/Multicriteria/models/*.onnx \
  pi@raspberrypi.local:~/Multicriteria/models/
```

Replace `pi@raspberrypi.local` with the Raspberry Pi's actual SSH username and address.

The models are:

- `YOLOv5nu.onnx`
- `YOLOv8n.onnx`
- `YOLO11n.onnx`

### Step 5: Transfer test images

From the Ubuntu computer:

```bash
scp /home/faveren/Desktop/Multicriteria/dataset_merged/test/images/* \
  pi@raspberrypi.local:~/Multicriteria/test_images/
```

Use the same test images for all models to ensure fair comparison.

### Step 6: Run the benchmark

First, perform a quick test with 20 images:

```bash
PI_MAX_IMAGES=20 PI_REPEATS=1 \
  bash ~/Multicriteria/scripts/run_benchmark.sh
```

Then run the full benchmark:

```bash
PI_MAX_IMAGES=0 PI_REPEATS=5 \
  bash ~/Multicriteria/scripts/run_benchmark.sh
```

Results will be saved in:

```text
~/Multicriteria/results/
├── pi_benchmarks.csv
└── benchmark_run.log
```

### Step 7: Analyze results for DEA + RL

Use the benchmarking results to compare the models based on:

| Criterion | DEA role |
|---|---|
| Inference latency (ms) | Input — minimize |
| Energy consumption (J/image) | Input — minimize |
| Detection accuracy (mAP50-95) | Output — maximize |

Energy consumption must be measured separately or estimated using a documented hardware power model. Detection accuracy must come from standardized test-set evaluation.

Apply DEA to identify efficient candidate models, then use RL to select models dynamically according to UAV battery level, mission progress, and computational requirements.

### Important notes

- Raspberry Pi 4 must run a compatible 64-bit operating system.
- All models should use the same image size (640 × 640) and benchmark configuration.
- Keep the device's thermal and power conditions consistent.
- The supplied benchmark measures preprocessing and ONNX inference, not complete detection including NMS.
- Benchmark measurements should be collected on the Raspberry Pi, not substituted with workstation results.

**Expected output:** A reproducible Raspberry Pi performance dataset for comparing YOLO models and supporting the proposed DEA-guided RL framework.
