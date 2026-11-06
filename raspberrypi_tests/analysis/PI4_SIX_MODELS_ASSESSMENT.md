# Raspberry Pi 4 six-model evidence assessment

Assessment date: 2026-10-30.

## Integrity and protocol

The retained campaign contains complete repetitions 7, 8, and 9 for all six
models. `system.json` identifies a Raspberry Pi 4 Model B Rev. 1.2, aarch64,
four Cortex-A72 cores, Python 3.11.17, and 1.8 GiB RAM. Every repetition used
four threads, 20 warm-ups, and 100 measured inferences. All processes exited
successfully, no throttling was reported, and peak observed temperature was
61.3 °C.

## Consolidated latency and energy

| Model | Latency (ms) | FPS | Gross energy (J/inf.) | Gross energy (mWh/inf.) | Estimated incremental energy (mWh/inf.) |
|---|---:|---:|---:|---:|---:|
| SSD-MobileNetV2 | 439.72 ± 1.94 | 2.274 | 1.581 | 0.439 | 0.160 |
| EfficientDet-D0 | 1,940.54 ± 4.64 | 0.515 | 5.893 | 1.637 | 0.414 |
| NanoDet-Plus | 463.35 ± 0.06 | 2.158 | 1.765 | 0.490 | 0.188 |
| YOLOv5nu | 508.22 ± 5.28 | 1.968 | 1.935 | 0.538 | 0.213 |
| YOLOv8n | 557.12 ± 3.26 | 1.795 | 2.107 | 0.585 | 0.219 |
| YOLO11n | 514.98 ± 1.44 | 1.942 | 1.939 | 0.539 | 0.202 |

Values are mean ± standard deviation across repetitions 7--9.

## Inference-window reconstruction

The benchmark records `captured_at` after the measured loop and reports
throughput for 100 inferences. The analysis reconstructs:

```text
end = captured_at
start = end - (100 / throughput_fps)
```

Power is linearly interpolated at both boundaries and trapezoidally integrated
only inside this interval. Gross loop energy is divided by 100. Full-process
energy is not divided by 100 because it includes initialization, warm-up, and
shutdown.

## Electrical limitations

The Arduino reports current multiplied by a fixed 5-V assumption. Real voltage
and sensor calibration were not recorded. Only one pre-process idle sample is
available per execution; estimated idle power across 18 samples was
2.321 ± 0.067 W. Idle-subtracted energy is therefore exploratory. The primary
matrix uses gross loop energy and reports the idle-subtracted definition only
as sensitivity.

## Detection quality

The final selection pipeline recalibrates every confidence threshold on the
validation split. The audit table is
`rl/plan_a/data/raspberry_pi4_selection_validation.csv`. It contains validation
F1, frozen-threshold test metrics, DEA-BCC scores, and Pareto dominance.

The deployed artifact identity is `YOLOv5nu`; it must not be silently renamed
to YOLOv5n. Artifact hashes and formats are preserved in the campaign manifest.
