# Raspberry Pi 4 campaign — 2026-10-30

Joint six-model campaign using four threads, 20 warm-ups, 100 measured
inferences, and three monitored repetitions per model. Cross-model comparisons
use repetitions 7, 8, and 9 only.

- `raw/`: original benchmark, telemetry, power samples, and metadata;
- `../../analysis/pi4_six_models_summary.csv`: consolidated measurements;
- `../../analysis/PI4_SIX_MODELS_ASSESSMENT.md`: methodological assessment.

The original YOLO filenames marked `provisorio_thr0.250` document the received
state. Final Plan A thresholds are recalibrated on validation by
`rl.plan_a.rebuild_selection_matrix`.
