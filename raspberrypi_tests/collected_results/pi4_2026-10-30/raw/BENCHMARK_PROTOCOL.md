# Retained Raspberry Pi 4 protocol

The campaign used 640×640 input, four threads, 20 warm-ups, 100 measured
inferences, batch size one, and three repetitions. System metadata, benchmark
JSON, resource telemetry, timestamped power samples, and per-run metadata are
preserved in this directory.

Gross inference-loop energy is reconstructed from timestamps and throughput;
full-process energy must not be divided directly by 100. See the canonical
protocol at `raspberrypi_tests/BENCHMARK_PROTOCOL.md`.
