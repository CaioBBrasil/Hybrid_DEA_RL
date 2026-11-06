# Raspberry Pi 4 benchmark protocol

## Required environment record

Record board revision and RAM, operating system, kernel, architecture, runtime
versions, power supply, cooling, ambient temperature, CPU governor, clock, and
thread count. Do not run desktop applications or unrelated heavy workloads.

## Procedure

1. Reboot the device and wait five minutes.
2. Confirm no throttling with `vcgencmd get_throttled`.
3. Capture `system.json`.
4. Run 20 warm-up inferences.
5. Run 100 measured inferences per model with batch size one.
6. Repeat the full experiment three times.
7. Record temperature before and after every repetition.
8. Keep 640×640 input, four threads, and manifest thresholds unchanged.
9. Exclude model loading and image-file reading from primary latency.
10. Preserve raw JSON/CSV files rather than transcribing only averages.

## Energy

Prefer an external USB-C meter with timestamped power export. Record at least
60 seconds of idle power before and after a long inference block.

```text
incremental_energy_per_inference =
  (block_energy - mean_idle_power × block_duration) / inference_count
```

Also report gross energy per inference. Never mix gross and idle-subtracted
energy silently. Document instrument model, sample rate, circuit location,
calibration, voltage measurement, and uncertainty.

## Validity criteria

- no undervoltage or throttling;
- identical images and order for every detector;
- identical resolution, threads, governor, and thermal setup;
- no manual result exclusion without justification;
- latency, energy, and detection quality remain separate evidence sources;
- quality evaluation covers 152 images and 989 annotated objects.
