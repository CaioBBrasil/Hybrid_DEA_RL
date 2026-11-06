from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from inference import detector, load_rgb


ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def run_model(spec: dict, args: argparse.Namespace, images: list[Path]) -> dict:
    runtime = detector(spec, ROOT, args.threads)
    prepared = [runtime.prepare_path(path) if hasattr(runtime, "prepare_path")
                else runtime.prepare(load_rgb(path)) for path in images]
    temperature_before = system_value("vcgencmd", "measure_temp")
    throttled_before = system_value("vcgencmd", "get_throttled")
    for index in range(args.warmup):
        runtime.raw(prepared[index % len(prepared)])
    samples = []
    started = time.perf_counter()
    for index in range(args.runs):
        begin = time.perf_counter_ns()
        runtime.raw(prepared[index % len(prepared)])
        samples.append((time.perf_counter_ns() - begin) / 1_000_000)
    duration = time.perf_counter() - started
    return {
        "model": spec["name"], "format": spec["format"], "artifact": spec["path"],
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "threads": args.threads, "warmup_runs": args.warmup, "measured_runs": args.runs,
        "input_size": [640, 640], "batch_size": 1, "timing_scope": "runtime inference and output materialization",
        "latency_ms": {"mean": statistics.fmean(samples), "median": statistics.median(samples),
                       "stdev": statistics.stdev(samples) if len(samples) > 1 else 0,
                       "min": min(samples), "max": max(samples), "p95": percentile(samples, 95),
                       "p99": percentile(samples, 99)},
        "throughput_fps": args.runs / duration, "raw_latency_ms": samples,
        "max_process_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "temperature_before": temperature_before,
        "temperature_after": system_value("vcgencmd", "measure_temp"),
        "throttled_before": throttled_before,
        "throttled_after": system_value("vcgencmd", "get_throttled"),
    }


def system_value(*command: str) -> str | None:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


parser = argparse.ArgumentParser()
parser.add_argument("--all", action="store_true", help="Executa todos os modelos do manifest.")
parser.add_argument("--model", action="append", help="Nome de modelo específico; repetível.")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--runs", type=int, default=100)
parser.add_argument("--repeat", type=int, default=1)
args = parser.parse_args()
os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
selected = manifest["models"] if args.all else [m for m in manifest["models"] if m["name"] in (args.model or [])]
if not selected:
    raise SystemExit("Selecione --all ou ao menos um --model.")
image_paths = sorted((ROOT / "test_images").glob("*"))
if not image_paths:
    raise SystemExit("Nenhuma imagem em test_images/.")
results = ROOT / "results"
results.mkdir(exist_ok=True)
for spec in selected:
    payload = run_model(spec, args, image_paths)
    payload["repeat"] = args.repeat
    target = results / f"{spec['name']}_{spec['format']}_{args.threads}threads_run{args.repeat}.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(spec["name"], f"mediana={payload['latency_ms']['median']:.2f} ms", target)

