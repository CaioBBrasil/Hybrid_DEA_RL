"""Executa os três treinamentos em sequência e publica progresso/ETA em JSON."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "treino" / "outputs" / "sequential_run"
SPLITS = ("train", "valid", "test")
TF_STEP = re.compile(r"\bStep\s+(\d+)\b", re.IGNORECASE)
NANODET_EPOCH = re.compile(r"\bEpoch\s*(\d+)(?:\D|$)", re.IGNORECASE)
TF_TOTAL_LOSS = re.compile(r"Loss/total_loss['\"]?\s*:\s*([-+a-zA-Z0-9.eE]+)")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "indisponível"
    return str(timedelta(seconds=round(seconds)))


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def image_count(dataset: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(
        1 for path in (dataset / "train" / "images").iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def build_jobs(args: argparse.Namespace) -> list[Dict[str, object]]:
    train_images = image_count(args.dataset)
    specs = [
        ("ssd_mobilenet_v2", args.ssd_batch, "tensorflow"),
        ("efficientdet_d0", args.efficientdet_batch, "tensorflow"),
        ("nanodet_plus", args.nanodet_batch, "nanodet"),
    ]
    if args.only_tensorflow:
        specs = [spec for spec in specs if spec[2] == "tensorflow"]
    jobs = []
    for name, batch, framework in specs:
        output = args.run_dir / name
        if framework == "tensorflow":
            total = math.ceil(train_images / batch) * args.epochs
            command = [
                sys.executable, str(ROOT / "treino" / name / "train.py"),
                "--models-repo", str(args.models_repo), "--dataset", str(args.dataset),
                "--output", str(output), "--batch-size", str(batch),
                "--epochs", str(args.epochs),
                "--learning-rate-factor", str(args.learning_rate_factor),
                "--gradient-clip-norm", str(args.gradient_clip_norm),
            ]
            unit = "step"
        else:
            total = args.epochs
            command = [
                sys.executable, str(ROOT / "treino" / name / "train.py"),
                "--nanodet-repo", str(args.nanodet_repo), "--dataset", str(args.dataset),
                "--output", str(output), "--batch-size", str(batch),
                "--epochs", str(args.epochs),
            ]
            unit = "epoch"
        jobs.append({
            "name": name, "framework": framework, "batch_size": batch,
            "output": str(output), "command": command, "unit": unit,
            "completed_units": 0, "total_units": total, "status": "pending",
            "started_at": None, "finished_at": None, "elapsed_seconds": 0,
            "eta_seconds": None, "return_code": None,
            "guard_alert": None,
        })
    return jobs


def parse_progress(job: Dict[str, object], line: str) -> Optional[int]:
    pattern = TF_STEP if job["framework"] == "tensorflow" else NANODET_EPOCH
    matches = list(pattern.finditer(ANSI.sub("", line)))
    if not matches:
        return None
    value = int(matches[-1].group(1))
    return min(value, int(job["total_units"]))


def update_estimates(state: Dict[str, object], current_index: int, started: float) -> None:
    jobs = state["jobs"]
    job = jobs[current_index]
    elapsed = max(0.0, time.monotonic() - started)
    job["elapsed_seconds"] = round(elapsed, 1)
    done = int(job["completed_units"])
    total = int(job["total_units"])
    job["eta_seconds"] = round(elapsed * (total - done) / done, 1) if done else None

    completed_equivalents = float(current_index) + (done / total if total else 0)
    overall_elapsed = max(0.0, time.monotonic() - float(state["monotonic_started"]))
    state["elapsed_seconds"] = round(overall_elapsed, 1)
    state["progress_percent"] = round(100 * completed_equivalents / len(jobs), 2)
    # Trata cada arquitetura como uma etapa de peso igual. É uma projeção inicial;
    # depois de cada modelo concluído, a média real recalibra automaticamente.
    if completed_equivalents > 0:
        state["eta_seconds"] = round(
            overall_elapsed * (len(jobs) - completed_equivalents) / completed_equivalents, 1
        )
    else:
        state["eta_seconds"] = None
    state["updated_at"] = now_iso()


def stream_process(
    process: subprocess.Popen[str], job: Dict[str, object], log_path: Path,
    state: Dict[str, object], state_path: Path, index: int, started: float,
    loss_threshold: float, loss_patience: int,
) -> int:
    lines: queue.Queue[Optional[str]] = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        for raw in process.stdout:
            # Lightning usa retorno de carro para atualizar a barra de progresso.
            for part in raw.replace("\r", "\n").splitlines(True):
                lines.put(part)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    excessive_losses = 0
    guard_stopped = False
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        finished_output = False
        last_write = 0.0
        while process.poll() is None or not finished_output:
            try:
                line = lines.get(timeout=1)
                if line is None:
                    finished_output = True
                else:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    if job["framework"] == "tensorflow" and not guard_stopped:
                        match = TF_TOTAL_LOSS.search(ANSI.sub("", line))
                        if match:
                            try:
                                loss = float(match.group(1))
                            except ValueError:
                                loss = math.nan
                            reason = None
                            if not math.isfinite(loss):
                                reason = f"Loss inválida detectada: {match.group(1)}"
                            elif loss > loss_threshold:
                                excessive_losses += 1
                                if excessive_losses >= loss_patience:
                                    reason = (
                                        f"Loss total acima de {loss_threshold:g} por "
                                        f"{excessive_losses} registros consecutivos (atual={loss:g})"
                                    )
                            else:
                                excessive_losses = 0
                            if reason:
                                alert = f"[GUARD] {reason}. Encerrando treino para diagnóstico.\n"
                                sys.stdout.write(alert)
                                sys.stdout.flush()
                                log.write(alert)
                                job["guard_alert"] = reason
                                guard_stopped = True
                                os.killpg(process.pid, signal.SIGTERM)
                    progress = parse_progress(job, line)
                    if progress is not None and progress > int(job["completed_units"]):
                        job["completed_units"] = progress
            except queue.Empty:
                pass
            if time.monotonic() - last_write >= 5:
                update_estimates(state, index, started)
                atomic_json(state_path, {k: v for k, v in state.items() if k != "monotonic_started"})
                last_write = time.monotonic()
    return int(process.wait())


def run(args: argparse.Namespace) -> int:
    if args.learning_rate_factor <= 0 or args.gradient_clip_norm < 0:
        raise SystemExit("O fator de LR deve ser positivo e o clipping não pode ser negativo.")
    if args.abort_loss_threshold <= 0 or args.abort_loss_patience <= 0:
        raise SystemExit("O limite de loss e a tolerância do guard devem ser positivos.")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.run_dir / "status.json"
    jobs = build_jobs(args)
    active_outputs = [Path(job["output"]) for job in jobs]
    if not args.resume and any(path.exists() for path in active_outputs):
        raise SystemExit(
            f"Saídas já existem em {args.run_dir}. Use outro --run-dir ou --resume conscientemente."
        )
    state: Dict[str, object] = {
        "status": "running", "pid": os.getpid(), "started_at": now_iso(),
        "updated_at": now_iso(), "finished_at": None, "current_job": None,
        "elapsed_seconds": 0, "eta_seconds": None, "progress_percent": 0,
        "jobs": jobs, "monotonic_started": time.monotonic(),
    }
    atomic_json(state_path, {k: v for k, v in state.items() if k != "monotonic_started"})
    child: Optional[subprocess.Popen[str]] = None

    def stop(signum: int, _frame: object) -> None:
        state["status"] = "interrupted"
        state["updated_at"] = now_iso()
        atomic_json(state_path, {k: v for k, v in state.items() if k != "monotonic_started"})
        if child and child.poll() is None:
            os.killpg(child.pid, signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    for index, job in enumerate(jobs):
        state["current_job"] = job["name"]
        job["status"] = "running"
        job["started_at"] = now_iso()
        started = time.monotonic()
        log_path = args.run_dir / f"{job['name']}.log"
        print(f"\n=== Iniciando {job['name']} ===", flush=True)
        print("Log:", log_path, flush=True)
        child = subprocess.Popen(
            job["command"], cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        return_code = stream_process(
            child, job, log_path, state, state_path, index, started,
            args.abort_loss_threshold, args.abort_loss_patience,
        )
        job["return_code"] = return_code
        job["elapsed_seconds"] = round(time.monotonic() - started, 1)
        job["finished_at"] = now_iso()
        if return_code != 0:
            guarded = bool(job["guard_alert"])
            job["status"] = "guard_stopped" if guarded else "failed"
            state["status"] = "guard_stopped" if guarded else "failed"
            state["finished_at"] = now_iso()
            state["updated_at"] = now_iso()
            atomic_json(state_path, {k: v for k, v in state.items() if k != "monotonic_started"})
            print(f"{job['name']} falhou com código {return_code}. A sequência foi interrompida.")
            return return_code
        job["status"] = "completed"
        job["completed_units"] = job["total_units"]
        job["eta_seconds"] = 0

    state.update({
        "status": "completed", "current_job": None, "finished_at": now_iso(),
        "updated_at": now_iso(), "progress_percent": 100, "eta_seconds": 0,
        "elapsed_seconds": round(time.monotonic() - float(state["monotonic_started"]), 1),
    })
    atomic_json(state_path, {k: v for k, v in state.items() if k != "monotonic_started"})
    print("\nTodos os treinamentos foram concluídos.")
    return 0


def show_status(args: argparse.Namespace) -> int:
    path = args.run_dir / "status.json"
    if not path.is_file():
        raise SystemExit(f"Status ainda não existe: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    # Versões do NanoDet podem registrar `Epoch221` sem espaço. Reconstitui o
    # progresso do log para execuções iniciadas antes desta correção.
    for job in state["jobs"]:
        if job["name"] == "nanodet_plus" and job["status"] == "running":
            log_path = args.run_dir / "nanodet_plus.log"
            if log_path.is_file():
                epochs = [int(match.group(1)) for match in NANODET_EPOCH.finditer(log_path.read_text(encoding="utf-8", errors="replace"))]
                if epochs:
                    job["completed_units"] = min(max(epochs), int(job["total_units"]))
                    elapsed = float(job.get("elapsed_seconds") or 0)
                    done = int(job["completed_units"])
                    job["eta_seconds"] = elapsed * (int(job["total_units"]) - done) / done
                    completed = sum(
                        1 if item["status"] == "completed" else int(item["completed_units"]) / int(item["total_units"])
                        for item in state["jobs"]
                    )
                    state["progress_percent"] = 100 * completed / len(state["jobs"])
                    state["eta_seconds"] = job["eta_seconds"]
    print(f"Estado: {state['status']}")
    print(f"Progresso total estimado: {state['progress_percent']:.2f}%")
    print(f"Decorrido: {duration(state.get('elapsed_seconds'))}")
    print(f"ETA aproximado: {duration(state.get('eta_seconds'))}")
    print(f"Etapa atual: {state.get('current_job') or '-'}")
    print(f"Última atualização: {state['updated_at']}")
    print()
    for job in state["jobs"]:
        total, done = int(job["total_units"]), int(job["completed_units"])
        percent = 100 * done / total if total else 0
        print(
            f"{job['name']:<22} {job['status']:<10} "
            f"{done}/{total} {job['unit']} ({percent:.2f}%) "
            f"decorrido={duration(job.get('elapsed_seconds'))} "
            f"ETA={duration(job.get('eta_seconds'))}"
        )
        if job.get("guard_alert"):
            print(f"  alerta: {job['guard_alert']}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="Executa os três treinamentos.")
    run_parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    run_parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_merged")
    run_parser.add_argument("--models-repo", type=Path, default=ROOT / "external" / "tensorflow-models")
    run_parser.add_argument("--nanodet-repo", type=Path, default=ROOT / "external" / "nanodet")
    run_parser.add_argument("--epochs", type=int, default=300)
    run_parser.add_argument("--ssd-batch", type=int, default=8)
    run_parser.add_argument("--efficientdet-batch", type=int, default=4)
    run_parser.add_argument("--nanodet-batch", type=int, default=16)
    run_parser.add_argument("--learning-rate-factor", type=float, default=0.1)
    run_parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    run_parser.add_argument("--abort-loss-threshold", type=float, default=50.0)
    run_parser.add_argument("--abort-loss-patience", type=int, default=3)
    run_parser.add_argument("--only-tensorflow", action="store_true")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.set_defaults(function=run)
    status_parser = subparsers.add_parser("status", help="Mostra progresso e ETA.")
    status_parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    status_parser.set_defaults(function=show_status)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.function(arguments))
