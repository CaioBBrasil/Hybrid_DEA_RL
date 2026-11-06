"""Exporta artefatos de implantação depois da conclusão dos três treinos."""
from __future__ import annotations

import shutil
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TF_RUN = ROOT / "treino" / "outputs" / "tensorflow_retrain_v2"
NANODET_RUN = ROOT / "treino" / "outputs" / "sequential_run"
OUT = ROOT / "raspberrypi_tests" / "artifacts"
TF_REPO = ROOT / "external" / "tensorflow-models"
NANODET = ROOT / "external" / "nanodet"
WRAPPER = ROOT / "treino" / "common" / "tf_entrypoint.py"


def execute(command: list[str]) -> None:
    print("Executando:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def export_tflite(name: str) -> None:
    source = TF_RUN / name
    intermediate = OUT / name / "tflite_saved_model"
    execute([sys.executable, str(WRAPPER), str(TF_REPO / "research/object_detection/export_tflite_graph_tf2.py"),
             f"--pipeline_config_path={source / 'pipeline.config'}",
             f"--trained_checkpoint_dir={source / 'checkpoint'}",
             f"--output_directory={intermediate}", "--max_detections=100"])
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(str(intermediate / "saved_model"))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True  # TFLite_Detection_PostProcess da Object Detection API.
    target = OUT / name / f"{name}_fp32.tflite"
    target.write_bytes(converter.convert())


def export_nanodet() -> None:
    source = NANODET_RUN / "nanodet_plus"
    checkpoint = source / "nanodet_plus_best.pt"
    if not checkpoint.is_file():
        candidates = list(source.rglob("nanodet_model_best.pth")) + list(source.rglob("model_best.ckpt"))
        if not candidates:
            raise FileNotFoundError("Checkpoint final do NanoDet não encontrado.")
        checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)
    target = OUT / "nanodet_plus" / "nanodet_plus.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    execute([sys.executable, str(NANODET / "tools/export_onnx.py"), "--cfg_path", str(source / "train.yml"),
             "--model_path", str(checkpoint), "--out_path", str(target), "--input_shape", "640,640"])
    shutil.copy2(checkpoint, target.parent / checkpoint.name)


OUT.mkdir(parents=True, exist_ok=True)
status_path = TF_RUN / "status.json"
if not status_path.is_file() or json.loads(status_path.read_text(encoding="utf-8")).get("status") != "completed":
    raise SystemExit("O retreinamento TensorFlow ainda não foi concluído.")
for model_name in ("ssd_mobilenet_v2", "efficientdet_d0"):
    if not (TF_RUN / model_name / "checkpoint" / "checkpoint").is_file():
        raise SystemExit(f"Treino ainda não concluído ou checkpoint ausente: {model_name}")
    export_tflite(model_name)
export_nanodet()
print("Artefatos exportados em", OUT)

