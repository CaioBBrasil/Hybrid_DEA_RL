from __future__ import annotations

import argparse
import math
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict

from .project import DEFAULT_DATASET, DEFAULT_OUTPUTS, absolute, run, seed_everything, validate_dataset
from .results import evaluate_tensorflow_saved_model, extract_tensorboard_metrics, write_json
from .yolo_tfrecord import ensure_tfrecords

MODELS = {
    "ssd_mobilenet_v1": {
        "archive": "ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz",
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz",
        "directory": "ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8",
    },
    "ssd_mobilenet_v2": {
        "archive": "ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
        "directory": "ssd_mobilenet_v2_320x320_coco17_tpu-8",
    },
    "efficientdet_d0": {
        "archive": "efficientdet_d0_coco17_tpu-32.tar.gz",
        "url": "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz",
        "directory": "efficientdet_d0_coco17_tpu-32",
    },
}

TF_ENTRYPOINT = Path(__file__).with_name("tf_entrypoint.py")


def train_parser(model: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Treina {model} para detectar açaí.")
    parser.add_argument("--models-repo", required=True, help="Clone local de tensorflow/models")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUTS / model))
    parser.add_argument("--image-size", type=int, default=640)
    default_batch = 8 if model.startswith("ssd_mobilenet") else 4
    parser.add_argument("--batch-size", type=int, default=default_batch)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=10)
    parser.add_argument("--learning-rate-factor", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--smoke-test", action="store_true", help="Executa apenas um passo para diagnóstico.")
    return parser


def test_parser(model: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Avalia {model} no split de teste.")
    parser.add_argument("--models-repo", required=True, help="Clone local de tensorflow/models")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUTS / model))
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    return parser


def _export_saved_model(repo: Path, output: Path, checkpoint_dir: Path) -> Path:
    exporter = repo / "research" / "object_detection" / "exporter_main_v2.py"
    if not exporter.is_file():
        raise FileNotFoundError(f"Exportador TensorFlow ausente: {exporter}")
    export_dir = output / "exported_model"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    run([
        __import__("sys").executable, str(exporter), "--input_type=image_tensor",
        f"--pipeline_config_path={output / 'pipeline.config'}",
        f"--trained_checkpoint_dir={checkpoint_dir}",
        f"--output_directory={export_dir}",
    ])
    saved_model = export_dir / "saved_model" / "saved_model.pb"
    if not saved_model.is_file():
        raise RuntimeError(f"SavedModel não foi gerado: {saved_model}")
    return saved_model.parent


def _official_model(model: str, cache: Path) -> Path:
    spec = MODELS[model]
    extracted = cache / spec["directory"]
    if extracted.is_dir():
        return extracted
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / spec["archive"]
    if not archive.exists():
        print(f"Baixando checkpoint oficial: {spec['url']}", flush=True)
        urllib.request.urlretrieve(spec["url"], archive)
    with tarfile.open(archive, "r:gz") as package:
        # Compatível com Python 3.8 e impede que o arquivo escreva fora do cache.
        cache_resolved = cache.resolve()
        for member in package.getmembers():
            target = (cache / member.name).resolve()
            try:
                target.relative_to(cache_resolved)
            except ValueError as exc:
                raise RuntimeError(f"Caminho inseguro no arquivo: {member.name}") from exc
        package.extractall(cache)
    return extracted


def _write_config(
    source: Path,
    destination: Path,
    records: Dict[str, Path],
    image_size: int,
    batch_size: int,
    steps: int,
    learning_rate_factor: float,
    gradient_clip_norm: float,
    pretrained_checkpoint: Path,
    eval_split: str,
) -> None:
    try:
        from object_detection.utils import config_util
    except ImportError as exc:
        raise RuntimeError("TensorFlow Object Detection API não está importável.") from exc

    configs = config_util.get_configs_from_pipeline_file(str(source))
    configs["model"].ssd.num_classes = 1
    fixed = configs["model"].ssd.image_resizer.fixed_shape_resizer
    fixed.height = image_size
    fixed.width = image_size
    train = configs["train_config"]
    original_batch_size = train.batch_size
    train.batch_size = batch_size
    train.num_steps = steps
    train.gradient_clipping_by_norm = gradient_clip_norm
    train.fine_tune_checkpoint = str(pretrained_checkpoint)
    train.fine_tune_checkpoint_type = "detection"
    # Preserva a receita oficial ao adaptar batch e duração: escala linear da
    # taxa de aprendizado e mantém as proporções do cosine decay/warmup.
    optimizer_name = train.optimizer.WhichOneof("optimizer")
    if optimizer_name:
        optimizer = getattr(train.optimizer, optimizer_name)
        learning_rate = optimizer.learning_rate
        schedule_name = learning_rate.WhichOneof("learning_rate")
        if schedule_name:
            schedule = getattr(learning_rate, schedule_name)
            batch_scale = batch_size / original_batch_size if original_batch_size else 1.0
            if hasattr(schedule, "learning_rate_base"):
                schedule.learning_rate_base *= batch_scale * learning_rate_factor
            if hasattr(schedule, "warmup_learning_rate"):
                schedule.warmup_learning_rate *= batch_scale * learning_rate_factor
            if hasattr(schedule, "total_steps"):
                original_total = schedule.total_steps
                if hasattr(schedule, "warmup_steps") and original_total:
                    warmup_ratio = schedule.warmup_steps / original_total
                    schedule.warmup_steps = max(1, round(steps * warmup_ratio))
                schedule.total_steps = steps
    train_input = configs["train_input_config"]
    train_input.label_map_path = str(records["label_map"])
    del train_input.tf_record_input_reader.input_path[:]
    train_input.tf_record_input_reader.input_path.append(str(records["train"]))
    eval_input = configs["eval_input_configs"][0]
    eval_input.label_map_path = str(records["label_map"])
    eval_input.shuffle = False
    eval_input.num_epochs = 1
    del eval_input.tf_record_input_reader.input_path[:]
    eval_input.tf_record_input_reader.input_path.append(str(records[eval_split]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    config_util.save_pipeline_config(
        config_util.create_pipeline_proto_from_configs(configs), str(destination.parent)
    )
    generated = destination.parent / "pipeline.config"
    if generated != destination:
        shutil.move(str(generated), str(destination))


def train(model: str, args: argparse.Namespace) -> None:
    dataset, output, repo = map(absolute, (args.dataset, args.output, args.models_repo))
    validate_dataset(dataset)
    if args.batch_size <= 0 or args.epochs <= 0 or args.checkpoint_every_epochs <= 0:
        raise ValueError("Batch, épocas e intervalo de checkpoint devem ser positivos.")
    if args.learning_rate_factor <= 0 or args.gradient_clip_norm < 0:
        raise ValueError("O fator de LR deve ser positivo e o clipping não pode ser negativo.")
    seed_everything(args.seed)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    train_images = sum(
        1 for path in (dataset / "train" / "images").iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    if not train_images:
        raise RuntimeError("O split de treino não contém imagens.")
    steps_per_epoch = math.ceil(train_images / args.batch_size)
    # O loop distribuído da OD API trabalha em blocos de 100 passos. Mantemos
    # um schedule suficientemente longo no diagnóstico e limitamos a execução
    # a 100 passos para não criar LR=0/NaN artificialmente.
    total_steps = 1000 if args.smoke_test else steps_per_epoch * args.epochs
    checkpoint_steps = 100 if args.smoke_test else steps_per_epoch * args.checkpoint_every_epochs
    entrypoint = repo / "research" / "object_detection" / "model_main_tf2.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"TensorFlow Models inválido; arquivo ausente: {entrypoint}")
    records = ensure_tfrecords(dataset, output / "records")
    official = _official_model(model, DEFAULT_OUTPUTS / "_pretrained")
    _write_config(
        official / "pipeline.config", output / "pipeline.config", records,
        args.image_size, args.batch_size, total_steps,
        args.learning_rate_factor, args.gradient_clip_norm,
        official / "checkpoint" / "ckpt-0", "valid",
    )
    write_json(output / "training_run.json", {
        "model": model,
        "dataset": str(dataset),
        "train_images": train_images,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "image_size": args.image_size,
        "seed": args.seed,
        "checkpoint_every_epochs": args.checkpoint_every_epochs,
        "learning_rate_factor": args.learning_rate_factor,
        "gradient_clip_norm": args.gradient_clip_norm,
    })
    train_command = [
        __import__("sys").executable, str(TF_ENTRYPOINT), str(entrypoint),
        f"--pipeline_config_path={output / 'pipeline.config'}",
        f"--model_dir={output / 'checkpoint'}",
        f"--checkpoint_every_n={checkpoint_steps}", "--alsologtostderr",
    ]
    if args.smoke_test:
        train_command.append("--num_train_steps=100")
    run(train_command)
    saved_model = _export_saved_model(repo, output, output / "checkpoint")
    write_json(output / "artifact.json", {
        "model": model,
        "framework": "tensorflow",
        "format": "SavedModel",
        "path": str(saved_model),
        "note": ".pt is a PyTorch format and is not valid for this TensorFlow model.",
    })


def test(model: str, args: argparse.Namespace) -> None:
    dataset, output, repo = map(absolute, (args.dataset, args.output, args.models_repo))
    checkpoint_dir = absolute(args.checkpoint_dir)
    validate_dataset(dataset)
    entrypoint = repo / "research" / "object_detection" / "model_main_tf2.py"
    trained_config = output / "pipeline.config"
    if not trained_config.is_file():
        raise FileNotFoundError(f"Configuração treinada ausente: {trained_config}")
    records = ensure_tfrecords(dataset, output / "records")
    # Mantém todos os hiperparâmetros treinados e troca apenas o leitor de avaliação.
    try:
        from object_detection.utils import config_util
    except ImportError as exc:
        raise RuntimeError("TensorFlow Object Detection API não está importável.") from exc
    configs = config_util.get_configs_from_pipeline_file(str(trained_config))
    eval_input = configs["eval_input_configs"][0]
    eval_input.label_map_path = str(records["label_map"])
    del eval_input.tf_record_input_reader.input_path[:]
    eval_input.tf_record_input_reader.input_path.append(str(records["test"]))
    test_dir = output / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    config_util.save_pipeline_config(
        config_util.create_pipeline_proto_from_configs(configs), str(test_dir)
    )
    run([
        __import__("sys").executable, str(TF_ENTRYPOINT), str(entrypoint),
        f"--pipeline_config_path={test_dir / 'pipeline.config'}",
        f"--model_dir={test_dir / 'metrics'}",
        f"--checkpoint_dir={checkpoint_dir}", "--alsologtostderr",
        "--eval_timeout=1",
    ])
    saved_model = output / "exported_model" / "saved_model"
    if not (saved_model / "saved_model.pb").is_file():
        saved_model = _export_saved_model(repo, output, checkpoint_dir)
    official_metrics = extract_tensorboard_metrics(test_dir / "metrics")
    counts = evaluate_tensorflow_saved_model(
        saved_model, dataset / "test" / "images",
        output / "records" / "normalized_labels" / "test",
        args.score_threshold, args.iou_threshold,
    )
    write_json(test_dir / "results.json", {
        "model": model,
        "split": "test",
        "official_coco_metrics": official_metrics,
        "threshold_counts": counts,
    })

