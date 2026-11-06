from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project import DEFAULT_DATASET, DEFAULT_OUTPUTS, absolute, run, seed_everything, validate_dataset
from common.results import write_json
from common.yolo import prepare_nanodet_layout
from nanodet_plus.config import write_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina NanoDet-Plus para detectar açaí.")
    parser.add_argument("--nanodet-repo", required=True)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUTS / "nanodet_plus"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--pretrained", help="Checkpoint COCO opcional")
    args = parser.parse_args()
    dataset, output, repo = map(absolute, (args.dataset, args.output, args.nanodet_repo))
    validate_dataset(dataset)
    if args.batch_size <= 0 or args.epochs <= 0 or args.validation_interval <= 0:
        raise ValueError("Batch, épocas e intervalo de validação devem ser positivos.")
    seed_everything(args.seed)
    os.environ.setdefault("TORCH_HOME", str(DEFAULT_OUTPUTS / "_torch_cache"))
    entrypoint = repo / "tools" / "train.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Repositório NanoDet inválido: {entrypoint}")
    config = output / "train.yml"
    prepared = prepare_nanodet_layout(dataset, output / "nanodet_dataset")
    write_config(
        config, prepared["train"], prepared["train"],
        prepared["valid"], prepared["valid"], output,
        args.image_size, args.batch_size, args.epochs, args.workers,
        args.learning_rate, args.validation_interval,
        absolute(args.pretrained) if args.pretrained else None,
    )
    train_images = sum(1 for path in (dataset / "train/images").iterdir() if path.is_file())
    write_json(output / "training_run.json", {
        "model": "nanodet_plus",
        "dataset": str(dataset),
        "train_images": train_images,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "validation_interval_epochs": args.validation_interval,
        "learning_rate": args.learning_rate,
    })
    run([sys.executable, str(entrypoint), str(config)], cwd=repo)
    candidates = list(output.rglob("nanodet_model_best.pth"))
    if not candidates:
        raise RuntimeError("NanoDet terminou sem gerar nanodet_model_best.pth")
    best_weights = max(candidates, key=lambda path: path.stat().st_mtime)
    # .pth e .pt são extensões usuais para a mesma serialização torch.save.
    shutil.copy2(best_weights, output / "nanodet_plus_best.pt")


if __name__ == "__main__":
    main()

