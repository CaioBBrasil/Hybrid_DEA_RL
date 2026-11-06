from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project import DEFAULT_DATASET, DEFAULT_OUTPUTS, absolute, run, validate_dataset
from common.results import evaluate_nanodet_results, parse_nanodet_metrics, write_json
from common.yolo import prepare_nanodet_layout
from nanodet_plus.config import write_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia NanoDet-Plus no teste.")
    parser.add_argument("--nanodet-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUTS / "nanodet_plus"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    args = parser.parse_args()
    dataset, output, repo = map(absolute, (args.dataset, args.output, args.nanodet_repo))
    checkpoint = absolute(args.checkpoint)
    validate_dataset(dataset)
    entrypoint = repo / "tools" / "test.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Repositório NanoDet inválido: {entrypoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint ausente: {checkpoint}")
    config = output / f"{args.split}.yml"
    prepared = prepare_nanodet_layout(dataset, output / "nanodet_dataset")
    selected = prepared[args.split]
    result_dir = output / args.split
    write_config(
        config, prepared["train"], prepared["train"],
        selected, selected, result_dir,
        args.image_size, args.batch_size, 1, args.workers, 0.001, 1,
    )
    run([
        sys.executable, str(entrypoint), "--task", "val", "--config", str(config),
        "--model", str(checkpoint),
    ], cwd=repo)
    eval_files = list(result_dir.rglob("eval_results.txt"))
    result_files = [path for path in result_dir.rglob("results.json")]
    if not eval_files or not result_files:
        raise RuntimeError("O avaliador NanoDet não gerou eval_results.txt/results.json")
    eval_file = max(eval_files, key=lambda path: path.stat().st_mtime)
    predictions_file = max(result_files, key=lambda path: path.stat().st_mtime)
    counts = evaluate_nanodet_results(
        predictions_file, selected, args.score_threshold, args.iou_threshold
    )
    write_json(result_dir / "summary.json", {
        "model": "nanodet_plus",
        "split": args.split,
        "official_coco_metrics": parse_nanodet_metrics(eval_file),
        "threshold_counts": counts,
        "predictions_file": str(predictions_file),
    })


if __name__ == "__main__":
    main()

