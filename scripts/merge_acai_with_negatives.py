"""Cria um dataset YOLO de açaí com negativos de outras espécies de árvores.

As caixas de Banana-plant, Oil Palm Tree e coconut_tree são removidas. As
imagens selecionadas permanecem no dataset com um arquivo de rótulo vazio,
representando imagens negativas para a única classe de interesse: acai.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "valid", "test")
MULTI_CLASS_NAMES = {
    "0": "Banana-plant",
    "1": "Oil Palm Tree",
    "2": "coconut_tree",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Combina açaí com exatamente 1.000 imagens negativas."
    )
    parser.add_argument(
        "--acai",
        type=Path,
        default=project_root / "parts_dataset" / "acai",
        help="Raiz do dataset de açaí.",
    )
    parser.add_argument(
        "--multi-tree",
        type=Path,
        default=project_root / "parts_dataset" / "Multiple tree detection.v1i.yolov8",
        help="Raiz do dataset com outras espécies.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "dataset_merged",
        help="Destino do dataset combinado.",
    )
    parser.add_argument("--negative-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove o destino existente antes de gerar novamente.",
    )
    return parser.parse_args()


def image_files(directory: Path) -> List[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_root(root: Path) -> None:
    for split in SPLITS:
        for child in ("images", "labels"):
            path = root / split / child
            if not path.is_dir():
                raise FileNotFoundError(f"Diretório obrigatório ausente: {path}")


def label_for(image: Path, label_dir: Path) -> Path:
    label = label_dir / f"{image.stem}.txt"
    if not label.is_file():
        raise FileNotFoundError(f"Rótulo ausente para {image}: {label}")
    return label


def allocate_proportionally(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    """Alocação proporcional pelo método dos maiores restos."""
    weight_sum = sum(weights.values())
    if total <= 0 or weight_sum <= 0:
        raise ValueError("A quantidade de negativos e os pesos devem ser positivos.")
    exact = {split: total * weights[split] / weight_sum for split in SPLITS}
    allocation = {split: int(exact[split]) for split in SPLITS}
    remaining = total - sum(allocation.values())
    order = sorted(SPLITS, key=lambda split: (exact[split] - allocation[split]), reverse=True)
    for split in order[:remaining]:
        allocation[split] += 1
    return allocation


def original_classes(label: Path) -> List[str]:
    class_ids = set()
    for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] not in MULTI_CLASS_NAMES:
            raise ValueError(f"Anotação inválida em {label}:{line_number}")
        class_ids.add(parts[0])
    return [MULTI_CLASS_NAMES[class_id] for class_id in sorted(class_ids)]


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"O destino já existe: {output}. Use --overwrite para recriá-lo."
            )
        resolved = output.resolve()
        if resolved == resolved.parent or len(resolved.parts) < 3:
            raise RuntimeError(f"Recusa em remover destino inseguro: {resolved}")
        shutil.rmtree(resolved)
    for split in SPLITS:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_acai(acai: Path, output: Path, manifest: List[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in SPLITS:
        images = image_files(acai / split / "images")
        counts[split] = len(images)
        for image in images:
            label = label_for(image, acai / split / "labels")
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            for line_number, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5 or parts[0] != "0":
                    raise ValueError(f"Classe inesperada em {label}:{line_number}")
            final_stem = f"acai__{image.stem}"
            final_image = output / split / "images" / f"{final_stem}{image.suffix.lower()}"
            final_label = output / split / "labels" / f"{final_stem}.txt"
            shutil.copy2(image, final_image)
            shutil.copy2(label, final_label)
            manifest.append({
                "final_file": final_image.name,
                "split": split,
                "source_dataset": "acai",
                "source_file": image.name,
                "original_classes": "acai",
                "final_class": "acai",
                "annotation_count": str(len(lines)),
            })
    return counts


def select_negatives(
    multi_tree: Path,
    allocation: Dict[str, int],
    seed: int,
) -> Dict[str, Sequence[Tuple[Path, Path, List[str]]]]:
    selected: Dict[str, Sequence[Tuple[Path, Path, List[str]]]] = {}
    for split_index, split in enumerate(SPLITS):
        candidates: List[Tuple[Path, Path, List[str]]] = []
        for image in image_files(multi_tree / split / "images"):
            label = label_for(image, multi_tree / split / "labels")
            classes = original_classes(label)
            # Priorizamos negativos que realmente contenham uma das outras árvores.
            if classes:
                candidates.append((image, label, classes))
        required = allocation[split]
        if len(candidates) < required:
            raise ValueError(
                f"O split {split} possui {len(candidates)} negativos anotados; "
                f"são necessários {required}."
            )
        rng = random.Random(seed + split_index)
        rng.shuffle(candidates)
        selected[split] = sorted(candidates[:required], key=lambda item: item[0].name)
    return selected


def copy_negatives(
    selected: Dict[str, Sequence[Tuple[Path, Path, List[str]]]],
    output: Path,
    manifest: List[Dict[str, str]],
) -> None:
    for split in SPLITS:
        for image, label, classes in selected[split]:
            final_stem = f"negative__{image.stem}"
            final_image = output / split / "images" / f"{final_stem}{image.suffix.lower()}"
            final_label = output / split / "labels" / f"{final_stem}.txt"
            if final_image.exists() or final_label.exists():
                raise FileExistsError(f"Colisão de nome no destino: {final_stem}")
            shutil.copy2(image, final_image)
            final_label.write_text("", encoding="utf-8")
            manifest.append({
                "final_file": final_image.name,
                "split": split,
                "source_dataset": "multiple_tree_detection",
                "source_file": image.name,
                "original_classes": ";".join(classes),
                "final_class": "background",
                "annotation_count": "0",
            })


def write_metadata(
    output: Path,
    manifest: List[Dict[str, str]],
    positive_counts: Dict[str, int],
    allocation: Dict[str, int],
    seed: int,
) -> None:
    (output / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        "nc: 1\nnames:\n  0: acai\n",
        encoding="utf-8",
    )
    fields = (
        "final_file", "split", "source_dataset", "source_file",
        "original_classes", "final_class", "annotation_count",
    )
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "seed": seed,
        "class_names": ["acai"],
        "negative_images_total": sum(allocation.values()),
        "splits": {
            split: {
                "positive_images": positive_counts[split],
                "negative_images": allocation[split],
                "total_images": positive_counts[split] + allocation[split],
            }
            for split in SPLITS
        },
        "negative_policy": (
            "All annotations from the multiple-tree dataset are removed; "
            "the selected images are background-only examples."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "SOURCES.md").write_text(
        "# Fontes\n\n"
        "- `parts_dataset/acai`: acai0-2, licença CC BY 4.0.\n"
        "- `parts_dataset/Multiple tree detection.v1i.yolov8`: Multiple tree "
        "detection v1, licença CC BY 4.0.\n\n"
        "Consulte os arquivos README dos datasets originais para autoria, URLs e atribuição.\n",
        encoding="utf-8",
    )


def verify(output: Path, expected_negatives: int) -> None:
    negatives = 0
    for split in SPLITS:
        images = image_files(output / split / "images")
        labels = sorted((output / split / "labels").glob("*.txt"))
        if len(images) != len(labels):
            raise RuntimeError(f"Quantidade divergente em {split}: {len(images)} imagens, {len(labels)} rótulos")
        label_stems = {label.stem for label in labels}
        for image in images:
            if image.stem not in label_stems:
                raise RuntimeError(f"Rótulo final ausente: {image}")
        for label in labels:
            content = label.read_text(encoding="utf-8").strip()
            if label.stem.startswith("negative__"):
                negatives += 1
                if content:
                    raise RuntimeError(f"Negativo contém anotação: {label}")
            elif any(line.split()[0] != "0" for line in content.splitlines() if line.strip()):
                raise RuntimeError(f"Classe diferente de 0 no resultado: {label}")
    if negatives != expected_negatives:
        raise RuntimeError(f"Esperados {expected_negatives} negativos; encontrados {negatives}")


def main() -> None:
    args = parse_args()
    acai = args.acai.expanduser().resolve()
    multi_tree = args.multi_tree.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validate_root(acai)
    validate_root(multi_tree)

    positive_counts = {
        split: len(image_files(acai / split / "images")) for split in SPLITS
    }
    allocation = allocate_proportionally(args.negative_images, positive_counts)
    selected = select_negatives(multi_tree, allocation, args.seed)

    prepare_output(output, args.overwrite)
    manifest: List[Dict[str, str]] = []
    copied_counts = copy_acai(acai, output, manifest)
    copy_negatives(selected, output, manifest)
    write_metadata(output, manifest, copied_counts, allocation, args.seed)
    verify(output, args.negative_images)

    print(f"Dataset criado em: {output}")
    for split in SPLITS:
        print(
            f"{split}: {copied_counts[split]} positivas + "
            f"{allocation[split]} negativas = {copied_counts[split] + allocation[split]}"
        )
    print(f"Total de imagens negativas: {sum(allocation.values())}")


if __name__ == "__main__":
    main()

