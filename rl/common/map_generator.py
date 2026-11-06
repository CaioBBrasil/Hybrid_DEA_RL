from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Cell:
    image_id: str
    image_path: str
    label_path: str
    ground_truth_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _annotation_count(label: Path) -> int:
    return sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())


@lru_cache(maxsize=None)
def load_cells(dataset: Path, split: str) -> tuple[Cell, ...]:
    """Carrega metadados uma vez por split; as células são imutáveis."""
    image_dir, label_dir = dataset / split / "images", dataset / split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Split ausente ou incompleto: {dataset / split}")
    cells: list[Cell] = []
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Anotação ausente: {label}")
        cells.append(Cell(image.stem, str(image.resolve()), str(label.resolve()), _annotation_count(label)))
    if not cells:
        raise ValueError(f"Nenhuma imagem em {image_dir}")
    return tuple(cells)


def balanced_order(cells: Sequence[Cell], seed: int) -> list[Cell]:
    """Espalha positivos/negativos por blocos aleatórios, sem alternância rígida."""
    rng = np.random.default_rng(seed)
    positives = [c for c in cells if c.ground_truth_count > 0]
    negatives = [c for c in cells if c.ground_truth_count == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    total = len(cells)
    # Intercala por quantis: mantém a proporção local e adiciona jitter reproduzível.
    decorated: list[tuple[float, Cell]] = []
    for group in (positives, negatives):
        for index, cell in enumerate(group):
            base = (index + 0.5) / max(1, len(group))
            decorated.append((base + float(rng.uniform(-0.35, 0.35)) / total, cell))
    decorated.sort(key=lambda item: item[0])
    ordered = [cell for _, cell in decorated]
    if len({cell.image_id for cell in ordered}) != len(cells):
        raise RuntimeError("O mapa contém imagens repetidas.")
    return ordered


def generate_map(dataset: Path, split: str, seed: int, cells_per_episode: int | None = None) -> list[Cell]:
    ordered = balanced_order(load_cells(dataset, split), seed)
    if cells_per_episode is None:
        return ordered
    if cells_per_episode <= 0:
        raise ValueError("cells_per_episode must be positive.")
    if cells_per_episode > len(ordered):
        raise ValueError(
            f"cells_per_episode={cells_per_episode} exceeds split {split} size={len(ordered)}"
        )
    return ordered[:cells_per_episode]


def generate_synthetic_map(count: int, split: str, seed: int) -> list[Cell]:
    """Gera células abstratas sem imagens, uma ocorrência-alvo por célula."""
    if count <= 0:
        raise ValueError("dataset.cells_per_episode deve ser positivo.")
    return [Cell(f"{split}:{seed}:{index}", "", "", 1) for index in range(count)]
