from __future__ import annotations

from pathlib import Path
from typing import Dict
import os
import shutil


def prepare_clipped_labels(dataset: Path, destination: Path) -> Dict[str, Path]:
    """Valida YOLO e cria uma cópia com caixas limitadas à área da imagem."""
    result: Dict[str, Path] = {}
    clipped = 0
    for split in ("train", "valid", "test"):
        source = dataset / split / "labels"
        target = destination / split
        target.mkdir(parents=True, exist_ok=True)
        result[split] = target
        for label_path in sorted(source.glob("*.txt")):
            output_lines = []
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5 or parts[0] != "0":
                    raise ValueError(f"Rótulo YOLO inválido em {label_path}:{line_number}")
                xc, yc, bw, bh = map(float, parts[1:])
                if not all(0.0 <= value <= 1.0 for value in (xc, yc, bw, bh)):
                    raise ValueError(f"Valor YOLO fora de [0,1] em {label_path}:{line_number}")
                original = (xc - bw / 2, xc + bw / 2, yc - bh / 2, yc + bh / 2)
                left, right, top, bottom = (
                    max(0.0, original[0]), min(1.0, original[1]),
                    max(0.0, original[2]), min(1.0, original[3]),
                )
                if (left, right, top, bottom) != original:
                    clipped += 1
                if right <= left or bottom <= top:
                    raise ValueError(f"Caixa sem área em {label_path}:{line_number}")
                new_x, new_y = (left + right) / 2, (top + bottom) / 2
                new_w, new_h = right - left, bottom - top
                output_lines.append(f"0 {new_x:.12g} {new_y:.12g} {new_w:.12g} {new_h:.12g}")
            (target / label_path.name).write_text(
                "\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8"
            )
    if clipped:
        print(f"Aviso: {clipped} caixas que cruzavam a borda foram recortadas na cópia de trabalho.")
    return result


def prepare_nanodet_layout(dataset: Path, destination: Path) -> Dict[str, Path]:
    """Cria pastas com imagem e TXT juntos, como exige YoloDataset do NanoDet."""
    label_dirs = prepare_clipped_labels(dataset, destination / "labels")
    result: Dict[str, Path] = {}
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    for split in ("train", "valid", "test"):
        target = destination / split
        target.mkdir(parents=True, exist_ok=True)
        result[split] = target
        for label in label_dirs[split].glob("*.txt"):
            shutil.copy2(label, target / label.name)
        for image in sorted((dataset / split / "images").iterdir()):
            if not image.is_file() or image.suffix.lower() not in extensions:
                continue
            final = target / image.name
            if final.exists():
                continue
            try:
                os.link(image, final)
            except OSError:
                shutil.copy2(image, final)
    return result
