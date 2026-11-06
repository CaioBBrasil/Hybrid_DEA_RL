from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from .yolo import prepare_clipped_labels

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _image_files(directory: Path) -> List[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def ensure_tfrecords(dataset: Path, destination: Path) -> Dict[str, Path]:
    """Converte os três splits YOLO para TFRecord, se ainda não existirem."""
    try:
        import tensorflow as tf
        from PIL import Image
        from object_detection.utils import dataset_util
    except ImportError as exc:
        raise RuntimeError(
            "Instale TensorFlow, Pillow e a TensorFlow Object Detection API."
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    label_dirs = prepare_clipped_labels(dataset, destination / "normalized_labels")
    label_map = destination / "label_map.pbtxt"
    label_map.write_text("item {\n  id: 1\n  name: 'acai'\n}\n", encoding="utf-8")
    records: Dict[str, Path] = {}

    for split in ("train", "valid", "test"):
        output = destination / f"{split}.tfrecord"
        records[split] = output
        if output.exists() and output.stat().st_size > 0:
            continue
        image_dir = dataset / split / "images"
        label_dir = label_dirs[split]
        with tf.io.TFRecordWriter(str(output)) as writer:
            for image_path in _image_files(image_dir):
                encoded = image_path.read_bytes()
                with Image.open(image_path) as image:
                    width, height = image.size
                    image_format = image.format.lower().encode("ascii")
                xmin: List[float] = []
                xmax: List[float] = []
                ymin: List[float] = []
                ymax: List[float] = []
                classes: List[int] = []
                names: List[bytes] = []
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    raise FileNotFoundError(f"Rótulo ausente: {label_path}")
                for line_number, line in enumerate(
                    label_path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 5 or parts[0] != "0":
                        raise ValueError(f"Rótulo YOLO inválido em {label_path}:{line_number}")
                    _, xc, yc, bw, bh = map(float, parts)
                    left, right = xc - bw / 2, xc + bw / 2
                    top, bottom = yc - bh / 2, yc + bh / 2
                    xmin.append(left)
                    xmax.append(right)
                    ymin.append(top)
                    ymax.append(bottom)
                    classes.append(1)  # TF Object Detection usa IDs a partir de 1.
                    names.append(b"acai")

                features = {
                    "image/height": dataset_util.int64_feature(height),
                    "image/width": dataset_util.int64_feature(width),
                    "image/filename": dataset_util.bytes_feature(image_path.name.encode()),
                    "image/source_id": dataset_util.bytes_feature(image_path.stem.encode()),
                    "image/key/sha256": dataset_util.bytes_feature(hashlib.sha256(encoded).hexdigest().encode()),
                    "image/encoded": dataset_util.bytes_feature(encoded),
                    "image/format": dataset_util.bytes_feature(image_format),
                    "image/object/bbox/xmin": dataset_util.float_list_feature(xmin),
                    "image/object/bbox/xmax": dataset_util.float_list_feature(xmax),
                    "image/object/bbox/ymin": dataset_util.float_list_feature(ymin),
                    "image/object/bbox/ymax": dataset_util.float_list_feature(ymax),
                    "image/object/class/text": dataset_util.bytes_list_feature(names),
                    "image/object/class/label": dataset_util.int64_list_feature(classes),
                }
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
    return {**records, "label_map": label_map}

