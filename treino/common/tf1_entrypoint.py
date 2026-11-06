"""Executa pipelines legados da Detection API sobre tensorflow.compat.v1."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Uso: tf1_entrypoint.py CAMINHO_DO_SCRIPT [argumentos]")
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from object_detection.utils import tf_version
    # Alguns extratores oficiais (incluindo MobileDet) continuam marcados como
    # TF1-only. O processo é isolado para não afetar os treinadores TF2.
    tf_version.is_tf1 = lambda: True
    tf_version.is_tf2 = lambda: False
    entrypoint = Path(sys.argv[1]).resolve()
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
