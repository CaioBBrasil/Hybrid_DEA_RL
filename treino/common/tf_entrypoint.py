"""Executa um entrypoint TensorFlow após aplicar compatibilidade com tf-slim 1.1."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Uso: tf_entrypoint.py CAMINHO_DO_SCRIPT [argumentos]")

    # tf-slim 1.1 usa este símbolo interno, movido no TensorFlow 2.15.
    from tensorflow.python.ops import cond
    from tensorflow.python.ops import control_flow_case
    from tensorflow.python.ops import control_flow_ops

    if not hasattr(control_flow_ops, "case"):
        control_flow_ops.case = control_flow_case.case
    if not hasattr(control_flow_ops, "cond"):
        control_flow_ops.cond = cond.cond

    entrypoint = Path(sys.argv[1]).resolve()
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
