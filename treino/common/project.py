from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
DEFAULT_DATASET = PROJECT_ROOT / "dataset_merged"
DEFAULT_OUTPUTS = TRAINING_ROOT / "outputs"


def absolute(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def validate_dataset(root: Path) -> None:
    for split in ("train", "valid", "test"):
        for kind in ("images", "labels"):
            directory = root / split / kind
            if not directory.is_dir():
                raise FileNotFoundError(f"Diretório ausente: {directory}")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def run(command: Sequence[str], cwd: Optional[Path] = None) -> None:
    print("Executando:", subprocess.list2cmdline(list(command)), flush=True)
    subprocess.run(list(command), cwd=cwd, check=True)


def add_repo_to_pythonpath(repo: Path) -> Dict[str, str]:
    env = os.environ.copy()
    research = repo / "research"
    slim = research / "slim"
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(research), str(slim)] + ([current] if current else [])
    )
    return env


def python_executable() -> str:
    return sys.executable

