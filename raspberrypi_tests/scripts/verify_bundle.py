from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
failed = False
for model in manifest["models"]:
    path = ROOT / model["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
    ok = digest == model["sha256"]
    print(f"{'OK' if ok else 'ERRO'} {model['name']}: {path}")
    failed |= not ok
if failed:
    raise SystemExit("Bundle inválido: modelo ausente ou hash diferente.")
print("Todos os modelos passaram na verificação SHA-256.")
