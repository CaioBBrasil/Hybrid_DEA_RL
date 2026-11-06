from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command(*parts: str) -> str | None:
    try:
        return subprocess.run(parts, check=False, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("results/system.json"))
args = parser.parse_args()
payload = {
    "captured_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "cpu_count": os.cpu_count(),
    "model": command("cat", "/proc/device-tree/model"),
    "kernel": command("uname", "-a"),
    "cpu": command("lscpu"),
    "memory": command("free", "-h"),
    "temperature": command("vcgencmd", "measure_temp"),
    "throttled": command("vcgencmd", "get_throttled"),
    "governor": command("cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(args.output)

