#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def pkg_version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


data = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "macos": platform.mac_ver()[0],
    "packages": {
        "torch": pkg_version("torch"),
        "torchvision": pkg_version("torchvision"),
        "transformers": pkg_version("transformers"),
        "mlx": pkg_version("mlx"),
        "numpy": pkg_version("numpy"),
    },
    "git": {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status": run(["git", "status", "--short"]),
    },
    "hardware": {
        "chip": run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "mem_bytes": run(["sysctl", "-n", "hw.memsize"]),
        "cpu_count": os.cpu_count(),
    },
}

try:
    import torch
    data["torch_runtime"] = {
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    if torch.backends.mps.is_available():
        for key, fn_name in [
            ("current_allocated_memory", "current_allocated_memory"),
            ("driver_allocated_memory", "driver_allocated_memory"),
            ("recommended_max_memory", "recommended_max_memory"),
        ]:
            fn = getattr(torch.mps, fn_name, None)
            if fn is not None:
                try:
                    data["torch_runtime"][key] = int(fn())
                except Exception as exc:
                    data["torch_runtime"][key] = f"<error: {exc}>"
except Exception as exc:
    data["torch_runtime"] = {"error": repr(exc)}

out_dir = Path("results")
out_dir.mkdir(exist_ok=True)
out = out_dir / "environment.json"
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
Path("results/pip-freeze.txt").write_text(run([sys.executable, "-m", "pip", "freeze"]) + "\n")

print(json.dumps(data, indent=2, ensure_ascii=False))
print(f"\nSaved: {out}")
print("Saved: results/pip-freeze.txt")

if platform.machine() != "arm64":
    print("\nWARNING: Python is not running natively as arm64. Do not benchmark under Rosetta.")
