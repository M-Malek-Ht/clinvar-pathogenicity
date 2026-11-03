"""Record the hardware and software environment to reports/environment.md.

Run from the repo root:  python scripts/env_report.py
Re-run whenever packages change, and commit the output with the results it describes.
"""
import datetime
import importlib
import platform
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "environment.md"


def run(cmd):
    """Run a shell command and return its output, or '' if it fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def cpu_name():
    if platform.system() == "Windows":
        name = run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if name:
            return name
    return platform.processor() or "unknown"


def gpu_driver():
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
               "--format=csv,noheader"])
    return out or "no NVIDIA GPU detected"


def version(pkg):
    try:
        return importlib.import_module(pkg).__version__
    except Exception:
        return "not installed"


def torch_rows():
    try:
        import torch
    except ImportError:
        return [("PyTorch", "not installed")]
    rows = [("PyTorch", torch.__version__),
            ("PyTorch CUDA build", str(torch.version.cuda)),
            ("CUDA available", str(torch.cuda.is_available()))]
    if torch.cuda.is_available():
        rows.append(("GPU (seen by PyTorch)", torch.cuda.get_device_name(0)))
        rows.append(("Compute capability",
                     ".".join(map(str, torch.cuda.get_device_capability(0)))))
        try:  # a real kernel launch; fails if this PyTorch build doesn't support the GPU
            x = torch.randn(1000, 1000, device="cuda")
            (x @ x).sum().item()
            rows.append(("GPU smoke test", "passed"))
        except Exception as e:
            rows.append(("GPU smoke test", f"FAILED: {e}"))
    return rows


def main():
    rows = [
        ("Report generated", datetime.datetime.now().isoformat(timespec="seconds")),
        ("Git commit", run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]) or "none"),
        ("OS", platform.platform()),
        ("CPU", cpu_name()),
        ("CPU cores (physical / logical)",
         f"{psutil.cpu_count(logical=False)} / {psutil.cpu_count(logical=True)}"),
        ("RAM", f"{psutil.virtual_memory().total / 1024**3:.1f} GB"),
        ("GPU (name, memory, driver)", gpu_driver()),
        ("Python", sys.version.split()[0]),
    ]
    rows += [(p, version(p)) for p in ["numpy", "pandas", "sklearn", "matplotlib", "pyarrow"]]
    rows += torch_rows()

    lines = ["# Computing environment", "", "| Item | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
