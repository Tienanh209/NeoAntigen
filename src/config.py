"""Paths shared by notebooks and scripts. Runs unchanged on Colab and locally.

Colab: code in /content/NeoAntigen (git clone), data + checkpoints on Drive.
Local: data/ and checkpoints/ inside the repo (both git-ignored).
Override with the env vars NEOANTIGEN_DRIVE, DATA_DIR, CKPT_DIR, WORK_DIR.
"""
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
try:
    ON_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:   # no `google` package at all (local machine)
    ON_COLAB = False

DRIVE_DIR = Path(os.environ.get("NEOANTIGEN_DRIVE", "/content/drive/MyDrive/s1156052/NeoAntigen"))
DATA_DIR = Path(os.environ.get("DATA_DIR", DRIVE_DIR / "data" if ON_COLAB else REPO_DIR / "data"))
CKPT_DIR = Path(os.environ.get("CKPT_DIR", DRIVE_DIR / "ckpt" if ON_COLAB else REPO_DIR / "checkpoints"))
# Scratch space (CD-HIT files, staged data) lives on the fast local disk, never on Drive
WORK_DIR = Path(os.environ.get("WORK_DIR", "/content/work" if ON_COLAB else REPO_DIR / "work"))
STAGED_DATA_DIR = Path("/content/data") if ON_COLAB else DATA_DIR

EMB_DIR = DATA_DIR / "emb"          # PLM embedding caches, one subdir per dataset build
RESULTS_DIR = CKPT_DIR / "results"  # experiment runs, predictions, figures


def git_sha():
    """Short commit of the checked-out code, recorded with every run. '' if git is unavailable."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def file_sha256(path, chunk=1 << 20):
    """SHA-256 of a file: identifies the exact snapshot / manifest a result was built from."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tool_versions():
    """Versions of what shapes a dataset (recorded in every meta.json). '' when a tool is not installed."""
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = ""
    for exe, args in (("cd-hit", ["-h"]), ("mmseqs", ["version"])):
        path = shutil.which(exe)
        try:
            r = subprocess.run([path, *args], capture_output=True, text=True, timeout=20) if path else None
            lines = (r.stdout or r.stderr).strip().splitlines() if r else []
            out[exe] = lines[0].strip("= ") if lines else ""
        except (OSError, subprocess.SubprocessError):
            out[exe] = ""
    return out


def stage_data():
    """Copy DATA_DIR from Drive to the VM disk before training (much faster reads). Returns the path to read from."""
    if STAGED_DATA_DIR != DATA_DIR:
        shutil.copytree(DATA_DIR, STAGED_DATA_DIR, dirs_exist_ok=True)
    return STAGED_DATA_DIR
