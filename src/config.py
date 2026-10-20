"""Paths shared by notebooks and scripts. Runs unchanged on Colab and locally.

Colab: code in /content/NeoAntigen (git clone), data + checkpoints on Drive.
Local: data/ and checkpoints/ inside the repo (both git-ignored).
Override with the env vars NEOANTIGEN_DRIVE, DATA_DIR, CKPT_DIR, WORK_DIR.
"""
import importlib.util
import os
import shutil
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


def stage_data():
    """Copy DATA_DIR from Drive to the VM disk before training (much faster reads). Returns the path to read from."""
    if STAGED_DATA_DIR != DATA_DIR:
        shutil.copytree(DATA_DIR, STAGED_DATA_DIR, dirs_exist_ok=True)
    return STAGED_DATA_DIR
