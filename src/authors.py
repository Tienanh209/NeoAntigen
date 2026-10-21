"""Load the DeepNeoAG authors' public dataset (https://github.com/B1607/DeepNeoAG).

The repo ships one FASTA file per peptide under fasta/{Train,Test}/{pos,neg}/, i.e.
the already-filtered, already-split dataset of the paper's Table 1 (7,012 peptides:
241 + 5,367 train, 61 + 1,343 test). Loading it lets the experiment suite run on
exactly the authors' data instead of a rebuild from CEDAR.

The download is pinned to a commit so the dataset can never change under a run.
The output follows the same contract as the CEDAR build (uid, sequence, label,
split, length, mhc_classes), so every notebook and `src/runner.py` work unchanged.
"""
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from . import config, data

REPO = "B1607/DeepNeoAG"
REF = "0bebb4ff3fa2bc9ba05e51a18771a14ac8b1c897"   # pinned commit of the authors' repo
DATASET_NAME = "authors_dataset.csv"
_PATH_RE = re.compile(r"fasta/(Train|Test)/(pos|neg)/([^/]+)\.fasta$")


def _download(ref, dest):
    url = f"https://codeload.github.com/{REPO}/zip/{ref}"
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    Path(dest).write_bytes(r.content)


def parse_zip(zip_path):
    """One row per FASTA file. The peptide id comes from the file name, which the
    FASTA header repeats (checked), and is unique across the whole repo."""
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            m = _PATH_RE.search(name)
            if not m:
                continue
            split, cls, pid = m.group(1).lower(), m.group(2), m.group(3)
            lines = z.read(name).decode().strip().splitlines()
            header, seq = lines[0], "".join(lines[1:]).strip()
            if header.lstrip(">") != pid:
                raise ValueError(f"{name}: header {header!r} does not match file name")
            rows.append({"uid": pid, "structure_id": pid, "sequence": seq,
                         "label": int(cls == "pos"), "split": split})
    df = pd.DataFrame(rows).sort_values("uid").reset_index(drop=True)
    df["length"] = df["sequence"].str.len()
    # The authors' repo carries no MHC annotation, so leave it empty; cedar.mhc_class
    # then falls back to its length rule.
    df["mhc_classes"] = float("nan")
    df["source"] = "authors"
    return df


def build_dataset(data_dir=None, ref=REF, force=False, zip_path=None):
    """Download the repo at `ref`, write authors_dataset.csv + authors_meta.json, return the frame."""
    data_dir = Path(data_dir or config.DATA_DIR)
    out = data_dir / DATASET_NAME
    if out.exists() and not force:
        return data.load_dataset(data_dir, DATASET_NAME)

    data_dir.mkdir(parents=True, exist_ok=True)
    if zip_path is None:
        work = Path(config.WORK_DIR)
        work.mkdir(parents=True, exist_ok=True)
        zip_path = work / f"DeepNeoAG-{ref[:10]}.zip"
        if not zip_path.exists():
            _download(ref, zip_path)
    df = parse_zip(zip_path)

    std = df["sequence"].str.fullmatch(data.cedar.STD_AA)
    assert std.all(), f"{(~std).sum()} sequences contain non-standard residues"
    assert df["uid"].is_unique and df["sequence"].is_unique

    df.to_csv(out, index=False)
    meta = {"source": f"github.com/{REPO}", "ref": ref, "n": len(df),
            "counts": {f"{s}/{l}": int(n) for (s, l), n in df.groupby(["split", "label"]).size().items()},
            "length_min": int(df["length"].min()), "length_max": int(df["length"].max())}
    (data_dir / "authors_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"authors dataset: {len(df)} peptides -> {out}")
    return data.load_dataset(data_dir, DATASET_NAME)


def load(data_dir=None, **kw):
    """The authors' dataset, downloading it on first use."""
    return build_dataset(data_dir, **kw)
