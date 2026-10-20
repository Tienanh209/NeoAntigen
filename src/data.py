"""Dataset loading, splits, cross-validation folds and the non-PLM encodings.

Every feature source (one-hot, PLM cache) returns the same pair
`(features [N, D, L] float32, lengths [N] int64)` so the model never needs to
know where its input came from.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import cedar, config

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}
REQUIRED_COLS = ("uid", "sequence", "label", "split")


def load_dataset(data_dir=None, name="deepneoag_like_dataset.csv"):
    """The built dataset, with its contract checked up front."""
    path = Path(data_dir or config.DATA_DIR) / name
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; rebuild with load_datasets.ipynb")
    if "length" not in df.columns:
        df["length"] = df["sequence"].str.len()
    return df.reset_index(drop=True)


def build_id(df):
    """Short hash of the dataset *content*, used to key embedding caches and runs.

    Derived from the rows rather than from meta.json, so the strict and relaxed
    builds get different ids automatically, any change to filtering or splitting
    invalidates the cache, and a missing or hand-edited meta.json breaks nothing.
    """
    rows = df[["uid", "sequence", "label", "split"]].sort_values("uid")
    payload = "\n".join(f"{r.uid}\t{r.sequence}\t{r.label}\t{r.split}" for r in rows.itertuples())
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def meta(data_dir=None):
    path = Path(data_dir or config.DATA_DIR) / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def max_len(df, data_dir=None):
    """Padded length for the convolution. Prefers the build's MAX_LEN so the
    cache and the model agree even if the longest peptide was filtered out."""
    m = meta(data_dir).get("MAX_LEN")
    return int(m) if m else int(df["length"].max())


def splits(df):
    """(train, test) by the `split` column. The test frame is touched only at final predict."""
    return (df[df["split"] == "train"].reset_index(drop=True),
            df[df["split"] == "test"].reset_index(drop=True))


def cv_folds(train_df, n_splits=5, seed=42):
    """Stratified folds over the training split only."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(train_df)), train_df["label"]))


def group_cv_folds(train_df, groups, n_splits=5, seed=42):
    """Folds that keep similar sequences together, so near-duplicates cannot
    straddle the train/validation boundary."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(train_df)), train_df["label"], groups=groups))


def kmer_groups(df, k=8):
    """Cluster id per row: two peptides share a cluster when they share a k-mer
    (transitively, via union-find).

    Used instead of CD-HIT for the leakage experiment. CD-HIT cannot go below 0.4
    identity and its word size of 2 makes it close to meaningless on 9-mers,
    whereas shared-k-mer grouping is well defined at every peptide length.
    Sequences shorter than k are keyed by the whole sequence, so identical short
    peptides still group together.
    """
    parent = list(range(len(df)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first = {}
    for i, seq in enumerate(df["sequence"]):
        keys = {seq[j:j + k] for j in range(len(seq) - k + 1)} if len(seq) >= k else {seq}
        for key in keys:
            if key in first:
                union(first[key], i)
            else:
                first[key] = i
    return pd.Series([find(i) for i in range(len(df))], index=df.index, name="group")


def add_mhc_class(df):
    """Adds an `mhc` column ('I'/'II') via the existing CEDAR helper."""
    return df.assign(mhc=df.apply(cedar.mhc_class, axis=1))


def aac_features(seqs):
    """Amino-acid composition, [N, 20] fractions. The order-free baseline."""
    out = np.zeros((len(seqs), len(AA)), dtype=np.float32)
    for i, s in enumerate(seqs):
        for ch in s:
            idx = AA_INDEX.get(ch)
            if idx is not None:
                out[i, idx] += 1
        if len(s):
            out[i] /= len(s)
    return out


def onehot(seqs, max_len):
    """One-hot encoding as [N, 20, L] float32 plus true lengths [N].

    Same shape contract as the PLM cache, so the identical model class serves the
    one-hot CNN / mCNN baselines.
    """
    seqs = list(seqs)
    x = np.zeros((len(seqs), len(AA), max_len), dtype=np.float32)
    lengths = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:max_len]
        lengths[i] = len(s)
        for j, ch in enumerate(s):
            idx = AA_INDEX.get(ch)
            if idx is not None:
                x[i, idx, j] = 1.0
    return x, lengths
