"""Per-residue protein-language-model embeddings and their on-disk cache.

The PLMs are frozen, so every embedding is computed exactly once and every
experiment reads the cache. That is what makes dozens of runs on a flaky Colab
session practical, and it lets most of the analysis run on CPU.

Cache layout, one directory per (dataset build, PLM):

    DATA_DIR/emb/{build_id}/{plm}/
        vectors.npy     [total_residues, D] float16, residues concatenated
        index.csv       uid, offset, length
        manifest.json   written LAST, so its presence means the cache is complete

Residues are stored ragged rather than padded to a fixed length: it is ~2x
smaller and it does not bake MAX_LEN into the cache. Padding happens at batch time.
"""
import datetime
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# TAPE is deliberately absent: tape_proteins pins torch<1.6 and cannot be installed
# alongside Colab's torch without breaking it.
PLMS = {
    "prott5":    {"hf": "Rostlab/prot_t5_xl_half_uniref50-enc", "kind": "t5",  "dim": 1024},
    "esm2_650m": {"hf": "facebook/esm2_t33_650M_UR50D",         "kind": "esm", "dim": 1280},
    "esm1b":     {"hf": "facebook/esm1b_t33_650M_UR50S",        "kind": "esm", "dim": 1280},
    # Tiny checkpoint for validating the pipeline end to end in seconds.
    "esm2_8m":   {"hf": "facebook/esm2_t6_8M_UR50D",            "kind": "esm", "dim": 320},
}


def cache_dir(plm, build_id, data_dir=None):
    base = Path(data_dir) / "emb" if data_dir else config.EMB_DIR
    return base / build_id / plm


def is_cached(plm, build_id, data_dir=None):
    return (cache_dir(plm, build_id, data_dir) / "manifest.json").exists()


def _load_model(plm, device):
    import torch
    from transformers import AutoTokenizer, AutoModel, T5EncoderModel, T5Tokenizer

    spec = PLMS[plm]
    if spec["kind"] == "t5":
        tok = T5Tokenizer.from_pretrained(spec["hf"], do_lower_case=False, legacy=True)
        model = T5EncoderModel.from_pretrained(spec["hf"])
        if device.type == "cuda":
            model = model.half()     # the -half- checkpoint; fp16 is GPU-only
    else:
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        model = AutoModel.from_pretrained(spec["hf"])
    return tok, model.to(device).eval()


def _encode(plm, tok, model, seqs, device):
    """Per-residue embeddings for a batch. Returns a list of [len(seq), D] arrays.

    Special tokens are stripped here. Getting this wrong shifts every feature by
    one residue and still produces plausible-looking results, so the caller asserts
    the shape of every sequence.
    """
    import torch

    spec = PLMS[plm]
    if spec["kind"] == "t5":
        # ProtTrans convention: residues space-separated, rare residues -> X.
        text = [" ".join(s.replace("U", "X").replace("Z", "X").replace("O", "X")) for s in seqs]
        enc = tok(text, add_special_tokens=True, padding=True, return_tensors="pt").to(device)
    else:
        enc = tok(list(seqs), padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        h = model(**enc).last_hidden_state.float().cpu().numpy()

    out = []
    for i, s in enumerate(seqs):
        # T5 appends </s> and has no BOS; ESM wraps the sequence in CLS ... EOS.
        out.append(h[i, 0:len(s)] if spec["kind"] == "t5" else h[i, 1:len(s) + 1])
    return out


def build_cache(plm, df, data_dir=None, build_id=None, batch_size=64, shard=2048,
                device=None, force=False, verbose=True):
    """Embed every sequence in `df` and write the cache. Resumes from finished shards."""
    import torch

    from . import data as data_mod

    if plm not in PLMS:
        raise ValueError(f"unknown PLM {plm!r}; known: {sorted(PLMS)}")
    build_id = build_id or data_mod.build_id(df)
    out = cache_dir(plm, build_id, data_dir)
    if is_cached(plm, build_id, data_dir) and not force:
        if verbose:
            print(f"[{plm}] already cached at {out}")
        return out

    # Build on fast local scratch, then publish to Drive in one copy.
    scratch = Path(config.WORK_DIR) / "emb" / build_id / plm
    scratch.mkdir(parents=True, exist_ok=True)
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    rows = df[["uid", "sequence"]].drop_duplicates("uid").reset_index(drop=True)
    shards = [(i, min(i + shard, len(rows))) for i in range(0, len(rows), shard)]
    tok = model = None
    loaded = False

    for si, (lo, hi) in enumerate(shards):
        vec_path, idx_path = scratch / f"part{si:03d}.npy", scratch / f"part{si:03d}.csv"
        if vec_path.exists() and idx_path.exists() and not force:
            if verbose:
                print(f"[{plm}] shard {si+1}/{len(shards)} already done")
            continue
        if not loaded:
            if verbose:
                print(f"[{plm}] loading {PLMS[plm]['hf']} on {device}")
            tok, model = _load_model(plm, device)
            loaded = True

        chunk, vectors = rows.iloc[lo:hi], []
        for b in range(0, len(chunk), batch_size):
            seqs = chunk["sequence"].iloc[b:b + batch_size].tolist()
            embs = _encode(plm, tok, model, seqs, device)
            for s, e in zip(seqs, embs):
                # An off-by-one in special-token stripping is silent otherwise.
                assert e.shape == (len(s), PLMS[plm]["dim"]), \
                    f"{plm}: got {e.shape} for a {len(s)}-residue peptide"
                vectors.append(e.astype(np.float16))
            if verbose and (b // batch_size) % 10 == 0:
                print(f"[{plm}] shard {si+1}/{len(shards)}  {b + len(seqs)}/{len(chunk)}")

        lengths = [len(v) for v in vectors]
        tmp = scratch / f"part{si:03d}.tmp.npy"        # np.save appends .npy, so name it that way
        np.save(tmp, np.concatenate(vectors, axis=0))
        tmp.replace(vec_path)
        pd.DataFrame({"uid": chunk["uid"].to_numpy(), "length": lengths}).to_csv(idx_path, index=False)

    _finalize(plm, build_id, scratch, out, len(rows), verbose)
    return out


def _finalize(plm, build_id, scratch, out, n_seqs, verbose=True):
    """Concatenate the shards, compute offsets, publish to the cache directory."""
    parts = sorted(scratch.glob("part*.npy"))
    vectors = [np.load(p) for p in parts]
    index = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in parts], ignore_index=True)
    index["offset"] = np.concatenate([[0], np.cumsum(index["length"].to_numpy())[:-1]]).astype(np.int64)

    out.mkdir(parents=True, exist_ok=True)
    allv = np.concatenate(vectors, axis=0)
    tmp = out / "vectors.tmp.npy"
    np.save(tmp, allv)
    tmp.replace(out / "vectors.npy")
    index[["uid", "offset", "length"]].to_csv(out / "index.csv", index=False)

    try:
        import transformers
        tv = transformers.__version__
    except ImportError:
        tv = ""
    manifest = {"plm": plm, "hf": PLMS[plm]["hf"], "dim": PLMS[plm]["dim"], "dtype": "float16",
                "build_id": build_id, "n_seqs": int(len(index)),
                "total_residues": int(allv.shape[0]), "transformers": tv,
                "created": datetime.datetime.now().isoformat(timespec="seconds")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))   # commit marker, written last
    if verbose:
        mb = allv.nbytes / 1e6
        print(f"[{plm}] cached {len(index)} sequences, {allv.shape[0]} residues, {mb:.0f} MB -> {out}")
    shutil.rmtree(scratch, ignore_errors=True)


def stage(plm, build_id, data_dir=None, verbose=True):
    """Copy a cache from Drive to local scratch once per session.

    Never memmap straight off Drive's FUSE mount: every random read becomes a
    network round-trip.
    """
    src = cache_dir(plm, build_id, data_dir)
    if not config.ON_COLAB:
        return src
    dst = Path(config.WORK_DIR) / "emb_staged" / build_id / plm
    if not (dst / "manifest.json").exists():
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("vectors.npy", "index.csv", "manifest.json"):
            shutil.copy(src / f, dst / f)
        if verbose:
            print(f"[{plm}] staged to {dst}")
    return dst


class EmbCache:
    """Read-only view over a built cache."""

    def __init__(self, plm, build_id, data_dir=None, staged=True):
        self.path = stage(plm, build_id, data_dir) if staged else cache_dir(plm, build_id, data_dir)
        if not (self.path / "manifest.json").exists():
            raise FileNotFoundError(
                f"no cache for {plm} / build {build_id} at {self.path}; run embeddings.ipynb first")
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.dim = int(self.manifest["dim"])
        idx = pd.read_csv(self.path / "index.csv")
        self._idx = {r.uid: (int(r.offset), int(r.length)) for r in idx.itertuples()}
        self._vec = np.load(self.path / "vectors.npy", mmap_mode="r")

    def get(self, uid):
        off, ln = self._idx[uid]
        return np.asarray(self._vec[off:off + ln])

    def stack(self, uids, max_len):
        """[N, D, L] float32 padded, plus true lengths — the model's input contract."""
        uids = list(uids)
        x = np.zeros((len(uids), self.dim, max_len), dtype=np.float32)
        lengths = np.zeros(len(uids), dtype=np.int64)
        for i, uid in enumerate(uids):
            e = self.get(uid)[:max_len]
            lengths[i] = len(e)
            x[i, :, :len(e)] = e.T
        return x, lengths

    def pooled(self, uids, how="mean"):
        """[N, D] sequence-level vectors for the classical classifiers and the
        'PLM without local scanning' ablation."""
        uids = list(uids)
        out = np.zeros((len(uids), self.dim), dtype=np.float32)
        for i, uid in enumerate(uids):
            e = self.get(uid).astype(np.float32)
            out[i] = e.max(axis=0) if how == "max" else e.mean(axis=0)
        return out
