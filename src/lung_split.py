"""Frozen splits for the lung dataset, following the Ou Lab manuscript guide (steps 4-5).

Sets
    train / test    development data (first qualifying evidence <= T). `test` is the conventional independent test:
                    reserved at build time, never used for tuning.
    temporal_all    first evidence in (T, E]; peptides already present in development are removed (they are new
                    *annotations* of known sequences, not new sequences) and counted separately.
    temporal_lowsim subset of temporal_all whose maximum identity to development is below `sim_id`.
`temporal_all` and `temporal_lowsim` overlap: they are not two independent replications.

Similarity check: CD-HIT reps are not proven to satisfy the threshold pair by pair, so `max_identity` re-measures it
without CD-HIT: best ungapped-diagonal identity (matches / length of the shorter peptide) over pairs sharing a 5-mer.
It is a screening check for close relatives (a lower bound on alignment identity, blind to pairs with no shared 5-mer);
on Colab also run `cd-hit-2d` for the full alignment-based answer.

Outputs go to DATA_DIR/<name>/; `deepneoag_like_dataset.csv` keeps the train/test schema so `data.load_dataset` works.
"""
import datetime
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import cedar, config

K = 5   # shared k-mer length used to find candidate pairs (a 3-mer makes nearly every pair a candidate)


def max_identity(queries, refs, k=K):
    """For each query peptide: best ungapped identity to any reference (exact duplicates and substrings give 1.0)."""
    index = defaultdict(set)
    for j, s in enumerate(refs):
        for i in range(len(s) - k + 1):
            index[s[i:i + k]].add(j)
    out = []
    for q in queries:
        cand = set()
        for i in range(len(q) - k + 1):
            cand |= index.get(q[i:i + k], set())
        best = 0.0
        for j in cand:
            r = refs[j]
            short = min(len(q), len(r))
            for d in range(-(len(q) - 1), len(r)):          # offset of q relative to r
                a, b = max(0, -d), max(0, d)
                n = min(len(q) - a, len(r) - b)
                m = sum(x == y for x, y in zip(q[a:a + n], r[b:b + n]))
                best = max(best, m / short)
            if best >= 1.0:
                break
        out.append(best)
    return pd.Series(out, dtype=float)


def cross_split_check(a, b, name_a, name_b, sim_id):
    """One row of the similarity report: how close the sequences of `b` are to those of `a`."""
    if len(a) == 0 or len(b) == 0:
        return {"pair": f"{name_a} vs {name_b}", "n_query": len(b), "exact_dup": 0, "max_id": None,
                "q95": None, "median": None, "violations": 0, "threshold": sim_id}
    ident = max_identity(list(b["sequence"]), list(a["sequence"]))
    return {"pair": f"{name_a} vs {name_b}", "n_query": len(b), "exact_dup": int(b["sequence"].isin(set(a["sequence"])).sum()),
            "max_id": round(float(ident.max()), 3), "q95": round(float(ident.quantile(.95)), 3),
            "median": round(float(ident.median()), 3), "violations": int((ident > sim_id).sum()), "threshold": sim_id}


def _reduce(df, work_dir, sim_id):
    """CD-HIT representatives plus cluster ids (cluster id = the CD-HIT cluster, or the row number without CD-HIT)."""
    reps = cedar.cdhit_filter(df, work_dir, sim_id=sim_id).copy()
    try:
        cl = cedar.cdhit_clusters(work_dir)
        reps["cluster_id"] = reps["uid"].map(cl)
    except FileNotFoundError:
        reps["cluster_id"] = range(len(reps))
    return reps


def counts(d):
    n, o = int((d["label"] == 1).sum()), int((d["label"] == 0).sum())
    return n, o, round(n / max(n + o, 1), 4)


def make_splits(df, data_dir, work_dir, T=2022, E=None, sim_id=0.4, max_len=35, test_size=0.2, seed=42,
                name="lung_cedar_v1", reduce=None, params=None):
    """df: output of `lung.build` (one row per sequence, with year_first, tier, label). Returns a dict of the sets/tables.

    `reduce(df, work_dir)` defaults to CD-HIT; tests may pass a stand-in when cd-hit is not installed.
    """
    E = E or datetime.date.today().year
    reduce = reduce or (lambda d, w: _reduce(d, w, sim_id))
    work_dir, out_dir = Path(work_dir), Path(data_dir) / name
    df = df.copy()
    log = []                                  # (uid, reason) for rows dropped at the split stage

    # 1) length filter (same rule for every set), then time decides the cohort, not similarity
    too_long = (df["length"] > max_len) if max_len else pd.Series(False, index=df.index)
    log += [(u, "length") for u in df.loc[too_long, "uid"]]
    df = df[~too_long]
    dev = df[df["year_first"] <= T].assign(undated=lambda d: d["year_first"] == 0)   # undated rows can only be development
    tmp = df[(df["year_first"] > T) & (df["year_first"] <= E)]

    # 2) temporal: peptides already in development are new annotations of known sequences, not new sequences
    known = tmp["sequence"].isin(set(dev["sequence"]))
    log += [(u, "temporal_known_sequence") for u in tmp.loc[known, "uid"]]
    tmp_new = tmp[~known]

    # 3) redundancy reduction inside each side, then the conventional 80/20 split of development
    dev_nr = reduce(dev, work_dir / "dev")
    log += [(u, "cdhit_redundant_dev") for u in set(dev["uid"]) - set(dev_nr["uid"])]
    tmp_nr = reduce(tmp_new, work_dir / "temporal")
    log += [(u, "cdhit_redundant_temporal") for u in set(tmp_new["uid"]) - set(tmp_nr["uid"])]
    train, test = cedar.split(dev_nr, test_size=test_size, seed=seed)
    assert not set(train["cluster_id"]) & set(test["cluster_id"]), "a cluster crosses train/test"

    # 4) low-similarity temporal subset: prespecified rule, max identity to development < sim_id
    tmp_nr = tmp_nr.assign(max_id_dev=max_identity(list(tmp_nr["sequence"]), list(dev_nr["sequence"])).values)
    tmp_nr["lowsim"] = tmp_nr["max_id_dev"] < sim_id
    temporal_all = tmp_nr.assign(split="temporal_all")
    temporal_low = temporal_all[temporal_all["lowsim"]].assign(split="temporal_lowsim")

    # 5) verification that does not depend on CD-HIT
    sim = pd.DataFrame([cross_split_check(train, test, "train", "test", sim_id),
                        cross_split_check(train, temporal_all, "train", "temporal_all", sim_id),
                        cross_split_check(test, temporal_all, "test", "temporal_all", sim_id),
                        cross_split_check(dev_nr, temporal_low, "development", "temporal_lowsim", sim_id)])
    final = pd.concat([train, test], ignore_index=True)
    for a, b in [(train, test), (train, temporal_all), (test, temporal_all)]:
        assert not set(a["sequence"]) & set(b["sequence"]), "a sequence appears in two sets"
    assert not set(final["uid"]) & set(temporal_all["uid"]), "a uid appears in development and temporal"

    # 6) tables 1 and 2 (guide section 5)
    stages = {"after curation + length": df, "development (<=T)": dev, "development CD-HIT": dev_nr,
              "temporal candidates (T,E]": tmp, "temporal new sequences": tmp_new, "temporal CD-HIT": tmp_nr}
    table1 = pd.DataFrame({k: v.groupby("tier").size() for k, v in stages.items()}).fillna(0).astype(int)
    table1.loc["undated (development only)"] = [int(dev["undated"].sum()) if c == "development (<=T)" else 0
                                                for c in table1.columns]
    rows = []
    for nm, d, purpose in [("train", train, f"first evidence <= {T}; learning and selection"),
                           ("test (conventional independent)", test, f"first evidence <= {T}; reserved, no tuning"),
                           ("temporal_all", temporal_all, f"first evidence in ({T}, {E}]; new sequences only"),
                           ("temporal_lowsim", temporal_low, f"subset of temporal_all, max identity to development < {sim_id}")]:
        n, o, prev = counts(d)
        rows.append({"set": nm, "purpose": purpose, "entities": len(d), "N+": n, "N-": o, "prevalence": prev})
    table2 = pd.DataFrame(rows)
    limits = []
    for nm, d in [("temporal_all", temporal_all), ("temporal_lowsim", temporal_low)]:
        n, o, _ = counts(d)
        if n == 0 or o == 0:
            limits.append(f"{nm} has {n} positives / {o} negatives: not a usable test; collect a later cohort")

    # 7) manifest + files
    manifest = pd.concat([train.assign(split="train"), test.assign(split="test"), temporal_all], ignore_index=True)
    manifest = manifest.reindex(columns=["uid", "accession", "sequence", "length", "label", "tier", "split", "cluster_id",
                                         "year_first", "pubmed_ids", "disease_names", "mhc_allele_names", "mhc_classes",
                                         "dataset_version", "download_date", "lowsim", "undated"])
    dropped = pd.DataFrame(log, columns=["uid", "reason"])
    out_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_dir / "deepneoag_like_dataset.csv", index=False)
    temporal_all.to_csv(out_dir / "temporal_all.csv", index=False)
    temporal_low.to_csv(out_dir / "temporal_lowsim.csv", index=False)
    manifest.to_csv(out_dir / "split_manifest.csv", index=False)
    dropped.to_csv(out_dir / "split_exclusions.csv", index=False)
    table1.to_csv(out_dir / "table1_sources.csv")
    table2.to_csv(out_dir / "table2_splits.csv", index=False)
    sim.to_csv(out_dir / "similarity_report.csv", index=False)
    for nm, g in final.groupby("split"):
        cedar.write_fasta(g, out_dir / f"{nm}.fasta")
    (out_dir / "meta.json").write_text(json.dumps({
        **(params or {}), "dataset_version": name, "disease": "lung", "T": T, "E": E, "sim_id": sim_id, "max_len": max_len,
        "seed": seed, "test_size": test_size, "cdhit": _cdhit_version(), "git_sha": config.git_sha(),
        "similarity_method": f"best ungapped-diagonal identity over pairs sharing a {K}-mer (lower bound)",
        "exclusions": dropped["reason"].value_counts().to_dict(), "limitations": limits,
        "created": datetime.datetime.now().isoformat(timespec="seconds")}, indent=2))
    return {"train": train, "test": test, "temporal_all": temporal_all, "temporal_lowsim": temporal_low,
            "table1": table1, "table2": table2, "similarity": sim, "exclusions": dropped, "limitations": limits}


def _cdhit_version():
    try:
        r = subprocess.run(["cd-hit", "-h"], capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip().splitlines()[0]
    except Exception:
        return "unavailable"


def positives_by_year(df):
    """Look at this before choosing T: lung positives per first-evidence year (0 = undated)."""
    return df[df["label"] == 1].groupby("year_first").size().rename("positives")
