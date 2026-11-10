"""Frozen pan-cancer dataset and per-arm views (see docs/pancancer_study_spec.md).

`make_dataset` turns the raw snapshot (`pancancer.get_raw`) into ONE manifest: a row per unique peptide with its
disease group, label, similarity cluster and an outer fold. Folds are assigned per cluster (CD-HIT, 0.4), so similar
peptides never sit in different folds, whatever their disease.

An *arm* is a view of the manifest: which diseases are trained on. `arm_frame(manifest, arm, test_fold)` gives the
dataframe the runner expects (`uid, sequence, label, split`) plus evaluation columns:
    split             "train" | "test" (the runner predicts every non-train row)
    disease_group     melanoma / colorectal / lung / other solid cancers / hematologic / background
    in_dist           the row's disease is one of the arm's training diseases
    similar_to_train  its cluster also holds a training peptide of this arm -> leaks; evaluate on ~similar_to_train
Evaluate each `disease_group` separately on `split == "test" & ~similar_to_train`, and report the all-rows number next
to it (the difference is the leakage).

Why clusters and not papers: grouping by paper is infeasible here. Rows cite several papers, so 83% of the core rows
end up in ONE connected paper component (papers also span diseases). The paper effect is reported as a diagnostic
(`paper_share`) instead.

Negatives (`neg_mode`): "exp" = experimentally tested, T cell negative neoantigens of the same disease (primary);
"bg" = viral / TAA background epitopes (the paper's choice; "virus vs human" shortcut, kept for comparison).
"""
import datetime
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import cedar, config, data

CORE = ("melanoma", "colorectal", "lung")
NOT_USED = ("multi",)               # peptides annotated with several cancers: in no arm
HEMATOLOGIC = "hematologic"         # not solid tumours: evaluated apart from the other cancers
ARMS = {"colorectal": ("colorectal",), "melanoma": ("melanoma",), "lung": ("lung",),
        "pooled": CORE, "pooled_bal": CORE}      # pooled_bal: positives downsampled to the smallest disease
KEEP = ["uid", "accession", "sequence", "length", "label", "tier", "disease_group", "role", "cluster_id", "outer_fold",
        "year_first", "pubmed_ids", "disease_names", "mhc_allele_names", "mhc_classes", "dataset_version", "download_date"]


def build(raw, neg_mode="exp", min_len=8, max_len=35):
    """One row per sequence for the chosen negative policy. Returns (df, info)."""
    if neg_mode not in ("exp", "bg"):
        raise ValueError("neg_mode must be 'exp' or 'bg'")
    neo = raw["group"] == "neo"
    pos = neo & (raw["tier"] == "positive")
    neg = (neo & (raw["tier"] == "exp_negative")) if neg_mode == "exp" else raw["group"].isin(["viral", "taa"])
    sel = raw[(pos | neg) & ~raw["disease_group"].isin(NOT_USED)].copy()
    sel["label"] = pos[sel.index].astype(int)
    df, info = cedar.build_initial(sel, require_positive_assay=False)
    n0 = len(df)
    keep = df["length"] >= (min_len or 0)
    if max_len:      # None: the caller filters the length itself (e.g. after CD-HIT)
        keep &= df["length"] <= max_len
    df = df[keep].reset_index(drop=True)
    info["length_dropped"] = n0 - len(df)
    df["role"] = np.select([df["disease_group"] == "background", df["disease_group"].isin(CORE),
                            df["disease_group"] == HEMATOLOGIC], ["bg", "core", "heme"], default="ood")
    return df, info


def cluster_ids(df, work_dir, sim_id=0.4, method="auto"):
    """(cluster id per row, method used). CD-HIT when installed (Colab: apt-get install cd-hit), else shared 8-mers.

    The 8-mer fallback only groups near-identical peptides, a weaker guarantee than CD-HIT 0.4: use it for a quick
    local check, not for the frozen dataset.
    """
    if method in ("auto", "cdhit") and shutil.which("cd-hit"):
        cedar.cdhit_filter(df, work_dir, sim_id=sim_id)
        cl = cedar.cdhit_clusters(work_dir)
        ids = df["uid"].map(cl)
        if ids.isna().any():
            raise RuntimeError(f"{int(ids.isna().sum())} peptides missing from the CD-HIT clusters")
        return ids.astype(int), "cdhit"
    if method == "cdhit":
        raise RuntimeError("cd-hit is not installed")
    warnings.warn("cd-hit not found: using shared 8-mer clusters (weaker than CD-HIT 0.4); do not freeze this dataset")
    return data.kmer_groups(df, k=8).astype(int), "kmer8"


def assign_folds(df, n_folds=5, seed=42):
    """Outer fold per cluster for core + background rows (stratified by disease x label); -1 for the others."""
    out = pd.Series(-1, index=df.index)
    used = df["role"].isin(["core", "bg"])
    sub = df[used]
    strata = pd.factorize(sub["disease_group"] + "_" + sub["label"].astype(str))[0]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(sub)), strata, groups=sub["cluster_id"])):
        out.loc[sub.index[te]] = k
    return out


def make_dataset(raw, data_dir, work_dir, name="pancancer_cedar_v1", neg_mode="exp", n_folds=5, sim_id=0.4,
                 min_len=8, max_len=35, seed=42, method="auto"):
    """Build and write DATA_DIR/<name>/{manifest.csv, summary.csv, meta.json}. Returns (manifest, summary)."""
    df, info = build(raw, neg_mode, min_len, max_len)
    df["cluster_id"], used = cluster_ids(df, Path(work_dir) / name, sim_id, method)
    df["outer_fold"] = assign_folds(df, n_folds, seed)
    manifest = df.reindex(columns=KEEP)
    assert not manifest.groupby("cluster_id")["outer_fold"].apply(lambda f: f[f >= 0].nunique() > 1).any(), \
        "a cluster crosses outer folds"
    summary = summarize(manifest)
    out = Path(data_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "manifest.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "meta.json").write_text(json.dumps({
        "dataset_version": name, "neg_mode": neg_mode, "n_folds": n_folds, "sim_id": sim_id, "min_len": min_len,
        "max_len": max_len, "seed": seed, "cluster_method": used, "core": CORE, "not_used": NOT_USED, **info,
        "snapshot_version": str(raw["dataset_version"].iloc[0]), "download_date": str(raw["download_date"].iloc[0]),
        "git_sha": config.git_sha(), "created": datetime.datetime.now().isoformat(timespec="seconds")}, indent=2))
    return manifest, summary


def load_manifest(data_dir, name="pancancer_cedar_v1"):
    return pd.read_csv(Path(data_dir) / name / "manifest.csv")


def summarize(m):
    """Per disease group: positives / negatives / prevalence / papers / share of the top paper (batch-effect warning)."""
    rows = []
    for g, d in m.groupby("disease_group"):
        n, o = int((d["label"] == 1).sum()), int((d["label"] == 0).sum())
        papers = d["pubmed_ids"].astype(str).str.split(";").explode()
        rows.append({"disease_group": g, "role": d["role"].iloc[0], "N+": n, "N-": o, "prevalence": round(n / max(n + o, 1), 3),
                     "clusters": d["cluster_id"].nunique(), "papers": papers.nunique(),
                     "paper_share": round(papers.value_counts().iloc[0] / len(d), 2)})
    return pd.DataFrame(rows).sort_values("N+", ascending=False).reset_index(drop=True)


def arm_frame(m, arm, test_fold=0, neg_ratio=5, seed=42):
    """The runner's dataframe for one arm and one outer test fold (see module docstring for the columns).

    Train = the arm's diseases outside `test_fold` (+ background negatives when the manifest is a "bg" build).
    neg_ratio: per training disease keep at most neg_ratio negatives per positive (None keeps all), which removes the
    prevalence difference between diseases. Test = every other row; nothing is removed from it, only flagged.
    """
    diseases = ARMS[arm]
    rng = np.random.default_rng(seed)
    m = m.copy()
    cand = (m["disease_group"].isin(diseases) | (m["role"] == "bg")) & (m["outer_fold"] >= 0) & (m["outer_fold"] != test_fold)
    train_clusters = set(m.loc[cand, "cluster_id"])          # before subsampling: dropped rows still have relatives
    train = m[cand]
    if arm == "pooled_bal":
        k = int(min((train[(train["disease_group"] == d) & (train["label"] == 1)]).shape[0] for d in diseases))
        keep_pos = [train[(train["disease_group"] == d) & (train["label"] == 1)].sample(
            k, random_state=int(rng.integers(1 << 31))) for d in diseases]
        train = pd.concat(keep_pos + [train[train["label"] == 0]])
    if neg_ratio:
        kept = [train[train["label"] == 1]]
        neg = train[train["label"] == 0]
        for d, g in neg.groupby("disease_group"):
            # a disease keeps neg_ratio x its own positives; background negatives share the arm's total positives
            n_pos = int((train["label"] == 1).sum() if d == "background"
                        else ((train["label"] == 1) & (train["disease_group"] == d)).sum())
            kept.append(g.sample(min(len(g), neg_ratio * n_pos), random_state=int(rng.integers(1 << 31))))
        train = pd.concat(kept)
    test = m[~m.index.isin(m.index[cand])]
    out = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    out["in_dist"] = out["disease_group"].isin(diseases)
    out["similar_to_train"] = (out["split"] == "test") & out["cluster_id"].isin(train_clusters)
    out["arm"], out["test_fold"] = arm, test_fold
    check_arm(out)
    return out


def check_arm(df):
    """Sanity checks that must hold for every arm frame."""
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    assert not set(tr["sequence"]) & set(te["sequence"]), "a sequence is in both train and test"
    clean = te[~te["similar_to_train"]]
    assert not set(tr["cluster_id"]) & set(clean["cluster_id"]), "a cluster crosses train and the clean test"
    assert df["uid"].is_unique, "duplicate uid"


def write_arm(df, data_dir, name="pancancer_cedar_v1"):
    """DATA_DIR/<name>/<arm>_f<k>/deepneoag_like_dataset.csv: loadable with `data.load_dataset(dir)`."""
    d = Path(data_dir) / name / f"{df['arm'].iloc[0]}_f{df['test_fold'].iloc[0]}"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "deepneoag_like_dataset.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"arm": df["arm"].iloc[0], "test_fold": int(df["test_fold"].iloc[0]),
                                             "build_id": data.build_id(df), "n_train": int((df["split"] == "train").sum()),
                                             "n_test": int((df["split"] == "test").sum())}, indent=2))
    return d


def eval_sets(df, clean=True):
    """{disease_group: test rows to score}. clean=True drops rows similar to this arm's training data.

    A disease uses its own negatives (neg_mode "exp"); when it has none ("bg" builds) it is paired with the arm's
    background test negatives. Rows keep the runner's uid so predictions can be joined back.
    """
    te = df[(df["split"] == "test") & ~(df["similar_to_train"] if clean else False)]
    bg = te[te["disease_group"] == "background"]
    out = {}
    for g, d in te[te["disease_group"] != "background"].groupby("disease_group"):
        out[g] = d if (d["label"] == 0).any() else pd.concat([d, bg])
    return out


def eval_table(df, min_pos=50):
    """Counts per disease group on the test split, all rows vs clean rows (the difference is the leakage).

    `reportable`: at least min_pos positives and negatives after the filter (smaller sets give unstable AUCs).
    """
    full, clean = eval_sets(df, clean=False), eval_sets(df, clean=True)
    rows = []
    for g in full:
        n = lambda s, v: int((s[g]["label"] == v).sum()) if g in s else 0
        rows.append({"disease_group": g, "in_dist": bool(df.loc[df["disease_group"] == g, "in_dist"].iloc[0]),
                     "N+": n(full, 1), "N-": n(full, 0), "N+ clean": n(clean, 1), "N- clean": n(clean, 0)})
    t = pd.DataFrame(rows)
    t["reportable"] = (t["N+ clean"] >= min_pos) & (t["N- clean"] >= min_pos)
    return t.sort_values(["in_dist", "N+"], ascending=False).reset_index(drop=True)
