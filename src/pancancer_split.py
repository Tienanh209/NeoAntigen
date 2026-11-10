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
import hashlib
import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import cedar, config, data, lung_split

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


def mmseqs_clusters(df, work_dir, sim_id=0.3, cov=0.8, sensitivity=8.0, max_seqs=10000, threads=0, extra=()):
    """Cluster ids from MMseqs2 `easy-cluster` (Colab: `apt-get install mmseqs2`; or conda -c bioconda mmseqs2).

    Unlike CD-HIT (identity floor 0.4) it can cluster at 0.3. Short peptides need a sensitive prefilter, hence the
    high `-s` and `--max-seqs`. `--cov-mode 1` asks the alignment to cover `cov` of the MEMBER, so a short peptide can
    join a longer representative (like CD-HIT's identity over the shorter sequence); `cov-mode 0` would only join peptides
    of similar length. Returns a Series of ints aligned to df.index.
    """
    exe = shutil.which("mmseqs")
    if exe is None:
        raise RuntimeError("mmseqs not found: apt-get install mmseqs2 (Colab) or conda install -c bioconda mmseqs2")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "all.fa", "w") as f:
        for r in df.itertuples():
            f.write(f">{r.uid}\n{r.sequence}\n")
    out, tmp = work / "mmseqs", work / "tmp"
    cmd = [exe, "easy-cluster", str(work / "all.fa"), str(out), str(tmp), "--min-seq-id", str(sim_id), "-c", str(cov),
           "--cov-mode", "1", "-s", str(sensitivity), "--max-seqs", str(max_seqs),
           "--threads", str(threads or os.cpu_count() or 1), "-v", "1", *extra]
    with open(work / "mmseqs.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    tsv = pd.read_csv(f"{out}_cluster.tsv", sep="\t", header=None, names=["rep", "member"], dtype=str)
    rep_id = {r: i for i, r in enumerate(sorted(tsv["rep"].unique()))}
    cl = dict(zip(tsv["member"], tsv["rep"].map(rep_id)))
    ids = df["uid"].astype(str).map(cl)
    if ids.isna().any():
        raise RuntimeError(f"{int(ids.isna().sum())} peptides missing from the MMseqs2 clusters (see {work / 'mmseqs.log'})")
    return ids.astype(int)


def mmseqs_filter(df, work_dir, sim_id=0.3, **kw):
    """One representative per MMseqs2 cluster, positives first then longer: the CD-HIT-style filter for
    `pancancer_dataset.make_paper_dataset(..., reduce=...)` (use `lambda d, w: mmseqs_filter(d, w, 0.3)`)."""
    d = df.assign(_c=mmseqs_clusters(df, work_dir, sim_id, **kw).values)
    return d.sort_values(["label", "length"], ascending=False, kind="stable").drop_duplicates("_c").drop(columns="_c")


def cluster_ids(df, work_dir, sim_id=0.4, method="auto", **kw):
    """(cluster id per row, method used). method: "cdhit" (identity >= 0.4), "mmseqs" (any identity, e.g. 0.3; `kw` goes
    to `mmseqs_clusters`), "kmer8", or "auto" = CD-HIT when installed, else shared 8-mers.

    The 8-mer fallback only groups near-identical peptides, a weaker guarantee than CD-HIT 0.4: use it for a quick
    local check, not for the frozen dataset.
    """
    if method == "mmseqs":
        return mmseqs_clusters(df, work_dir, sim_id, **kw), "mmseqs"
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
                 min_len=8, max_len=35, seed=42, method="auto", cluster_kw=None):
    """Build and write DATA_DIR/<name>/{manifest.csv, summary.csv, meta.json}. Returns (manifest, summary)."""
    df, info = build(raw, neg_mode, min_len, max_len)
    df["cluster_id"], used = cluster_ids(df, Path(work_dir) / name, sim_id, method, **(cluster_kw or {}))
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
        "max_len": max_len, "seed": seed, "cluster_method": used, "cluster_kw": cluster_kw or {}, "core": CORE, "not_used": NOT_USED, **info,
        "snapshot_version": str(raw["dataset_version"].iloc[0]), "download_date": str(raw["download_date"].iloc[0]),
        "snapshot_file": raw.attrs.get("snapshot_file", ""), "snapshot_sha256": raw.attrs.get("snapshot_sha256", ""),
        "manifest_sha256": config.file_sha256(out / "manifest.csv"), "tool_versions": config.tool_versions(),
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


def arm_frame(m, arm, test_fold=0, neg_ratio=None, seed=42):
    """The runner's dataframe for one arm and one outer test fold (see module docstring for the columns).

    Train = the arm's diseases outside `test_fold` (+ background negatives when the manifest is a "bg" build).
    neg_ratio: per training disease keep at most neg_ratio negatives per positive (None, the default, keeps all; a ratio
    removes the prevalence difference between diseases but drops data). Test = every other row; nothing is removed
    from it, only flagged.
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
    out["neg_ratio"] = int(neg_ratio or 0)                    # 0 = every negative kept
    out["test_set"] = np.where(out["split"] == "test", test_set_name(out), "")
    check_arm(out)
    return out


def check_arm(df):
    """Sanity checks that must hold for every arm frame."""
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    assert not set(tr["sequence"]) & set(te["sequence"]), "a sequence is in both train and test"
    clean = te[~te["similar_to_train"]]
    assert not set(tr["cluster_id"]) & set(clean["cluster_id"]), "a cluster crosses train and the clean test"
    assert df["uid"].is_unique, "duplicate uid"


def arm_dir(data_dir, name, arm, fold=0, neg_ratio=None):
    """DATA_DIR/<name>/<arm>_f<k>, with an `_nr<r>` suffix when negatives were capped (never overwrites the full frame)."""
    return Path(data_dir) / name / (f"{arm}_f{fold}" + (f"_nr{neg_ratio}" if neg_ratio else ""))


def write_arm(df, data_dir, name="pancancer_cedar_v1"):
    """DATA_DIR/<name>/<arm>_f<k>[_nr<r>]/deepneoag_like_dataset.csv: loadable with `data.load_dataset(dir)`."""
    nr = int(df["neg_ratio"].iloc[0]) if "neg_ratio" in df else 0
    d = arm_dir(data_dir, name, df["arm"].iloc[0], int(df["test_fold"].iloc[0]), nr)
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "deepneoag_like_dataset.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"arm": df["arm"].iloc[0], "test_fold": int(df["test_fold"].iloc[0]),
                                             "neg_ratio": nr, "build_id": data.build_id(df),
                                             "n_train": int((df["split"] == "train").sum()),
                                             "n_test": int((df["split"] == "test").sum())}, indent=2))
    return d


def load_arm(data_dir, name="pancancer_cedar_v1", arm="colorectal", fold=0, neg_ratio=None):
    """An arm frame written by `write_arm`, with the padded length attached (`df.attrs["max_len"]`) so
    `data.max_len` does not fall back to another dataset's meta.json."""
    df = data.load_dataset(arm_dir(data_dir, name, arm, fold, neg_ratio))
    meta = Path(data_dir) / name / "meta.json"
    df.attrs["max_len"] = int(json.loads(meta.read_text()).get("max_len", 35)) if meta.exists() else 35
    mpath = Path(data_dir) / name / "manifest.csv"
    if mpath.exists():       # every arm frame is a subset of the manifest: the runner reads the manifest-wide embedding cache
        df.attrs["emb_id"] = emb_id(pd.read_csv(mpath, usecols=["uid", "sequence"]))
    return df


def build_arm(data_dir, name="pancancer_cedar_v1", arm="colorectal", fold=0, neg_ratio=None, seed=42):
    """Rebuild an arm frame from manifest.csv (deterministic), write it and return it loaded.

    Never trusts an old directory: an arm written earlier with another `neg_ratio` or by older code is simply replaced."""
    write_arm(arm_frame(load_manifest(data_dir, name), arm, fold, neg_ratio, seed), data_dir, name)
    return load_arm(data_dir, name, arm, fold, neg_ratio)


ARM_PREFIX = {"colorectal": "CRC", "lung": "LUNG", "melanoma": "MEL", "pooled": "POOL", "pooled_bal": "POOLB"}
# (train pos, train neg, own-test pos, own-test neg) of the 2026-11-10 snapshot, CD-HIT 0.4, fold 0, every negative kept.
# A guard against a silently changed manifest, not a contract: a rebuilt manifest may legitimately differ.
DOCUMENTED = {"colorectal": (384, 6867, 77, 1584), "lung": (547, 1480, 97, 311), "melanoma": (1265, 4456, 268, 1075),
              "pooled": (2196, 12803, 442, 2970)}


def emb_id(manifest):
    """Cache key for the embeddings of a whole manifest. Every arm frame is a subset of it, so all arms and folds
    read ONE cache (`embeddings_pancancer.ipynb` builds it once)."""
    payload = "\n".join(f"{u}\t{s}" for u, s in sorted(zip(manifest["uid"], manifest["sequence"])))
    return "pc_" + hashlib.sha1(payload.encode()).hexdigest()[:10]


def arm_counts(df):
    """(train pos, train neg, own-test pos, own-test neg) of an arm frame."""
    tr, own = df[df["split"] == "train"], df[(df["split"] == "test") & df["in_dist"]]
    return (int((tr["label"] == 1).sum()), int((tr["label"] == 0).sum()),
            int((own["label"] == 1).sum()), int((own["label"] == 0).sum()))


def own_uids(df):
    """uids of the arm's own test (every disease the arm was trained on; for `pooled`, the three tests together)."""
    return df.loc[(df["split"] == "test") & df["in_dist"], "uid"]


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


def eval_uids(df):
    """{set name: uids} of the test rows to score, clean rows only (not similar to this arm's training data).

    Order: the arm's own test sets ("<disease> (own test)"), the other core cancers, every other cancer by size,
    then "other solid (all)" and hematologic. Feed it to `results.by_set` / `results.breakdown`.
    """
    sets = eval_sets(df, clean=True)
    info = df.drop_duplicates("disease_group").set_index("disease_group")[["role", "in_dist"]]
    out, solid = {}, []
    for g in sorted(sets, key=lambda g: -(sets[g]["label"] == 1).sum()):
        role, own = info.loc[g, "role"], bool(info.loc[g, "in_dist"])
        if own:
            out[f"{g} (own test)"] = sets[g]["uid"]
        elif role == "core":
            out[g] = sets[g]["uid"]
    for g in sorted(sets, key=lambda g: -(sets[g]["label"] == 1).sum()):
        if info.loc[g, "role"] == "ood":
            out[g] = sets[g]["uid"]
            solid.append(sets[g])
    if solid:
        out["other solid (all)"] = pd.concat(solid)["uid"]
    for g in sets:
        if info.loc[g, "role"] == "heme":
            out[g] = sets[g]["uid"]
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


def check_similarity(df, sim_id=0.4):
    """Independent screen of an arm frame, not based on the clustering tool: best ungapped identity of the clean test rows
    to the training rows (`lung_split.max_identity`). A lower bound; `violations` should be 0 at the clustering threshold."""
    tr = df[df["split"] == "train"]
    te = df[(df["split"] == "test") & ~df["similar_to_train"]]
    return pd.DataFrame([lung_split.cross_split_check(tr, te[te["in_dist"]], "train", "own test", sim_id),
                         lung_split.cross_split_check(tr, te[~te["in_dist"]], "train", "other cancers", sim_id)])


def _pn(d):
    return f"{int((d['label'] == 1).sum()):,} pos / {int((d['label'] == 0).sum()):,} neg"


def cluster_funnel(raw, manifest, frames):
    """The three main cancers (+ pooled) from CEDAR to train/test in the cluster split.

    CEDAR = unique peptides per tier in the raw snapshot; cleaned = the manifest (standard amino acids, no label conflict,
    8-35 aa); train / test = the arm's own frame. Clustering removes nothing, it only keeps similar peptides on one side.
    """
    neo = raw[raw["group"] == "neo"]
    rows = []
    for arm in ("colorectal", "lung", "melanoma", "pooled"):
        ds = CORE if arm == "pooled" else (arm,)
        c = neo[neo["disease_group"].isin(ds)]
        cedar_n = {t: c.loc[c["tier"] == t, "linear_sequence"].nunique() for t in ("positive", "exp_negative")}
        f = frames[arm]
        rows.append({"cancer": arm, "CEDAR": f"{cedar_n['positive']:,} pos / {cedar_n['exp_negative']:,} neg",
                     "cleaned, 8-35 aa": _pn(manifest[manifest["disease_group"].isin(ds)]),
                     "train": _pn(f[f["split"] == "train"]), "test": _pn(f[(f["split"] == "test") & f["in_dist"]])})
    return pd.DataFrame(rows)


def test_set_name(d):
    """Evaluation-set name of a row: the cancer (core), other_<cancer>, hematologic, or background."""
    name = d["disease_group"].replace({"other": "misc"})
    return (name.where(d["role"] == "core", "other_" + name)
            .where(d["role"] != "heme", HEMATOLOGIC).where(d["role"] != "bg", "background"))


MAIN_ARMS = ("colorectal", "lung", "melanoma", "pooled")


def write_main_arms(m, data_dir, name="pancancer_cedar_v1", test_fold=0, neg_ratio=None, seed=42):
    """The requested layout, split by cluster so no peptide is thrown away.

    colorectal / lung / melanoma: trained and tested on their own ~80:20 (one outer fold = the 20%), 5-fold CV on train
    in the runner (split_mode="grouped" groups the folds by `cluster_id`). pooled: the three trains together, tested on
    the three tests (24:6 = 80:20), plus the other cancers as extra test rows (`test_set`, `similar_to_train`).
    neg_ratio None keeps every negative (the paper's choice); an int caps negatives per positive in train.
    Returns ({arm: frame}, sizes, eval tables of all arms).
    """
    frames, sizes, evals = {}, [], []
    for arm in MAIN_ARMS:
        df = arm_frame(m, arm, test_fold, neg_ratio, seed)
        write_arm(df, data_dir, name)
        frames[arm] = df
        tr, own = df[df["split"] == "train"], df[(df["split"] == "test") & df["in_dist"]]
        sizes.append({"arm": arm, "train N+": int((tr["label"] == 1).sum()), "train N-": int((tr["label"] == 0).sum()),
                      "own test N+": int((own["label"] == 1).sum()), "own test N-": int((own["label"] == 0).sum())})
        evals.append(eval_table(df).assign(arm=arm))
    return frames, pd.DataFrame(sizes), pd.concat(evals, ignore_index=True)
