"""Paper-style pan-cancer datasets (see docs/pancancer_study_spec.md, "paper-style design").

    raw snapshot -> positive / negative -> CD-HIT 0.4 (once, all groups) -> length <= 35 -> 80:20 split

Sets
    colorectal / lung / melanoma   each its own stratified 80:20 train/test (5-fold CV on train runs in the runner)
    pooled                         train = the three trains, test = the three tests (24:6 is 80:20 by construction)
    other_<disease>, hematologic   test only: cancers never trained on; the pooled directory carries them too

CD-HIT runs ONCE over every group, so no similar peptide sits on both sides of any split, across diseases either.
It keeps one representative per cluster, so the positives drop a lot (the paper: 671 -> 303): read the test sets'
sizes in `table2_splits.csv` before trusting any AUC from them.

Outputs go to DATA_DIR/<name>/{colorectal,lung,melanoma,pooled}/deepneoag_like_dataset.csv (+ meta.json with MAX_LEN),
so `data.load_dataset(DATA_DIR / name / "pooled")` works. `test_set` names the evaluation set of every test row.
"""
import datetime
import json
import re
from pathlib import Path

import pandas as pd

from . import cedar, config, lung_split, pancancer_split as ps

CORE = ps.CORE
HEME = ps.HEMATOLOGIC
OUT_COLS = ["uid", "accession", "structure_id", "sequence", "length", "label", "tier", "disease_group", "split", "test_set",
            "year_first", "pubmed_ids", "disease_names", "mhc_allele_names", "mhc_classes", "dataset_version", "download_date"]


def stage_table(stages):
    """One row per cancer, one column per step: how many positives / negatives each step leaves.

    `stages` maps a step name to the frame after that step (first = raw, last = final). The last two columns say what
    share of the raw positives / negatives survived to the end.
    """
    names = list(stages)
    cnt = {k: v.groupby("disease_group")["label"].agg(pos=lambda y: int((y == 1).sum()), neg=lambda y: int((y == 0).sum()))
           for k, v in stages.items()}
    raw = cnt[names[0]]
    order = [d for d in CORE if d in raw.index] + [d for d in raw.sort_values("pos", ascending=False).index if d not in CORE]
    rows = []
    for d in order + ["total"]:
        c = {k: (v.sum() if d == "total" else v.reindex([d]).fillna(0).iloc[0]).astype(int) for k, v in cnt.items()}
        row = {"cancer": "other (misc)" if d == "other" else d}
        row.update({k: f"{c[k].pos:,} pos / {c[k].neg:,} neg" for k in names})
        row["positives kept"] = f"{c[names[-1]].pos / max(c[names[0]].pos, 1):.0%}"
        row["negatives kept"] = f"{c[names[-1]].neg / max(c[names[0]].neg, 1):.0%}"
        rows.append(row)
    return pd.DataFrame(rows)


def paper_funnel(res):
    """The three main cancers (+ pooled) through the paper-style pipeline: raw -> similarity filter -> length -> train / test.

    `res` is the dict returned by `make_paper_dataset`."""
    stages, sets = res["stages"], res["sets"]
    names = list(stages)
    pn = lambda d: f"{int((d['label'] == 1).sum()):,} pos / {int((d['label'] == 0).sum()):,} neg"
    rows = []
    for arm in list(CORE) + ["pooled"]:
        ds = CORE if arm == "pooled" else (arm,)
        row = {"cancer": arm}
        row.update({k: pn(v[v["disease_group"].isin(ds)]) for k, v in stages.items()})
        row["train"], row["test"] = pn(sets[f"{arm}_train"]), pn(sets[f"{arm}_test"])
        rows.append(row)
    return pd.DataFrame(rows)


def default_filters():
    """The two similarity filters to compare: CD-HIT 0.4 and MMseqs2 0.3 (both keep one representative per cluster)."""
    return {"CD-HIT 0.4": lambda d, w: cedar.cdhit_filter(d, w, sim_id=0.4),
            "MMseqs2 0.3": lambda d, w: ps.mmseqs_filter(d, w, 0.3)}


def filter_stage_tables(raw, work_dir, filters=None, min_len=8, max_len=35, out_dir=None):
    """One stage table per similarity filter (cancer x raw / after filter / after length), no splitting.

    Same columns as `make_paper_dataset`'s table1, so the effect of each tool reads side by side.
    `filters` maps a name to `reduce(df, work_dir)` (default: `default_filters()`); with `out_dir` each table is also saved
    as stage_table_<name>.csv.
    """
    df, _ = ps.build(raw, "exp", min_len=min_len, max_len=None)
    out = {}
    for name, reduce in (filters or default_filters()).items():
        slug = re.sub(r"\W+", "_", name).strip("_")
        reps = reduce(df, Path(work_dir) / f"stage_{slug}")
        ln = reps[reps["length"] <= max_len]
        out[name] = stage_table({"raw": df, f"after {name}": reps, f"after length<={max_len}": ln})
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            out[name].to_csv(Path(out_dir) / f"stage_table_{slug}.csv", index=False)
    return out


def _counts(d):
    n, o = int((d["label"] == 1).sum()), int((d["label"] == 0).sum())
    return n, o


def _test_set(d):
    """Evaluation-set name of a test row."""
    name = d["disease_group"].replace({"other": "misc"})        # the catch-all group is "other_misc", not "other_other"
    return name.where(d["role"] == "core", "other_" + name).where(d["role"] != "heme", HEME)


def make_paper_dataset(raw, data_dir, work_dir, name="pancancer_paper_v1", sim_id=0.4, min_len=8, max_len=35,
                       test_size=0.2, seed=42, reduce=None, check_similarity=True):
    """Build and write the datasets. Returns a dict (sets, tables, similarity, limitations).

    `reduce(df, work_dir)` defaults to CD-HIT (needs cd-hit: Colab `apt-get install cd-hit`); tests may pass a stand-in.
    """
    reduce = reduce or (lambda d, w: cedar.cdhit_filter(d, w, sim_id=sim_id))
    out_dir = Path(data_dir) / name
    log = []                                                    # (uid, reason) for every dropped row

    # 1) positive / negative (disease-matched experimental negatives); the length filter comes after CD-HIT
    df, info = ps.build(raw, "exp", min_len=min_len, max_len=None)

    # 2) CD-HIT once, over every group
    reps = reduce(df, Path(work_dir) / name)
    log += [(u, "cdhit_redundant") for u in set(df["uid"]) - set(reps["uid"])]

    # 3) length
    too_long = reps["length"] > max_len
    log += [(u, "length") for u in reps.loc[too_long, "uid"]]
    ln = reps[~too_long].reset_index(drop=True)

    # 4) per-disease 80:20, 5) pooled = union
    sets, per = {}, {}
    for d in CORE:
        tr, te = cedar.split(ln[ln["disease_group"] == d], test_size=test_size, seed=seed)
        per[d] = (tr, te.assign(test_set=d))
    pooled_tr = pd.concat([per[d][0] for d in CORE], ignore_index=True)
    pooled_te = pd.concat([per[d][1] for d in CORE], ignore_index=True)

    # 6) other cancers: test only
    other = ln[ln["role"] == "ood"]
    other = other.assign(split="test", test_set=_test_set(other))
    heme = ln[ln["role"] == "heme"]
    heme = heme.assign(split="test", test_set=HEME)
    sets = {"pooled_train": pooled_tr, "pooled_test": pooled_te, "other": other, "hematologic": heme,
            **{f"{d}_{k}": per[d][i] for d in CORE for i, k in enumerate(("train", "test"))}}

    # 8) checks: a sequence belongs to exactly one set; the pooled test is the union of the three tests
    names = ["pooled_train", "pooled_test", "other", "hematologic"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(sets[a]["sequence"]) & set(sets[b]["sequence"]), f"a sequence is in both {a} and {b}"
    assert set(pooled_te["uid"]) == set().union(*[set(per[d][1]["uid"]) for d in CORE]), "pooled test != union of tests"
    sim = pd.DataFrame(_similarity(sets, sim_id) if check_similarity else [])

    # tables: stages, and train/test sizes
    stages = {"raw": df, "after CD-HIT 0.4": reps, f"after length<={max_len}": ln}
    t1 = stage_table(stages)
    rows = []
    for nm, d in [("colorectal train", per["colorectal"][0]), ("colorectal test", per["colorectal"][1]),
                  ("lung train", per["lung"][0]), ("lung test", per["lung"][1]),
                  ("melanoma train", per["melanoma"][0]), ("melanoma test", per["melanoma"][1]),
                  ("pooled train", pooled_tr), ("pooled test", pooled_te), ("other cancers (test only)", other),
                  ("hematologic (test only)", heme)]:
        n, o = _counts(d)
        rows.append({"set": nm, "entities": len(d), "N+": n, "N-": o, "prevalence": round(n / max(n + o, 1), 4)})
    t2 = pd.DataFrame(rows)
    limits = [f"{r['set']}: only {r['N+']} positives, AUC will be unstable (use bootstrap CIs)" for _, r in t2.iterrows()
              if "test" in r["set"] and 0 < r["entities"] and r["N+"] < 30]

    _write(out_dir, per, pooled_tr, pooled_te, other, heme, t1, t2, sim, log, max_len)
    (out_dir / "meta.json").write_text(json.dumps({
        "dataset_version": name, "neg_mode": "exp", "sim_id": sim_id, "min_len": min_len, "max_len": max_len, "MAX_LEN": max_len,
        "test_size": test_size, "seed": seed, "core": CORE, **info, "cdhit": lung_split._cdhit_version(),
        "snapshot_version": str(raw["dataset_version"].iloc[0]), "download_date": str(raw["download_date"].iloc[0]),
        "snapshot_file": raw.attrs.get("snapshot_file", ""), "snapshot_sha256": raw.attrs.get("snapshot_sha256", ""),
        "tool_versions": config.tool_versions(),
        "git_sha": config.git_sha(), "exclusions": pd.Series([r for _, r in log]).value_counts().to_dict(),
        "limitations": limits, "created": datetime.datetime.now().isoformat(timespec="seconds")}, indent=2))
    return {"sets": sets, "stages": stages, "table1": t1, "table2": t2, "similarity": sim, "limitations": limits,
            "exclusions": pd.DataFrame(log, columns=["uid", "reason"])}


def _similarity(sets, sim_id):
    """Alignment-free screen (see lung_split.max_identity): how close each held-out set is to the pooled training set."""
    tr = sets["pooled_train"]
    return [lung_split.cross_split_check(tr, sets[k], "pooled_train", k, sim_id) for k in ("pooled_test", "other", "hematologic")]


def _write(out, per, pooled_tr, pooled_te, other, heme, t1, t2, sim, log, max_len):
    def save(d, df):
        df = df.reindex(columns=[c for c in OUT_COLS if c in df.columns])
        df["test_set"] = df["test_set"].fillna("") if "test_set" in df else ""
        (out / d).mkdir(parents=True, exist_ok=True)
        df.to_csv(out / d / "deepneoag_like_dataset.csv", index=False)
        (out / d / "meta.json").write_text(json.dumps({"MAX_LEN": max_len, "set": d}))
        for sp, g in df.groupby("split"):
            cedar.write_fasta(g, out / d / f"{sp}.fasta")

    for d in CORE:
        tr, te = per[d]
        save(d, pd.concat([tr.assign(split="train"), te.assign(split="test")], ignore_index=True))
    save("pooled", pd.concat([pooled_tr.assign(split="train", test_set=""), pooled_te.assign(split="test"), other, heme],
                             ignore_index=True))
    t1.to_csv(out / "table1_stages.csv", index=False)
    t2.to_csv(out / "table2_splits.csv", index=False)
    sim.to_csv(out / "similarity_report.csv", index=False)
    pd.DataFrame(log, columns=["uid", "reason"]).to_csv(out / "split_exclusions.csv", index=False)


def eval_table(df, min_pos=50):
    """Counts per evaluation set of a pooled frame (test rows): each core disease, each other cancer, and all other solid.

    `reportable`: at least min_pos positives and min_pos negatives (smaller sets give unstable AUCs).
    """
    te = df[df["split"] == "test"]
    sets = {k: g for k, g in te.groupby("test_set")}
    sets["other_solid (all)"] = te[te["test_set"].str.startswith("other_")]
    rows = [{"test_set": k, "N+": _counts(g)[0], "N-": _counts(g)[1]} for k, g in sets.items()]
    t = pd.DataFrame(rows)
    t["reportable"] = (t["N+"] >= min_pos) & (t["N-"] >= min_pos)
    return t.sort_values("N+", ascending=False).reset_index(drop=True)
