"""Pan-cancer CEDAR collection: every neoantigen epitope, labelled with a clean `disease_group`.

One dated raw snapshot (`pancancer_cedar_raw_YYYYMMDD.csv`) feeds the cross-disease study (per-disease models, a pooled
model, and held-out "other cancers"). Same design as `lung.py`: download once, filter later. This module only collects
and labels; splits, CD-HIT and leakage filtering come afterwards.

Rows:
  * neoantigens of any cancer: `group == "neo"`, with `disease_group` (melanoma, colorectal, lung, ..., other, multi)
  * background negatives (viral / TAA): `group` viral / taa, `disease_group == "background"` (not disease specific)

`disease_names` is kept untouched. `disease_groups` lists every group a peptide belongs to (a peptide can be annotated
with several diseases); `disease_group` is the single group when unique, "multi" when it spans several.
"""
import datetime
import json
import re
import warnings
from pathlib import Path

import pandas as pd

from . import cedar_v2 as v2, config, lung

DATASET_VERSION = "pancancer_v1"

# Ordered: the first matching group wins for a given CEDAR disease name. Names that match none become "other".
DISEASE_GROUPS = {
    "melanoma": r"melanoma",
    "colorectal": r"colorectal|colon|rectal|rectum|\bcrc\b",
    "lung": lung.LUNG_RE,
    "pancreatic": r"pancrea",
    "bile_duct": r"bile duct|cholangio|biliary|gallbladder",
    "hepatocellular": r"hepat|liver",
    "head_neck": r"head and neck|oral |oropharyn|laryn|pharyn|tongue|nasopharyn",
    "brain": r"glioma|glioblastoma|brain|astrocytoma",
    "stomach_esophageal": r"stomach|gastric|gastroesoph|esophag|oesophag",
    "breast": r"breast",
    "ovarian_gyn": r"ovarian|endometri|cervi|uterine",
    "renal": r"renal|kidney",
    "bladder": r"bladder|urothelial",
    "hematologic": r"leukemia|lymphoma|myeloma|myelodysplastic|hodgkin",   # kept apart: not solid tumours
}
OTHER = "other"
# Annotations that are not a cancer (CEDAR also lists healthy donors, fibrosis, infections, autoimmunity...)
NOT_CANCER_RE = (r"healthy|fibrosis|hepatitis|infect|virus|viral|diabetes|arthritis|covid|sars|autoimmun|lupus|"
                 r"sclerosis|allerg|asthma|\bcontrol\b|transplant|animal model|experimental")


def name_to_group(name):
    """CEDAR disease name -> group, or None when the name is not a cancer."""
    if re.search(NOT_CANCER_RE, name, re.I):
        return None
    for group, rx in DISEASE_GROUPS.items():
        if re.search(rx, name, re.I):
            return group
    return OTHER


def row_groups(names):
    """Sorted unique groups over a row's disease names (non-cancer names ignored). [] when none is a cancer."""
    if not isinstance(names, list):
        return []
    return sorted({g for g in map(name_to_group, map(str, names)) if g})


def _main_group(groups):
    return "multi" if len(groups) > 1 else groups[0] if groups else ""


def download_raw():
    """All neoantigens (any disease) + viral + TAA pools, with the columns `cedar_v2` / `lung` expect."""
    pos = lung._fetch({**v2.LINEAR, "neoantigen_bool": "eq.1"}).assign(label=1, group="neo")
    viral = lung._fetch({**v2.LINEAR, "neoantigen_bool": "eq.0", "viral_antigen_bool": "eq.1"}).assign(label=0, group="viral")
    taa = lung._fetch({**v2.LINEAR, "neoantigen_bool": "eq.0", "other_antigen_bool": "eq.1", "viral_antigen_bool": "eq.0",
                       "tcell_ids": "not.is.null"}).assign(label=0, group="taa")
    raw = pd.concat([pos, viral, taa], ignore_index=True)
    ls = lambda v: v if isinstance(v, list) else []

    neo = raw["group"] == "neo"
    groups = raw["disease_names"].apply(row_groups)
    raw["disease_groups"] = groups.apply(";".join)
    raw["disease_group"] = groups.apply(_main_group)
    raw.loc[~neo, "disease_group"] = "background"
    raw = raw[~neo | (raw["disease_group"] != "")].copy()    # drops neoantigens with no cancer annotation
    neo = raw["group"] == "neo"

    raw["has_tcell"] = raw["tcell_ids"].notna()
    raw["has_positive"] = raw["qualitative_measures"].apply(lambda q: any(x.startswith("Positive") for x in ls(q)))
    years = raw["reference_dates"].apply(lambda d: [int(y) for y in ls(d) if str(y).isdigit()])
    raw["year_first"] = years.apply(lambda y: min(y) if y else 0)   # 0 = undated; never invented
    raw["year_last"] = years.apply(lambda y: max(y) if y else 0)
    raw["has_mutation"] = raw["mutation"].notna()
    raw["structure_id"] = raw["structure_id"].astype(int)
    raw["accession"] = "CEDAR:" + raw["structure_id"].astype(str)
    raw["download_date"] = f"{datetime.date.today():%Y-%m-%d}"
    raw["dataset_version"] = DATASET_VERSION
    # tiers as in lung.py: neoantigens tested by T cell assay with no positive result are experimental negatives;
    # neoantigens without any T cell assay are unknown (never turned into negatives)
    raw["tier"] = "unknown"
    raw.loc[neo & raw["has_tcell"] & raw["has_positive"], "tier"] = "positive"
    raw.loc[neo & raw["has_tcell"] & ~raw["has_positive"], "tier"] = "exp_negative"
    raw.loc[raw["group"] == "viral", "tier"] = "bg_viral"
    raw.loc[raw["group"] == "taa", "tier"] = "bg_taa"
    for c in v2.LIST_COLS + ["mhc_allele_names"]:
        raw[c] = raw[c].apply(lambda v: ";".join(map(str, v)) if isinstance(v, list) else v)
    return raw.drop(columns=v2.ID_COLS).assign(source="cedar", evidence="validated").reset_index(drop=True)


def _write_meta(raw, path):
    neo = raw[raw["group"] == "neo"]
    meta = {"dataset_version": DATASET_VERSION, "download_date": f"{datetime.date.today():%Y-%m-%d}",
            "snapshot": path.name, "rows": int(len(raw)), "rows_per_group": raw["group"].value_counts().to_dict(),
            "neoantigen_rows_per_disease_group": neo["disease_group"].value_counts().to_dict(),
            "queries": {"neo": {**v2.LINEAR, "neoantigen_bool": "eq.1"},
                        "viral": {**v2.LINEAR, "neoantigen_bool": "eq.0", "viral_antigen_bool": "eq.1"},
                        "taa": {**v2.LINEAR, "neoantigen_bool": "eq.0", "other_antigen_bool": "eq.1",
                                "viral_antigen_bool": "eq.0", "tcell_ids": "not.is.null"}},
            "disease_groups": DISEASE_GROUPS, "not_cancer_re": NOT_CANCER_RE,
            # the labelling rules above are applied at download time, so the code version belongs with the snapshot
            "sha256": config.file_sha256(path), "git_sha": config.git_sha(), "tool_versions": config.tool_versions()}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def _new_path(data_dir):
    """pancancer_cedar_raw_YYYYMMDD.csv, or _2, _3... when that day already has one (a snapshot is never overwritten)."""
    stem = f"pancancer_cedar_raw_{datetime.date.today():%Y%m%d}"
    path, n = data_dir / f"{stem}.csv", 1
    while path.exists():
        n += 1
        path = data_dir / f"{stem}_{n}.csv"
    return path


def _snapshot_path(data_dir, snapshot):
    """A snapshot given as a date ("20261110"), a file name, or a path."""
    p = Path(str(snapshot))
    if p.suffix != ".csv":
        p = data_dir / f"pancancer_cedar_raw_{snapshot}.csv"
    elif not p.is_absolute():
        p = data_dir / p
    if not p.exists():
        raise FileNotFoundError(f"snapshot {p} not found; available: {[q.name for q in sorted(data_dir.glob('pancancer_cedar_raw_*.csv'))]}")
    return p


def _load_snapshot(path):
    """Read a snapshot, check it against the checksum in its meta.json, and remember which file it came from."""
    raw = pd.read_csv(path)
    sha = config.file_sha256(path)
    meta = path.with_suffix(".meta.json")
    expected = json.loads(meta.read_text()).get("sha256") if meta.exists() else None
    if expected and expected != sha:
        warnings.warn(f"{path.name} does not match the checksum in its meta.json: it was modified after download")
    raw.attrs.update(snapshot_file=path.name, snapshot_sha256=sha)     # copied into every dataset meta.json
    return raw


def get_raw(data_dir, refresh=False, snapshot=None):
    """A dated snapshot pancancer_cedar_raw_YYYYMMDD[_n].csv in data_dir (CEDAR changes over time: the file IS the data).

    snapshot: a date, file name or path that pins the exact snapshot a result was built from. None takes the latest and
    warns when several exist, because a newer file would silently change the data. Downloads a new snapshot when none
    exists or refresh=True; an existing file is never overwritten (a second download on the same day gets a _2 suffix).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if snapshot:
        path = _snapshot_path(data_dir, snapshot)
        print(f"using snapshot {path.name}")
        return _load_snapshot(path)
    snapshots = sorted(data_dir.glob("pancancer_cedar_raw_*.csv"))
    if snapshots and not refresh:
        if len(snapshots) > 1:
            warnings.warn(f"{len(snapshots)} snapshots in {data_dir}; using the latest, {snapshots[-1].name}. "
                          f"Pass snapshot=... to pin one: {[s.name for s in snapshots]}")
        print(f"using snapshot {snapshots[-1].name}")
        return _load_snapshot(snapshots[-1])
    raw = download_raw()
    path = _new_path(data_dir)
    raw.to_csv(path, index=False)
    _write_meta(raw, path)
    raw.attrs.update(snapshot_file=path.name, snapshot_sha256=config.file_sha256(path))
    print(f"downloaded {len(raw)} rows ({(raw['group'] == 'neo').sum()} neoantigens) -> {path.name}")
    return raw


def summary(raw):
    """Per disease group: how much usable data there is (decides which diseases can be their own train/test arm)."""
    neo = raw[raw["group"] == "neo"]
    rows = []
    for g, d in neo.groupby("disease_group"):
        yr = d.loc[d["year_first"] > 0, "year_first"]
        rows.append({"disease_group": g, "rows": len(d), "unique_seq": d["linear_sequence"].nunique(),
                     "positive_seq": d.loc[d["tier"] == "positive", "linear_sequence"].nunique(),
                     "exp_negative_seq": d.loc[d["tier"] == "exp_negative", "linear_sequence"].nunique(),
                     "unknown_seq": d.loc[d["tier"] == "unknown", "linear_sequence"].nunique(),
                     "median_year": int(yr.median()) if len(yr) else 0})
    return pd.DataFrame(rows).sort_values("positive_seq", ascending=False).reset_index(drop=True)


def discover_terms(raw):
    """Distinct disease names that landed in `other` (with row counts), to check/extend DISEASE_GROUPS."""
    counts = {}
    for names in raw.loc[raw["group"] == "neo", "disease_names"].dropna():
        for n in str(names).split(";"):
            if name_to_group(n) == OTHER:
                counts[n] = counts.get(n, 0) + 1
    return pd.Series(counts, name="rows", dtype=int).sort_values(ascending=False)
