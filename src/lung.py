"""Lung cancer (NSCLC + SCLC) neoantigen dataset: CEDAR positives + optional extra positives.

Same design as `cedar_v2.py` (dated raw snapshot, configurable filters afterwards, downstream steps reused from
`cedar.py`), but the positive pool is lung-cancer neoantigens instead of melanoma. Negatives (viral / TAA epitopes)
are not disease specific, so they are the same pools as in `cedar_v2`.

Extra sources: TSNAdb, NeoPepDB, dbPepNeo have no stable bulk-download API (TSNAdb's download links are
JavaScript-driven). Download a file by hand into DATA_DIR/lung_extra/ and load it with `load_extra`. Note that TSNAdb
holds *predicted* neoantigens, not T cell-validated ones; mixing them with CEDAR positives adds label noise, so each
extra source is opt-in and its rows carry `source` and `evidence` columns.

Outputs go to DATA_DIR/<preset name>/ with the same file names as the melanoma builds.
"""
import datetime
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import requests

from . import cedar, cedar_v2 as v2

# Exact CEDAR `disease_names` values seen for neoantigens (discover_lung_terms() refreshes this list).
LUNG_TERMS = ("lung non-small cell carcinoma", "lung squamous cell carcinoma", "lung non-squamous non-small cell carcinoma",
              "lung cancer", "lung adenocarcinoma", "lung small cell carcinoma", "adenosquamous lung carcinoma")
LUNG_RE = r"lung|nsclc|sclc"
NOT_CANCER_RE = r"fibrosis"
DATASET_VERSION = "lung_v1"
EXTRA_ID_BASE =9_000_000_000   # synthetic structure_id for rows not from CEDAR (never collides with CEDAR ids)
EXTRA_COLS = ["sequence", "hla", "gene", "mutation", "disease", "source", "pubmed_id", "evidence"]


@dataclass(frozen=True)
class LungPreset:
    name: str
    pos_tcell: bool = True            # CEDAR positives need a T cell assay
    pos_positive: bool = True         # ... with a "Positive*" result
    extra_sources: tuple = ()         # names of files loaded with load_extra() to add as positives
    neg_sources: tuple = ("viral", "taa")   # background negatives (guide: not experimental negatives)
    exp_negative: bool = False        # also add lung neoantigens with only negative T cell results as class 0
    sim_id: float = 0.4
    min_len: int = 8
    max_len: int | None = 35
    seed: int = 42


PRESETS = {p.name: p for p in [
    LungPreset("lung_cedar"),
    LungPreset("lung_cedar_any_assay", pos_tcell=False, pos_positive=False),
]}


def discover_lung_terms():
    """Distinct lung-cancer `disease_names` among CEDAR neoantigens (so LUNG_TERMS can be checked/refreshed)."""
    d = v2._fetch({**v2.LINEAR, "neoantigen_bool": "eq.1"})
    names = {x for v in d["disease_names"] for x in (v if isinstance(v, list) else [])}
    return sorted(n for n in names if re.search(LUNG_RE, n, re.I) and not re.search(NOT_CANCER_RE, n, re.I))


def _fetch(filters, page=10000):
    """Like cedar_v2._fetch but also selects the allele names (kept for provenance and later HLA-aware work)."""
    rows, offset = [], 0
    while True:
        r = requests.get(cedar.API, params={**filters, "select": v2.COLS + ",mhc_allele_names", "order": "structure_id",
                                            "limit": page, "offset": offset}, timeout=300)
        r.raise_for_status()
        chunk = r.json()
        rows += chunk
        if len(chunk) < page:
            return pd.DataFrame(rows)
        offset += page


def download_raw():
    """Lung neoantigens + viral + TAA pools with the columns `cedar_v2` expects."""
    lung = "or=(" + ",".join(f"disease_names.cs.{{{t}}}" for t in LUNG_TERMS) + ")"
    # `_fetch` passes filters as params, so the `or` filter goes in as a key/value pair
    pos = _fetch({**v2.LINEAR, "neoantigen_bool": "eq.1", "or": lung[3:]}).assign(label=1, group="neo")
    viral = _fetch({**v2.LINEAR, "neoantigen_bool": "eq.0", "viral_antigen_bool": "eq.1"}).assign(label=0, group="viral")
    taa = _fetch({**v2.LINEAR, "neoantigen_bool": "eq.0", "other_antigen_bool": "eq.1", "viral_antigen_bool": "eq.0",
                  "tcell_ids": "not.is.null"}).assign(label=0, group="taa")
    raw = pd.concat([pos, viral, taa], ignore_index=True)
    ls = lambda v: v if isinstance(v, list) else []
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
    # label tiers (guide step 4): experimental negatives are lung neoantigens tested by T cell assay with no positive
    # result; lung neoantigens without any T cell assay are unknown (never turned into negatives)
    neo = raw["group"] == "neo"
    raw["tier"] = "unknown"
    raw.loc[neo & raw["has_tcell"] & raw["has_positive"], "tier"] = "positive"
    raw.loc[neo & raw["has_tcell"] & ~raw["has_positive"], "tier"] = "exp_negative"
    raw.loc[raw["group"] == "viral", "tier"] = "bg_viral"
    raw.loc[raw["group"] == "taa", "tier"] = "bg_taa"
    for c in v2.LIST_COLS + ["mhc_allele_names"]:
        raw[c] = raw[c].apply(lambda v: ";".join(map(str, v)) if isinstance(v, list) else v)
    # safety net: keep lung positives only (drops anything non-cancer the `cs` match could let through)
    ok = raw["disease_names"].fillna("").str.contains(LUNG_RE, case=False, regex=True)
    return raw[~neo | ok].drop(columns=v2.ID_COLS).assign(source="cedar", evidence="validated").reset_index(drop=True)


def get_raw(data_dir, refresh=False):
    """Latest dated snapshot lung_cedar_raw_YYYYMMDD.csv in data_dir; downloads a new one if none exists or refresh."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshots = sorted(data_dir.glob("lung_cedar_raw_*.csv"))
    if snapshots and not refresh:
        raw = pd.read_csv(snapshots[-1])
        if "tier" in raw.columns:
            print(f"using snapshot {snapshots[-1].name}")
            return raw
        print(f"{snapshots[-1].name} predates the provenance columns (year_first, tier); downloading a new snapshot")
    raw = download_raw()
    path = data_dir / f"lung_cedar_raw_{datetime.date.today():%Y%m%d}.csv"
    raw.to_csv(path, index=False)
    print(f"downloaded {len(raw)} rows ({(raw['group'] == 'neo').sum()} lung positives) -> {path.name}")
    return raw


def load_extra(path, source, seq_col, disease_col=None, hla_col=None, sep=None, evidence="predicted"):
    """Read one hand-downloaded table (csv/tsv/xlsx) of extra lung positives into the common EXTRA_COLS schema.

    seq_col / disease_col / hla_col are the column names in that file; rows are kept when the disease matches lung
    (if disease_col is given). `evidence` is "validated" or "predicted" and is kept for later filtering.
    """
    path = Path(path)
    df = pd.read_excel(path) if path.suffix in (".xlsx", ".xls") else pd.read_csv(path, sep=sep, engine="python")
    if seq_col not in df.columns:
        raise ValueError(f"{path.name}: no column {seq_col!r}; columns are {list(df.columns)}")
    if disease_col:
        df = df[df[disease_col].astype(str).str.contains(LUNG_RE, case=False, regex=True)]
    out = pd.DataFrame({"sequence": df[seq_col].astype(str), "hla": df[hla_col] if hla_col else None})
    return out.assign(source=source, evidence=evidence).reindex(columns=EXTRA_COLS)


def add_extra(raw, extras):
    """Append extra positives to the raw table as CEDAR-like rows (group 'neo', synthetic structure_id)."""
    rows = []
    for e in extras:
        e = e.drop_duplicates("sequence").reset_index(drop=True)
        rows.append(pd.DataFrame({
            "structure_id": EXTRA_ID_BASE + len(raw) + sum(map(len, rows)) + e.index, "linear_sequence": e["sequence"],
            "label": 1, "group": "neo", "has_tcell": True, "has_positive": True, "has_mutation": False,
            "tier": "positive", "year_first": 0, "year_last": 0, "disease_names": "lung cancer", "source": e["source"], "evidence": e["evidence"]}))
    return pd.concat([raw.assign(source="cedar", evidence="validated")] + rows, ignore_index=True) if rows \
        else raw.assign(source="cedar", evidence="validated")


def select(raw, p):
    """Apply the preset's filters. Extra-source positives are kept only if their source is in p.extra_sources."""
    pos = raw["group"] == "neo"
    cedar_pos = pos & (raw["source"] == "cedar")
    if p.pos_tcell:
        cedar_pos &= raw["has_tcell"]
    if p.pos_positive:
        cedar_pos &= raw["has_positive"]
    keep_pos = cedar_pos | (pos & raw["source"].isin(p.extra_sources))
    neg = raw["group"].isin(p.neg_sources) & (raw["has_tcell"] | (raw["group"] == "viral"))
    exp_neg = (raw["tier"] == "exp_negative") & p.exp_negative
    out = raw[(keep_pos & ~exp_neg) | neg | exp_neg].copy()
    if p.exp_negative:
        out.loc[out["tier"] == "exp_negative", "label"] = 0   # experimental negatives are class 0 (never positives)
    return out


def build(raw, p):
    """Cleaned one-row-per-sequence frame (before CD-HIT). Returns (df, info) incl. positives per source."""
    sel = select(raw, p)
    # keep the earliest dated row of a sequence (undated last) so year_first is the first public evidence
    sel = sel.assign(_y=sel["year_first"].where(sel["year_first"] > 0, 9999)).sort_values("_y", kind="stable").drop(columns="_y")
    sel = sel.assign(linear_sequence_length=sel["linear_sequence"].astype(str).str.len())
    df, info = cedar.build_initial(sel, require_positive_assay=False)
    df = df[(df["length"] >= p.min_len) | (df["label"] == 0)] if p.min_len else df
    pos_src = df[df["label"] == 1]["source"].value_counts().to_dict()
    return df.reset_index(drop=True), {**info, "positives_per_source": pos_src}


def sweep(raw, presets=PRESETS):
    """Class counts per preset (cheap: no CD-HIT)."""
    rows = []
    for p in presets.values():
        df, _ = build(raw, p)
        ln = df if p.max_len is None else df[df["length"] <= p.max_len]
        n, o = int((ln["label"] == 1).sum()), int((ln["label"] == 0).sum())
        rows.append({"preset": p.name, "pos": n, "neg": o, "neg/pos": round(o / max(n, 1), 1)})
    return pd.DataFrame(rows)


def run(raw, p, data_dir, work_dir):
    """Full pipeline: clean -> CD-HIT -> length -> 80/20 split -> tables -> files in data_dir/<name>/ (needs cd-hit)."""
    df, info = build(raw, p)
    sim = cedar.cdhit_filter(df, Path(work_dir) / p.name, sim_id=p.sim_id)
    ln = (sim if p.max_len is None else sim[sim["length"] <= p.max_len]).reset_index(drop=True)
    train, test = cedar.split(ln, test_size=0.2, seed=p.seed)
    final = pd.concat([train, test], ignore_index=True)
    tables = cedar.stats_tables(df, sim, ln, train, test, p.sim_id, p.max_len)
    cedar.save_outputs(Path(data_dir) / p.name, df, final, tables, {**asdict(p), **info, "disease": "lung"})
    return df, final, tables
