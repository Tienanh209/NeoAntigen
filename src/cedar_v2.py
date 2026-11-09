"""Second CEDAR collection, tuned to land close to Table 1 of the DeepNeoAG paper.

Why: `cedar.py` queries today's CEDAR, which has grown far past the paper's snapshot
(positives ~1100 vs 671, negatives differ in kind). This module downloads a richer raw
table once (`cedar_v2_raw_YYYYMMDD.csv`) and then applies *configurable* filters, so
presets can be compared against the paper's numbers without re-downloading.

Main knobs (see `Preset`): the snapshot year (`max_year`, emulates an older CEDAR via the
reference dates), which assay types count, whether the positive needs a mutation, and
which negative sources (viral / TAA) are used. Downstream steps (CD-HIT, length filter,
split, tables) are reused from `cedar.py`; outputs go to DATA_DIR/<preset name>/ with the
same file names, so `data.load_dataset(DATA_DIR / name)` works unchanged.
"""
import datetime
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import requests

from . import cedar

PAPER = {"pos": 671, "neg": 32206, "pos_sim": 303, "neg_sim": 6717, "pos_len": 302, "neg_len": 6710,
         "pos_train": 241, "neg_train": 5367, "pos_test": 61, "neg_test": 1343}

COLS = ("structure_id,linear_sequence,linear_sequence_length,qualitative_measures,disease_names,"
        "mhc_classes,reference_dates,pubmed_ids,mutation,tcell_ids,bcell_ids,elution_ids,"
        "viral_antigen_bool,other_antigen_bool,germline_antigen_bool,parent_source_antigen_names,"
        "source_organism_names")
LIST_COLS = ["qualitative_measures", "disease_names", "mhc_classes", "reference_dates", "pubmed_ids",
             "parent_source_antigen_names", "source_organism_names"]
ID_COLS = ["tcell_ids", "bcell_ids", "elution_ids"]
LINEAR = {"structure_type": "eq.Linear peptide"}


@dataclass(frozen=True)
class Preset:
    name: str
    # positives (neoantigen epitopes from melanoma)
    pos_tcell: bool = True            # needs a T cell assay (False also admits MHC-ligand-only rows)
    pos_positive: bool = True         # needs a "Positive*" qualitative result
    pos_mutation: bool = False        # needs a mutation annotation (a real neoepitope)
    # negatives (non-neoantigen epitopes)
    neg_sources: tuple = ("viral", "taa")
    neg_assay: str = "tcell"          # "tcell": T cell assay rows only | "viral_any": viral rows of any assay (TAA stays
                                      # T cell only; the TAA pool is T cell only in the download)
    neg_positive: bool = False        # needs a "Positive*" result
    # CEDAR snapshot emulation: keep entries whose reference years pass this cut-off
    max_year: int | None = None
    year_rule: str = "first"          # "first": earliest reference <= max_year | "last": all references <= max_year
    neg_snapshot: bool = True         # False: negatives come from today's CEDAR, only positives are cut at max_year
    # downstream (same meaning as in load_datasets.ipynb)
    sim_id: float = 0.4
    max_len: int | None = 35
    seed: int = 42


PRESETS = {p.name: p for p in [
    Preset("today"),                                              # what cedar.py effectively does
    Preset("snap2022", max_year=2022),
    Preset("snap2023", max_year=2023),
    Preset("snap2022_last", max_year=2022, year_rule="last"),
    Preset("snap2023_last", max_year=2023, year_rule="last"),
    Preset("mut_only", pos_mutation=True),
    Preset("snap2022_last_viral_any", max_year=2022, year_rule="last", neg_assay="viral_any"),
    Preset("snap2023_last_viral_any", max_year=2023, year_rule="last", neg_assay="viral_any"),
    Preset("snap2023_viral_any", max_year=2023, neg_assay="viral_any"),
    Preset("today_viral_any", neg_assay="viral_any"),
    # closest to the paper's 671 / 32,206: old positives, negatives from today's CEDAR
    Preset("pos2022_negtoday", max_year=2022, year_rule="last", neg_snapshot=False),
    Preset("strict_pos_neg", pos_positive=True, neg_positive=True, max_year=2023),
]}


def _fetch(filters, page=10000):
    rows, offset = [], 0
    while True:
        chunk = requests.get(cedar.API, params={**filters, "select": COLS, "order": "structure_id",
                                                       "limit": page, "offset": offset}, timeout=300)
        chunk.raise_for_status()
        chunk = chunk.json()
        rows += chunk
        if len(chunk) < page:
            return pd.DataFrame(rows)
        offset += page


def download_raw():
    """Broad pools, filtered later: melanoma neoantigens; viral epitopes (any assay); TAA/other epitopes (T cell)."""
    pos = _fetch({**LINEAR, "neoantigen_bool": "eq.1", "disease_names": "cs.{melanoma}"}).assign(label=1, group="neo")
    viral = _fetch({**LINEAR, "neoantigen_bool": "eq.0", "viral_antigen_bool": "eq.1"}).assign(label=0, group="viral")
    # TAA/other restricted to T cell assays: without it this pool is >1M MHC-ligand rows
    taa = _fetch({**LINEAR, "neoantigen_bool": "eq.0", "other_antigen_bool": "eq.1",
                  "viral_antigen_bool": "eq.0", "tcell_ids": "not.is.null"}).assign(label=0, group="taa")
    raw = pd.concat([pos, viral, taa], ignore_index=True)
    ls = lambda v: v if isinstance(v, list) else []
    raw["has_tcell"] = raw["tcell_ids"].notna()
    raw["has_positive"] = raw["qualitative_measures"].apply(lambda q: any(x.startswith("Positive") for x in ls(q)))
    years = raw["reference_dates"].apply(lambda d: [int(y) for y in ls(d) if str(y).isdigit()])
    raw["year_first"] = years.apply(lambda y: min(y) if y else 0)
    raw["year_last"] = years.apply(lambda y: max(y) if y else 0)
    raw["has_mutation"] = raw["mutation"].notna()
    raw["structure_id"] = raw["structure_id"].astype(int)
    for c in LIST_COLS:
        raw[c] = raw[c].apply(lambda v: ";".join(map(str, v)) if isinstance(v, list) else v)
    return raw.drop(columns=ID_COLS)


def get_raw(data_dir, refresh=False):
    """Latest dated snapshot cedar_v2_raw_YYYYMMDD.csv in data_dir; downloads a new one if none exists or refresh=True."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshots = sorted(data_dir.glob("cedar_v2_raw_*.csv"))
    if snapshots and not refresh:
        print(f"using snapshot {snapshots[-1].name}")
        return pd.read_csv(snapshots[-1])
    raw = download_raw()
    path = data_dir / f"cedar_v2_raw_{datetime.date.today():%Y%m%d}.csv"
    raw.to_csv(path, index=False)
    print(f"downloaded {len(raw)} rows -> {path.name}")
    return raw


def select(raw, p):
    """Apply a preset's filters to the raw pools. Returns rows (still one per CEDAR structure)."""
    year = raw["year_first" if p.year_rule == "first" else "year_last"]
    keep = (year <= p.max_year) & (year > 0) if p.max_year else pd.Series(True, index=raw.index)

    pos = (raw["group"] == "neo") & keep
    if p.pos_tcell:
        pos &= raw["has_tcell"]
    if p.pos_positive:
        pos &= raw["has_positive"]
    if p.pos_mutation:
        pos &= raw["has_mutation"]

    neg = raw["group"].isin(p.neg_sources) & (keep if p.neg_snapshot else True)
    if p.neg_assay == "tcell":
        neg &= raw["has_tcell"]
    elif p.neg_assay == "viral_any":
        neg &= raw["has_tcell"] | (raw["group"] == "viral")
    if p.neg_positive:
        neg &= raw["has_positive"]
    return raw[pos | neg]


def build(raw, p):
    """Cleaned one-row-per-sequence frame for a preset (before CD-HIT). Returns (df, info)."""
    df, info = cedar.build_initial(select(raw, p), require_positive_assay=False)
    return df, {**info, "viral": int(((df["group"] == "viral") & (df["label"] == 0)).sum()),
                "taa": int(((df["group"] == "taa") & (df["label"] == 0)).sum())}


def sweep(raw, presets=PRESETS):
    """Class counts per preset next to the paper's pre-CD-HIT numbers (671 / 32,206). Cheap: no CD-HIT."""
    rows = [{"preset": "PAPER", "pos": PAPER["pos"], "neg": PAPER["neg"], "pos<=len": "", "neg<=len": "",
             "neg/pos": round(PAPER["neg"] / PAPER["pos"], 1)}]
    for p in presets.values():
        df, info = build(raw, p)
        n, o = int((df["label"] == 1).sum()), int((df["label"] == 0).sum())
        ln = df if p.max_len is None else df[df["length"] <= p.max_len]
        rows.append({"preset": p.name, "pos": n, "neg": o, "pos<=len": int((ln["label"] == 1).sum()),
                     "neg<=len": int((ln["label"] == 0).sum()), "neg/pos": round(o / max(n, 1), 1),
                     "viral": info["viral"], "taa": info["taa"]})
    return pd.DataFrame(rows)


def run(raw, p, data_dir, work_dir):
    """Full pipeline for one preset: clean -> CD-HIT -> length -> 80/20 split -> tables -> files in data_dir/<name>/."""
    df, info = build(raw, p)
    sim = cedar.cdhit_filter(df, Path(work_dir) / p.name, sim_id=p.sim_id)
    ln = (sim if p.max_len is None else sim[sim["length"] <= p.max_len]).reset_index(drop=True)
    train, test = cedar.split(ln, test_size=0.2, seed=p.seed)
    final = pd.concat([train, test], ignore_index=True)
    tables = cedar.stats_tables(df, sim, ln, train, test, p.sim_id, p.max_len)
    cedar.save_outputs(Path(data_dir) / p.name, df, final, tables, {**asdict(p), **info})
    return df, final, tables
