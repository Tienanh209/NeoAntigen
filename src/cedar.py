"""Collect and preprocess the CEDAR dataset the way the DeepNeoAG paper describes (Table 1)."""
import datetime
import json
import subprocess
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

API = "https://cedar-api.iedb.org/epitope_search"   # CEDAR public PostgREST API
COLS = ("structure_id,linear_sequence,linear_sequence_length,qualitative_measures,disease_names,"
        "neoantigen_bool,viral_antigen_bool,other_antigen_bool,mhc_classes,"
        "parent_source_antigen_names,source_organism_names,pubmed_ids")
LIST_COLS = ["qualitative_measures", "disease_names", "mhc_classes", "parent_source_antigen_names",
             "source_organism_names", "pubmed_ids"]
STD_AA = "[ACDEFGHIKLMNPQRSTVWY]+"
# Common: linear peptides with T cell assay data (the paper: "T cell epitopes related to melanoma")
BASE = {"structure_type": "eq.Linear peptide", "tcell_ids": "not.is.null"}


def fetch(filters, page=10000):
    rows, offset = [], 0
    while True:
        r = requests.get(API, params={**filters, "select": COLS, "order": "structure_id",
                                      "limit": page, "offset": offset}, timeout=300)
        r.raise_for_status()
        chunk = r.json()
        rows += chunk
        if len(chunk) < page:
            return pd.DataFrame(rows)
        offset += page


def download_raw():
    """Unfiltered CEDAR rows for both classes. Filters are applied later so they can be toggled without re-downloading."""
    # Positive: neoantigen epitopes from melanoma
    pos = fetch({**BASE, "neoantigen_bool": "eq.1", "disease_names": "cs.{melanoma}"})
    # Negative: non-neoantigen epitopes from viral antigens and tumor-associated antigens (TAAs)
    neg = fetch({**BASE, "neoantigen_bool": "eq.0", "or": "(viral_antigen_bool.eq.1,other_antigen_bool.eq.1)"})
    raw = pd.concat([pos.assign(label=1), neg.assign(label=0)], ignore_index=True)
    # "Confirmed" = at least one positive T cell assay result
    raw["has_positive_assay"] = raw["qualitative_measures"].apply(
        lambda q: isinstance(q, list) and any(x.startswith("Positive") for x in q))
    for c in LIST_COLS:
        raw[c] = raw[c].apply(lambda v: ";".join(map(str, v)) if isinstance(v, list) else v)
    return raw


def get_raw(data_dir, refresh=False):
    """Latest dated snapshot cedar_raw_YYYYMMDD.csv in data_dir; downloads a new one if none exists or refresh=True."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshots = sorted(data_dir.glob("cedar_raw_*.csv"))
    if snapshots and not refresh:
        print(f"using snapshot {snapshots[-1].name}")
        return pd.read_csv(snapshots[-1])
    raw = download_raw()
    path = data_dir / f"cedar_raw_{datetime.date.today():%Y%m%d}.csv"
    raw.to_csv(path, index=False)
    print(f"downloaded {len(raw)} rows -> {path.name}")
    return raw


def build_initial(raw, require_positive_assay=True):
    """Clean the raw rows into one row per sequence. Returns (df, info)."""
    df = raw.copy()
    if require_positive_assay:
        df = df[(df["label"] == 0) | df["has_positive_assay"]]
    df = df.rename(columns={"linear_sequence": "sequence", "linear_sequence_length": "length"})
    info = {"rows_before_cleaning": len(df)}

    # CEDAR has a few sequences with stray spaces (typos) -> strip them
    df["sequence"] = df["sequence"].fillna("").str.replace(r"\s+", "", regex=True)
    df["length"] = df["sequence"].str.len()

    # Keep only the 20 standard amino acids (drops X, B, Z, U, ...)
    ok = df["sequence"].str.fullmatch(STD_AA)
    info["non_standard_dropped"] = int((~ok).sum())
    df = df[ok]

    # One row per sequence; a sequence seen as both classes is ambiguous -> drop
    df = df.drop_duplicates(["sequence", "label"])
    dup = df[df.duplicated("sequence", keep=False)]["sequence"].unique()
    info["ambiguous_dropped"] = len(dup)
    df = df[~df["sequence"].isin(dup)].reset_index(drop=True)
    df["uid"] = df["structure_id"].astype(str) + "_" + df["label"].astype(str)
    return df, info


def cdhit_filter(df, work_dir, sim_id=0.4, min_len=5):
    """Redundancy removal with CD-HIT (needs `apt-get install cd-hit`).

    CD-HIT cannot go below 0.4 identity for proteins (word size -n 2), so the paper's 30% is approximated by 0.4.
    -l matters: the default (-l 10) silently DROPS every peptide of <=10 aa from the output.
    """
    assert sim_id >= 0.4, "CD-HIT does not support identity below 0.4"
    n = 5 if sim_id >= 0.7 else 4 if sim_id >= 0.6 else 3 if sim_id >= 0.5 else 2
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # CD-HIT keeps the first sequence of a cluster: neoantigens first, then longer first
    with open(work / "all.fa", "w") as f:
        for r in df.sort_values(["label", "length"], ascending=False, kind="stable").itertuples():
            f.write(f">{r.uid}\n{r.sequence}\n")
    with open(work / "cdhit.log", "w") as log:
        subprocess.run(["cd-hit", "-i", str(work / "all.fa"), "-o", str(work / "nr.fa"), "-c", str(sim_id),
                        "-n", str(n), "-l", str(min_len), "-d", "0", "-M", "0", "-T", "0"],
                       stdout=log, check=True)
    reps = {l[1:].split()[0] for l in open(work / "nr.fa") if l.startswith(">")}
    return df[df["uid"].isin(reps)]


def cdhit_clusters(work_dir, clstr_name="nr.fa.clstr"):
    """Parse CD-HIT's .clstr file into {uid: cluster_id}.

    `cdhit_filter` keeps only one representative per cluster and throws the
    membership away, but the leakage experiment needs to know which peptides are
    near-duplicates of each other so they can be kept on the same side of a split.
    Run `cdhit_filter` first; this reads the file it left behind.
    """
    path = Path(work_dir) / clstr_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run cdhit_filter first")
    out, cid = {}, -1
    for line in path.read_text().splitlines():
        if line.startswith(">Cluster"):
            cid = int(line.split()[1])
        elif ">" in line:
            # e.g.  0\t9aa, >12345_1... *
            out[line.split(">", 1)[1].split("...")[0]] = cid
    return out


def split(df, test_size=0.2, seed=42):
    train, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    return train.assign(split="train"), test.assign(split="test")


def mhc_class(r):
    """CEDAR annotation first; if missing/both, fall back to the paper's length rule (MHC-I 8-11 aa, MHC-II 12-25 aa)."""
    c = set(str(r["mhc_classes"]).replace("nan", "").split(";")) & {"I", "II"}
    if len(c) == 1:
        return c.pop()
    return "I" if r["length"] <= 11 else "II"


def stats_tables(df, sim, ln, train, test, sim_id, max_len):
    """Tables in the same layout as Table 1 of the paper."""
    def counts(d):
        n, o = (d["label"] == 1).sum(), (d["label"] == 0).sum()
        return [n, o, n + o]

    rows = ["Neoantigen", "Others", "Total"]
    t1 = pd.DataFrame({"Dataset": rows, "Epitopes": counts(df),
                       f"Similarity <{int(sim_id * 100)} %": counts(sim),
                       f"Length <={max_len}" if max_len else "No length filter": counts(ln)})
    t2 = pd.DataFrame({"Dataset": rows, "Training data (80 %)": counts(train),
                       "Independent data (20 %)": counts(test)})
    test = test.assign(mhc=test.apply(mhc_class, axis=1))
    t3 = pd.DataFrame({"Independent dataset": rows,
                       "MHC-I presented peptides": counts(test[test["mhc"] == "I"]),
                       "MHC-II presented peptides": counts(test[test["mhc"] == "II"])})
    return t1, t2, t3


def write_fasta(df, path):
    with open(path, "w") as f:
        for r in df.itertuples():
            f.write(f">CEDAR_{r.structure_id}|label={r.label}\n{r.sequence}\n")


def save_outputs(data_dir, df, final, tables, params):
    """Write the datasets, the statistics and meta.json (the parameters that produced them)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "cedar_initial.csv", index=False)
    write_fasta(df, data_dir / "cedar_initial.fasta")
    final.to_csv(data_dir / "deepneoag_like_dataset.csv", index=False)
    for name, g in final.groupby("split"):
        write_fasta(g, data_dir / f"{name}.fasta")
    (data_dir / "dataset_statistics.txt").write_text("\n\n".join(t.to_string(index=False) for t in tables))
    (data_dir / "meta.json").write_text(json.dumps(
        {**params, "created": datetime.datetime.now().isoformat(timespec="seconds")}, indent=2))
