"""Regenerate the *_authors.ipynb notebooks from the CEDAR ones so the two never drift apart.

    python tools/make_authors_notebooks.py

The authors' versions differ in three ways only: the dataset comes from
`authors.load()`, experiment ids carry an `AU_` prefix (so their runs never mix
with the CEDAR runs in the results store), and the MHC-I / MHC-II block is replaced
by a length-confound check, because that dataset has no MHC annotation.
Outputs are stripped, as CLAUDE.md requires for anything new.
"""
import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_IDS = ["A_window", "B_combo", "C_filters", "J_hidden", "K_dropout", "D_plm", "E_clf",
           "G_baselines", "H_weight", "I_leak"]
AUTH_LOAD = "df = authors.load()      # github.com/B1607/DeepNeoAG at a pinned commit, cached on Drive"

LENGTH_BLOCK = '''# Block {n} — Experiment F: length confound (replaces the paper's MHC-I / MHC-II table)
# This dataset carries no MHC annotation, and a length rule puts only a handful of the
# 61 test positives in the 8-11 aa range, so that table cannot be rebuilt here. More
# important is what the length distribution shows: no peptide is shorter than 11 aa
# (consistent with CD-HIT's default -l 10, which drops them) and positives are much
# longer than negatives. If length alone separates the classes, a good AUC from a
# sequence model proves much less than it appears to.
print(df.groupby(["split", "label"])["length"].describe()[["count", "mean", "min", "50%", "max"]].round(1))

cfgs = [dict(label="length only (LogReg)", features={{"kind": "length"}},
             model={{"kind": "sklearn", "name": "logreg"}})]
runner.run_series("AU_F_length", cfgs, df=df, seeds=SEEDS, force=FORCE)
display(results.fmt(results.table("AU_F_length")))
display(results.fmt(results.table("AU_F_length", eval_split="cv")))'''


def src(c):
    return "".join(c["source"])


def setsrc(c, s):
    c["source"] = s.splitlines(keepends=True)


def sub(s, old, new):
    assert old in s, f"expected text not found: {old[:70]!r}"
    return s.replace(old, new, 1)


def clean(nb):
    for c in nb["cells"]:
        c["outputs"], c["execution_count"] = [], None
    return nb


def make_embeddings():
    nb = json.loads((ROOT / "embeddings.ipynb").read_text())
    for c in nb["cells"]:
        s = src(c)
        if "df = data.load_dataset()" in s:
            s = sub(s, "from src import config, data, embed", "from src import authors, config, data, embed")
            s = sub(s, "df = data.load_dataset()", AUTH_LOAD)
            s = sub(s, "# Block 1 — dataset and its build id (the cache key)",
                    "# Block 1 — the AUTHORS' dataset (7,012 peptides, Table 1 of the paper) and its build id")
            setsrc(c, s)
    (ROOT / "embeddings_authors.ipynb").write_text(json.dumps(clean(nb), indent=1, ensure_ascii=False))


def make_experiments():
    nb = copy.deepcopy(json.loads((ROOT / "experiments.ipynb").read_text()))
    for i, c in enumerate(nb["cells"]):
        s = src(c)
        if "df = data.load_dataset()" in s:
            s = sub(s, "from src import config, data, embed, results, runner",
                    "from src import authors, config, data, embed, results, runner")
            s = sub(s, "df = data.load_dataset()", AUTH_LOAD)
            s = sub(s, "# Block 1 — dataset, embedding caches, shared settings.",
                    "# Block 1 — the AUTHORS' dataset, embedding caches, shared settings.")
        if re.match(r"# Block \d+ — Experiment F: MHC", s):
            s = LENGTH_BLOCK.format(n=i)
        for e in EXP_IDS:
            s = s.replace(f'"{e}"', f'"AU_{e}"')
        if "TITLES = {" in s:
            s = sub(s, '"AU_G_baselines": "Baselines",', '"AU_G_baselines": "Baselines", "AU_F_length": "Length-only control",')
        setsrc(c, s)
    (ROOT / "experiments_authors.ipynb").write_text(json.dumps(clean(nb), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    make_embeddings()
    make_experiments()
    print("wrote embeddings_authors.ipynb and experiments_authors.ipynb")
