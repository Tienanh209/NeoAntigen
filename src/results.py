"""Append-only store for experiment runs, plus table and ROC rendering.

Per-sample predictions are the source of truth; metrics are recomputed from them.
Changing a metric definition or a decision threshold therefore never requires
retraining, and the report block works even when each experiment ran in a
different Colab session.

Layout (under RESULTS_DIR):
    runs/{run_id}.json    config + provenance + cached metrics; written LAST, so
                          its existence proves the predictions are complete
    preds/{run_id}.csv    uid, split (cv|test), fold, seed, y_true, y_score
    ckpt/{run_id}/        per-epoch resumable state, pruned once the run commits
    figs/{exp_id}_*.png   regenerable, never a source of truth

One immutable file per run rather than one shared log: appends on Google Drive's
FUSE mount can interleave or truncate when two cells finish at once.
"""
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, metrics

THR_MODES = ("mcc_cv", "youden_cv", "fixed")


def _dir(name, results_dir=None):
    p = Path(results_dir or config.RESULTS_DIR) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def cfg_hash(cfg):
    """Stable 12-hex digest of a run configuration.

    Excludes device, worker counts, verbosity and paths, so moving between
    machines does not invalidate a completed run.
    """
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def run_id(exp_id, cfg):
    return f"{exp_id}__{cfg_hash(cfg)}"


def done(exp_id, cfg, results_dir=None):
    """True when this exact config has already been committed."""
    return (_dir("runs", results_dir) / f"{run_id(exp_id, cfg)}.json").exists()


def save_run(exp_id, cfg, preds, extra=None, results_dir=None):
    """Commit a run: predictions first, then the JSON marker, each via atomic replace."""
    rid = run_id(exp_id, cfg)
    preds = preds[["uid", "split", "fold", "seed", "y_true", "y_score"]]

    pred_path = _dir("preds", results_dir) / f"{rid}.csv"
    tmp = pred_path.with_suffix(".csv.tmp")
    preds.to_csv(tmp, index=False)
    os.replace(tmp, pred_path)

    record = {"run_id": rid, "exp_id": exp_id, "config": cfg,
              "git_sha": config.git_sha(),
              "created": datetime.datetime.now().isoformat(timespec="seconds"),
              **(extra or {})}
    run_path = _dir("runs", results_dir) / f"{rid}.json"
    tmp = run_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str))
    os.replace(tmp, run_path)

    ckpt = _dir("ckpt", results_dir) / rid       # training state is scratch once committed
    if ckpt.exists():
        shutil.rmtree(ckpt, ignore_errors=True)
    return record


def load_runs(exp_id=None, results_dir=None):
    """Committed runs as a DataFrame, config flattened into dotted columns."""
    pattern = f"{exp_id}__*.json" if exp_id else "*.json"
    rows = []
    for p in sorted(_dir("runs", results_dir).glob(pattern)):
        rec = json.loads(p.read_text())
        flat = {"run_id": rec["run_id"], "exp_id": rec["exp_id"],
                "label": rec.get("label", ""), "created": rec.get("created", ""),
                "git_sha": rec.get("git_sha", "")}
        flat.update(_flatten(rec.get("config", {})))
        for k, v in rec.items():
            if k not in ("run_id", "exp_id", "config", "created", "git_sha", "label"):
                flat[k] = v
        rows.append(flat)
    return pd.DataFrame(rows)


def latest_per_label(runs, label_col="label"):
    """One run per label, the most recently committed.

    Changing a setting such as the number of seeds changes the config hash, so the
    same experiment can have several committed runs. A table or ROC plot should
    show the current one, not every historical attempt under the same name.
    """
    if runs.empty:
        return runs
    runs = runs.copy()
    key = runs[label_col].where(runs[label_col].astype(bool), runs["run_id"]) if label_col in runs else runs["run_id"]
    runs["_key"] = key
    runs = runs.sort_values("created").drop_duplicates("_key", keep="last")
    return runs.drop(columns="_key")


def load_preds(rid, results_dir=None):
    return pd.read_csv(_dir("preds", results_dir) / f"{rid}.csv")


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = json.dumps(v) if isinstance(v, (list, tuple)) else v
    return out


def _threshold(cv, thr_mode):
    """Operating point chosen on cross-validation predictions only."""
    if thr_mode == "fixed" or cv.empty:
        return 0.5
    objective = "mcc" if thr_mode == "mcc_cv" else "youden"
    return metrics.best_threshold(cv["y_true"], cv["y_score"], objective=objective)


def _transfer_threshold(thr, cv_scores, test_scores):
    """Carry a CV-tuned threshold over to the test scores.

    The CV scores are single-model out-of-fold predictions, whereas each test score
    is an average over several fold models (and seeds). Averaging compresses the
    distribution, so applying the raw CV threshold can sit above every test score:
    the ranking (AUC) is fine yet sensitivity collapses to 0. Instead, flag the same
    *fraction* of peptides as positive that the CV threshold flagged. This uses test
    scores but never test labels, and assumes the test split has the CV prevalence,
    which the stratified split guarantees.
    """
    q = float(np.mean(np.asarray(cv_scores) >= thr))
    k = int(round(q * len(test_scores)))
    if k <= 0:
        return float(np.max(test_scores)) + 1e-9
    return float(np.sort(np.asarray(test_scores))[::-1][k - 1])


def run_metrics(rid, eval_split="test", thr_mode="mcc_cv", subset=None, thr_on=None, results_dir=None):
    """Metrics for one run. The threshold always comes from the CV predictions.

    `subset` restricts evaluation to a set of uids (used for the MHC-I / MHC-II
    breakdown, which needs no retraining). The threshold is fixed on the whole
    split first, so a subset is judged at the same operating point as the full set.
    `thr_on` (uids) fixes that operating point on those rows only: when the test split mixes
    several diseases, pass the arm's own test so every disease is judged at its operating point.
    """
    preds = load_preds(rid, results_dir)
    cv = preds[preds["split"] == "cv"]
    thr = _threshold(cv, thr_mode)

    ev = cv if eval_split == "cv" else preds[preds["split"] == "test"]
    # Several seeds predict the same test peptide; average them into one score per peptide.
    if eval_split == "test" and not ev.empty:
        ev = ev.groupby("uid", as_index=False).agg(y_true=("y_true", "first"),
                                                   y_score=("y_score", "mean"))
        if thr_mode != "fixed" and not cv.empty:
            ref = ev if thr_on is None else ev[ev["uid"].isin(set(thr_on))]
            if not ref.empty:
                thr = _transfer_threshold(thr, cv["y_score"], ref["y_score"])

    if subset is not None:
        ev = ev[ev["uid"].isin(set(subset))]
    if ev.empty:
        return None
    out = metrics.all_metrics(ev["y_true"], ev["y_score"], thr)
    out["n_seeds"] = int(preds["seed"].nunique())
    return out


def seed_spread(rid, eval_split="test", thr_mode="mcc_cv", results_dir=None, subset=None):
    """Per-seed AUC, as mean / std (over the uids in `subset` when given).

    With a single seed the std is NaN, not 0: there is no spread to measure, and
    printing 0.0 would read as perfect stability.
    """
    preds = load_preds(rid, results_dir)
    if subset is not None:
        preds = preds[preds["uid"].isin(set(subset))]
    aucs = []
    for _, g in preds[preds["split"] == eval_split].groupby("seed"):
        r = metrics.ranking_metrics(g["y_true"], g["y_score"])
        if not np.isnan(r["AUC"]):
            aucs.append(r["AUC"])
    if not aucs:
        return {"AUC_mean": float("nan"), "AUC_std": float("nan")}
    return {"AUC_mean": float(np.mean(aucs)),
            "AUC_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan")}


def table(exp_id, eval_split="test", thr_mode="mcc_cv", label_col="label",
          order_by=None, extra_cols=(), with_baseline=True, results_dir=None, subset=None, thr_on=None):
    """Comparison table in the paper's column layout, rebuilt from stored predictions.

    `subset` / `thr_on` (uids): score only those test rows / fix the operating point on them (see `run_metrics`).
    """
    runs = latest_per_label(load_runs(exp_id, results_dir), label_col)
    if runs.empty:
        return pd.DataFrame()

    rows = []
    for r in runs.to_dict("records"):   # dict, not itertuples: config columns contain dots
        m = run_metrics(r["run_id"], eval_split, thr_mode, subset=subset, thr_on=thr_on, results_dir=results_dir)
        if m is None:
            continue
        row = {label_col: r.get(label_col) or r["run_id"]}
        row.update({c: m[c] for c in metrics.PAPER_COLS})
        row.update({"PR_AUC": m["PR_AUC"], "F1": m["F1"], "Precision": m["Precision"]})
        row.update(seed_spread(r["run_id"], eval_split, thr_mode, results_dir, subset=subset))
        for c in extra_cols:
            row[c] = r.get(c)
        row["_order"] = r.get("series_order", 10**9)
        rows.append(row)

    t = pd.DataFrame(rows)
    # Default to the order the configurations were written in; run files are
    # globbed by config hash, which is otherwise arbitrary.
    t = t.sort_values(order_by if order_by and order_by in t.columns else "_order")
    t = t.drop(columns="_order")
    if with_baseline and rows:
        any_run = runs.iloc[0]["run_id"]
        preds = load_preds(any_run, results_dir)
        ev = preds[preds["split"] == eval_split]
        if not ev.empty:
            ev = ev.groupby("uid", as_index=False).agg(y_true=("y_true", "first"))
            if subset is not None:
                ev = ev[ev["uid"].isin(set(subset))]
            b = metrics.majority_baseline(ev["y_true"])
            base = {label_col: "all-negative baseline"}
            base.update({c: b[c] for c in metrics.PAPER_COLS})
            base.update({"PR_AUC": b["PR_AUC"], "F1": b["F1"], "Precision": b["Precision"]})
            t = pd.concat([t, pd.DataFrame([base])], ignore_index=True)
    return t.reset_index(drop=True)


def breakdown(rid, subsets, eval_split="test", thr_mode="mcc_cv", results_dir=None,
              index_name="Dataset", thr_on=None):
    """Re-evaluate one committed run on subsets of its predictions.

    `subsets` maps a row name to a collection of uids. Used for the MHC-I / MHC-II
    table, which needs no training at all — the predictions are already stored.
    """
    rows = []
    for name, uids in subsets.items():
        m = run_metrics(rid, eval_split, thr_mode, subset=uids, thr_on=thr_on, results_dir=results_dir)
        if m is None:
            continue
        row = {index_name: name, "n": m["n"], "n_pos": m["n_pos"]}
        row.update({c: m[c] for c in metrics.PAPER_COLS})
        row["PR_AUC"] = m["PR_AUC"]
        rows.append(row)
    return pd.DataFrame(rows)


def best_run(exp_id, eval_split="cv", thr_mode="mcc_cv", by="AUC", results_dir=None):
    """run_id of the highest-scoring run of an experiment.

    Selected on cross-validation by default: picking a configuration on the
    independent test set would make that test no longer independent.
    """
    runs = load_runs(exp_id, results_dir)
    best, best_v = None, -np.inf
    for r in runs.to_dict("records"):
        m = run_metrics(r["run_id"], eval_split, thr_mode, results_dir=results_dir)
        if m and m[by] > best_v:
            best, best_v = r["run_id"], m[by]
    return best


def roc(exp_id, eval_split="test", label_col="label", title=None, save=True, results_dir=None, subset=None):
    """ROC overlay, one curve per run, AUC in the legend (only the uids in `subset` when given)."""
    import matplotlib.pyplot as plt

    runs = latest_per_label(load_runs(exp_id, results_dir), label_col)
    if "series_order" in runs.columns:
        runs = runs.sort_values("series_order")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for r in runs.to_dict("records"):
        preds = load_preds(r["run_id"], results_dir)
        ev = preds[preds["split"] == eval_split]
        if ev.empty:
            continue
        ev = ev.groupby("uid", as_index=False).agg(y_true=("y_true", "first"),
                                                   y_score=("y_score", "mean"))
        if subset is not None:
            ev = ev[ev["uid"].isin(set(subset))]
        fpr, tpr, auc = metrics.roc_points(ev["y_true"], ev["y_score"])
        name = r.get(label_col) or r["run_id"]
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title or f"{exp_id} — {eval_split}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    if save:
        fig.savefig(_dir("figs", results_dir) / f"{exp_id}_{eval_split}_roc.png", dpi=150)
    return fig


def _ordered_runs(exp_id, label_col, results_dir):
    runs = latest_per_label(load_runs(exp_id, results_dir), label_col)
    if "series_order" in runs.columns:
        runs = runs.sort_values("series_order")
    return runs


def by_set(exp_id, sets, thr_on=None, metric="AUC", label_col="label", eval_split="test", thr_mode="mcc_cv",
           results_dir=None):
    """One row per run, one column per evaluation set (`sets`: name -> uids): the cross-disease table.

    The first row holds the number of positives of each set, so a small set is visible next to its score.
    """
    runs = _ordered_runs(exp_id, label_col, results_dir)
    if runs.empty:
        return pd.DataFrame()
    rows, n_pos = [], {}
    for r in runs.to_dict("records"):
        row = {label_col: r.get(label_col) or r["run_id"]}
        for name, uids in sets.items():
            m = run_metrics(r["run_id"], eval_split, thr_mode, subset=uids, thr_on=thr_on, results_dir=results_dir)
            row[name] = m[metric] if m else float("nan")
            if m:
                n_pos[name] = m["n_pos"]
        rows.append(row)
    head = {label_col: "n positives", **{k: n_pos.get(k, 0) for k in sets}}
    return pd.concat([pd.DataFrame([head]), pd.DataFrame(rows)], ignore_index=True)


def cluster_bootstrap(rid, groups, subset=None, n_boot=1000, seed=0, eval_split="test", results_dir=None):
    """AUC and PR_AUC with a 95% CI that resamples whole clusters of similar peptides, not single peptides.

    Peptides of one cluster are near-duplicates, so a peptide-level bootstrap understates the uncertainty.
    `groups`: Series indexed by uid with the cluster id. Seeds are averaged per peptide first, as in `run_metrics`.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    preds = load_preds(rid, results_dir)
    ev = preds[preds["split"] == eval_split].groupby("uid", as_index=False).agg(y_true=("y_true", "first"),
                                                                                y_score=("y_score", "mean"))
    if subset is not None:
        ev = ev[ev["uid"].isin(set(subset))]
    ev = ev.assign(g=ev["uid"].map(groups)).dropna(subset=["g"])
    y, s = ev["y_true"].to_numpy().astype(int), ev["y_score"].to_numpy()
    point = metrics.ranking_metrics(y, s)
    members = list(ev.reset_index(drop=True).groupby("g").indices.values())
    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
        aps.append(average_precision_score(y[idx], s[idx]))
    lo_hi = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float("nan"),) * 2
    (a_lo, a_hi), (p_lo, p_hi) = lo_hi(aucs), lo_hi(aps)
    return {"AUC": point["AUC"], "AUC_lo": a_lo, "AUC_hi": a_hi, "PR_AUC": point["PR_AUC"], "PR_lo": p_lo, "PR_hi": p_hi,
            "n": int(len(y)), "n_pos": int(y.sum()), "n_clusters": len(members)}


def bootstrap_table(exp_id, groups, subset=None, n_boot=1000, label_col="label", eval_split="test", results_dir=None):
    """`cluster_bootstrap` for every run of an experiment (one row per run)."""
    runs = _ordered_runs(exp_id, label_col, results_dir)
    rows = [{label_col: r.get(label_col) or r["run_id"],
             **cluster_bootstrap(r["run_id"], groups, subset, n_boot, eval_split=eval_split, results_dir=results_dir)}
            for r in runs.to_dict("records")]
    return pd.DataFrame(rows)


def report(exp_id, own, sets, extra_cols=(), cv=True):
    """The three views of an experiment on a mixed test split (call from a notebook):
    own test, 5-fold CV on train, and AUC on the other diseases. `own`: uids of the arm's own test; `sets`: name -> uids."""
    from IPython.display import display

    print(f"{exp_id} — own test (operating point fixed on the own test)")
    display(fmt(table(exp_id, subset=own, thr_on=own, extra_cols=extra_cols)))
    if cv:
        print("5-fold cross-validation on train")
        display(fmt(table(exp_id, eval_split="cv", extra_cols=extra_cols)))
    print("AUC on other diseases (rows similar to train removed); first row = positives per set")
    display(fmt(by_set(exp_id, sets, thr_on=own)))
    roc(exp_id, subset=own, title=f"{exp_id} — own test")


def fmt(t, decimals=4):
    """Round the float columns for display, matching the paper's 4-decimal tables."""
    t = t.copy()
    for c in t.select_dtypes("number").columns:
        t[c] = t[c].round(decimals)
    return t
