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


def run_metrics(rid, eval_split="test", thr_mode="mcc_cv", subset=None, results_dir=None):
    """Metrics for one run. The threshold always comes from the CV predictions.

    `subset` restricts evaluation to a set of uids (used for the MHC-I / MHC-II
    breakdown, which needs no retraining).
    """
    preds = load_preds(rid, results_dir)
    cv = preds[preds["split"] == "cv"]
    thr = _threshold(cv, thr_mode)

    ev = cv if eval_split == "cv" else preds[preds["split"] == "test"]
    if subset is not None:
        ev = ev[ev["uid"].isin(set(subset))]
    if ev.empty:
        return None

    # Several seeds predict the same test peptide; average them into one score per peptide.
    if eval_split == "test":
        ev = ev.groupby("uid", as_index=False).agg(y_true=("y_true", "first"),
                                                   y_score=("y_score", "mean"))
    out = metrics.all_metrics(ev["y_true"], ev["y_score"], thr)
    out["n_seeds"] = int(preds["seed"].nunique())
    return out


def seed_spread(rid, eval_split="test", thr_mode="mcc_cv", results_dir=None):
    """Per-seed AUC, as mean / std. Essential at ~127 positives, where a 0.02 AUC
    gap between configurations is routinely smaller than the seed-to-seed spread."""
    preds = load_preds(rid, results_dir)
    cv = preds[preds["split"] == "cv"]
    aucs = []
    for _, g in preds[preds["split"] == eval_split].groupby("seed"):
        r = metrics.ranking_metrics(g["y_true"], g["y_score"])
        if not np.isnan(r["AUC"]):
            aucs.append(r["AUC"])
    if not aucs:
        return {"AUC_mean": float("nan"), "AUC_std": float("nan")}
    return {"AUC_mean": float(np.mean(aucs)), "AUC_std": float(np.std(aucs))}


def table(exp_id, eval_split="test", thr_mode="mcc_cv", label_col="label",
          order_by=None, extra_cols=(), with_baseline=True, results_dir=None):
    """Comparison table in the paper's column layout, rebuilt from stored predictions."""
    runs = load_runs(exp_id, results_dir)
    if runs.empty:
        return pd.DataFrame()

    rows = []
    for r in runs.to_dict("records"):   # dict, not itertuples: config columns contain dots
        m = run_metrics(r["run_id"], eval_split, thr_mode, results_dir=results_dir)
        if m is None:
            continue
        row = {label_col: r.get(label_col) or r["run_id"]}
        row.update({c: m[c] for c in metrics.PAPER_COLS})
        row.update({"PR_AUC": m["PR_AUC"], "F1": m["F1"], "Precision": m["Precision"]})
        row.update(seed_spread(r["run_id"], eval_split, thr_mode, results_dir))
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
            b = metrics.majority_baseline(ev["y_true"])
            base = {label_col: "all-negative baseline"}
            base.update({c: b[c] for c in metrics.PAPER_COLS})
            base.update({"PR_AUC": b["PR_AUC"], "F1": b["F1"], "Precision": b["Precision"]})
            t = pd.concat([t, pd.DataFrame([base])], ignore_index=True)
    return t.reset_index(drop=True)


def breakdown(rid, subsets, eval_split="test", thr_mode="mcc_cv", results_dir=None,
              index_name="Dataset"):
    """Re-evaluate one committed run on subsets of its predictions.

    `subsets` maps a row name to a collection of uids. Used for the MHC-I / MHC-II
    table, which needs no training at all — the predictions are already stored.
    """
    rows = []
    for name, uids in subsets.items():
        m = run_metrics(rid, eval_split, thr_mode, subset=uids, results_dir=results_dir)
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


def roc(exp_id, eval_split="test", label_col="label", title=None, save=True, results_dir=None):
    """ROC overlay, one curve per run, AUC in the legend."""
    import matplotlib.pyplot as plt

    runs = load_runs(exp_id, results_dir)
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


def fmt(t, decimals=4):
    """Round the float columns for display, matching the paper's 4-decimal tables."""
    t = t.copy()
    for c in t.select_dtypes("number").columns:
        t[c] = t[c].round(decimals)
    return t
