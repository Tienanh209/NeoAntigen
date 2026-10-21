"""Experiment orchestration: the single entry point a notebook block calls.

One call runs the whole protocol for one configuration and commits it:

    for every seed:
        5 stratified folds over the TRAINING split only
        train on 4/5, early-stop on the held-out 1/5, predict the held-out 1/5
          -> concatenated, these are the cross-validation (out-of-fold) predictions
        predict the test split with each fold model and average
          -> the independent-test prediction for that seed

The test split is read in exactly one place, the final predict, and the
early-stopping signal only ever comes from training-split folds.

Several seeds are not optional here: with ~127 positives, the seed-to-seed spread
is comparable to the differences between the configurations being compared.
"""
import numpy as np
import pandas as pd

from . import config, data, models, results, train

DEFAULT_SEEDS = (42, 43, 44)


def _features(spec, df, max_len, build_id, data_dir=None):
    """Resolve a feature spec to (x, lengths). Shapes: [N, D, L], or [N, D] when pooled."""
    kind = spec.get("kind", "plm")
    if kind == "onehot":
        return data.onehot(df["sequence"], max_len)
    if kind == "length":
        return data.length_features(df["sequence"]), np.full(len(df), 1, dtype=np.int64)
    if kind == "aac":
        x = data.aac_features(df["sequence"])
        return x, np.full(len(df), 1, dtype=np.int64)
    if kind in ("plm", "plm_pooled"):
        from . import embed            # imported lazily: needs transformers
        cache = embed.EmbCache(spec["plm"], build_id, data_dir=data_dir)
        if kind == "plm_pooled":
            x = cache.pooled(df["uid"], how=spec.get("pool", "mean"))
            return x, np.full(len(df), 1, dtype=np.int64)
        return cache.stack(df["uid"], max_len)
    raise ValueError(f"unknown feature kind {kind!r}")


def _sklearn_scores(name, seed, xtr, ytr, xs):
    """Fit a classical classifier and score each array in `xs`. Any monotone
    score works downstream, so decision_function is an acceptable fallback."""
    clf = models.sklearn_model(name, seed=seed)
    clf.fit(xtr, ytr)
    out = []
    for x in xs:
        if hasattr(clf, "predict_proba"):
            out.append(clf.predict_proba(x)[:, 1])
        else:
            out.append(clf.decision_function(x))
    return out


def run_experiment(exp_id, *, features, model, train_cfg=None, label="",
                   seeds=DEFAULT_SEEDS, folds=5, split_mode="stratified",
                   df=None, data_dir=None, results_dir=None, meta=None,
                   force=False, verbose=True):
    """Run and commit one configuration. Skips instantly if already committed."""
    df = data.load_dataset(data_dir) if df is None else df
    bid = data.build_id(df)
    mlen = data.max_len(df, data_dir)

    cfg = {"build_id": bid, "features": features, "model": model,
           "train": {**train.DEFAULTS, **(train_cfg or {})} if model.get("kind") != "sklearn" else {},
           "seeds": list(seeds), "folds": folds, "split_mode": split_mode}
    if results.done(exp_id, cfg, results_dir) and not force:
        if verbose:
            print(f"  [skip] {label or exp_id} — already committed")
        return results.load_runs(exp_id, results_dir)

    x, lengths = _features(features, df, mlen, bid, data_dir)
    y = df["label"].to_numpy().astype(np.float32)
    is_train = (df["split"] == "train").to_numpy()
    tr_idx = np.flatnonzero(is_train)
    te_idx = np.flatnonzero(~is_train)
    train_df = df.loc[tr_idx].reset_index(drop=True)

    groups = None
    if split_mode == "grouped":
        groups = data.kmer_groups(train_df).to_numpy()

    rows, fold_info = [], []
    for seed in seeds:
        train.seed_all(seed)
        splits = (data.group_cv_folds(train_df, groups, folds, seed) if groups is not None
                  else data.cv_folds(train_df, folds, seed))
        oof = np.full(len(tr_idx), np.nan)
        test_scores = []

        for k, (a, b) in enumerate(splits):
            ia, ib = tr_idx[a], tr_idx[b]
            if model.get("kind") == "sklearn":
                s_val, s_test = _sklearn_scores(model["name"], seed, x[ia], y[ia], [x[ib], x[te_idx]])
            else:
                net = models.build(model, d_in=x.shape[1])
                ckpt = (results._dir("ckpt", results_dir) /
                        results.run_id(exp_id, cfg) / f"seed{seed}_fold{k}.pt")
                out = train.fit(net,
                                (x[ia], lengths[ia], y[ia]),
                                (x[ib], lengths[ib], y[ib]),
                                {**(train_cfg or {}), "seed": seed},
                                ckpt_path=ckpt, verbose=False)
                dev = out["device"]
                s_val = train.predict(out["model"], x[ib], lengths[ib], dev)
                s_test = train.predict(out["model"], x[te_idx], lengths[te_idx], dev)
                fold_info.append({"seed": seed, "fold": k, "best_epoch": out["best_epoch"],
                                  "val_auc": out["best_auc"],
                                  "final_train_auc": out["history"][-1]["train_auc"] if out["history"] else None})
            oof[b] = s_val
            test_scores.append(s_test)

        rows.append(pd.DataFrame({"uid": df.loc[tr_idx, "uid"].to_numpy(), "split": "cv",
                                  "fold": -1, "seed": seed, "y_true": y[tr_idx], "y_score": oof}))
        # One test row per seed: the folds are averaged into a single ensemble score.
        rows.append(pd.DataFrame({"uid": df.loc[te_idx, "uid"].to_numpy(), "split": "test",
                                  "fold": -1, "seed": seed, "y_true": y[te_idx],
                                  "y_score": np.mean(test_scores, axis=0)}))
        if verbose:
            from . import metrics as M
            print(f"  seed {seed}: cv AUC {M.ranking_metrics(y[tr_idx], oof)['AUC']:.4f} | "
                  f"test AUC {M.ranking_metrics(y[te_idx], np.mean(test_scores, axis=0))['AUC']:.4f}")

    preds = pd.concat(rows, ignore_index=True)
    extra = {"label": label, "n_train": len(tr_idx), "n_test": len(te_idx),
             "max_len": mlen, "d_in": int(x.shape[1]), "folds_detail": fold_info,
             **(meta or {})}   # free-form columns for ordering and reporting
    if model.get("kind") == "mcnn":
        extra["conv_params"] = models.conv_param_count(
            int(x.shape[1]), model.get("windows", (34, 32, 24, 16)),
            model.get("filters", 512), model.get("bottleneck"))
    results.save_run(exp_id, cfg, preds, extra, results_dir)
    return results.load_runs(exp_id, results_dir)


def run_series(exp_id, configs, *, df=None, data_dir=None, results_dir=None,
               force=False, verbose=True, **common):
    """Run a list of configurations, cheapest first so a crash keeps the fast results.

    `configs` is a list of dicts accepted by run_experiment (features / model /
    train_cfg / label / split_mode).
    """
    df = data.load_dataset(data_dir) if df is None else df
    for i, c in enumerate(configs, 1):
        if verbose:
            print(f"[{exp_id} {i}/{len(configs)}] {c.get('label', '')}")
        c = dict(c)
        # Remember the position in the series so tables read in the order written here
        # rather than in config-hash order.
        c["meta"] = {"series_order": i, **(c.get("meta") or {})}
        run_experiment(exp_id, df=df, data_dir=data_dir, results_dir=results_dir,
                       force=force, verbose=verbose, **{**common, **c})
    return results.load_runs(exp_id, results_dir)
