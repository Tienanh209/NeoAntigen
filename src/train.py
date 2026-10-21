"""Training loop for the torch models: fit one fold, with resumable checkpoints.

No DataLoader workers: the features are already in memory as numpy arrays, so
batching is a slice and an index shuffle. Workers would only add Colab failure modes.
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import metrics

DEFAULTS = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "max_epochs": 150,
    "patience": 15,          # early stopping on validation AUC
    "min_delta": 1e-4,
    "class_weight": True,    # pos_weight = n_neg / n_pos, computed per training fold
    "lr_patience": 5,
    "amp": None,             # None -> on when CUDA is available
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_of(prefer=None):
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(n, batch_size, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def _to_torch(x, lengths, sel, device):
    xb = torch.from_numpy(np.ascontiguousarray(x[sel])).float().to(device)
    lb = torch.from_numpy(np.ascontiguousarray(lengths[sel])).to(device)
    return xb, lb


def predict(model, x, lengths, device, batch_size=256):
    """Sigmoid scores for every row of x."""
    model.eval()
    out = np.zeros(len(x), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            sel = np.arange(i, min(i + batch_size, len(x)))
            xb, lb = _to_torch(x, lengths, sel, device)
            out[sel] = torch.sigmoid(model(xb, lb)).float().cpu().numpy()
    return out


def fit(model, train_data, val_data, cfg=None, ckpt_path=None, device=None, verbose=False):
    """Train one fold. Returns {"state", "best_epoch", "best_auc", "history"}.

    train_data / val_data are (x, lengths, y) triples of numpy arrays.
    Early stopping watches validation AUC: with ~25 positives in a validation fold,
    a rank statistic is far steadier than MCC or PR-AUC.
    A checkpoint is written every epoch so an interrupted fold resumes where it stopped.
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    device = device or device_of(cfg.get("device"))
    model = model.to(device)

    xtr, ltr, ytr = train_data
    xva, lva, yva = val_data

    pos_weight = None
    if cfg["class_weight"]:
        n_pos = max(int(ytr.sum()), 1)
        pos_weight = torch.tensor([(len(ytr) - n_pos) / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=cfg["lr_patience"])
    use_amp = cfg["amp"] if cfg["amp"] is not None else (device.type == "cuda")
    try:   # torch >= 2.3
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best_auc, best_state, best_epoch, bad, history = 0, -np.inf, None, -1, 0, []
    if ckpt_path and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_epoch, best_auc = ck["epoch"] + 1, ck["best_auc"]
        best_state, best_epoch, bad = ck["best_state"], ck["best_epoch"], ck["bad"]
        history = ck["history"]
        if verbose:
            print(f"    resumed at epoch {start_epoch}")

    rng = np.random.default_rng(cfg.get("seed", 42))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))

    for epoch in range(start_epoch, cfg["max_epochs"]):
        model.train()
        total = 0.0
        for sel in _batches(len(xtr), cfg["batch_size"], True, rng):
            xb, lb = _to_torch(xtr, ltr, sel, device)
            yb = ytr_t[sel].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(xb, lb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(sel)

        val_scores = predict(model, xva, lva, device)
        val_auc = metrics.ranking_metrics(yva, val_scores)["AUC"]
        val_auc = -np.inf if np.isnan(val_auc) else val_auc
        tr_scores = predict(model, xtr, ltr, device)
        tr_auc = metrics.ranking_metrics(ytr, tr_scores)["AUC"]
        history.append({"epoch": epoch, "loss": total / len(xtr),
                        "train_auc": float(tr_auc), "val_auc": float(val_auc)})
        sched.step(val_auc)

        if val_auc > best_auc + cfg["min_delta"]:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:3d} loss {total/len(xtr):.4f} "
                  f"train_auc {tr_auc:.4f} val_auc {val_auc:.4f}")
        if ckpt_path:
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict(), "optim": opt.state_dict(),
                        "best_state": best_state, "best_auc": best_auc, "best_epoch": best_epoch,
                        "bad": bad, "history": history}, ckpt_path)
        if bad >= cfg["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"state": best_state, "best_epoch": best_epoch, "best_auc": float(best_auc),
            "history": history, "model": model, "device": device}
