"""The multi-window CNN and the baseline models.

Architecture (paper §2.3): per-residue embeddings `L x D` are scanned by parallel
convolution windows, each window is reduced by 1-max pooling, the pooled vectors
are concatenated and a fully connected head does the binary classification.
1-max pooling is what lets MHC-I peptides (8-11 aa) and MHC-II peptides (12-25 aa)
share one model without being forced to a common length.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MCNN(nn.Module):
    """Multi-window CNN over per-residue features.

    `d_in=20` with one-hot input gives the one-hot CNN / mCNN baselines, so the
    ablation ladder reuses this one class.

    Each window is applied with *full* padding (`W-1` on each side), so even a
    3-residue peptide produces valid outputs for a 34-wide window; the alternative
    (`L-W+1` with no padding) is undefined whenever the window exceeds the peptide.
    Padded positions are then masked out before the max, so a peptide's score can
    never be decided by padding.
    """

    def __init__(self, d_in, windows=(34, 32, 24, 16), filters=512, hidden=256,
                 dropout=0.3, bottleneck=None, batchnorm=False):
        super().__init__()
        self.windows = tuple(windows)
        self.filters = filters

        # Optional 1x1 projection: at filters=2048 and D=1280 the convolutions hold
        # ~300M parameters, so this makes the large-filter configurations trainable.
        self.bottleneck = nn.Conv1d(d_in, bottleneck, 1) if bottleneck else None
        d_conv = bottleneck or d_in

        self.convs = nn.ModuleList([nn.Conv1d(d_conv, filters, w) for w in self.windows])
        self.norms = nn.ModuleList([nn.BatchNorm1d(filters) if batchnorm else nn.Identity()
                                    for _ in self.windows])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(filters * len(self.windows), hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        """x: [B, D, L]. lengths: [B]. Returns logits [B]."""
        # Zero the padded positions here rather than trusting the caller. A window
        # that straddles the sequence end legitimately covers padding, so anything
        # non-zero out there would leak into the result.
        in_valid = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
        x = x * in_valid[:, None, :].to(x.dtype)

        if self.bottleneck is not None:
            x = F.relu(self.bottleneck(x))

        pooled = []
        for w, conv, norm in zip(self.windows, self.convs, self.norms):
            h = conv(F.pad(x, (w - 1, w - 1)))        # [B, F, L+W-1]
            h = F.relu(norm(h))
            # Output j spans input positions [j-(W-1), j]; it touches a real
            # residue exactly when j < length + W - 1.
            pos = torch.arange(h.shape[-1], device=h.device)
            valid = pos[None, :] < (lengths[:, None] + w - 1)
            # -1e4, not -inf: under autocast -inf propagates NaNs through the
            # fused backward, and post-ReLU activations are always >= 0.
            pooled.append(h.masked_fill(~valid[:, None, :], -1e4).amax(dim=2))

        z = self.dropout(torch.cat(pooled, dim=1))
        z = self.dropout(F.relu(self.fc1(z)))
        return self.fc2(z).squeeze(-1)


class MLPHead(nn.Module):
    """Classifier over a mean-pooled embedding — the 'PLM without local scanning'
    rung of the ablation ladder."""

    def __init__(self, d_in, hidden=(256,), dropout=0.3):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        """x: [B, D] pooled features (lengths ignored, kept for a common call signature)."""
        return self.net(x).squeeze(-1)


def build(spec, d_in):
    """Model from a plain dict, so a run's configuration is JSON-serialisable."""
    kind = spec.get("kind", "mcnn")
    args = {k: v for k, v in spec.items() if k != "kind"}
    if kind == "mcnn":
        return MCNN(d_in, **args)
    if kind == "mlp":
        return MLPHead(d_in, **args)
    raise ValueError(f"unknown model kind {kind!r}")


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sklearn_model(name, seed=42):
    """Classical classifiers on mean-pooled embeddings (paper Table 6).

    Probability estimates are enabled where available; `score()` in runner.py falls
    back to `decision_function`, and every metric here works on any monotone score.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                      n_jobs=-1, random_state=seed)
    if name == "svm":
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", class_weight="balanced", random_state=seed))
    if name == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=seed))
    raise ValueError(f"unknown classifier {name!r}")


def conv_param_count(d_in, windows, filters, bottleneck=None):
    """Convolution parameters only — the column that explains the filter sweep."""
    d = bottleneck or d_in
    return int(sum(filters * d * w + filters for w in windows))
