"""Classification metrics for neoantigen prediction. Pure functions, no I/O.

The paper reports Sensitivity, Specificity, Accuracy, MCC and AUC; the course brief
additionally requires Precision, F1 and PR-AUC, which matter far more at this
class imbalance (~1 positive per 13 negatives).

`y_score` is any monotone score, not necessarily a probability, so an SVM
`decision_function` margin works unchanged.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# Columns of the paper's comparison tables, in order.
PAPER_COLS = ["Sensitivity", "Specificity", "Accuracy", "MCC", "AUC"]


def confusion_metrics(y_true, y_score, thr):
    """Threshold-dependent metrics. Returns sensitivity, specificity, accuracy, MCC, precision, F1."""
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_score) >= thr
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))

    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    # MCC denominator collapses to 0 when a row or column of the matrix is empty; by convention MCC = 0 there.
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0

    return {"Sensitivity": sens, "Specificity": spec, "Accuracy": acc, "MCC": mcc,
            "Precision": prec, "F1": f1, "BalancedAcc": (sens + spec) / 2,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn}


def ranking_metrics(y_true, y_score):
    """Threshold-free metrics. Undefined (nan) when only one class is present."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"AUC": float("nan"), "PR_AUC": float("nan")}
    return {"AUC": float(roc_auc_score(y_true, y_score)),
            "PR_AUC": float(average_precision_score(y_true, y_score))}


def best_threshold(y_true, y_score, objective="mcc"):
    """Threshold maximising `objective` ('mcc' or 'youden'), scanning the observed scores.

    Tuned on cross-validation out-of-fold predictions and then frozen, so the
    independent test set never influences the operating point.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5

    # Candidates: midpoints between consecutive distinct scores, plus the extremes.
    uniq = np.unique(y_score)
    cands = np.concatenate([[uniq[0] - 1e-9], (uniq[:-1] + uniq[1:]) / 2, [uniq[-1] + 1e-9]])

    def score(thr):
        m = confusion_metrics(y_true, y_score, thr)
        return m["MCC"] if objective == "mcc" else m["Sensitivity"] + m["Specificity"] - 1

    values = [score(t) for t in cands]
    return float(cands[int(np.argmax(values))])


def all_metrics(y_true, y_score, thr):
    """Every metric at one threshold, plus the sample counts needed to read the table."""
    y_true = np.asarray(y_true).astype(int)
    out = {**confusion_metrics(y_true, y_score, thr), **ranking_metrics(y_true, y_score)}
    out.update(n=len(y_true), n_pos=int(y_true.sum()), threshold=float(thr))
    return out


def majority_baseline(y_true):
    """Metrics for 'predict every peptide as non-neoantigen'.

    Belongs in every table: at 1:13 imbalance it already scores ~0.93 accuracy with
    zero sensitivity, which is how an accuracy-only reading of these results misleads.
    """
    y_true = np.asarray(y_true).astype(int)
    out = all_metrics(y_true, np.zeros(len(y_true)), thr=0.5)
    out.update(AUC=0.5, PR_AUC=float(y_true.mean()) if len(y_true) else float("nan"))
    return out


def roc_points(y_true, y_score):
    """(fpr, tpr, auc) for plotting."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, float(roc_auc_score(y_true, y_score))
