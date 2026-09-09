import numpy as np


def _as_distribution(y, n_classes=None):
    y = np.asarray(y)
    if y.ndim == 1:
        if n_classes is None:
            n_classes = int(y.max()) + 1
        out = np.zeros((len(y), n_classes), dtype=np.float64)
        out[np.arange(len(y)), y.astype(int)] = 1.0
        return out
    return y.astype(np.float64)


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 2:
        y_true = y_true.argmax(axis=1)
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(axis=1)
    acc = float(np.mean(y_true == y_pred))
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(2 * prec * rec / (prec + rec + 1e-12))
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)) if f1s else 0.0}


def distribution_metrics(y_true, y_pred, eps=1e-8):
    p = _as_distribution(y_true, n_classes=np.asarray(y_pred).shape[1])
    q = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0)
    q = q / q.sum(axis=1, keepdims=True)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    diff = np.abs(p - q)
    chebyshev = float(np.mean(np.max(diff, axis=1)))
    clark = float(np.mean(np.sqrt(np.sum(((p - q) / (p + q)) ** 2, axis=1))))
    canberra = float(np.mean(np.sum(diff / (p + q), axis=1)))
    cosine = float(np.mean(np.sum(p * q, axis=1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1))))
    intersection = float(np.mean(np.sum(np.minimum(p, q), axis=1)))
    kl = float(np.mean(np.sum(p * np.log(p / q), axis=1)))
    mse = float(np.mean((p - q) ** 2))
    return {
        "chebyshev": chebyshev,
        "clark": clark,
        "canberra": canberra,
        "cosine": cosine,
        "intersection": intersection,
        "kl": kl,
        "mse": mse,
        "nll": float(np.mean(-np.sum(p * np.log(q), axis=1))),
        "brier": mse * p.shape[1],
    }


def expected_calibration_error(y_true, proba, n_bins=10):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    if y_true.ndim == 2:
        y_true = y_true.argmax(axis=1)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(pred[mask] == y_true[mask]))
        ece += float(np.mean(mask)) * abs(acc - float(np.mean(conf[mask])))
    return float(ece)


def score_annotations(y_true, proba, classes):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

    y_true = np.asarray(y_true).astype(str)
    classes = np.asarray(classes).astype(str)
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.clip(proba, 1e-8, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    class_index = {c: i for i, c in enumerate(classes)}
    known = np.array([v in class_index for v in y_true])
    y_pred_all = classes[proba.argmax(axis=1)]
    if not np.any(known):
        raise ValueError("No query labels overlap the training class set.")
    y_true_k = y_true[known]
    proba_k = proba[known]
    y_pred_k = y_pred_all[known]
    y_idx = np.array([class_index[v] for v in y_true_k])
    onehot = np.zeros_like(proba_k)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    out = {
        "accuracy": float(accuracy_score(y_true_k, y_pred_k)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_k, y_pred_k)),
        "macro_f1": float(f1_score(y_true_k, y_pred_k, average="macro", labels=classes, zero_division=0)),
        "log_loss": float(log_loss(y_true_k, proba_k, labels=list(classes))),
        "brier": float(np.mean(np.sum((onehot - proba_k) ** 2, axis=1))),
        "ece": expected_calibration_error(y_idx, proba_k),
        "mean_confidence": float(np.mean(proba_k.max(axis=1))),
        "mean_entropy": float(np.mean(-np.sum(proba_k * np.log(proba_k), axis=1))),
        "n_test": int(known.sum()),
        "unknown_label_fraction": float(1.0 - known.mean()),
    }
    out.update(distribution_metrics(onehot, proba_k))
    return out, y_pred_all
