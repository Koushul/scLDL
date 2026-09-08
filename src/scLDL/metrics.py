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
    }
