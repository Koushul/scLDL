from __future__ import annotations

import numpy as np

from scLDL.interpret import dissonance, entropy, row_normalize


HIPPO_SIMILAR = {
    "CA1": "CA3",
    "CA3": "CA1",
    "DG": "CA1",
    "OPC": "Oligodendrocyte",
    "Oligodendrocyte": "OPC",
    "Astrocyte": "Ependymal",
    "Ependymal": "Astrocyte",
    "Interneuron": "Neurogenesis",
    "Neurogenesis": "Interneuron",
    "Endothelial": "Mural",
    "Mural": "Endothelial",
}


def flip_labels(y, rate, rng, *, mode: str = "uniform", similar: dict | None = None):
    """Return noisy labels and a boolean mask of cells whose label changed.

    Flips are stratified by class so every type is corrupted at the same rate.
    ``mode="uniform"`` draws the wrong label from the other observed types.
    ``mode="similar"`` uses ``similar`` (default ``HIPPO_SIMILAR``) when the
    target type is present, otherwise falls back to uniform.
    """
    if mode not in {"uniform", "similar"}:
        raise ValueError("mode must be 'uniform' or 'similar'")
    y = np.asarray(y).astype(str)
    n = len(y)
    noisy = y.copy()
    flipped = np.zeros(n, dtype=bool)
    rate = float(rate)
    if n == 0 or rate <= 0:
        return noisy, flipped

    classes = np.unique(y)
    similar = dict(HIPPO_SIMILAR if similar is None else similar)
    others = {c: classes[classes != c] for c in classes}

    for c in classes:
        idx = np.flatnonzero(y == c)
        k = int(round(rate * len(idx)))
        if k <= 0 or len(others[c]) == 0:
            continue
        k = min(k, len(idx))
        take = np.asarray(rng.choice(idx, size=k, replace=False), dtype=int)
        flipped[take] = True
        if mode == "similar" and similar.get(c) in set(classes) and similar[c] != c:
            noisy[take] = similar[c]
            continue
        noisy[take] = rng.choice(others[c], size=k)

    return noisy, flipped


def class_index(labels, classes):
    lut = {str(c): i for i, c in enumerate(classes)}
    out = np.empty(len(labels), dtype=np.intp)
    out[:] = -1
    for i, lab in enumerate(np.asarray(labels).astype(str)):
        out[i] = lut.get(lab, -1)
    return out


def mass_on_label(proba, labels, classes):
    proba = np.asarray(proba, dtype=np.float64)
    idx = class_index(labels, classes)
    mass = np.full(len(labels), np.nan, dtype=np.float64)
    ok = idx >= 0
    mass[ok] = proba[np.flatnonzero(ok), idx[ok]]
    return mass


def align_proba(proba, src_classes, dst_classes):
    proba = np.asarray(proba, dtype=np.float64)
    src_classes = np.asarray(src_classes).astype(str)
    dst_classes = np.asarray(dst_classes).astype(str)
    out = np.zeros((len(proba), len(dst_classes)), dtype=np.float64)
    lut = {c: i for i, c in enumerate(dst_classes)}
    for j, c in enumerate(src_classes):
        if c in lut:
            out[:, lut[c]] = proba[:, j]
    return row_normalize(out)


def _safe_auc(y_true, scores):
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(scores)
    y_true, scores = y_true[ok], scores[ok]
    if y_true.size == 0 or y_true.min() == y_true.max():
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, scores))


def _safe_ap(y_true, scores):
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(scores)
    y_true, scores = y_true[ok], scores[ok]
    if y_true.size == 0 or not y_true.any():
        return None
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, scores))


def precision_at_k(y_true, scores, k):
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    k = int(k)
    if k <= 0 or len(y_true) == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")[: min(k, len(y_true))]
    return float(y_true[order].mean())


def noise_recovery_metrics(y_true, y_noisy, y_pred, proba, classes, *, vacuity=None, name: str = "model"):
    """Score label-error discovery and correction against known flips."""
    y_true = np.asarray(y_true).astype(str)
    y_noisy = np.asarray(y_noisy).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    proba = row_normalize(np.asarray(proba, dtype=np.float64))
    classes = np.asarray(classes).astype(str)
    flipped = y_true != y_noisy
    p_given = mass_on_label(proba, y_noisy, classes)
    p_true = mass_on_label(proba, y_true, classes)
    H = entropy(proba)
    diss = dissonance(proba)
    suspect = 1.0 - p_given
    disagree = y_pred != y_noisy
    n_flip = int(flipped.sum())
    clean = ~flipped
    out = {
        "source": name,
        "n": int(len(y_true)),
        "n_flipped": n_flip,
        "noise_rate": float(flipped.mean()),
        "acc_vs_true": float(np.mean(y_pred == y_true)),
        "acc_vs_noisy": float(np.mean(y_pred == y_noisy)),
        "mean_p_given": float(np.nanmean(p_given)),
        "mean_p_true": float(np.nanmean(p_true)),
        "mean_entropy": float(np.mean(H)),
        "mean_dissonance": float(np.mean(diss)),
        "discovery_auroc_1m_p_given": _safe_auc(flipped, suspect),
        "discovery_ap_1m_p_given": _safe_ap(flipped, suspect),
        "discovery_auroc_entropy": _safe_auc(flipped, H),
        "discovery_auroc_dissonance": _safe_auc(flipped, diss),
        "discovery_auroc_disagree": _safe_auc(flipped, disagree.astype(float)),
        "precision_at_nflip": precision_at_k(flipped, suspect, n_flip),
        "disagree_recall": float(np.mean(disagree[flipped])) if n_flip else None,
        "disagree_precision": float(np.mean(flipped[disagree])) if disagree.any() else None,
        "correction_rate": float(np.mean(y_pred[flipped] == y_true[flipped])) if n_flip else None,
        "wrong_fix_rate": float(np.mean((y_pred[flipped] != y_true[flipped]) & (y_pred[flipped] != y_noisy[flipped]))) if n_flip else None,
        "clean_kept_rate": float(np.mean(y_pred[clean] == y_true[clean])) if clean.any() else None,
        "false_correction_rate": float(np.mean(y_pred[clean] != y_true[clean])) if clean.any() else None,
    }
    if vacuity is not None:
        vacuity = np.asarray(vacuity, dtype=np.float64).ravel()
        out["mean_vacuity"] = float(np.mean(vacuity))
        out["discovery_auroc_vacuity"] = _safe_auc(flipped, vacuity)
    return out
