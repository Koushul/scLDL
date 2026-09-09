from __future__ import annotations

import numpy as np


def row_normalize(p, eps: float = 1e-12):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
    return (p / np.clip(p.sum(axis=1, keepdims=True), eps, None)).astype(np.float32)


def entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return (-p * np.log(p)).sum(axis=1)


def dissonance(p):
    """How contested the top two states are (0 = peaked, 1 = tie)."""
    p = row_normalize(p)
    part = np.sort(p, axis=1)
    a = part[:, -1]
    b = part[:, -2] if p.shape[1] > 1 else np.zeros(len(p))
    return (b / np.clip(a + b, 1e-8, None)).astype(np.float32)


def vacuity(alpha, prior: float = 0.2):
    alpha = np.asarray(alpha, dtype=np.float64)
    k = alpha.shape[1]
    s = alpha.sum(axis=1)
    return ((prior * k) / np.clip(s, 1e-8, None)).astype(np.float32)


def allowed_pair_mask(n_classes: int, edges=None):
    mask = np.eye(n_classes, dtype=np.float64)
    if edges:
        for i, j in edges:
            mask[i, j] = 1.0
            mask[j, i] = 1.0
    return mask


def illegal_mass(p, edges=None):
    p = row_normalize(p)
    k = p.shape[1]
    allowed = allowed_pair_mask(k, edges)
    illegal = 1.0 - allowed
    pair = p[:, :, None] * p[:, None, :]
    return (pair * illegal[None, :, :]).sum(axis=(1, 2)).astype(np.float32)


def project_lineage(p, edges=None, strength: float = 0.35):
    p = row_normalize(p)
    if not edges or strength <= 0:
        return p
    allowed = allowed_pair_mask(p.shape[1], edges)
    legal = p @ allowed
    legal = row_normalize(legal)
    out = (1.0 - strength) * p + strength * legal
    return row_normalize(out)


def top2_pairs(p, classes):
    p = row_normalize(p)
    classes = np.asarray(classes).astype(str)
    order = np.argsort(-p, axis=1)
    i0 = order[:, 0]
    i1 = order[:, 1] if p.shape[1] > 1 else i0
    names = np.char.add(np.char.add(classes[i0], "|"), classes[i1])
    mass = np.stack([p[np.arange(len(p)), i0], p[np.arange(len(p)), i1]], axis=1)
    return names, mass.astype(np.float32)


def residual_mass(model_p, marker_p=None, knn_p=None):
    model_p = row_normalize(model_p)
    parts = [model_p]
    if marker_p is not None:
        parts.append(row_normalize(marker_p))
    if knn_p is not None:
        parts.append(row_normalize(knn_p))
    support = np.mean(np.stack(parts, axis=0), axis=0)
    return (model_p - support).astype(np.float32)


def graph_refine(p, z, vacuity=None, xy=None, n_neighbors: int = 15, n_iter: int = 2, mix: float = 0.55):
    """Smooth the simplex on the query kNN graph; confident cells stay put.

    Neighbor edges are expression affinities (and spatial affinities when ``xy``
    is given). High-vacuity cells contribute less as sources and mix more.
    """
    from sklearn.neighbors import NearestNeighbors

    p = row_normalize(p)
    n = len(p)
    if n < 4:
        return p
    z = np.asarray(z, dtype=np.float64)
    k = max(2, min(int(n_neighbors) + 1, n))
    dist, idx = NearestNeighbors(n_neighbors=k).fit(z).kneighbors(z)
    sigma = np.maximum(dist[:, -1], 1e-8)
    w = np.exp(-(dist * dist) / np.maximum(sigma[:, None] ** 2, 1e-12))
    w[:, 0] = 0.0
    if xy is not None:
        xy = np.asarray(xy, dtype=np.float64)
        dxy = np.sqrt(((xy[idx] - xy[:, None, :]) ** 2).sum(axis=2))
        tau = np.maximum(np.median(dxy[:, 1:], axis=1, keepdims=True), 1e-8)
        w = w * np.exp(-(dxy * dxy) / np.maximum(tau ** 2, 1e-12))
        w[:, 0] = 0.0
    conf = p.max(axis=1)
    if vacuity is not None:
        vac = np.clip(np.asarray(vacuity, dtype=np.float64).ravel(), 0.0, 1.0)
        source = np.clip(1.0 - vac, 0.05, 1.0)
        lam = np.clip(float(mix) * np.maximum(1.0 - conf, vac), 0.0, 0.85)
    else:
        source = np.clip(conf, 0.05, 1.0)
        lam = np.clip(float(mix) * (1.0 - conf), 0.0, 0.85)
    w = w * source[idx]
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    out = np.asarray(p, dtype=np.float64)
    for _ in range(max(1, int(n_iter))):
        neigh = (w[..., None] * out[idx]).sum(axis=1)
        out = (1.0 - lam)[:, None] * out + lam[:, None] * neigh
        out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out.astype(np.float32)


def adaptive_blend(model_p, knn_p=None, marker_p=None, task: str = "type", batch_corrected: bool = False):
    """Blend sources; high-confidence cells stay peaked (cell-type accuracy)."""
    model_p = row_normalize(model_p)
    conf = model_p.max(axis=1, keepdims=True)
    if task == "state":
        w_model = 0.35 + 0.25 * conf
        w_knn = 0.20 + 0.10 * (1.0 - conf)
        w_mark = 0.20 + 0.10 * (1.0 - conf)
    elif batch_corrected:
        w_model = 0.25 + 0.20 * conf
        w_knn = 0.55 + 0.15 * (1.0 - conf)
        w_mark = 0.0
    else:
        w_model = 0.82 + 0.13 * conf
        w_knn = 0.05 + 0.08 * (1.0 - conf)
        w_mark = 0.0
    acc = w_model * model_p
    if knn_p is not None:
        acc = acc + w_knn * row_normalize(knn_p)
    if marker_p is not None and task == "state":
        acc = acc + w_mark * row_normalize(marker_p)
    return row_normalize(acc)


def marker_spearman(scores, proba):
    from scipy.stats import spearmanr

    scores = np.asarray(scores, dtype=np.float64)
    proba = np.asarray(proba, dtype=np.float64)
    out = []
    for k in range(min(scores.shape[1], proba.shape[1])):
        if np.std(scores[:, k]) < 1e-12 or np.std(proba[:, k]) < 1e-12:
            out.append(np.nan)
            continue
        r, _ = spearmanr(scores[:, k], proba[:, k])
        out.append(float(r))
    return np.asarray(out, dtype=np.float64)
