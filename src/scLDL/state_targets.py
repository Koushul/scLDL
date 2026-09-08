from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _row_softmax(x, temperature: float = 1.0):
    z = np.asarray(x, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def marker_score_matrix(expr, gene_names, markers: dict[str, list[str]], classes):
    names = np.asarray(gene_names).astype(str)
    index = {g: i for i, g in enumerate(names)}
    scores = np.zeros((expr.shape[0], len(classes)), dtype=np.float64)
    for k, cls in enumerate(classes):
        genes = [g for g in markers.get(cls, []) if g in index]
        if not genes:
            continue
        cols = np.stack([np.asarray(expr[:, index[g]], dtype=np.float64) for g in genes], axis=1)
        mu = cols.mean(axis=0)
        sd = cols.std(axis=0)
        sd[sd < 1e-6] = 1.0
        z = (cols - mu) / sd
        scores[:, k] = z.mean(axis=1)
    return scores


def marker_targets(expr, gene_names, markers, classes, temperature: float = 0.7):
    scores = marker_score_matrix(expr, gene_names, markers, classes)
    return _row_softmax(scores, temperature=temperature).astype(np.float32)


def knn_smooth_labels(X, Y, n_neighbors: int = 15, alpha: float = 0.7, n_iter: int = 15):
    """Inductive graph smoothing on the training set only."""
    Y = np.asarray(Y, dtype=np.float64)
    n = len(Y)
    k = max(2, min(n_neighbors + 1, n))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sigma = np.median(dist) + 1e-6
    W = np.exp(-(dist**2) / (2 * sigma**2))
    A = np.zeros((n, n), dtype=np.float64)
    rows = np.repeat(np.arange(n), idx.shape[1])
    A[rows, idx.ravel()] = W.ravel()
    A = 0.5 * (A + A.T)
    deg = A.sum(axis=1, keepdims=True)
    S = A / np.clip(deg, 1e-12, None)
    D = Y.copy()
    for _ in range(n_iter):
        D = (1.0 - alpha) * Y + alpha * (S @ D)
        D = np.clip(D, 0, None)
        D = D / np.clip(D.sum(axis=1, keepdims=True), 1e-12, None)
    return D.astype(np.float32)


def blend_targets(*parts_and_weights):
    acc = None
    for part, w in parts_and_weights:
        part = np.asarray(part, dtype=np.float64) * float(w)
        acc = part if acc is None else acc + part
    acc = np.clip(acc, 0, None)
    acc = acc / np.clip(acc.sum(axis=1, keepdims=True), 1e-12, None)
    return acc.astype(np.float32)


def lineage_laplacian(edges, n_classes: int):
    A = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = np.diag(A.sum(axis=1))
    return (deg - A).astype(np.float32)


def pancreas_lineage_edges():
    """Ductal-Ngn3low-Ngn3high-Pre-endocrine-{Beta,Alpha,Delta,Epsilon}."""
    return [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
