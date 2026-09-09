from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from scLDL.interpret import row_normalize


def _spatial_nn(xy, n_neighbors: int):
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    k = max(1, min(int(n_neighbors) + 1, n))
    dist, idx = NearestNeighbors(n_neighbors=k).fit(xy).kneighbors(xy)
    return dist, idx, k


def spatial_knn_graph(xy, z=None, n_neighbors: int = 12):
    """Row-stochastic spatial graph, optionally gated by expression similarity."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    if n == 0:
        return sparse.csr_matrix((0, 0))
    dist, idx, k = _spatial_nn(xy, n_neighbors)
    w = _pairwise_weights(dist, idx, z)
    rows = np.repeat(np.arange(n), k)
    return sparse.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))


def _pairwise_weights(dist, idx, z):
    sigma = np.maximum(dist[:, -1], 1e-8)
    w = np.exp(-(dist * dist) / np.maximum(sigma[:, None] ** 2, 1e-12))
    w[:, 0] = 0.0
    if z is not None and dist.shape[1] > 1:
        z = np.asarray(z, dtype=np.float64)
        dz = np.sqrt(((z[idx] - z[:, None, :]) ** 2).sum(axis=2))
        tau = np.maximum(np.median(dz[:, 1:], axis=1, keepdims=True), 1e-8)
        w = w * np.exp(-(dz * dz) / np.maximum(tau ** 2, 1e-12))
        w[:, 0] = 0.0
    return w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)


def neighbor_agreement(labels, xy, n_neighbors: int = 8):
    labels = np.asarray(labels)
    n = len(labels)
    if n < 2:
        return np.ones(n, dtype=np.float32)
    _, idx, _ = _spatial_nn(xy, n_neighbors)
    neigh = labels[idx[:, 1:]]
    return (neigh == labels[:, None]).mean(axis=1).astype(np.float32)


def spatial_stats(labels, xy, n_neighbors: int = 8):
    agree = neighbor_agreement(labels, xy, n_neighbors=n_neighbors)
    return {
        "mean_same_neighbor_frac": float(agree.mean()),
        "isolated_frac": float((agree == 0).mean()),
        "interior_frac": float((agree >= 0.75).mean()),
    }


def spatial_refine(p, xy, z=None, n_neighbors: int = 12, n_iter: int = 2, task: str = "type"):
    """Reassign isolated speckles; leave confident spatial interiors unchanged."""
    p = row_normalize(p)
    n = len(p)
    if n < 3:
        return p
    k = max(4, int(n_neighbors))
    dist, idx, _ = _spatial_nn(xy, k)
    labels = p.argmax(axis=1)
    agree = (labels[idx[:, 1:]] == labels[:, None]).mean(axis=1)
    speckle = agree <= (1.0 / k + 1e-9)
    w = _pairwise_weights(dist, idx, z)
    neigh = (w[..., None] * p[idx]).sum(axis=1)
    if task == "type":
        maj = np.zeros_like(p)
        maj[np.arange(n), neigh.argmax(axis=1)] = 1.0
        out = p.copy()
        out[speckle] = row_normalize(0.12 * p[speckle] + 0.88 * maj[speckle])
        return out
    alpha = np.zeros(n, dtype=np.float64)
    weak = (agree < 0.34) & ~speckle
    alpha[speckle] = 0.55
    alpha[weak] = 0.30
    interior = agree >= 0.5
    alpha[interior] = 0.0
    out = p.copy()
    for _ in range(max(3, int(n_iter))):
        neigh = (w[..., None] * out[idx]).sum(axis=1)
        out = row_normalize((1.0 - alpha[:, None]) * out + alpha[:, None] * neigh)
    out[interior] = p[interior]
    return out
