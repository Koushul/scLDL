from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from scLDL.interpret import row_normalize


def spatial_knn_graph(xy, z=None, n_neighbors: int = 12):
    """Row-stochastic spatial graph, optionally gated by expression similarity."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    if n == 0:
        return sparse.csr_matrix((0, 0))
    k = max(1, min(int(n_neighbors) + 1, n))
    dist, idx = NearestNeighbors(n_neighbors=k).fit(xy).kneighbors(xy)
    sigma = np.maximum(dist[:, -1], 1e-8)
    w = np.exp(-(dist * dist) / np.maximum(sigma[:, None] ** 2, 1e-12))
    w[:, 0] = 0.0
    if z is not None and k > 1:
        z = np.asarray(z, dtype=np.float64)
        dz = np.sqrt(((z[idx] - z[:, None, :]) ** 2).sum(axis=2))
        tau = np.maximum(np.median(dz[:, 1:], axis=1, keepdims=True), 1e-8)
        w = w * np.exp(-(dz * dz) / np.maximum(tau ** 2, 1e-12))
        w[:, 0] = 0.0
    mass = np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    w = w / mass
    rows = np.repeat(np.arange(n), k)
    return sparse.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))


def local_purity(p, graph):
    p = row_normalize(p)
    neigh = graph @ p
    return np.clip((p * neigh).sum(axis=1), 0.0, 1.0)


def neighbor_agreement(labels, xy, n_neighbors: int = 8):
    labels = np.asarray(labels).astype(str)
    n = len(labels)
    if n < 2:
        return np.ones(n, dtype=np.float32)
    k = max(1, min(int(n_neighbors) + 1, n))
    idx = NearestNeighbors(n_neighbors=k).fit(np.asarray(xy)).kneighbors(xy, return_distance=False)
    neigh = labels[idx[:, 1:]]
    return (neigh == labels[:, None]).mean(axis=1).astype(np.float32)


def spatial_stats(labels, xy, n_neighbors: int = 8):
    agree = neighbor_agreement(labels, xy, n_neighbors=n_neighbors)
    return {
        "mean_same_neighbor_frac": float(agree.mean()),
        "isolated_frac": float((agree == 0).mean()),
        "interior_frac": float((agree >= 0.75).mean()),
    }


def spatial_refine(p, xy, z=None, n_neighbors: int = 12, n_iter: int = 6, task: str = "type"):
    """Smooth a label simplex on the tissue, leaving confident interiors intact.

    Isolated / contested spots take more mass from bilateral spatial neighbors
    (near in space and similar in expression). Layer interiors stay peaked.
    """
    p = row_normalize(p)
    n = len(p)
    if n < 3:
        return p
    graph = spatial_knn_graph(xy, z=z, n_neighbors=n_neighbors)
    conf = p.max(axis=1)
    power = 1.35 if task == "type" else 0.85
    floor = 0.02 if task == "type" else 0.08
    ceil = 0.88 if task == "type" else 0.75
    out = p.copy()
    for _ in range(max(1, int(n_iter))):
        neigh = np.asarray(graph @ out)
        purity = np.clip((out * neigh).sum(axis=1), 0.0, 1.0)
        alpha = np.clip((1.0 - purity) ** power * (0.55 + 0.45 * (1.0 - conf)), floor, ceil)
        out = row_normalize((1.0 - alpha[:, None]) * out + alpha[:, None] * neigh)
    return out
