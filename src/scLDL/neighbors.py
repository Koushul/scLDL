from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def query_to_ref_weights(X_query, X_ref, n_neighbors: int = 30):
    """Row-stochastic P(ref j | query i) from a self-tuning Gaussian kNN."""
    X_query = np.asarray(X_query, dtype=np.float64)
    X_ref = np.asarray(X_ref, dtype=np.float64)
    k = max(1, min(int(n_neighbors), len(X_ref)))
    dist, idx = NearestNeighbors(n_neighbors=k).fit(X_ref).kneighbors(X_query)
    sigma = np.maximum(dist[:, -1], 1e-8)
    aff = np.exp(-(dist * dist) / np.maximum(sigma[:, None] * sigma[:, None], 1e-12))
    aff = aff / np.clip(aff.sum(axis=1, keepdims=True), 1e-12, None)
    return aff.astype(np.float64), idx


def transfer_labels(Y_ref, weights, idx):
    Y_ref = np.asarray(Y_ref, dtype=np.float64)
    picked = Y_ref[idx]
    return np.einsum("qk,qkc->qc", weights, picked).astype(np.float32)


def query_label_transfer(X_query, X_ref, Y_ref, n_neighbors: int = 30):
    w, idx = query_to_ref_weights(X_query, X_ref, n_neighbors=n_neighbors)
    return transfer_labels(Y_ref, w, idx)
