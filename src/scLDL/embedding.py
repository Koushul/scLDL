from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class ReferenceEmbedding:
    """Reference PCA with optional batch centering and query correction.

    Fits scaler + PCA on the labeled reference only. If `batches` is given,
    each reference batch is mean-centered in scaled space before PCA so
    technical offsets are not the leading PCs. Query cells are scaled with
    the reference scaler, projected, then optionally shifted onto the
    reference manifold (global centering or mutual nearest neighbors).
    """

    def __init__(self, n_pcs: int = 40, n_neighbors: int = 20):
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.scaler_ = None
        self.pca_ = None
        self.ref_z_ = None
        self.ref_mean_z_ = None

    def fit(self, X, batches=None):
        X = np.asarray(X, dtype=np.float64)
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(X)
        if batches is not None:
            batches = np.asarray(batches).astype(str)
            xs = xs.copy()
            for b in np.unique(batches):
                m = batches == b
                xs[m] -= xs[m].mean(axis=0, keepdims=True)
        n_comp = int(min(self.n_pcs, xs.shape[0] - 1, xs.shape[1]))
        n_comp = max(n_comp, 1)
        self.pca_ = PCA(n_components=n_comp, random_state=0)
        self.ref_z_ = self.pca_.fit_transform(xs).astype(np.float32)
        self.ref_mean_z_ = self.ref_z_.mean(axis=0)
        return self

    def transform(self, X, correct: str = "auto"):
        if self.pca_ is None:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=np.float64)
        xs = self.scaler_.transform(X)
        z = np.nan_to_num(self.pca_.transform(xs).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if correct == "auto":
            radius = float(np.sqrt(np.mean(self.ref_z_ ** 2)) + 1e-6)
            shift = float(np.linalg.norm(z.mean(axis=0) - self.ref_mean_z_) / radius)
            correct = "mnn" if shift > 0.35 else "none"
        self.last_correct_ = correct if correct not in (None, False) else "none"
        self.last_aligned_ = z
        if correct in (None, "none", False):
            return z
        if correct == "center":
            aligned = match_moments(z, self.ref_z_)
            self.last_aligned_ = aligned
            return aligned
        if correct == "mnn":
            aligned = match_moments(z, self.ref_z_)
            self.last_aligned_ = aligned
            return mnn_map(aligned, self.ref_z_, n_neighbors=self.n_neighbors)
        raise ValueError(f"Unknown query correction {correct!r}")


def match_moments(query, ref):
    query = np.asarray(query, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    qsd = query.std(axis=0)
    qsd[qsd < 1e-6] = 1.0
    rsd = ref.std(axis=0)
    rsd[rsd < 1e-6] = 1.0
    aligned = (query - query.mean(axis=0)) / qsd * rsd + ref.mean(axis=0)
    return aligned.astype(np.float32)


def mnn_map(query, ref, n_neighbors: int = 20, shrink: float = 0.85):
    """Shift query embeddings toward mutual nearest neighbors in the reference."""
    query = np.asarray(query, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if len(query) == 0 or len(ref) == 0:
        return query.astype(np.float32)
    k = max(1, min(int(n_neighbors), len(ref), len(query)))
    idx_q2r = NearestNeighbors(n_neighbors=k).fit(ref).kneighbors(query, return_distance=False)
    idx_r2q = NearestNeighbors(n_neighbors=k).fit(query).kneighbors(ref, return_distance=False)
    nq = query.shape[0]
    mutual = (idx_r2q[idx_q2r] == np.arange(nq)[:, None, None]).any(axis=2)
    w = mutual.astype(np.float64)
    none = w.sum(axis=1) == 0
    if np.any(none):
        w[none] = 0.0
        w[none, : min(3, k)] = 1.0
    w /= np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    target = (w[..., None] * ref[idx_q2r]).sum(axis=1)
    return (query + shrink * (target - query)).astype(np.float32)


def mnn_map_supervised(query, ref, query_labels, ref_labels, n_neighbors: int = 20, shrink: float = 0.85):
    """Second-pass MNN restricted to provisional query/reference type pairs (iSMNN-style)."""
    query = np.asarray(query, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    query_labels = np.asarray(query_labels)
    ref_labels = np.asarray(ref_labels)
    out = query.copy()
    for t in np.unique(query_labels):
        q_idx = np.flatnonzero(query_labels == t)
        r_idx = np.flatnonzero(ref_labels == t)
        if len(q_idx) < 2 or len(r_idx) < 2:
            continue
        out[q_idx] = mnn_map(query[q_idx], ref[r_idx], n_neighbors=n_neighbors, shrink=shrink)
    return out.astype(np.float32)
