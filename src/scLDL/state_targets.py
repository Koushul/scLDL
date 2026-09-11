from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def _row_softmax(x, temperature: float = 1.0):
    z = np.asarray(x, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def marker_score_matrix(expr, gene_names, markers: dict[str, list[str]], classes):
    from scLDL.data import to_dense

    names = np.asarray(gene_names).astype(str)
    index = {g: i for i, g in enumerate(names)}
    x = np.asarray(to_dense(expr), dtype=np.float64)
    scores = np.zeros((x.shape[0], len(classes)), dtype=np.float64)
    for k, cls in enumerate(classes):
        genes = [g for g in markers.get(cls, []) if g in index]
        if not genes:
            continue
        cols = x[:, [index[g] for g in genes]]
        mu = cols.mean(axis=0)
        sd = cols.std(axis=0)
        sd[sd < 1e-6] = 1.0
        z = (cols - mu) / sd
        scores[:, k] = z.mean(axis=1)
    return scores


def marker_targets(expr, gene_names, markers, classes, temperature: float = 0.7):
    scores = marker_score_matrix(expr, gene_names, markers, classes)
    return _row_softmax(scores, temperature=temperature).astype(np.float32)


def _embedding(X, n_pcs: int | None):
    X = np.asarray(X, dtype=np.float64)
    if n_pcs is None:
        return X
    n_comp = int(min(n_pcs, X.shape[0] - 1, X.shape[1]))
    if n_comp < 2:
        return X
    return PCA(n_components=n_comp, random_state=0).fit_transform(X)


def _row_normalize_csr(A):
    A = A.tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = 1.0 / np.clip(deg, 1e-12, None)
    return A.multiply(inv[:, np.newaxis]).tocsr()


def _knn_affinity_csr(Z, n_neighbors: int, metric: str, self_tuning: bool):
    n = Z.shape[0]
    k = max(2, min(int(n_neighbors) + 1, n))
    dist, idx = NearestNeighbors(n_neighbors=k, metric=metric).fit(Z).kneighbors(Z)
    dist, idx = dist[:, 1:], idx[:, 1:]
    rows = np.repeat(np.arange(n), idx.shape[1])
    js = idx.ravel()
    d = dist.ravel()
    if self_tuning:
        sigma = np.maximum(dist[:, -1], 1e-8)
        aff = np.exp(-(d * d) / np.maximum(sigma[rows] * sigma[js], 1e-12))
    else:
        sigma = np.median(dist) + 1e-6
        aff = np.exp(-(d * d) / (2.0 * sigma * sigma))
    return sparse.csr_matrix((aff, (rows, js)), shape=(n, n))


def probabilistic_neighbors(
    X,
    n_neighbors: int = 30,
    n_pcs: int | None = 40,
    metric: str = "euclidean",
):
    """Row-stochastic P(j|i) from a self-tuning Gaussian kNN graph.

    Local bandwidth σ_i is the distance to the k-th neighbor. Affinities use
    Zelnik-Manor/Perona self-tuning, P(j|i) is a softmax over those neighbors,
    and the graph is symmetrized with the UMAP fuzzy union
    P_ij = P_j|i + P_i|j - P_j|i P_i|j before a final random-walk normalize.
    """
    S = _fuzzy_knn_graph(X, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric)
    return np.asarray(S.toarray(), dtype=np.float64)


def _fuzzy_knn_graph(
    X,
    n_neighbors: int = 30,
    n_pcs: int | None = 40,
    metric: str = "euclidean",
):
    Z = _embedding(X, n_pcs)
    A = _knn_affinity_csr(Z, n_neighbors, metric, self_tuning=True)
    P = _row_normalize_csr(A)
    PT = P.transpose().tocsr()
    P_sym = P + PT - P.multiply(PT)
    P_sym.setdiag(0.0)
    P_sym.eliminate_zeros()
    return _row_normalize_csr(P_sym)


def _global_rbf_neighbors(X, n_neighbors: int):
    X = np.asarray(X, dtype=np.float64)
    A = _knn_affinity_csr(X, n_neighbors, "euclidean", self_tuning=False)
    A = 0.5 * (A + A.T)
    return _row_normalize_csr(A)


def knn_smooth_labels(
    X,
    Y,
    n_neighbors: int = 15,
    alpha: float = 0.7,
    n_iter: int = 15,
    method: str = "adaptive",
    n_pcs: int | None = 40,
    metric: str = "euclidean",
    return_graph: bool = False,
):
    """Inductive graph smoothing on the training set only."""
    Y = np.asarray(Y, dtype=np.float64)
    if method == "adaptive":
        S = _fuzzy_knn_graph(X, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric)
    elif method == "global_rbf":
        S = _global_rbf_neighbors(X, n_neighbors=n_neighbors)
    else:
        raise ValueError(f"Unknown neighbor method: {method}")
    D = Y.copy()
    for _ in range(n_iter):
        D = (1.0 - alpha) * Y + alpha * (S @ D)
        D = np.clip(D, 0, None)
        D = D / np.clip(D.sum(axis=1, keepdims=True), 1e-12, None)
    D = D.astype(np.float32)
    if return_graph:
        graph = S.toarray() if sparse.issparse(S) else S
        return D, np.asarray(graph, dtype=np.float32)
    return D


def robust_type_targets(
    X,
    Y,
    n_neighbors: int = 30,
    alpha: float = 0.8,
    n_iter: int = 12,
    mix: float = 0.55,
    min_weight: float = 0.2,
):
    """Soften type labels with neighbor consensus and downweight isolated flips.

    ``agree`` is the smoothed mass on the given (possibly noisy) class. Cells
    whose neighbors disagree with the annotation get lower training weight and
    a target pulled toward the graph posterior.
    """
    Y = np.asarray(Y, dtype=np.float64)
    smoothed, graph = knn_smooth_labels(
        X,
        Y,
        n_neighbors=n_neighbors,
        alpha=alpha,
        n_iter=n_iter,
        method="adaptive",
        n_pcs=None,
        return_graph=True,
    )
    mix = float(np.clip(mix, 0.0, 1.0))
    targets = blend_targets((Y, 1.0 - mix), (smoothed, mix))
    agree = np.sum(np.asarray(smoothed, dtype=np.float64) * Y, axis=1)
    weights = np.clip(agree, min_weight, 1.0).astype(np.float32)
    return targets, weights, np.asarray(graph, dtype=np.float32), smoothed.astype(np.float32)


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


def cd8_til_lineage_edges():
    """NaiveLike-CM-EM-{TEMRA,TPEX-TEX} with MAIT off the memory axis."""
    return [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (2, 5), (2, 6)]
