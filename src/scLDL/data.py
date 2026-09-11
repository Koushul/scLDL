import numpy as np
from scipy import sparse


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray(), dtype=np.float32)
    arr = np.asarray(x)
    if arr.dtype == np.float32:
        return arr
    return np.asarray(arr, dtype=np.float32)


def looks_like_counts(x) -> bool:
    n = int(x.shape[0])
    if n == 0 or int(x.shape[1]) == 0:
        return False
    if sparse.issparse(x):
        mx = x.max()
        if sparse.issparse(mx):
            mx = mx.toarray().ravel()[0]
        mx = float(mx)
        amin = float(x.data.min()) if x.data.size else 0.0
        sample = np.asarray(x.data[: min(4096, int(x.data.size))], dtype=np.float64) if x.data.size else np.array([0.0])
    else:
        arr = np.asarray(x)
        mx = float(np.nanmax(arr))
        amin = float(np.nanmin(arr))
        flat = np.asarray(arr, dtype=np.float64).ravel()
        nz = flat[np.abs(flat) > 0]
        sample = nz[:4096] if nz.size else flat[: min(32, flat.size)]
    if sample.size == 0:
        return False
    if amin < -0.05:
        return False
    if mx > 20:
        return True
    frac_int = np.mean(np.abs(sample - np.round(sample)) < 1e-6)
    return bool(frac_int > 0.9)


def log1p_normalize(X, target_sum: float = 1e4):
    """Library-size normalize then log1p, matching scanpy's default path."""
    if sparse.issparse(X):
        X = X.tocsr(copy=True).astype(np.float64, copy=False)
        counts = np.asarray(X.sum(axis=1)).ravel()
        scale = np.ones(counts.shape[0], dtype=np.float64)
        nz = counts > 0
        scale[nz] = target_sum / counts[nz]
        X = X.multiply(scale[:, np.newaxis]).tocsr()
        if X.nnz:
            X.data = np.log1p(X.data)
        return X.astype(np.float32)
    X = np.array(X, dtype=np.float64, copy=True)
    counts = X.sum(axis=1)
    scale = np.ones(len(counts), dtype=np.float64)
    nz = counts > 0
    scale[nz] = target_sum / counts[nz]
    return np.log1p(X * scale[:, None]).astype(np.float32)


def preprocess_reference(adata, n_top_genes: int = 2000, copy: bool = True, always_include=None):
    import scanpy as sc

    ad = adata.copy() if copy else adata
    if looks_like_counts(ad.X):
        ad.X = log1p_normalize(ad.X)
    if n_top_genes and ad.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, subset=False)
        keep = ad.var["highly_variable"].to_numpy()
        if always_include is not None:
            keep = keep | np.isin(ad.var_names.astype(str), np.asarray(list(always_include), dtype=str))
        ad = ad[:, keep].copy()
    return ad


def align_matrix(X, query_names, target_names):
    query_names = np.asarray(query_names).astype(str)
    target = np.asarray(target_names).astype(str)
    lookup = {}
    for i, g in enumerate(query_names):
        if g not in lookup:
            lookup[g] = i
    src = np.array([lookup.get(g, -1) for g in target], dtype=np.intp)
    overlap = int((src >= 0).sum())
    if overlap == 0:
        raise ValueError("No overlapping genes between reference and query.")
    x = np.zeros((X.shape[0], len(target)), dtype=np.float32)
    hit = src >= 0
    cols = src[hit]
    if sparse.issparse(X):
        x[:, hit] = to_dense(X.tocsc()[:, cols])
    else:
        x[:, hit] = np.asarray(X[:, cols], dtype=np.float32)
    return x, overlap


def align_to_genes(adata, var_names, copy: bool = True):
    import anndata as ad_mod

    x, overlap = align_matrix(adata.X, adata.var_names, var_names)
    out = ad_mod.AnnData(x, obs=adata.obs.copy() if copy else adata.obs)
    out.var_names = np.asarray(var_names).astype(str)
    out.obs_names = adata.obs_names
    return out, overlap


def labels_to_onehot(values, classes=None):
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    values = np.asarray(values).astype(str)
    if classes is None:
        y = encoder.fit_transform(values)
    else:
        encoder.fit(np.asarray(classes).astype(str))
        y = encoder.transform(values)
    n_classes = len(encoder.classes_)
    onehot = np.zeros((len(y), n_classes), dtype=np.float32)
    onehot[np.arange(len(y)), y] = 1.0
    return onehot, y, encoder
