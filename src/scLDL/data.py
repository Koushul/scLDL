import numpy as np
from scipy import sparse


def to_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def looks_like_counts(x) -> bool:
    sample = to_dense(x[: min(32, x.shape[0])])
    if sample.size == 0:
        return False
    if np.nanmax(sample) > 20:
        return True
    frac_int = np.mean(np.abs(sample - np.round(sample)) < 1e-6)
    return bool(frac_int > 0.9)


def preprocess_reference(adata, n_top_genes: int = 2000, copy: bool = True):
    import scanpy as sc

    ad = adata.copy() if copy else adata
    if looks_like_counts(ad.X):
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    if n_top_genes and ad.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, subset=True)
    return ad


def align_to_genes(adata, var_names, copy: bool = True):
    import anndata as ad_mod
    import pandas as pd

    ad = adata.copy() if copy else adata
    query_names = pd.Index(ad.var_names.astype(str))
    target = pd.Index(np.asarray(var_names).astype(str))
    overlap = int(target.isin(query_names).sum())
    if overlap == 0:
        raise ValueError("No overlapping genes between reference and query.")

    dense = pd.DataFrame(to_dense(ad.X), columns=query_names)
    x = dense.reindex(columns=target, fill_value=0.0).to_numpy(dtype=np.float32)

    out = ad_mod.AnnData(x, obs=ad.obs.copy())
    out.var_names = target
    out.obs_names = ad.obs_names
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
