from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scLDL.data import align_to_genes, looks_like_counts, preprocess_reference, to_dense


@dataclass
class BenchmarkData:
    X_train_log: np.ndarray
    X_test_log: np.ndarray
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    X_train_counts: np.ndarray | None
    X_test_counts: np.ndarray | None
    y_train: np.ndarray
    y_test: np.ndarray
    classes: np.ndarray
    var_names: np.ndarray
    n_overlap: int


def flip_labels(y, rate: float, rng: np.random.Generator):
    y = np.asarray(y).copy()
    if rate <= 0:
        return y
    classes = np.unique(y)
    if len(classes) < 2:
        return y
    n_flip = int(len(y) * rate)
    idx = rng.choice(len(y), size=n_flip, replace=False)
    for i in idx:
        others = classes[classes != y[i]]
        y[i] = rng.choice(others)
    return y


def _normalize_if_counts(adata):
    import scanpy as sc

    ad = adata.copy()
    if looks_like_counts(ad.X):
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        return ad, True
    return ad, False


def prepare_reference_query(ref, query, label_key: str, n_top_genes: int = 2000) -> BenchmarkData:
    if label_key not in ref.obs:
        raise KeyError(f"{label_key!r} not found in reference obs")
    if label_key not in query.obs:
        raise KeyError(f"{label_key!r} not found in query obs")

    raw_ref = ref.copy()
    raw_query = query.copy()
    ref_log, _ = _normalize_if_counts(ref)
    query_log, _ = _normalize_if_counts(query)
    ref_hvg = preprocess_reference(ref_log, n_top_genes=n_top_genes, copy=True)
    var_names = np.asarray(ref_hvg.var_names.astype(str))
    query_aln, n_overlap = align_to_genes(query_log, var_names)

    x_train = to_dense(ref_hvg.X)
    x_test = to_dense(query_aln.X)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_test_s = scaler.transform(x_test).astype(np.float32)

    y_train = np.asarray(ref_hvg.obs[label_key].astype(str))
    y_test = np.asarray(query_aln.obs[label_key].astype(str))
    classes = np.unique(y_train)

    counts_train = counts_test = None
    if looks_like_counts(raw_ref.X):
        counts_train = to_dense(align_to_genes(raw_ref, var_names)[0].X)
        counts_test = to_dense(align_to_genes(raw_query, var_names)[0].X)

    return BenchmarkData(
        X_train_log=x_train,
        X_test_log=x_test,
        X_train_scaled=x_train_s,
        X_test_scaled=x_test_s,
        X_train_counts=counts_train,
        X_test_counts=counts_test,
        y_train=y_train,
        y_test=y_test,
        classes=classes,
        var_names=var_names,
        n_overlap=n_overlap,
    )


def prepare_holdout(
    adata,
    label_key: str,
    test_size: float = 0.2,
    n_top_genes: int = 2000,
    seed: int = 0,
    noise_rate: float = 0.0,
) -> BenchmarkData:
    if label_key not in adata.obs:
        raise KeyError(f"{label_key!r} not found in adata.obs")
    labels = np.asarray(adata.obs[label_key].astype(str))
    idx = np.arange(adata.n_obs)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    data = prepare_reference_query(
        adata[train_idx].copy(),
        adata[test_idx].copy(),
        label_key=label_key,
        n_top_genes=n_top_genes,
    )
    if noise_rate > 0:
        rng = np.random.default_rng(seed)
        data.y_train = flip_labels(data.y_train, noise_rate, rng)
        data.classes = np.unique(data.y_train)
    return data
