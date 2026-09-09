import numpy as np
from anndata import AnnData

from scLDL import AnnotationPipeline
from scLDL.data import align_to_genes


def _make_adata(n=60, n_genes=24, n_classes=3, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    means = rng.normal(size=(n_classes, n_genes)) * 2.5
    x = means[y] + rng.normal(scale=0.3, size=(n, n_genes))
    adata = AnnData(x.astype(np.float32))
    adata.obs["cell_type"] = [f"type_{i}" for i in y]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    return adata


def test_pipeline_fit_annotate_evaluate():
    adata = _make_adata()
    pipe = AnnotationPipeline(
        model="mlp",
        n_top_genes=50,
        n_hidden=32,
        epochs=12,
        batch_size=16,
        verbose=False,
    )
    pipe.fit(adata, label_key="cell_type")
    out = pipe.annotate(adata, copy=True)
    assert "scldl_pred" in out.obs
    assert out.obsm["X_scldl"].shape == (adata.n_obs, 3)
    metrics = pipe.evaluate(adata)
    assert metrics["accuracy"] > 0.7


def test_gene_alignment_fills_missing():
    adata = _make_adata()
    query = adata[:, :10].copy()
    aligned, overlap = align_to_genes(query, adata.var_names)
    assert aligned.n_vars == adata.n_vars
    assert overlap == 10
    assert np.allclose(aligned.X[:, 10:], 0)
    assert np.allclose(aligned.X[:, :10], np.asarray(query.X))


def test_gene_alignment_uses_first_duplicate():
    from scipy import sparse

    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    ad = AnnData(sparse.csr_matrix(x))
    ad.var_names = ["A", "A", "B"]
    aligned, overlap = align_to_genes(ad, ["B", "A", "C"])
    assert overlap == 2
    np.testing.assert_allclose(aligned.X, np.array([[2, 0, 0], [5, 3, 0]], dtype=np.float32))


def test_looks_like_counts_uses_global_max():
    from scipy import sparse

    from scLDL.data import looks_like_counts

    x = np.zeros((40, 4), dtype=np.float32)
    x[35, 1] = 80.0
    assert looks_like_counts(sparse.csr_matrix(x))
    assert looks_like_counts(x)
    logx = np.log1p(np.abs(np.random.default_rng(0).normal(size=(40, 4)).astype(np.float32)))
    assert not looks_like_counts(logx)


def test_log1p_normalize_matches_scanpy():
    import scanpy as sc
    from scipy import sparse

    from scLDL.data import log1p_normalize, to_dense

    rng = np.random.default_rng(0)
    counts = rng.integers(0, 30, size=(12, 8)).astype(np.float32)
    ad = AnnData(sparse.csr_matrix(counts))
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    got = log1p_normalize(sparse.csr_matrix(counts))
    np.testing.assert_allclose(to_dense(got), to_dense(ad.X), rtol=1e-5, atol=1e-5)
    dense = log1p_normalize(counts)
    np.testing.assert_allclose(dense, to_dense(ad.X), rtol=1e-5, atol=1e-5)
