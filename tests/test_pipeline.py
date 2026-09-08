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
