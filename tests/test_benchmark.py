import numpy as np
import pytest
from anndata import AnnData

from scLDL.benchmark import CORE_METHODS, run_benchmark, summarize
from scLDL.benchmark.protocol import prepare_holdout
from scLDL.metrics import score_annotations


def _blob_adata(n=90, n_genes=30, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    means = rng.normal(size=(n_classes, n_genes)) * 1.6
    x = np.exp(means[y] + rng.normal(scale=0.25, size=(n, n_genes)))
    x = np.log1p(x / x.sum(axis=1, keepdims=True) * 1e4).astype(np.float32)
    adata = AnnData(x)
    adata.obs["cell_type"] = [f"type_{i}" for i in y]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    return adata


def test_holdout_selects_genes_from_train_only():
    adata = _blob_adata()
    data = prepare_holdout(adata, label_key="cell_type", n_top_genes=12, seed=0)
    assert data.X_train_log.shape[1] == 12
    assert data.X_test_log.shape[1] == 12
    assert set(data.var_names).issubset(set(adata.var_names.astype(str)))


def test_score_annotations_unknown_labels():
    classes = np.array(["a", "b"])
    y = np.array(["a", "b", "c"])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    metrics, pred = score_annotations(y, proba, classes)
    assert metrics["n_test"] == 2
    assert metrics["unknown_label_fraction"] == pytest.approx(1 / 3)
    assert list(pred) == ["a", "b", "a"]


def test_benchmark_logistic_beats_majority():
    adata = _blob_adata()
    results = run_benchmark(
        adata,
        label_key="cell_type",
        methods=["majority", "logistic", "knn", "pca_knn", "scldl_mlp"],
        n_top_genes=40,
        n_repeats=1,
        epochs=15,
        n_hidden=32,
        batch_size=16,
        verbose=False,
    )
    ok = results[results["status"] == "ok"].set_index("method")
    assert "majority" in ok.index
    assert "logistic" in ok.index
    assert ok.loc["logistic", "accuracy"] > ok.loc["majority", "accuracy"]
    assert ok.loc["logistic", "accuracy"] > 0.7
    summary = summarize(results)
    assert "accuracy_mean" in summary.columns


def test_benchmark_rejects_unknown_method():
    adata = _blob_adata()
    with pytest.raises(ValueError, match="Unknown methods"):
        run_benchmark(adata, methods=["not_a_method"], verbose=False)


def test_cross_dataset_missing_genes():
    ref = _blob_adata(seed=1)
    query = ref[:, :18].copy()
    query.obs_names = [f"q{i}" for i in range(query.n_obs)]
    results = run_benchmark(
        ref,
        label_key="cell_type",
        query=query,
        methods=["majority", "logistic"],
        n_top_genes=40,
        verbose=False,
    )
    assert (results["status"] == "ok").all()
    assert results["protocol"].iloc[0] == "cross_dataset"


def test_scanpy_ingest_runs():
    adata = _blob_adata()
    results = run_benchmark(
        adata,
        label_key="cell_type",
        methods=["scanpy_ingest"],
        n_top_genes=40,
        n_repeats=1,
        verbose=False,
    )
    row = results.iloc[0]
    assert row["method"] == "scanpy_ingest"
    if row["status"] == "ok":
        assert row["accuracy"] > 0.6
    else:
        assert row["error"]


def test_core_method_names():
    assert "scanpy_ingest" in CORE_METHODS
    assert "scldl_concentration" in CORE_METHODS
    assert "majority" in CORE_METHODS


def test_celltypist_runs_if_installed():
    pytest.importorskip("celltypist")
    adata = _blob_adata(n=120, n_genes=40, n_classes=3, seed=1)
    results = run_benchmark(
        adata,
        label_key="cell_type",
        methods=["celltypist", "logistic"],
        n_top_genes=40,
        n_repeats=1,
        verbose=False,
    )
    ok = results[results["status"] == "ok"].set_index("method")
    assert "celltypist" in ok.index
    assert ok.loc["celltypist", "accuracy"] > 0.7
