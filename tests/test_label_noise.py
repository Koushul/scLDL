import numpy as np
from anndata import AnnData

from scLDL.label_noise import (
    HIPPO_SIMILAR,
    LYMPH_SIMILAR,
    align_proba,
    flip_labels,
    mass_on_label,
    noise_recovery_metrics,
    precision_at_k,
)
from scLDL.pipeline import AnnotationPipeline


def test_flip_labels_stratified_and_never_self():
    rng = np.random.default_rng(0)
    y = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 50)
    noisy, flipped = flip_labels(y, 0.2, rng, mode="uniform")
    assert flipped.sum() == 20 + 20 + 10
    assert np.all(noisy[flipped] != y[flipped])
    assert np.all(noisy[~flipped] == y[~flipped])
    assert abs(flipped[:100].mean() - 0.2) < 1e-9
    assert flip_labels(y, 0.0, rng)[1].sum() == 0


def test_similar_flips_use_hippo_map():
    rng = np.random.default_rng(1)
    y = np.array(["CA1"] * 40 + ["CA3"] * 40 + ["DG"] * 20)
    noisy, flipped = flip_labels(y, 0.5, rng, mode="similar", similar=HIPPO_SIMILAR)
    assert np.all(noisy[(y == "CA1") & flipped] == "CA3")
    assert np.all(noisy[(y == "CA3") & flipped] == "CA1")
    assert np.all(noisy[(y == "DG") & flipped] == "CA1")
    y2 = np.array(["Resting T"] * 40 + ["CD8+ T"] * 40 + ["B"] * 20)
    noisy2, flipped2 = flip_labels(y2, 0.5, rng, mode="similar", similar=LYMPH_SIMILAR)
    assert np.all(noisy2[(y2 == "Resting T") & flipped2] == "CD8+ T")
    assert np.all(noisy2[(y2 == "CD8+ T") & flipped2] == "Resting T")
    assert np.all(noisy2[(y2 == "B") & flipped2] != "B")


def test_mass_on_label_and_align_proba():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
    mass = mass_on_label(p, ["A", "C"], ["A", "B", "C"])
    np.testing.assert_allclose(mass, [0.7, 0.1])
    q = align_proba(p[:, :2], ["A", "B"], ["B", "A", "C"])
    np.testing.assert_allclose(q[0, 1] / q[0].sum(), p[0, 0] / p[0, :2].sum(), atol=1e-6)


def test_precision_at_k_and_recovery_metrics():
    y_true = np.array(["A", "A", "B", "B"])
    y_noisy = np.array(["A", "B", "B", "A"])
    y_pred = np.array(["A", "A", "B", "B"])
    proba = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.7, 0.3],
        ]
    )
    m = noise_recovery_metrics(y_true, y_noisy, y_pred, proba, ["A", "B"], name="toy")
    assert m["n_flipped"] == 2
    assert m["correction_rate"] == 1.0
    assert m["clean_kept_rate"] == 1.0
    assert m["discovery_auroc_1m_p_given"] == 1.0
    assert precision_at_k(y_true != y_noisy, 1.0 - mass_on_label(proba, y_noisy, ["A", "B"]), 2) == 1.0


def _blob_adata(n=120, n_genes=18, n_classes=3, seed=0, scale=0.25, sep=3.2):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n // n_classes)
    means = rng.normal(size=(n_classes, n_genes)) * sep
    x = means[y] + rng.normal(scale=scale, size=(len(y), n_genes))
    adata = AnnData(x.astype(np.float32))
    adata.obs["cell_type"] = [f"type_{i}" for i in y]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs_names = [f"c{i}" for i in range(len(y))]
    return adata


def test_oof_scldl_recovers_uniform_flips():
    from sklearn.model_selection import StratifiedKFold

    from scLDL.label_noise import align_proba

    adata = _blob_adata()
    y_true = adata.obs["cell_type"].to_numpy()
    rng = np.random.default_rng(2)
    noisy, flipped = flip_labels(y_true, 0.3, rng, mode="uniform")
    adata.obs["cell_type"] = noisy
    n = adata.n_obs
    classes = np.unique(noisy)
    proba = np.zeros((n, len(classes)), dtype=np.float32)
    pred = np.empty(n, dtype=object)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for tr, te in skf.split(np.zeros(n), noisy):
        pipe = AnnotationPipeline(
            model="scldl",
            n_top_genes=50,
            n_pcs=8,
            n_neighbors=8,
            n_hidden=32,
            epochs=18,
            batch_size=16,
            verbose=False,
            query_correct="none",
            spatial="off",
            graph_refine="off",
        )
        pipe.fit(adata[tr].copy(), label_key="cell_type")
        out = pipe.annotate(adata[te].copy(), copy=True)
        proba[te] = align_proba(out.obsm["X_scldl"], pipe.classes_, classes)
        pred[te] = out.obs["scldl_pred"].to_numpy()
    metrics = noise_recovery_metrics(y_true, noisy, pred, proba, classes)
    assert flipped.mean() == metrics["noise_rate"]
    assert metrics["correction_rate"] > 0.7
    assert metrics["discovery_auroc_1m_p_given"] > 0.8
    assert metrics["acc_vs_true"] > 0.8


def test_label_smooth_improves_noisy_type_fit():
    from sklearn.model_selection import StratifiedKFold

    adata = _blob_adata(n=180, n_genes=16, n_classes=3, seed=1, scale=0.9, sep=1.6)
    y_true = adata.obs["cell_type"].to_numpy()
    rng = np.random.default_rng(4)
    noisy, _ = flip_labels(y_true, 0.4, rng, mode="uniform")
    adata.obs["cell_type"] = noisy
    n = adata.n_obs
    classes = np.unique(noisy)

    def _oof(smooth):
        proba = np.zeros((n, len(classes)), dtype=np.float32)
        pred = np.empty(n, dtype=object)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        for tr, te in skf.split(np.zeros(n), noisy):
            pipe = AnnotationPipeline(
                model="scldl",
                n_top_genes=50,
                n_pcs=8,
                n_neighbors=8,
                n_hidden=32,
                epochs=16,
                batch_size=16,
                verbose=False,
                query_correct="none",
                spatial="off",
                graph_refine="off",
                label_smooth=smooth,
            )
            pipe.fit(adata[tr].copy(), label_key="cell_type")
            out = pipe.annotate(adata[te].copy(), copy=True)
            proba[te] = align_proba(out.obsm["X_scldl"], pipe.classes_, classes)
            pred[te] = out.obs["scldl_pred"].to_numpy()
            if smooth == "on":
                assert pipe.last_label_smooth_ == "on"
        return noise_recovery_metrics(y_true, noisy, pred, proba, classes)

    hard = _oof("off")
    robust = _oof("on")
    assert robust["acc_vs_true"] >= hard["acc_vs_true"]
    assert robust["correction_rate"] >= hard["correction_rate"]
    assert robust["acc_vs_true"] > 0.85
