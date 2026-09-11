import numpy as np
from scLDL.state_targets import (
    blend_targets,
    cd8_til_lineage_edges,
    knn_smooth_labels,
    lineage_laplacian,
    pancreas_lineage_edges,
    probabilistic_neighbors,
    robust_type_targets,
)


def test_blend_and_smooth():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6)).astype(np.float32)
    Y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=40)]
    D = knn_smooth_labels(X, Y, n_neighbors=5, n_iter=5)
    assert D.shape == Y.shape
    assert np.allclose(D.sum(axis=1), 1.0, atol=1e-5)
    M = rng.random((40, 3)).astype(np.float32)
    M = M / M.sum(axis=1, keepdims=True)
    B = blend_targets((Y, 0.4), (D, 0.4), (M, 0.2))
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-5)


def test_probabilistic_neighbors_row_stochastic():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(25, 8)).astype(np.float32)
    P = probabilistic_neighbors(X, n_neighbors=6, n_pcs=4)
    assert P.shape == (25, 25)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(P >= -1e-12)
    np.fill_diagonal(P, 0.0)
    assert np.all(P.sum(axis=1) > 0.99)

    Y = np.eye(3, dtype=np.float32)[rng.integers(0, 3, size=25)]
    adaptive, S = knn_smooth_labels(X, Y, n_neighbors=6, n_iter=4, method="adaptive", return_graph=True)
    legacy = knn_smooth_labels(X, Y, n_neighbors=6, n_iter=4, method="global_rbf")
    assert np.allclose(adaptive.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(S.sum(axis=1), 1.0, atol=1e-5)
    assert not np.allclose(adaptive, legacy, atol=1e-4)


def test_robust_type_targets_downweight_isolated_flips():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 40)
    X = rng.normal(scale=0.2, size=(120, 8)).astype(np.float32)
    X += np.eye(3, 8, dtype=np.float32)[y] * 4
    Y = np.eye(3, dtype=np.float32)[y]
    noisy = Y.copy()
    flip = np.zeros(120, dtype=bool)
    flip[::10] = True
    for i in np.flatnonzero(flip):
        noisy[i] = 0
        noisy[i, (y[i] + 1) % 3] = 1
    _, weights, _, smoothed = robust_type_targets(X, noisy, n_neighbors=8, n_iter=8)
    assert weights[flip].mean() < weights[~flip].mean()
    true_mass = smoothed[np.arange(120), y]
    assert true_mass[flip].mean() > 0.4


def test_state_model_predicts_simplex():
    from scLDL.models import StateConcentrationLE

    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=60)
    X = rng.normal(size=(60, 12)).astype(np.float32) + np.eye(3, 12, dtype=np.float32)[y] * 3
    L = np.eye(3, dtype=np.float32)[y]
    L = 0.7 * L + 0.3 / 3
    L = L / L.sum(axis=1, keepdims=True)
    model = StateConcentrationLE(
        n_features=12,
        n_outputs=3,
        n_hidden=32,
        epochs=6,
        batch_size=16,
        lineage_edges=[(0, 1), (1, 2)],
        verbose=False,
        mixup_alpha=0.0,
    )
    model.fit(X, L)
    pred = model.predict(X)
    assert pred.shape == L.shape
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-4)


def test_cd8_laplacian():
    L = lineage_laplacian(cd8_til_lineage_edges(), 7)
    assert L.shape == (7, 7)
    assert np.allclose(L, L.T)
    assert np.allclose(L.sum(axis=1), 0.0, atol=1e-5)
