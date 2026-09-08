import numpy as np
from scLDL.state_targets import blend_targets, knn_smooth_labels, lineage_laplacian, pancreas_lineage_edges


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


def test_pancreas_laplacian():
    L = lineage_laplacian(pancreas_lineage_edges(), 8)
    assert L.shape == (8, 8)
    assert np.allclose(L, L.T)
    assert np.allclose(L.sum(axis=1), 0.0, atol=1e-5)
