import numpy as np
import pytest
import torch

from scLDL.metrics import classification_metrics, distribution_metrics
from scLDL.models import ConcentrationLE, HybridLEVI, LEVI, LIBLE, MLPBaseline


def _blob_data(n=80, d=16, k=3, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(k, d)) * 3.0
    y = rng.integers(0, k, size=n)
    x = means[y] + rng.normal(scale=0.4, size=(n, d))
    l = np.eye(k, dtype=np.float32)[y]
    return x.astype(np.float32), l, y


@pytest.mark.parametrize("cls", [MLPBaseline, LIBLE, ConcentrationLE, HybridLEVI])
def test_annotation_models_predict_simplex(cls):
    x, l, y = _blob_data()
    model = cls(n_features=x.shape[1], n_outputs=l.shape[1], n_hidden=32, epochs=8, batch_size=16, verbose=False)
    if cls is HybridLEVI:
        model = cls(
            n_features=x.shape[1],
            n_outputs=l.shape[1],
            n_hidden=32,
            n_latent=16,
            epochs=8,
            batch_size=16,
            verbose=False,
            alpha=0.01,
        )
    model.fit(x, l)
    pred = model.predict(x)
    assert pred.shape == l.shape
    assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(pred >= -1e-6)
    acc = (pred.argmax(1) == y).mean()
    assert acc > 0.7


def test_concentration_uncertainty_separate_from_classes():
    x, l, y = _blob_data()
    model = ConcentrationLE(n_features=x.shape[1], n_outputs=3, n_hidden=32, epochs=5, batch_size=16, verbose=False)
    model.fit(x, l)
    mean, u = model.predict_evidence(x)
    assert mean.shape == (len(x), 3)
    assert u.shape == (len(x),)


def test_levi_requires_labels():
    x, l, y = _blob_data()
    model = LEVI(n_features=x.shape[1], n_outputs=3, n_hidden=32, epochs=5, batch_size=16, verbose=False, alpha=0.1)
    model.fit(x, l)
    pred = model.predict(x, l)
    assert pred.shape == l.shape
    with pytest.raises(TypeError):
        model.predict(x)


def test_metrics():
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.8, 0.2], [0.1, 0.9]])
    d = distribution_metrics(y_true, y_pred)
    assert d["mse"] < 0.05
    c = classification_metrics([0, 1], y_pred)
    assert c["accuracy"] == 1.0
