"""Toy LDL recovery: hard labels observed, soft p(y|z) is known."""

import numpy as np

from scLDL.metrics import distribution_metrics
from scLDL.models import ConcentrationLE, HybridLEVI, LEVI, LIBLE


def generate_toy_data(n_samples=1000, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.uniform(-2, 2, size=(n_samples, 1)).astype(np.float32)
    x = (z**3) + rng.normal(0, 0.1, size=(n_samples, 1)).astype(np.float32)
    x = ((x - x.mean()) / x.std()).astype(np.float32)
    logits = np.concatenate([z, -z, z**2], axis=1)
    y_true = np.exp(logits)
    y_true = (y_true / y_true.sum(axis=1, keepdims=True)).astype(np.float32)
    y_idx = np.array([rng.choice(3, p=p) for p in y_true])
    y_obs = np.zeros_like(y_true)
    y_obs[np.arange(n_samples), y_idx] = 1.0
    return x, y_obs, y_true


def main():
    x, y_obs, y_true = generate_toy_data()
    specs = [
        ("LIBLE", LIBLE, dict(n_hidden=64, n_latent=8, epochs=80, batch_size=32, alpha=1e-3, beta=1e-3)),
        ("ConcentrationLE", ConcentrationLE, dict(n_hidden=64, epochs=80, batch_size=32)),
        ("HybridLEVI", HybridLEVI, dict(n_hidden=64, n_latent=8, epochs=80, batch_size=32, alpha=0.01)),
        ("LEVI", LEVI, dict(n_hidden=64, epochs=80, batch_size=32, alpha=0.01)),
    ]
    print(f"Toy LDL recovery  n={len(x)}")
    for name, cls, kwargs in specs:
        model = cls(n_features=1, n_outputs=3, verbose=False, **kwargs)
        model.fit(x, y_obs)
        if name == "LEVI":
            pred = model.predict(x, y_obs)
        else:
            pred = model.predict(x)
        metrics = distribution_metrics(y_true, pred)
        print(f"{name:16s}  KL={metrics['kl']:.4f}  MSE={metrics['mse']:.4f}  Chebyshev={metrics['chebyshev']:.4f}")


if __name__ == "__main__":
    main()
