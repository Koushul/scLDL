import numpy as np
from anndata import AnnData

from scLDL.pipeline import AnnotationPipeline
from scLDL.spatial_smooth import spatial_refine, spatial_stats


def test_spatial_refine_cleans_speckle_without_merging_domains():
    rng = np.random.default_rng(0)
    n = 180
    xy = np.vstack(
        [
            rng.normal([-2.2, 0.0], 0.22, size=(n, 2)),
            rng.normal([2.2, 0.0], 0.22, size=(n, 2)),
        ]
    )
    true = np.array([0] * n + [1] * n)
    p = np.eye(2, dtype=np.float32)[true]
    flip = rng.choice(2 * n, size=50, replace=False)
    p[flip] = p[flip, ::-1]
    z = np.eye(2, dtype=np.float32)[true] + rng.normal(scale=0.05, size=(2 * n, 2)).astype(np.float32)
    before = (p.argmax(1) == true).mean()
    out = spatial_refine(p, xy, z=z, n_neighbors=10, n_iter=3, task="type")
    after = (out.argmax(1) == true).mean()
    assert after > before
    assert after >= 0.92
    st0 = spatial_stats(p.argmax(1), xy)
    st1 = spatial_stats(out.argmax(1), xy)
    assert st1["isolated_frac"] < st0["isolated_frac"]
    assert st1["mean_same_neighbor_frac"] > st0["mean_same_neighbor_frac"]


def test_spatial_refine_keeps_a_sharp_boundary():
    rng = np.random.default_rng(1)
    xs = np.linspace(-1, 1, 24)
    ys = np.linspace(-1, 1, 24)
    grid = np.array([[x, y] for y in ys for x in xs])
    true = (grid[:, 0] >= 0).astype(int)
    p = np.eye(2, dtype=np.float32)[true]
    z = np.eye(2, dtype=np.float32)[true] + rng.normal(scale=0.03, size=p.shape).astype(np.float32)
    out = spatial_refine(p, grid, z=z, n_neighbors=8, n_iter=2, task="type")
    pred = out.argmax(1)
    interior = np.abs(grid[:, 0]) > 0.25
    assert (pred[interior] == true[interior]).mean() >= 0.98


def test_pipeline_spatial_auto_uses_coordinates():
    rng = np.random.default_rng(2)
    n = 80
    y = np.array([0] * 40 + [1] * 40)
    x = rng.normal(scale=0.2, size=(n, 12)).astype(np.float32)
    x[:, 0] += y * 3
    ad = AnnData(x)
    ad.obs["cell_type"] = [f"type_{i}" for i in y]
    ad.var_names = [f"g{i}" for i in range(12)]
    ad.obs_names = [f"c{i}" for i in range(n)]
    ad.obsm["spatial"] = np.column_stack([y.astype(float) + rng.normal(scale=0.05, size=n), rng.normal(size=n)])
    pipe = AnnotationPipeline(model="scldl", n_top_genes=20, n_pcs=6, n_hidden=32, epochs=12, batch_size=16, verbose=False)
    pipe.fit(ad, label_key="cell_type")
    out = pipe.annotate(ad, copy=True)
    assert pipe.last_spatial_ == "on"
    assert "X_scldl_expr" in out.obsm
    assert "scldl_pred_expr" in out.obs
    pipe.spatial = "off"
    labeled = pipe.annotate(ad, copy=True)
    assert pipe.last_spatial_ == "off"
    assert "scldl_pred_expr" not in labeled.obs
