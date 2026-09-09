import numpy as np
from anndata import AnnData

from scLDL.embedding import mnn_map, mnn_map_supervised
from scLDL.interpret import graph_refine
from scLDL.pipeline import AnnotationPipeline
from scLDL.spatial_smooth import spatial_stats


def test_graph_refine_repairs_isolated_flips():
    rng = np.random.default_rng(0)
    n = 120
    true = np.array([0] * n + [1] * n)
    z = np.eye(2, dtype=np.float64)[true] + rng.normal(scale=0.04, size=(2 * n, 2))
    xy = np.vstack(
        [
            rng.normal([-2.0, 0.0], 0.2, size=(n, 2)),
            rng.normal([2.0, 0.0], 0.2, size=(n, 2)),
        ]
    )
    p = np.eye(2, dtype=np.float32)[true]
    flip = rng.choice(2 * n, size=36, replace=False)
    p[flip] = p[flip, ::-1]
    before = (p.argmax(1) == true).mean()
    out = graph_refine(p, z, xy=xy, n_neighbors=10, n_iter=3, mix=0.7)
    after = (out.argmax(1) == true).mean()
    assert after > before
    assert after >= 0.90
    st0 = spatial_stats(p.argmax(1), xy)
    st1 = spatial_stats(out.argmax(1), xy)
    assert st1["isolated_frac"] < st0["isolated_frac"]


def test_graph_refine_keeps_peaked_cells():
    rng = np.random.default_rng(1)
    p = np.vstack(
        [
            np.tile(np.array([0.96, 0.04], dtype=np.float32), (8, 1)),
            np.tile(np.array([0.05, 0.95], dtype=np.float32), (8, 1)),
        ]
    )
    z = np.vstack(
        [
            rng.normal([0.0, 0.0], 0.05, size=(8, 2)),
            rng.normal([4.0, 0.0], 0.05, size=(8, 2)),
        ]
    )
    out = graph_refine(p, z, n_neighbors=5, n_iter=4, mix=0.55)
    assert out[:8, 0].min() > 0.9
    assert out[8:, 1].min() > 0.9


def test_mnn_map_supervised_matches_per_type():
    rng = np.random.default_rng(1)
    ref_a = rng.normal(size=(40, 6))
    ref_b = rng.normal(size=(40, 6)) + 3.0
    ref = np.vstack([ref_a, ref_b])
    query = np.vstack([ref_a[:15] + 1.2, ref_b[:10] + 1.2])
    q_lab = np.array([0] * 15 + [1] * 10)
    r_lab = np.array([0] * 40 + [1] * 40)
    got = mnn_map_supervised(query, ref, q_lab, r_lab, n_neighbors=8, shrink=0.85)
    exp = np.vstack(
        [
            mnn_map(query[:15], ref_a, n_neighbors=8, shrink=0.85),
            mnn_map(query[15:], ref_b, n_neighbors=8, shrink=0.85),
        ]
    )
    np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-5)


def test_pipeline_ablation_flags_toggle():
    rng = np.random.default_rng(2)
    y = np.array([0] * 40 + [1] * 40)
    x = rng.normal(scale=0.25, size=(80, 16)).astype(np.float32)
    x[:, 0] += y * 3.2
    ad = AnnData(x)
    ad.obs["cell_type"] = [f"type_{i}" for i in y]
    ad.var_names = [f"g{i}" for i in range(16)]
    ad.obs_names = [f"c{i}" for i in range(80)]
    pipe = AnnotationPipeline(
        model="scldl",
        n_top_genes=20,
        n_pcs=8,
        n_hidden=32,
        epochs=8,
        batch_size=16,
        verbose=False,
        query_correct="mnn",
        spatial="off",
        graph_refine="off",
        supervised_mnn="off",
    )
    pipe.fit(ad, label_key="cell_type")
    pipe.evaluate(ad)
    assert pipe.last_graph_refine_ == "off"
    assert pipe.last_supervised_mnn_ is False
    pipe.graph_refine = "on"
    pipe.supervised_mnn = "on"
    pipe.evaluate(ad)
    assert pipe.last_graph_refine_ == "on"
    assert pipe.last_supervised_mnn_ is True
