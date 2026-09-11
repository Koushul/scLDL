import numpy as np
from anndata import AnnData

from scLDL.embedding import ReferenceEmbedding, mnn_map
from scLDL.interpret import adaptive_blend, dissonance, illegal_mass, project_lineage, row_normalize, top2_pairs
from scLDL.neighbors import query_label_transfer
from scLDL.pipeline import AnnotationPipeline


def _type_adata(n=90, n_genes=24, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    means = rng.normal(size=(n_classes, n_genes)) * 3.0
    x = means[y] + rng.normal(scale=0.25, size=(n, n_genes))
    adata = AnnData(x.astype(np.float32))
    adata.obs["cell_type"] = [f"type_{i}" for i in y]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs_names = [f"c{i}" for i in range(n)]
    return adata


def _state_adata(n=120, seed=0):
    rng = np.random.default_rng(seed)
    # continuum a -> b -> c using two marker modules
    t = rng.uniform(0, 2, size=n)
    y = np.where(t < 0.7, 0, np.where(t < 1.3, 1, 2))
    x = rng.normal(scale=0.2, size=(n, 16)).astype(np.float32)
    x[:, 0] += (2 - t) * 2.2
    x[:, 1] += (2 - t) * 1.8
    x[:, 2] += (1.0 - np.abs(t - 1.0)) * 2.4
    x[:, 3] += (1.0 - np.abs(t - 1.0)) * 1.6
    x[:, 4] += t * 2.2
    x[:, 5] += t * 1.8
    adata = AnnData(x)
    adata.obs["cell_state"] = np.array(["A", "B", "C"])[y]
    adata.var_names = [f"g{i}" for i in range(16)]
    adata.obs_names = [f"s{i}" for i in range(n)]
    return adata


def test_mnn_recovers_global_shift():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(80, 8))
    query = ref[:30] + 4.0
    mapped = mnn_map(query, ref, n_neighbors=10)
    raw = np.linalg.norm(query - ref[:30], axis=1).mean()
    fixed = np.linalg.norm(mapped - ref[:30], axis=1).mean()
    assert fixed < 0.35 * raw


def test_mnn_matches_mutual_neighbor_mean():
    rng = np.random.default_rng(4)
    ref = rng.normal(size=(40, 6))
    query = ref[:12] + rng.normal(scale=0.05, size=(12, 6))
    mapped = mnn_map(query, ref, n_neighbors=8, shrink=0.85)
    from sklearn.neighbors import NearestNeighbors

    k = 8
    q2r = NearestNeighbors(n_neighbors=k).fit(ref).kneighbors(query, return_distance=False)
    r2q = NearestNeighbors(n_neighbors=k).fit(query).kneighbors(ref, return_distance=False)
    r2q_sets = [set(row.tolist()) for row in r2q]
    expected = np.empty_like(query)
    for i in range(len(query)):
        partners = [int(j) for j in q2r[i] if i in r2q_sets[int(j)]]
        if not partners:
            partners = [int(j) for j in q2r[i][:3]]
        target = ref[partners].mean(axis=0)
        expected[i] = query[i] + 0.85 * (target - query[i])
    np.testing.assert_allclose(mapped, expected.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_query_label_transfer_and_pairs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=40)
    X = rng.normal(size=(40, 6)) + np.eye(3, 6)[y] * 4
    Y = np.eye(3, dtype=np.float32)[y]
    p = query_label_transfer(X, X, Y, n_neighbors=5)
    assert p.shape == Y.shape
    assert np.allclose(p.sum(1), 1.0, atol=1e-4)
    assert (p.argmax(1) == y).mean() > 0.8
    names, mass = top2_pairs(p, ["a", "b", "c"])
    assert len(names) == 40
    assert mass.shape == (40, 2)


def test_lineage_projection_reduces_illegal_mass():
    p = np.array([[0.5, 0.0, 0.5], [0.1, 0.8, 0.1]], dtype=np.float32)
    edges = [(0, 1), (1, 2)]
    before = illegal_mass(p, edges)
    after = illegal_mass(project_lineage(p, edges, strength=0.8), edges)
    assert after[0] < before[0]
    d = dissonance(p)
    assert d[0] > d[1]


def test_adaptive_blend_types_stay_peaked():
    model = np.array([[0.92, 0.04, 0.04], [0.4, 0.4, 0.2]], dtype=np.float32)
    knn = np.array([[0.3, 0.4, 0.3], [0.2, 0.6, 0.2]], dtype=np.float32)
    blended = adaptive_blend(model, knn_p=knn, task="type")
    assert blended[0].max() > 0.85
    assert blended[0].argmax() == 0


def test_interpretable_pipeline_matches_logistic_on_types():
    adata = _type_adata()
    train = adata[:70].copy()
    test = adata[70:].copy()
    pipe = AnnotationPipeline(
        model="interpretable",
        task="type",
        n_top_genes=40,
        n_pcs=10,
        n_hidden=32,
        epochs=20,
        batch_size=16,
        verbose=False,
        query_correct="center",
    )
    pipe.fit(train, label_key="cell_type")
    out = pipe.annotate(test, copy=True)
    assert "scldl_pred" in out.obs
    assert "scldl_pair" in out.obs
    assert out.obsm["X_scldl"].shape[1] == 3
    metrics = pipe.evaluate(test)
    assert metrics["accuracy"] >= 0.75


def test_batch_shift_query_beats_gene_space_mlp():
    rng = np.random.default_rng(4)
    adata = _type_adata(n=150, n_genes=30, seed=4)
    train = adata[:100].copy()
    test = adata[100:].copy()
    test.X = np.asarray(test.X) * 0.35 + 5.5

    gene_mlp = AnnotationPipeline(
        model="mlp",
        n_top_genes=40,
        n_hidden=32,
        epochs=18,
        batch_size=16,
        verbose=False,
    )
    gene_mlp.fit(train, label_key="cell_type")
    mlp_acc = gene_mlp.evaluate(test)["accuracy"]

    refmap = AnnotationPipeline(
        model="interpretable",
        task="type",
        n_top_genes=40,
        n_pcs=12,
        n_hidden=32,
        epochs=18,
        batch_size=16,
        verbose=False,
        query_correct="mnn",
        label_smooth="off",
    )
    refmap.fit(train, label_key="cell_type")
    ref_acc = refmap.evaluate(test)["accuracy"]
    assert ref_acc >= mlp_acc
    assert ref_acc >= 0.70


def test_state_pipeline_tracks_markers():
    adata = _state_adata()
    markers = {"A": ["g0", "g1"], "B": ["g2", "g3"], "C": ["g4", "g5"]}
    train = adata[:90].copy()
    test = adata[90:].copy()
    pipe = AnnotationPipeline(
        model="interpretable",
        task="state",
        n_top_genes=20,
        n_pcs=8,
        n_hidden=32,
        epochs=18,
        batch_size=16,
        verbose=False,
        markers=markers,
        lineage_edges=[(0, 1), (1, 2)],
        query_correct="center",
    )
    pipe.fit(train, label_key="cell_state")
    out = pipe.annotate(test, copy=True)
    assert "X_scldl_marker" in out.obsm
    assert "X_scldl_knn" in out.obsm
    metrics = pipe.evaluate(test)
    assert metrics["accuracy"] >= 0.55
    assert metrics["marker_spearman_mean"] > 0.25
    assert "scldl_illegal" in out.obs
