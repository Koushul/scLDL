"""Ablate query-graph refinement and two-pass supervised MNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from anndata import AnnData
from sklearn.model_selection import train_test_split

from scLDL.metrics import classification_metrics
from scLDL.pipeline import AnnotationPipeline
from scLDL.spatial_smooth import spatial_stats

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "graph_smnn"
CFGS = (
    ("baseline", "off", "off"),
    ("graph", "on", "off"),
    ("smnn", "off", "on"),
    ("both", "on", "on"),
)


def _metrics(y_true, y_pred, extra=None):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    out = classification_metrics(y_true, y_pred)
    classes, counts = np.unique(y_true, return_counts=True)
    rare = classes[np.argmin(counts)]
    tp = np.sum((y_true == rare) & (y_pred == rare))
    fp = np.sum((y_true != rare) & (y_pred == rare))
    fn = np.sum((y_true == rare) & (y_pred != rare))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    out["rare_label"] = str(rare)
    out["rare_f1"] = float(2 * prec * rec / (prec + rec + 1e-12))
    if extra:
        out.update(extra)
    return out


def _blob(n=320, n_genes=36, counts=(140, 120, 40, 20), seed=0):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    rng.shuffle(y)
    means = rng.normal(size=(len(counts), n_genes)) * 2.8
    x = means[y] + rng.normal(scale=0.28, size=(len(y), n_genes))
    ad = AnnData(x.astype(np.float32))
    ad.obs["cell_type"] = np.array([f"type_{i}" for i in y])
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    ad.obs_names = [f"c{i}" for i in range(len(y))]
    return ad


def _spatial_speckle(n=160, seed=4):
    rng = np.random.default_rng(seed)
    y = np.array([0] * n + [1] * n)
    x = rng.normal(scale=0.22, size=(2 * n, 18)).astype(np.float32)
    x[:, 0] += y * 3.4
    flip = rng.choice(2 * n, size=40, replace=False)
    x[flip, 0] = (1 - y[flip]) * 3.4 + rng.normal(scale=0.22, size=len(flip))
    ad = AnnData(x)
    ad.obs["cell_type"] = np.array([f"type_{i}" for i in y])
    ad.var_names = [f"g{i}" for i in range(18)]
    ad.obs_names = [f"c{i}" for i in range(2 * n)]
    ad.obsm["spatial"] = np.column_stack(
        [
            y.astype(float) * 4.0 + rng.normal(scale=0.12, size=2 * n),
            rng.normal(scale=0.12, size=2 * n),
        ]
    )
    return ad


def _split(ad, label="cell_type", seed=0):
    idx = np.arange(ad.n_obs)
    tr, te = train_test_split(idx, test_size=0.35, random_state=seed, stratify=ad.obs[label])
    return ad[tr].copy(), ad[te].copy()


def _ablate(pipe, query, label_key, spatial_xy=None):
    rows = []
    for name, graph, smnn in CFGS:
        pipe.graph_refine = graph
        pipe.supervised_mnn = smnn
        out = pipe.annotate(query, copy=True)
        row = _metrics(query.obs[label_key], out.obs["scldl_pred"])
        row["config"] = name
        row["last_correct"] = getattr(pipe.embed_, "last_correct_", None)
        row["last_graph_refine"] = pipe.last_graph_refine_
        row["last_supervised_mnn"] = bool(pipe.last_supervised_mnn_)
        row["mean_entropy"] = float(out.obs["scldl_entropy"].mean())
        row["mean_p1"] = float(out.obs["scldl_p1"].mean())
        if spatial_xy is not None:
            row["spatial"] = spatial_stats(out.obs["scldl_pred"].astype(str), spatial_xy)
        rows.append(row)
    return rows


def run_synthetic():
    rows = []
    types = _blob()
    train, test = _split(types, seed=0)
    pipe = AnnotationPipeline(
        model="scldl",
        n_top_genes=80,
        n_pcs=20,
        n_hidden=64,
        epochs=22,
        batch_size=32,
        verbose=False,
        query_correct="none",
        spatial="off",
        graph_refine="off",
        supervised_mnn="off",
    )
    pipe.fit(train, label_key="cell_type")
    for row in _ablate(pipe, test, "cell_type"):
        row["setting"] = "holdout"
        rows.append(row)

    shifted = test.copy()
    shifted.X = np.asarray(shifted.X) * 0.4 + 4.8
    pipe.query_correct = "mnn"
    for row in _ablate(pipe, shifted, "cell_type"):
        row["setting"] = "global_shift"
        rows.append(row)

    rare = test.copy()
    x = np.asarray(rare.X, dtype=np.float32)
    y = rare.obs["cell_type"].to_numpy()
    rare_mask = y == "type_3"
    x[rare_mask] = x[rare_mask] * 0.55 + 6.5
    x[~rare_mask] = x[~rare_mask] * 0.4 + 4.8
    rare.X = x
    for row in _ablate(pipe, rare, "cell_type"):
        row["setting"] = "rare_type_shift"
        rows.append(row)

    spatial = _spatial_speckle()
    strain, stest = _split(spatial, seed=5)
    spipe = AnnotationPipeline(
        model="scldl",
        n_top_genes=30,
        n_pcs=10,
        n_hidden=48,
        epochs=18,
        batch_size=32,
        verbose=False,
        query_correct="none",
        spatial="off",
        graph_refine="off",
        supervised_mnn="off",
    )
    spipe.fit(strain, label_key="cell_type")
    xy = np.asarray(stest.obsm["spatial"])
    for row in _ablate(spipe, stest, "cell_type", spatial_xy=xy):
        row["setting"] = "spatial_speckle"
        rows.append(row)
    return rows


def run_pbmc():
    import scanpy as sc

    pbmc = sc.datasets.pbmc68k_reduced()
    pbmc.obs["cell_type"] = pbmc.obs["bulk_labels"].astype(str)
    train, test = _split(pbmc, seed=2)
    shifted = test.copy()
    shifted.X = np.asarray(shifted.X) * 0.45 + 3.2
    pipe = AnnotationPipeline(
        model="scldl",
        n_top_genes=500,
        n_pcs=30,
        n_hidden=96,
        epochs=28,
        batch_size=64,
        verbose=False,
        query_correct="mnn",
        spatial="off",
        graph_refine="off",
        supervised_mnn="off",
    )
    pipe.fit(train, label_key="cell_type")
    rows = []
    for row in _ablate(pipe, test, "cell_type"):
        row["setting"] = "pbmc68k_holdout"
        rows.append(row)
    for row in _ablate(pipe, shifted, "cell_type"):
        row["setting"] = "pbmc68k_shift"
        rows.append(row)
    return rows


def run_slideseq(max_per_class=350, max_query=3500, epochs=20):
    import pandas as pd
    import scanpy as sc

    from scLDL.spatial import collapse_dropviz, map_rctd, standardize_gene_names, subsample_balanced, try_spatial_xy

    root = Path(__file__).resolve().parents[1] / "data" / "spatial"
    ref = collapse_dropviz(sc.read_h5ad(root / "AMB_HC.h5ad"))
    ref.obs["batch"] = pd.Index(ref.obs_names.astype(str)).str.replace(r"_.*", "", regex=True)
    query = sc.read_h5ad(root / "AdataMH1.h5ad")
    rctd = pd.read_csv(root / "ssHippo_RCTD.csv", index_col=0)
    shared = query.obs_names.intersection(rctd.index)
    query.obs["rctd_class"] = pd.Series(index=query.obs_names, dtype="object")
    query.obs["rctd_type"] = pd.Series(index=query.obs_names, dtype="object")
    query.obs.loc[shared, "rctd_class"] = rctd.loc[shared, "spot_class"].astype(str).to_numpy()
    query.obs.loc[shared, "rctd_type"] = map_rctd(rctd.loc[shared, "celltype_1"]).to_numpy()
    ref = standardize_gene_names(ref)
    query = standardize_gene_names(query)
    n_genes = np.asarray((query.X > 0).sum(axis=1)).ravel()
    query = query[n_genes >= 80].copy()
    ref = subsample_balanced(ref, "cell_type", max_per_class, seed=0)
    if query.n_obs > max_query:
        rng = np.random.default_rng(1)
        take = np.sort(rng.choice(query.n_obs, size=max_query, replace=False))
        query = query[take].copy()
    pipe = AnnotationPipeline(
        model="scldl",
        n_top_genes=2000,
        n_pcs=40,
        n_neighbors=20,
        n_hidden=192,
        epochs=epochs,
        batch_size=256,
        verbose=True,
        query_correct="auto",
        spatial="auto",
        graph_refine="off",
        supervised_mnn="off",
    )
    batch_key = "batch" if ref.obs["batch"].nunique() > 1 else None
    pipe.fit(ref, label_key="cell_type", batch_key=batch_key)
    rows = []
    xy = np.asarray(query.obsm["spatial"]) if "spatial" in query.obsm else None
    if xy is None and {"xcoord", "ycoord"} <= set(query.obs.columns):
        xy = np.column_stack([query.obs["xcoord"].to_numpy(), query.obs["ycoord"].to_numpy()])
    for name, graph, smnn in CFGS:
        pipe.graph_refine = graph
        pipe.supervised_mnn = smnn
        out = pipe.annotate(query, copy=True)
        pred = out.obs["scldl_pred"].astype(str)
        row = {
            "setting": "slideseqv2_rctd",
            "config": name,
            "n_ref": int(ref.n_obs),
            "n_query": int(query.n_obs),
            "last_correct": getattr(pipe.embed_, "last_correct_", None),
            "last_graph_refine": pipe.last_graph_refine_,
            "last_supervised_mnn": bool(pipe.last_supervised_mnn_),
            "last_spatial": pipe.last_spatial_,
            "mean_entropy": float(out.obs["scldl_entropy"].mean()),
            "mean_p1": float(out.obs["scldl_p1"].mean()),
        }
        if xy is not None:
            row["spatial"] = spatial_stats(pred, xy)
        if "rctd_type" in out.obs:
            mask = out.obs["rctd_type"].notna()
            row["rctd_all"] = _metrics(out.obs.loc[mask, "rctd_type"], pred.loc[mask])
            single = mask & (out.obs["rctd_class"].astype(str) == "singlet")
            if int(single.sum()) > 20:
                row["rctd_singlets"] = _metrics(out.obs.loc[single, "rctd_type"], pred.loc[single])
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-spatial", action="store_true")
    p.add_argument("--skip-pbmc", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = run_synthetic()
    if not args.skip_pbmc:
        try:
            rows.extend(run_pbmc())
        except Exception as exc:
            rows.append({"setting": "pbmc68k", "config": "skipped", "error": str(exc)})
    if not args.skip_spatial:
        try:
            rows.extend(run_slideseq())
        except Exception as exc:
            rows.append({"setting": "slideseqv2_rctd", "config": "skipped", "error": str(exc)})
    (OUT / "results.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
