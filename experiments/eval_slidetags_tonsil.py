#!/usr/bin/env python
"""Out-of-fold scLDL on SpaceTravLR Slide-tags tonsil (cell_type_2)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import scanpy as sc
from sklearn.model_selection import StratifiedKFold

from scLDL.pipeline import AnnotationPipeline
from scLDL.spatial import standardize_gene_names, try_spatial_xy

ROOT = Path(__file__).resolve().parents[1]
H5AD = Path("/ix1/ylee/kor11/tools/SpaceTravLR/data/snrna_germinal_center.h5ad")
OUT = ROOT / "artifacts" / "slidetags_tonsil"
SITE = ROOT / "artifacts" / "slidetags_tonsil_site"
LABEL = "cell_type_2"


def _map_proba(proba, src_classes, dst_classes):
    out = np.zeros((len(proba), len(dst_classes)), dtype=np.float32)
    idx = {str(c): i for i, c in enumerate(dst_classes)}
    for j, c in enumerate(src_classes):
        k = idx.get(str(c))
        if k is not None:
            out[:, k] = proba[:, j]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    adata = standardize_gene_names(sc.read_h5ad(H5AD))
    adata = adata[adata.obs[LABEL].notna()].copy()
    adata.obs[LABEL] = adata.obs[LABEL].astype(str)
    keep = adata.obs[LABEL] != "nan"
    adata = adata[keep].copy()
    y = adata.obs[LABEL].to_numpy()
    classes = np.array(sorted(np.unique(y)))
    xy = try_spatial_xy(adata)
    if xy is None:
        raise SystemExit("query lacks spatial coordinates")

    n, k = adata.n_obs, len(classes)
    p_all = np.zeros((n, k), dtype=np.float32)
    vac_all = np.zeros(n, dtype=np.float32)
    pred_all = np.empty(n, dtype=object)
    folds = np.full(n, -1, dtype=np.int8)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (train, test) in enumerate(skf.split(np.arange(n), y)):
        pipe = AnnotationPipeline(
            model="scldl",
            task="type",
            n_top_genes=min(2000, adata.n_vars),
            n_pcs=40,
            n_neighbors=20,
            n_hidden=192,
            epochs=30,
            batch_size=128,
            verbose=True,
            query_correct="none",
            spatial="off",
            graph_refine="off",
            supervised_mnn="off",
        )
        pipe.fit(adata[train].copy(), label_key=LABEL)
        q = pipe.annotate(adata[test].copy(), copy=True)
        p_all[test] = _map_proba(np.asarray(q.obsm["X_scldl"]), pipe.classes_, classes)
        vac_all[test] = q.obs["scldl_uncertainty"].to_numpy(dtype=np.float32)
        pred_all[test] = q.obs["scldl_pred"].astype(str).to_numpy()
        folds[test] = fold
        print(f"fold {fold}: n_test={len(test)} acc={np.mean(pred_all[test] == y[test]):.3f}")

    entropy = (-np.clip(p_all, 1e-8, 1.0) * np.log(np.clip(p_all, 1e-8, 1.0))).sum(axis=1)
    p1 = p_all.max(axis=1)
    agree = pred_all == y
    metrics = {
        "dataset": "SpaceTravLR Slide-tags human tonsil (germinal center)",
        "source": str(H5AD),
        "label_key": LABEL,
        "n_cells": int(n),
        "n_types": int(k),
        "types": list(map(str, classes)),
        "counts": {str(c): int((y == c).sum()) for c in classes},
        "oof_accuracy": float(np.mean(agree)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_vacuity": float(np.mean(vac_all)),
        "mean_p1": float(np.mean(p1)),
        "disagree_frac": float(1.0 - np.mean(agree)),
        "per_type": {},
    }
    for c in classes:
        m = y == c
        metrics["per_type"][str(c)] = {
            "n": int(m.sum()),
            "accuracy": float(np.mean(agree[m])),
            "mean_entropy": float(np.mean(entropy[m])),
            "mean_vacuity": float(np.mean(vac_all[m])),
            "mean_p1": float(np.mean(p1[m])),
        }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pub_i = np.fromiter(({str(c): i for i, c in enumerate(classes)}[v] for v in y), dtype=np.uint8, count=n)
    pred_i = np.fromiter(({str(c): i for i, c in enumerate(classes)}[v] for v in pred_all), dtype=np.uint8, count=n)
    x = xy[:, 0].astype(np.float32)
    yxy = xy[:, 1].astype(np.float32)
    buf = bytearray()
    buf += struct.pack("<II", n, k)
    buf += x.tobytes()
    buf += yxy.tobytes()
    buf += entropy.astype(np.float32).tobytes()
    buf += vac_all.tobytes()
    buf += p1.astype(np.float32).tobytes()
    buf += pub_i.tobytes()
    buf += pred_i.tobytes()
    buf += p_all.tobytes()
    (SITE / "tonsil.bin").write_bytes(buf)
    meta = {
        "title": "Slide-tags tonsil · cell_type_2",
        "file": "tonsil.bin",
        "n": n,
        "k": k,
        "types": list(map(str, classes)),
        "metrics": {
            "oof_accuracy": metrics["oof_accuracy"],
            "mean_entropy": metrics["mean_entropy"],
            "mean_vacuity": metrics["mean_vacuity"],
            "mean_p1": metrics["mean_p1"],
        },
        "per_type": metrics["per_type"],
    }
    (SITE / "explorer.json").write_text(json.dumps(meta), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
