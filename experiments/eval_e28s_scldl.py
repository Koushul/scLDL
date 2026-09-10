#!/usr/bin/env python
"""Fit scLDL on all E28S cells and annotate each well's label distribution."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import scanpy as sc

from scLDL.interpret import entropy as scldl_entropy
from scLDL.pipeline import AnnotationPipeline
from scLDL.spatial import standardize_gene_names

ROOT = Path(__file__).resolve().parents[1]
H5 = Path("/ix1/ylee/kor11/tools/hypoxia_fronts/data/E28S.h5ad")
OUT = ROOT / "artifacts" / "e28s_scldl"
SITE = ROOT / "artifacts" / "slidetags_tonsil_site"
LABEL = "cell_type"
SLIDES = ["edge", "core"]


def _xy(ad):
    if "spatial" in ad.obsm:
        return np.asarray(ad.obsm["spatial"], dtype=np.float64)[:, :2]
    return np.column_stack(
        [
            ad.obs["col"].to_numpy(dtype=np.float64) * 55.0,
            ad.obs["row"].to_numpy(dtype=np.float64) * 55.0,
        ]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    adata = standardize_gene_names(sc.read_h5ad(H5))
    adata.obs[LABEL] = adata.obs[LABEL].astype(str)
    adata = adata[adata.obs[LABEL].notna() & (adata.obs[LABEL] != "nan")].copy()
    adata.obs["slide"] = adata.obs["slide"].astype(str)
    adata.obsm["spatial"] = _xy(adata)

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
    pipe.fit(adata, label_key=LABEL)
    parts = []
    for slide in SLIDES:
        sub = adata[adata.obs["slide"] == slide].copy()
        if sub.n_obs == 0:
            continue
        parts.append(pipe.annotate(sub, copy=True))
    q = parts[0].concatenate(parts[1], batch_key="_concat", index_unique="-") if len(parts) > 1 else parts[0]

    classes = np.asarray(pipe.classes_).astype(str)
    p = np.asarray(q.obsm["X_scldl"], dtype=np.float32)
    y = q.obs[LABEL].astype(str).to_numpy()
    pred = q.obs["scldl_pred"].astype(str).to_numpy()
    vac = q.obs["scldl_uncertainty"].to_numpy(dtype=np.float32)
    H = np.asarray(scldl_entropy(p), dtype=np.float32)
    p1 = p.max(axis=1)
    slides = q.obs["slide"].astype(str).to_numpy()
    xy = np.asarray(q.obsm["spatial"], dtype=np.float32)[:, :2]
    agree = pred == y
    n, k = p.shape
    lut = {c: i for i, c in enumerate(classes)}
    pub_i = np.fromiter((lut[v] for v in y), dtype=np.uint8, count=n)
    pred_i = np.fromiter((lut[v] for v in pred), dtype=np.uint8, count=n)
    slide_i = np.fromiter((SLIDES.index(s) for s in slides), dtype=np.uint8, count=n)

    per_type = {}
    for c in classes:
        m = y == c
        per_type[str(c)] = {
            "n": int(m.sum()),
            "accuracy": float(np.mean(agree[m])) if m.any() else 0.0,
            "mean_entropy": float(np.mean(H[m])) if m.any() else 0.0,
            "mean_vacuity": float(np.mean(vac[m])) if m.any() else 0.0,
            "mean_p1": float(np.mean(p1[m])) if m.any() else 0.0,
        }
    metrics = {
        "dataset": "E28S MC38 spatial (full edge+core)",
        "source": str(H5),
        "label_key": LABEL,
        "n_cells": int(n),
        "n_types": int(k),
        "types": list(map(str, classes)),
        "slides": {s: int((slides == s).sum()) for s in SLIDES},
        "train": "all cells",
        "spatial": "off",
        "accuracy": float(np.mean(agree)),
        "mean_entropy": float(np.mean(H)),
        "mean_vacuity": float(np.mean(vac)),
        "mean_p1": float(np.mean(p1)),
        "per_type": per_type,
        "per_slide": {
            s: {
                "n": int((slides == s).sum()),
                "accuracy": float(np.mean(agree[slides == s])),
                "mean_entropy": float(np.mean(H[slides == s])),
            }
            for s in SLIDES
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    buf = bytearray()
    buf += struct.pack("<II", n, k)
    buf += xy[:, 0].tobytes()
    buf += xy[:, 1].tobytes()
    buf += H.tobytes()
    buf += vac.tobytes()
    buf += p1.astype(np.float32).tobytes()
    buf += pub_i.tobytes()
    buf += pred_i.tobytes()
    buf += slide_i.tobytes()
    buf += p.tobytes()
    (SITE / "e28s.bin").write_bytes(buf)
    old = SITE / "tonsil.bin"
    if old.exists():
        old.unlink()
    meta = {
        "title": "E28S · cell_type",
        "file": "e28s.bin",
        "n": n,
        "k": k,
        "types": list(map(str, classes)),
        "slides": SLIDES,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "mean_entropy": metrics["mean_entropy"],
            "mean_vacuity": metrics["mean_vacuity"],
            "mean_p1": metrics["mean_p1"],
            "n_edge": metrics["slides"]["edge"],
            "n_core": metrics["slides"]["core"],
        },
        "per_type": per_type,
        "per_slide": metrics["per_slide"],
    }
    (SITE / "explorer.json").write_text(json.dumps(meta), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
