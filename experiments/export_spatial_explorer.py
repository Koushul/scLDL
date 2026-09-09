#!/usr/bin/env python
"""Pack Visium HD / Slide-seqV2 spots for the here.now explorer."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "spatial_refmap_site"


def _xy_visium(ad):
    xy = np.asarray(ad.obsm["spatial"], dtype=np.float64)[:, :2]
    return xy


def _xy_slideseq(ad):
    return np.column_stack(
        [
            ad.obs["xcoord"].to_numpy(dtype=np.float64),
            ad.obs["ycoord"].to_numpy(dtype=np.float64),
        ]
    )


def _pack(ann: pd.DataFrame, xy: np.ndarray, types: list[str], path: Path) -> dict:
    type_index = {t: i for i, t in enumerate(types)}
    pred = ann["scldl_pred"].astype(str).to_numpy()
    pairs = ann["scldl_pair"].astype(str).to_numpy()
    t2 = np.empty(len(ann), dtype=np.uint8)
    for i, pair in enumerate(pairs):
        parts = pair.split("|", 1)
        second = parts[1] if len(parts) > 1 else parts[0]
        t2[i] = type_index.get(second, type_index[pred[i]])
    pred_i = np.fromiter((type_index[p] for p in pred), dtype=np.uint8, count=len(pred))
    x = xy[:, 0].astype(np.float32)
    y = xy[:, 1].astype(np.float32)
    entropy = ann["scldl_entropy"].to_numpy(dtype=np.float32)
    p1 = ann["scldl_p1"].to_numpy(dtype=np.float32)
    p2 = ann["scldl_p2"].to_numpy(dtype=np.float32)
    n = len(ann)
    buf = bytearray()
    buf += struct.pack("<I", n)
    buf += x.tobytes()
    buf += y.tobytes()
    buf += entropy.tobytes()
    buf += p1.tobytes()
    buf += p2.tobytes()
    buf += pred_i.tobytes()
    buf += t2.tobytes()
    path.write_bytes(buf)
    return {"file": path.name, "n": n, "bytes": len(buf)}


def _align(query, csv_path: Path, xy_fn) -> tuple[pd.DataFrame, np.ndarray]:
    ann = pd.read_csv(csv_path, index_col=0)
    shared = query.obs_names.intersection(ann.index)
    query = query[shared]
    ann = ann.loc[shared]
    return ann, xy_fn(query)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    visium_q = sc.read_h5ad(ROOT / "data/spatial/VisiumHD_mouse_brain.h5ad")
    visium_ann, visium_xy = _align(
        visium_q, ROOT / "artifacts/spatial_refmap/visiumhd/spot_annotations.csv", _xy_visium
    )
    slideseq_q = sc.read_h5ad(ROOT / "data/spatial/AdataMH1.h5ad")
    slideseq_ann, slideseq_xy = _align(
        slideseq_q, ROOT / "artifacts/spatial_refmap/slideseqv2/spot_annotations.csv", _xy_slideseq
    )

    visium_types = sorted(visium_ann["scldl_pred"].astype(str).unique())
    slideseq_types = sorted(slideseq_ann["scldl_pred"].astype(str).unique())
    extra_v = sorted(
        {p.split("|", 1)[1] for p in visium_ann["scldl_pair"].astype(str) if "|" in p} - set(visium_types)
    )
    extra_s = sorted(
        {p.split("|", 1)[1] for p in slideseq_ann["scldl_pair"].astype(str) if "|" in p} - set(slideseq_types)
    )
    visium_types = visium_types + extra_v
    slideseq_types = slideseq_types + extra_s

    meta = {
        "datasets": {
            "visiumhd": {
                "title": "Visium HD mouse brain",
                "types": visium_types,
                **_pack(visium_ann, visium_xy, visium_types, OUT / "visium.bin"),
            },
            "slideseqv2": {
                "title": "Slide-seqV2 hippocampus",
                "types": slideseq_types,
                **_pack(slideseq_ann, slideseq_xy, slideseq_types, OUT / "slideseq.bin"),
            },
        }
    }
    (OUT / "explorer.json").write_text(json.dumps(meta), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
