#!/usr/bin/env python
"""Visium HD figures for the README (query spots + scLDL annotations)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import Normalize

QUERY = Path("data/spatial/VisiumHD_mouse_brain.h5ad")
ANN = Path("artifacts/spatial_refmap/visiumhd/spot_annotations.csv")
OUT = Path("docs/figures")


def _xy(adata) -> np.ndarray:
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise SystemExit("query lacks spatial xy")
    return xy[:, :2]


def _scatter(ax, xy, color, *, s=1.6, cmap=None, norm=None, alpha=0.92):
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=color,
        s=s,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q = sc.read_h5ad(QUERY)
    ann = pd.read_csv(ANN, index_col=0)
    shared = q.obs_names.intersection(ann.index)
    q = q[shared].copy()
    ann = ann.loc[shared]
    xy = _xy(q)

    labels = ann["scldl_pred"].astype(str).to_numpy()
    types = sorted(pd.unique(labels))
    cmap = plt.get_cmap("tab20", max(len(types), 1))
    lut = {t: cmap(i) for i, t in enumerate(types)}
    colors = np.array([lut[t] for t in labels])

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    _scatter(ax, xy, colors, s=1.4)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=lut[t], markersize=5, label=t)
        for t in types
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        frameon=False,
        ncol=1,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    ax.set_title("Visium HD mouse brain · scLDL predicted type", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "visiumhd_pred.png", bbox_inches="tight")
    plt.close(fig)

    entropy = ann["scldl_entropy"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    scatt = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=entropy,
        s=1.4,
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=float(np.quantile(entropy, 0.98))),
        alpha=0.92,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(scatt, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("entropy  −Σ p log p", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Visium HD · label-distribution entropy", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "visiumhd_entropy.png", bbox_inches="tight")
    plt.close(fig)

    p1 = ann["scldl_p1"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    scatt = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=p1,
        s=1.4,
        cmap="viridis",
        norm=Normalize(vmin=float(np.quantile(p1, 0.02)), vmax=1.0),
        alpha=0.92,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(scatt, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("top-1 mass  p₍₁₎", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Visium HD · peak of the label distribution", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "visiumhd_p1.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=160)
    ax.hist(entropy, bins=60, color="#3b3b3b", alpha=0.9)
    ax.axvline(float(np.mean(entropy)), color="#c44e52", lw=1.4, label=f"mean {np.mean(entropy):.3f}")
    ax.set_xlabel("entropy")
    ax.set_ylabel("spots")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Entropy over Visium HD spots")
    fig.tight_layout()
    fig.savefig(OUT / "visiumhd_entropy_hist.png", bbox_inches="tight")
    plt.close(fig)

    print("wrote", list(OUT.glob("visiumhd_*.png")))


if __name__ == "__main__":
    main()
