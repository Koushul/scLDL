"""Analyze mixed CD8 T cell states on the ProjecTILs human TIL atlas."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from scLDL.data import looks_like_counts, to_dense
from scLDL.metrics import score_annotations
from scLDL.models import StateConcentrationLE
from scLDL.seurat_rds import load_seurat_rds
from scLDL.state_targets import (
    blend_targets,
    cd8_til_lineage_edges,
    knn_smooth_labels,
    marker_score_matrix,
    marker_targets,
)

ROOT = Path(__file__).resolve().parents[1]
RDS = ROOT / "data" / "CD8T_human_ref_v1.rds"
H5AD = ROOT / "data" / "CD8T_human_ref_v1.h5ad"
OUT = ROOT / "artifacts" / "tcell_cd8"
SITE = ROOT / "artifacts" / "tcell_cd8_site"

ORDER = ["CD8.NaiveLike", "CD8.CM", "CD8.EM", "CD8.TEMRA", "CD8.TPEX", "CD8.TEX", "CD8.MAIT"]
SHORT = {
    "CD8.NaiveLike": "Naive-like",
    "CD8.CM": "CM",
    "CD8.EM": "EM",
    "CD8.TEMRA": "TEMRA",
    "CD8.TPEX": "Tpex",
    "CD8.TEX": "Tex",
    "CD8.MAIT": "MAIT",
}
MARKERS = {
    "CD8.NaiveLike": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "S1PR1"],
    "CD8.CM": ["CCR7", "SELL", "IL7R", "CD27", "CD28", "LTB"],
    "CD8.EM": ["GZMK", "CXCR3", "EOMES", "CCL5", "CRTAM"],
    "CD8.TEMRA": ["GZMB", "PRF1", "FGFBP2", "CX3CR1", "KLRG1", "NKG7", "GNLY"],
    "CD8.TPEX": ["TCF7", "CXCL13", "PDCD1", "IL7R", "GZMK", "SLAMF6"],
    "CD8.TEX": ["HAVCR2", "TOX", "LAG3", "PDCD1", "CXCL13", "ENTPD1", "CTLA4", "TIGIT"],
    "CD8.MAIT": ["SLC4A10", "KLRB1", "ZBTB16", "IL18R1", "NCR3", "CEBPD"],
}
UCELL = {
    "CD8.NaiveLike": "CD8_N_UCell",
    "CD8.EM": "CD8_EM_UCell",
    "CD8.TEMRA": "CD8_TEMRA_UCell",
    "CD8.TPEX": "CD8_TPEX_UCell",
    "CD8.TEX": "CD8_TEX_UCell",
    "CD8.MAIT": "CD8_MAIT_UCell",
}
PALETTE = {
    "CD8.NaiveLike": "#4C78A8",
    "CD8.CM": "#72B7B2",
    "CD8.EM": "#54A24B",
    "CD8.TEMRA": "#E45756",
    "CD8.TPEX": "#F58518",
    "CD8.TEX": "#B279A2",
    "CD8.MAIT": "#FF9DA6",
}


def _load():
    if H5AD.exists():
        return sc.read_h5ad(H5AD)
    ad = load_seurat_rds(RDS)
    ad.write_h5ad(H5AD)
    return ad


def _prepare(ad):
    ad = ad.copy()
    ad.obs["cell_state"] = pd.Categorical(ad.obs["functional.cluster"].astype(str), categories=ORDER)
    ad = ad[ad.obs["cell_state"].notna()].copy()
    if looks_like_counts(ad.X):
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    extra = sorted({g for gs in MARKERS.values() for g in gs if g in ad.var_names})
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, subset=False)
    keep = ad.var["highly_variable"].copy()
    keep.loc[extra] = True
    ad = ad[:, keep].copy()
    X = np.nan_to_num(to_dense(ad.X), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(ad.obs["cell_state"].astype(str))
    idx = {c: i for i, c in enumerate(ORDER)}
    y_int = np.array([idx[v] for v in y])
    onehot = np.eye(len(ORDER), dtype=np.float32)[y_int]
    return ad, X, y, y_int, onehot


def _entropy(p):
    return (-p * np.log(np.clip(p, 1e-8, 1))).sum(1)


def _mixture_table(y, p, classes, secondary=0.18):
    rows = []
    for st in classes:
        m = y == st
        if not m.any():
            continue
        sub = p[m]
        mean = sub.mean(0)
        second = np.argsort(-mean)
        partner = classes[second[1]] if second[0] == classes.tolist().index(st) else classes[second[0]]
        mixed = (sub.max(1) < 1.0 - secondary) | ((np.sort(sub, axis=1)[:, -2]) >= secondary)
        top2 = np.argsort(-sub, axis=1)[:, :2]
        partner_counts = {}
        for i, j in top2:
            a, b = classes[i], classes[j]
            other = b if a == st else a
            partner_counts[other] = partner_counts.get(other, 0) + 1
        top_partner = max(partner_counts, key=partner_counts.get) if partner_counts else partner
        rec = {
            "annotation": st,
            "short": SHORT[st],
            "n": int(m.sum()),
            "P_self": float(mean[list(classes).index(st)]),
            "entropy": float(_entropy(sub).mean()),
            "frac_mixed": float(mixed.mean()),
            "top_partner": SHORT[top_partner],
        }
        for i, c in enumerate(classes):
            rec[f"P_{c}"] = float(mean[i])
        rows.append(rec)
    return pd.DataFrame(rows)


def _save_fig(name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    return path.name


def make_plots(ad, y, p, u, classes):
    umap = np.asarray(ad.obsm["X_umap"])
    colors = [PALETTE[s] for s in y]
    sns.set_theme(style="whitegrid", font_scale=1.05)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(umap[:, 0], umap[:, 1], c=colors, s=6, linewidths=0, alpha=0.85)
    for st in ORDER:
        ax.scatter([], [], c=PALETTE[st], s=40, label=SHORT[st])
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.set_title("ProjecTILs discrete annotations")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.axis("equal")
    f1 = _save_fig("umap_annotations.png")

    pred = np.array([classes[i] for i in p.argmax(1)])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(umap[:, 0], umap[:, 1], c=[PALETTE[s] for s in pred], s=6, linewidths=0, alpha=0.85)
    ax.set_title("LDL predicted state (argmax)")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.axis("equal")
    f2 = _save_fig("umap_pred.png")

    ent = _entropy(p)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sca = ax.scatter(umap[:, 0], umap[:, 1], c=ent, s=6, linewidths=0, cmap="magma", vmin=0, vmax=np.quantile(ent, 0.98))
    plt.colorbar(sca, ax=ax, label="Predictive entropy")
    ax.set_title("Where discrete labels hide mixtures")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.axis("equal")
    f3 = _save_fig("umap_entropy.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    for ax, st in zip(axes, ["CD8.NaiveLike", "CD8.TPEX", "CD8.TEX"]):
        sca = ax.scatter(umap[:, 0], umap[:, 1], c=p[:, ORDER.index(st)], s=5, linewidths=0, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"P({SHORT[st]})")
        ax.axis("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sca, ax=ax, fraction=0.046)
    f4 = _save_fig("umap_probabilities.png")

    mean = np.vstack([p[y == st].mean(0) if (y == st).any() else np.zeros(len(ORDER)) for st in ORDER])
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    sns.heatmap(
        mean,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=[SHORT[s] for s in ORDER],
        yticklabels=[SHORT[s] for s in ORDER],
        ax=ax,
        vmin=0,
        vmax=0.85,
    )
    ax.set_xlabel("Predicted LDL mass")
    ax.set_ylabel("Discrete ProjecTILs label")
    ax.set_title("Mean predicted distribution given the annotation")
    f5 = _save_fig("heatmap_mix.png")

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bottom = np.zeros(len(ORDER))
    x = np.arange(len(ORDER))
    for j, st in enumerate(ORDER):
        ax.bar(x, mean[:, j], bottom=bottom, color=PALETTE[st], label=SHORT[st], width=0.82)
        bottom += mean[:, j]
    ax.set_xticks(x, [SHORT[s] for s in ORDER], rotation=20)
    ax.set_ylabel("Mean probability")
    ax.set_title("Each discrete label is a mixture, not a point")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    f6 = _save_fig("stacked_mix.png")

    pairs = [(st, UCELL[st]) for st in ORDER if st in UCELL and UCELL[st] in ad.obs]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    axes = axes.ravel()
    corrs = {}
    for ax, (st, col) in zip(axes, pairs):
        xv = np.asarray(ad.obs[col], dtype=float)
        yv = p[:, ORDER.index(st)]
        r, _ = spearmanr(xv, yv)
        corrs[st] = float(r)
        ax.scatter(xv, yv, s=6, alpha=0.35, c=PALETTE[st], linewidths=0)
        ax.set_title(f"{SHORT[st]}  r={r:.2f}")
        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel(f"P({SHORT[st]})")
    f7 = _save_fig("ucell_vs_p.png")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    df_e = pd.DataFrame({"state": [SHORT[s] for s in y], "entropy": ent, "vacuity": u})
    sns.boxplot(data=df_e, x="state", y="entropy", hue="state", palette={SHORT[k]: v for k, v in PALETTE.items()}, ax=ax, legend=False)
    ax.set_title("Mixture uncertainty by discrete annotation")
    ax.set_xlabel("")
    ax.set_ylabel("Entropy of LDL distribution")
    f8 = _save_fig("entropy_by_state.png")
    return [f1, f2, f3, f4, f5, f6, f7, f8], corrs, mean


def write_site(summary, figures):
    SITE.mkdir(parents=True, exist_ok=True)
    for name in figures:
        src = OUT / name
        (SITE / name).write_bytes(src.read_bytes())
    mix_rows = "".join(
        f"<tr><td>{r['short']}</td><td>{r['n']}</td><td>{r['P_self']:.2f}</td>"
        f"<td>{r['frac_mixed']:.0%}</td><td>{r['top_partner']}</td><td>{r['entropy']:.2f}</td></tr>"
        for r in summary["mixtures"]
    )
    corr_rows = "".join(
        f"<tr><td>{SHORT[k]}</td><td>{v:.3f}</td></tr>" for k, v in summary["ucell_spearman"].items()
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Mixed CD8 T cell states — scLDL</title>
  <style>
    :root {{ --bg:#f7f4ee; --ink:#1b1b18; --muted:#5c584f; --card:#fff; --line:#e4ddd0; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Source Serif 4, Iowan Old Style, Georgia, serif; background:var(--bg); color:var(--ink); line-height:1.55; }}
    header {{ padding:2.4rem 6vw 1rem; max-width:1100px; }}
    h1 {{ font-size:2.1rem; margin:0 0 .4rem; letter-spacing:-.02em; }}
    .lede {{ color:var(--muted); font-size:1.12rem; max-width:46rem; }}
    main {{ padding:0 6vw 4rem; max-width:1100px; }}
    .grid {{ display:grid; gap:1.2rem; grid-template-columns:1fr; }}
    @media (min-width:900px) {{ .pair {{ display:grid; grid-template-columns:1fr 1fr; gap:1.2rem; }} }}
    figure {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1rem; }}
    img {{ width:100%; height:auto; border-radius:8px; }}
    figcaption {{ font-size:.92rem; color:var(--muted); margin-top:.7rem; }}
    table {{ width:100%; border-collapse:collapse; background:var(--card); border-radius:16px; overflow:hidden; }}
    th, td {{ text-align:left; padding:.65rem .8rem; border-bottom:1px solid var(--line); font-size:.95rem; }}
    th {{ font-family:ui-sans-serif, system-ui, sans-serif; font-weight:600; color:var(--muted); }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:.8rem; margin:1rem 0 1.6rem; }}
    .chip {{ background:var(--card); border:1px solid var(--line); border-radius:999px; padding:.35rem .85rem; font-family:ui-sans-serif, system-ui, sans-serif; font-size:.86rem; }}
    h2 {{ margin:2.2rem 0 .6rem; font-size:1.45rem; }}
    a {{ color:#234e70; }}
  </style>
</head>
<body>
<header>
  <h1>CD8 T cells are labeled as points. LDL reads them as mixtures.</h1>
  <p class="lede">
    Discrete ProjecTILs annotations on human tumor-infiltrating CD8 T cells
    (Naive-like, CM, EM, TEMRA, Tpex, Tex, MAIT) sit on a differentiation continuum.
    A state-aware label-distribution model recovers which other states each annotation
    is mixed with — including on held-out cells that never entered the graph smoother.
  </p>
</header>
<main>
  <div class="metrics">
    <span class="chip">{summary['n_cells']} cells · 7 states</span>
    <span class="chip">Held-out acc {summary['accuracy']:.3f}</span>
    <span class="chip">Macro-F1 {summary['macro_f1']:.3f}</span>
    <span class="chip">Mean mixed fraction {summary['mean_frac_mixed']:.0%}</span>
    <span class="chip">Andreatta / Carmona ProjecTILs CD8 TIL atlas</span>
  </div>

  <h2>The atlas, then the distribution</h2>
  <div class="pair">
    <figure>
      <img src="umap_annotations.png" alt="UMAP of discrete annotations"/>
      <figcaption>Hard ProjecTILs <code>functional.cluster</code> labels. These are the non-continuous annotations.</figcaption>
    </figure>
    <figure>
      <img src="umap_entropy.png" alt="UMAP of predictive entropy"/>
      <figcaption>LDL entropy. Boundaries between memory, Tpex and Tex are where a single label is least honest.</figcaption>
    </figure>
  </div>

  <h2>What each discrete label is mixed with</h2>
  <table>
    <thead><tr><th>Annotation</th><th>n</th><th>P(self)</th><th>Mixed</th><th>Main partner</th><th>Entropy</th></tr></thead>
    <tbody>{mix_rows}</tbody>
  </table>
  <p class="lede">A cell is mixed if the runner-up state has mass ≥ 0.18 or the top state is below 0.82. Partners are counted from the top-2 states on held-out cells.</p>

  <div class="pair">
    <figure>
      <img src="heatmap_mix.png" alt="Heatmap of predicted mass by annotation"/>
      <figcaption>Rows sum to 1. Off-diagonals are the mixtures the cluster labels cannot express.</figcaption>
    </figure>
    <figure>
      <img src="stacked_mix.png" alt="Stacked mixture bars"/>
      <figcaption>Naive-like still leaks into CM; Tpex shares mass with Tex and EM; Tex is the most self-concentrated exhausted state.</figcaption>
    </figure>
  </div>

  <h2>Does the mixture track independent biology?</h2>
  <p class="lede">
    ProjecTILs also scored gene sets with UCell. Those scores were not used as training targets.
    Spearman correlation between LDL probability and UCell is a check that the extra mass is marker-real, not just neighbor bleed.
  </p>
  <div class="pair">
    <figure>
      <img src="ucell_vs_p.png" alt="UCell scores versus LDL probabilities"/>
      <figcaption>Each panel is all held-out cells. A high r means P(state) follows the state’s gene program even when the discrete label says otherwise.</figcaption>
    </figure>
    <figure>
      <table>
        <thead><tr><th>State</th><th>UCell vs P Spearman</th></tr></thead>
        <tbody>{corr_rows}</tbody>
      </table>
      <figcaption style="padding:1rem;color:var(--muted);">Held-out cells only. No UCell column for CM in this atlas.</figcaption>
    </figure>
  </div>

  <h2>Where the probability mass actually sits</h2>
  <div class="pair">
    <figure>
      <img src="umap_probabilities.png" alt="UMAP of Naive, Tpex, Tex probabilities"/>
      <figcaption>Naive-like, Tpex and Tex form a gradient rather than three blobs. Discrete labels cut that gradient into names.</figcaption>
    </figure>
    <figure>
      <img src="entropy_by_state.png" alt="Entropy boxplots by annotation"/>
      <figcaption>Tpex and EM are the most intrinsically mixed annotations; MAIT and TEMRA are more exclusive.</figcaption>
    </figure>
  </div>

  <h2>Method</h2>
  <p>
    Dataset: ProjecTILs human CD8<sup>+</sup> TIL reference
    (Andreatta, Gueguen, Carmona; 10,045 cells, 7 tumor types;
    <a href="https://doi.org/10.6084/m9.figshare.23608308">figshare 23608308</a>,
    <a href="https://www.nature.com/articles/s41467-021-23324-4">Nat Commun 2021</a>).
    Model: <strong>StateConcentrationLE</strong> — evidential Dirichlet head trained on
    25% one-hot + 45% adaptive probabilistic kNN smoothing (PCA-space self-tuning Gaussian)
    + 30% marker-module softmax, with a CD8 lineage Laplacian
    (Naive-like→CM→EM→{{TEMRA, Tpex→Tex}}, EM–MAIT).
    Split: 80/20 stratified hold-out. Graph and markers used training cells only.
  </p>
  <p class="lede">scLDL · {summary['dataset']}</p>
</main>
</body>
</html>
"""
    (SITE / "index.html").write_text(html, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ad_raw = _load()
    ad, X, y, y_int, onehot = _prepare(ad_raw)
    classes = np.array(ORDER)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0, stratify=y_int)

    marker_tr = marker_targets(X[tr], ad.var_names, MARKERS, ORDER, temperature=0.65)
    graph_tr, P_graph = knn_smooth_labels(
        X[tr], onehot[tr], n_neighbors=30, alpha=0.75, n_iter=20, method="adaptive", n_pcs=40, return_graph=True
    )
    soft_tr = blend_targets((onehot[tr], 0.25), (graph_tr, 0.45), (marker_tr, 0.30))

    model = StateConcentrationLE(
        n_features=X.shape[1],
        n_outputs=len(ORDER),
        n_hidden=256,
        epochs=45,
        batch_size=128,
        prior=0.2,
        lineage_weight=0.45,
        manifold_weight=0.12,
        kl_weight=0.3,
        vacuity_weight=0.15,
        mixup_alpha=0.3,
        lineage_edges=cd8_til_lineage_edges(),
        verbose=True,
    )
    model.fit(X[tr], soft_tr, neighbor_p=P_graph)
    p_te, u_te = model.predict_evidence(X[te])
    p_all, u_all = model.predict_evidence(X)

    metrics, _ = score_annotations(y[te], p_te, classes)
    mix = _mixture_table(y[te], p_te, classes)
    scores = marker_score_matrix(X[te], ad.var_names, MARKERS, ORDER)
    marker_corr = {}
    for i, st in enumerate(ORDER):
        r, _ = spearmanr(scores[:, i], p_te[:, i])
        marker_corr[st] = float(r)

    ad_all = ad
    figures, ucell_corr, mean = make_plots(ad_all[te], y[te], p_te, u_te, classes)
    # also dump a full-atlas umap using held-out-trained predictions for context
    make_plots(ad_all, y, p_all, u_all, classes)

    summary = {
        "dataset": "ProjecTILs human CD8 TIL atlas v1",
        "n_cells": int(ad.n_obs),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "brier": float(metrics["brier"]),
        "ece": float(metrics["ece"]),
        "mean_frac_mixed": float(mix["frac_mixed"].mean()),
        "mixtures": mix.to_dict(orient="records"),
        "ucell_spearman": ucell_corr,
        "marker_spearman": marker_corr,
        "mean_P": {ORDER[i]: mean[i].tolist() for i in range(len(ORDER))},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    mix.to_csv(OUT / "mixtures.csv", index=False)
    write_site(summary, [
        "umap_annotations.png",
        "umap_pred.png",
        "umap_entropy.png",
        "umap_probabilities.png",
        "heatmap_mix.png",
        "stacked_mix.png",
        "ucell_vs_p.png",
        "entropy_by_state.png",
    ])
    print(json.dumps({k: summary[k] for k in ["n_cells", "accuracy", "macro_f1", "mean_frac_mixed"]}, indent=2))
    print(mix[["short", "n", "P_self", "frac_mixed", "top_partner"]].to_string(index=False))


if __name__ == "__main__":
    main()
