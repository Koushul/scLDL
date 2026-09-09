"""Annotate Visium HD and Slide-seqV2 with public scRNA references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from scLDL.metrics import classification_metrics
from scLDL.pipeline import AnnotationPipeline
from scLDL.spatial import (
    collapse_dropviz,
    map_rctd,
    marker_subset,
    spatial_coords,
    standardize_gene_names,
    subsample_balanced,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spatial"
OUT = ROOT / "artifacts" / "spatial_refmap"


def _filter_spots(ad, min_genes=80):
    if "n_genes" in ad.obs:
        n = ad.obs["n_genes"].to_numpy()
    else:
        x = ad.X
        n = np.asarray((x > 0).sum(axis=1)).ravel()
    return ad[n >= min_genes].copy()


def _load_visium():
    ref = sc.read_h5ad(DATA / "scRNA_mouse_brain.h5ad")
    ref.obs["cell_type"] = ref.obs["celltype"].astype(str)
    ref.obs["batch"] = ref.obs["orig.ident"].astype(str)
    query = sc.read_h5ad(DATA / "VisiumHD_mouse_brain.h5ad")
    ref = standardize_gene_names(ref)
    query = standardize_gene_names(query)
    query = _filter_spots(query, min_genes=50)
    return ref, query, "visiumhd"


def _load_slideseq():
    ref = collapse_dropviz(sc.read_h5ad(DATA / "AMB_HC.h5ad"))
    ref.obs["batch"] = pd.Index(ref.obs_names.astype(str)).str.replace(r"_.*", "", regex=True)
    query = sc.read_h5ad(DATA / "AdataMH1.h5ad")
    rctd_path = DATA / "ssHippo_RCTD.csv"
    if rctd_path.exists():
        rctd = pd.read_csv(rctd_path, index_col=0)
        shared = query.obs_names.intersection(rctd.index)
        query.obs["rctd_class"] = np.nan
        query.obs["rctd_type"] = np.nan
        query.obs.loc[shared, "rctd_class"] = rctd.loc[shared, "spot_class"].astype(str)
        query.obs.loc[shared, "rctd_type"] = map_rctd(rctd.loc[shared, "celltype_1"])
    ref = standardize_gene_names(ref)
    query = standardize_gene_names(query)
    query = _filter_spots(query, min_genes=80)
    return ref, query, "slideseqv2"


def _palette(labels):
    cmap = plt.get_cmap("tab20")
    uniq = list(dict.fromkeys(labels))
    return {lab: cmap(i / max(len(uniq), 1)) for i, lab in enumerate(uniq)}


def _spatial_cat(ad, color, path, title, s=2.4):
    x, y = spatial_coords(ad)
    labs = ad.obs[color].astype(str).to_numpy()
    pal = _palette(list(ad.obs[color].astype(str).value_counts().index))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    order = np.argsort([pal[v][0] for v in labs])
    ax.scatter(x[order], y[order], c=[pal[v] for v in labs[order]], s=s, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[k], markersize=6, label=k)
        for k in list(ad.obs[color].astype(str).value_counts().index)[:16]
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _spatial_cont(ad, values, path, title, s=2.4, cmap="magma"):
    x, y = spatial_coords(ad)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    sca = ax.scatter(x, y, c=values, s=s, linewidths=0, cmap=cmap, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _agreement(y_true, y_pred):
    mask = pd.notna(y_true) & pd.notna(y_pred)
    yt = np.asarray(y_true)[mask].astype(str)
    yp = np.asarray(y_pred)[mask].astype(str)
    shared = sorted(set(yt) & set(yp))
    if len(yt) == 0:
        return {"n": 0}
    acc = float(accuracy_score(yt, yp))
    out = {
        "n": int(len(yt)),
        "accuracy": acc,
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
    }
    if shared:
        in_shared = np.isin(yt, shared)
        out["accuracy_shared_types"] = float(accuracy_score(yt[in_shared], yp[in_shared]))
        out["n_shared_types"] = int(in_shared.sum())
    return out


def _marker_check(ad, classes):
    genes = pd.Index(ad.var_names.astype(str))
    pred = ad.obs["scldl_pred"].astype(str).to_numpy()
    rows = []
    for lab in classes:
        lab = str(lab)
        markers = marker_subset([lab], genes).get(lab, [])
        if not markers:
            continue
        present = [g for g in markers if g in genes]
        if not present:
            continue
        sub = ad[:, present]
        x = sub.X
        if hasattr(x, "toarray"):
            x = np.asarray(x.toarray())
        else:
            x = np.asarray(x)
        score = x.mean(axis=1)
        in_lab = pred == lab
        if in_lab.sum() < 20 or (~in_lab).sum() < 20:
            continue
        rows.append(
            {
                "type": lab,
                "markers": ",".join(present),
                "mean_in_pred": float(score[in_lab].mean()),
                "mean_out_pred": float(score[~in_lab].mean()),
            }
        )
    return rows


def _run_dataset(name, ref, query, args):
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_all = ref
    ref = subsample_balanced(ref_all, "cell_type", args.max_per_class, seed=0)
    if query.n_obs > args.max_query:
        rng = np.random.default_rng(1)
        take = np.sort(rng.choice(query.n_obs, size=args.max_query, replace=False))
        query = query[take].copy()

    markers = marker_subset(ref.obs["cell_type"].unique(), ref.var_names)
    pipe = AnnotationPipeline(
        model="interpretable",
        task="type",
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        n_neighbors=20,
        n_hidden=192,
        epochs=args.epochs,
        batch_size=256,
        verbose=True,
        markers=markers or None,
        query_correct="auto",
    )
    batch_key = "batch" if "batch" in ref.obs and ref.obs["batch"].nunique() > 1 else None
    pipe.fit(ref, label_key="cell_type", batch_key=batch_key)
    query = pipe.annotate(query, copy=True)
    correct = getattr(pipe.embed_, "last_correct_", None)

    pred_counts = query.obs["scldl_pred"].value_counts().to_dict()
    metrics = {
        "dataset": name,
        "n_ref": int(ref.n_obs),
        "n_query": int(query.n_obs),
        "n_types": int(len(pipe.classes_)),
        "types": list(map(str, pipe.classes_)),
        "query_correct": correct,
        "pred_counts": {str(k): int(v) for k, v in pred_counts.items()},
        "mean_entropy": float(query.obs["scldl_entropy"].mean()),
        "mean_dissonance": float(query.obs["scldl_dissonance"].mean()),
        "mean_p1": float(query.obs["scldl_p1"].mean()),
        "marker_check": _marker_check(query, pipe.classes_),
    }

    if "rctd_type" in query.obs:
        singlets = query.obs["rctd_class"].astype(str) == "singlet"
        metrics["rctd_all"] = _agreement(query.obs["rctd_type"], query.obs["scldl_pred"])
        metrics["rctd_singlets"] = _agreement(
            query.obs.loc[singlets, "rctd_type"], query.obs.loc[singlets, "scldl_pred"]
        )
        sub = query[singlets & query.obs["rctd_type"].notna()].copy()
        if sub.n_obs > 50:
            _spatial_cat(sub, "rctd_type", out_dir / "spatial_rctd_singlets.png", f"{name} RCTD singlets")

    held = ref_all.obs_names.difference(ref.obs_names)
    if len(held) >= 80:
        hold = subsample_balanced(ref_all[held].copy(), "cell_type", max(30, args.max_per_class // 6), seed=3)
        metrics["ref_holdout"] = classification_metrics(
            hold.obs["cell_type"].astype(str),
            pipe.annotate(hold, copy=True).obs["scldl_pred"].astype(str),
        )

    _spatial_cat(query, "scldl_pred", out_dir / "spatial_pred.png", f"{name} scLDL ({correct})")
    _spatial_cont(query, query.obs["scldl_entropy"], out_dir / "spatial_entropy.png", f"{name} entropy")
    _spatial_cont(query, query.obs["scldl_dissonance"], out_dir / "spatial_dissonance.png", f"{name} dissonance")
    if "leiden" in query.obs:
        _spatial_cat(query, "leiden", out_dir / "spatial_leiden.png", f"{name} leiden")

    query.obs[["scldl_pred", "scldl_entropy", "scldl_dissonance", "scldl_pair", "scldl_p1", "scldl_p2"]].to_csv(
        out_dir / "spot_annotations.csv"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["visiumhd", "slideseqv2", "all"], default="all")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-top-genes", dest="n_top_genes", type=int, default=2000)
    p.add_argument("--n-pcs", dest="n_pcs", type=int, default=40)
    p.add_argument("--max-per-class", dest="max_per_class", type=int, default=700)
    p.add_argument("--max-query", dest="max_query", type=int, default=28000)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    loaders = {"visiumhd": _load_visium, "slideseqv2": _load_slideseq}
    names = list(loaders) if args.dataset == "all" else [args.dataset]
    summary = []
    for name in names:
        ref, query, tag = loaders[name]()
        summary.append(_run_dataset(tag, ref, query, args))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
