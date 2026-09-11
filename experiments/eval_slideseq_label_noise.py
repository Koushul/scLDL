#!/usr/bin/env python
"""Inject increasing label noise into Slide-seqV2 RCTD singlets and score scLDL OOF recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import StratifiedKFold

from scLDL.label_noise import (
    HIPPO_SIMILAR,
    align_proba,
    flip_labels,
    noise_recovery_metrics,
)
from scLDL.pipeline import AnnotationPipeline
from scLDL.spatial import map_rctd, standardize_gene_names, subsample_balanced

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spatial"
OUT = ROOT / "artifacts" / "slideseq_label_noise"


def load_slideseq_singlets(min_class: int = 50, max_per_class: int | None = 200, seed: int = 0):
    query = sc.read_h5ad(DATA / "AdataMH1.h5ad")
    rctd = pd.read_csv(DATA / "ssHippo_RCTD.csv", index_col=0)
    shared = query.obs_names.intersection(rctd.index)
    query = query[shared].copy()
    cls = rctd.loc[shared, "spot_class"].astype(str)
    query.obs["rctd_class"] = cls.to_numpy()
    query.obs["cell_type"] = map_rctd(rctd.loc[shared, "celltype_1"]).astype(str).to_numpy()
    query = query[query.obs["rctd_class"] == "singlet"].copy()
    query = query[~query.obs["cell_type"].str.fullmatch(r"\d+")].copy()
    counts = query.obs["cell_type"].value_counts()
    keep = counts[counts >= min_class].index
    query = query[query.obs["cell_type"].isin(keep)].copy()
    query = standardize_gene_names(query)
    if max_per_class:
        query = subsample_balanced(query, "cell_type", max_per_class, seed=seed)
    if "spatial" not in query.obsm and {"xcoord", "ycoord"}.issubset(query.obs.columns):
        query.obsm["spatial"] = np.column_stack(
            [query.obs["xcoord"].to_numpy(dtype=float), query.obs["ycoord"].to_numpy(dtype=float)]
        )
    return query


def _pipe_kwargs(args):
    variant = getattr(args, "variant", "baseline")
    return dict(
        model="scldl",
        task="type",
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        n_hidden=args.n_hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        query_correct="none",
        spatial="off" if not args.spatial else "auto",
        graph_refine="off" if not args.graph_refine else "auto",
        supervised_mnn="off",
        label_smooth="on" if variant == "robust" else "off",
    )


def oof_predict(adata, label_key, n_splits, seed, pipe_kwargs, classes):
    y = np.asarray(adata.obs[label_key].astype(str))
    n = adata.n_obs
    k = len(classes)
    blended = np.zeros((n, k), dtype=np.float32)
    model = np.zeros((n, k), dtype=np.float32)
    knn = np.zeros((n, k), dtype=np.float32)
    vac = np.full(n, np.nan, dtype=np.float32)
    pred = np.empty(n, dtype=object)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(n), y)):
        pipe = AnnotationPipeline(**pipe_kwargs)
        pipe.fit(adata[tr].copy(), label_key=label_key)
        out = pipe.annotate(adata[te].copy(), copy=True)
        blended[te] = align_proba(out.obsm["X_scldl"], pipe.classes_, classes)
        model[te] = align_proba(out.obsm["X_scldl_model"], pipe.classes_, classes)
        if "X_scldl_knn" in out.obsm:
            knn[te] = align_proba(out.obsm["X_scldl_knn"], pipe.classes_, classes)
        else:
            knn[te] = blended[te]
        vac[te] = out.obs["scldl_uncertainty"].to_numpy(dtype=np.float32)
        pred[te] = out.obs["scldl_pred"].astype(str).to_numpy()
        if pipe_kwargs.get("verbose"):
            print(f"  fold {fold + 1}/{n_splits}: train {len(tr)} test {len(te)}")
    return {"blended": blended, "model": model, "knn": knn, "vacuity": vac, "pred": pred}


def _pred_from_proba(p, classes):
    return np.asarray(classes)[np.asarray(p).argmax(axis=1)]


def evaluate_fraction(adata, y_true, rate, args):
    rng = np.random.default_rng(args.seed + int(round(1000 * rate)))
    noisy, flipped = flip_labels(y_true, rate, rng, mode=args.noise)
    work = adata.copy()
    work.obs["cell_type"] = noisy
    classes = np.unique(noisy)
    parts = oof_predict(work, "cell_type", args.folds, args.seed, _pipe_kwargs(args), classes)
    rows = []
    sources = [
        ("scldl", parts["pred"], parts["blended"], parts["vacuity"]),
        ("scldl_model", _pred_from_proba(parts["model"], classes), parts["model"], None),
        ("knn", _pred_from_proba(parts["knn"], classes), parts["knn"], None),
    ]
    for name, pred, proba, vac in sources:
        row = noise_recovery_metrics(y_true, noisy, pred, proba, classes, vacuity=vac, name=name)
        row["noise_mode"] = args.noise
        row["requested_rate"] = float(rate)
        row["realized_flips"] = int(flipped.sum())
        row["variant"] = getattr(args, "variant", "baseline")
        row["seed"] = int(args.seed)
        rows.append(row)
    return rows, {"noisy": noisy, "flipped": flipped, "parts": parts, "classes": classes}


def plot_curves(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    sources = list(dict.fromkeys(df["source"]))
    for src in sources:
        sub = df[df["source"] == src].sort_values("requested_rate")
        axes[0].plot(sub["requested_rate"], sub["correction_rate"], marker="o", label=src)
        axes[1].plot(sub["requested_rate"], sub["discovery_auroc_1m_p_given"], marker="o", label=src)
        axes[2].plot(sub["requested_rate"], sub["acc_vs_true"], marker="o", label=src)
    axes[0].set_ylabel("correction rate")
    axes[1].set_ylabel("discovery AUROC (1 − p_given)")
    axes[2].set_ylabel("accuracy vs true label")
    for ax in axes:
        ax.set_xlabel("flip fraction")
        ax.set_xlim(-0.02, 0.55)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    axes[1].set_ylim(0.45, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def paired_tests(df):
    from scipy import stats

    rows = []
    keep = df[df["source"] == "scldl"].copy()
    for (mode, rate), sub in keep.groupby(["noise_mode", "requested_rate"]):
        base = sub[sub["variant"] == "baseline"].sort_values("seed")
        rob = sub[sub["variant"] == "robust"].sort_values("seed")
        merged = base.merge(rob, on="seed", suffixes=("_base", "_rob"))
        if merged.empty:
            continue
        rec = {"noise_mode": mode, "requested_rate": float(rate), "n_seeds": int(len(merged))}
        for metric in ["acc_vs_true", "correction_rate", "discovery_auroc_1m_p_given"]:
            a = merged[f"{metric}_base"].to_numpy(dtype=float)
            b = merged[f"{metric}_rob"].to_numpy(dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            rec[f"{metric}_baseline"] = float(np.mean(a)) if len(a) else None
            rec[f"{metric}_robust"] = float(np.mean(b)) if len(b) else None
            rec[f"{metric}_delta"] = float(np.mean(b - a)) if len(a) else None
            if len(a) >= 2 and np.std(b - a) > 1e-12:
                rec[f"{metric}_p_paired"] = float(stats.ttest_rel(b, a).pvalue)
            else:
                rec[f"{metric}_p_paired"] = None
        rows.append(rec)
    return rows


def plot_compare(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    metrics = [
        ("correction_rate", "correction rate"),
        ("discovery_auroc_1m_p_given", "discovery AUROC"),
        ("acc_vs_true", "accuracy vs true"),
    ]
    sub = df[df["source"] == "scldl"]
    for ax, (col, ylab) in zip(axes, metrics):
        for variant, color in (("baseline", "#4c4c4c"), ("robust", "#234e70")):
            part = sub[sub["variant"] == variant]
            if part.empty:
                continue
            g = part.groupby("requested_rate")[col].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), marker="o", label=variant, color=color)
        ax.set_xlabel("flip fraction")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and not np.isfinite(val):
            return None
        return val
    return obj


def _fmt(x):
    return "  nan" if x is None else f"{x:.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fractions", default="0,0.1,0.2,0.3,0.5")
    p.add_argument("--noise", choices=["uniform", "similar"], default="uniform")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--min-class", type=int, default=50)
    p.add_argument("--max-per-class", type=int, default=180)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--n-top-genes", type=int, default=1500)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None)
    p.add_argument("--variant", choices=["baseline", "robust", "compare"], default="baseline")
    p.add_argument("--spatial", action="store_true")
    p.add_argument("--graph-refine", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    adata = load_slideseq_singlets(min_class=args.min_class, max_per_class=args.max_per_class, seed=args.seed)
    y_true = adata.obs["cell_type"].astype(str).to_numpy()
    if args.variant == "compare":
        variants = ["baseline", "robust"]
        seeds = [int(x) for x in (args.seeds or "0,1,2").split(",") if x.strip()]
    else:
        variants = [args.variant]
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]
    meta = {
        "n": int(adata.n_obs),
        "types": adata.obs["cell_type"].value_counts().to_dict(),
        "noise": args.noise,
        "folds": args.folds,
        "epochs": args.epochs,
        "max_per_class": args.max_per_class,
        "variants": variants,
        "seeds": seeds,
    }
    print(json.dumps({"loaded": meta}, indent=2))
    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    rows = []
    from copy import copy

    for seed in seeds:
        for variant in variants:
            run_args = copy(args)
            run_args.seed = seed
            run_args.variant = variant
            for rate in fractions:
                print(f"\n=== {variant} seed={seed} flip={rate} ({args.noise}) ===", flush=True)
                part_rows, _ = evaluate_fraction(adata, y_true, rate, run_args)
                for row in part_rows:
                    print(
                        f"{row['source']:12s}  acc_true={row['acc_vs_true']:.3f}  "
                        f"correct={_fmt(row['correction_rate'])}  "
                        f"auroc={_fmt(row['discovery_auroc_1m_p_given'])}  "
                        f"P@k={_fmt(row['precision_at_nflip'])}",
                        flush=True,
                    )
                rows.extend(part_rows)
                (OUT / "metrics.json").write_text(
                    json.dumps(_json_ready({"meta": meta, "rows": rows}), indent=2), encoding="utf-8"
                )
    df = pd.DataFrame(rows)
    tag = f"{args.variant}_{args.noise}"
    df.to_csv(OUT / f"metrics_{tag}.csv", index=False)
    if args.variant == "compare":
        plot_compare(df, OUT / f"compare_{args.noise}.png")
        tests = paired_tests(df)
        (OUT / f"compare_{args.noise}.json").write_text(
            json.dumps(_json_ready({"meta": meta, "tests": tests, "rows": rows}), indent=2), encoding="utf-8"
        )
        print(json.dumps(_json_ready({"tests": tests}), indent=2))
    else:
        plot_curves(df[df["source"].isin(["scldl", "scldl_model", "knn"])], OUT / f"noise_curves_{tag}.png")
    payload = _json_ready({"meta": meta, "rows": rows})
    (OUT / f"metrics_{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / f'metrics_{tag}.csv'}")


if __name__ == "__main__":
    main()
