#!/usr/bin/env python
"""Hold-out annotation: CellTypist vs scLDL (and logistic / PCA-kNN) on the same genes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import scanpy as sc

from scLDL.benchmark import run_benchmark, summarize
from scLDL.spatial import map_rctd, standardize_gene_names, subsample_balanced

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LN_H5 = Path("/ix1/ylee/kor11/tools/SpaceTravLR/data/SlideSeqV2_mouse_lymphnode.h5ad")
DEFAULT_METHODS = ["majority", "logistic", "pca_knn", "celltypist", "scldl_mlp", "scldl_interpretable"]


def load_pancreas():
    adata = sc.read_h5ad(DATA / "Pancreas" / "endocrinogenesis_day15.h5ad")
    adata.var_names_make_unique()
    adata.obs["cell_type"] = adata.obs["clusters"].astype(str)
    return standardize_gene_names(adata)


def load_hippo(min_class: int = 50, max_per_class: int | None = None, seed: int = 0):
    query = sc.read_h5ad(DATA / "spatial" / "AdataMH1.h5ad")
    rctd = pd.read_csv(DATA / "spatial" / "ssHippo_RCTD.csv", index_col=0)
    shared = query.obs_names.intersection(rctd.index)
    query = query[shared].copy()
    query.obs["rctd_class"] = rctd.loc[shared, "spot_class"].astype(str).to_numpy()
    query.obs["cell_type"] = map_rctd(rctd.loc[shared, "celltype_1"]).astype(str).to_numpy()
    query = query[query.obs["rctd_class"] == "singlet"].copy()
    query = query[~query.obs["cell_type"].str.fullmatch(r"\d+")].copy()
    counts = query.obs["cell_type"].value_counts()
    keep = counts[counts >= min_class].index
    query = query[query.obs["cell_type"].isin(keep)].copy()
    query = standardize_gene_names(query)
    if max_per_class:
        query = subsample_balanced(query, "cell_type", max_per_class, seed=seed)
    return query


def load_lymphnode(min_class: int = 50, max_per_class: int | None = 400, seed: int = 0):
    from scLDL.data import looks_like_counts

    if not LN_H5.exists():
        raise FileNotFoundError(f"Missing lymph-node Slide-seqV2 file: {LN_H5}")
    query = sc.read_h5ad(LN_H5)
    query.obs["cell_type"] = query.obs["cell_type"].astype(str)
    query = query[query.obs["cell_type"].notna() & (query.obs["cell_type"] != "nan")].copy()
    counts = query.obs["cell_type"].value_counts()
    keep = counts[counts >= min_class].index
    query = query[query.obs["cell_type"].isin(keep)].copy()
    query = standardize_gene_names(query)
    if looks_like_counts(query.X):
        raise RuntimeError("Lymph-node matrix was classified as counts; it should already be log-normalized.")
    if max_per_class:
        query = subsample_balanced(query, "cell_type", max_per_class, seed=seed)
    return query


LOADERS = {
    "pancreas": lambda args: load_pancreas(),
    "hippo": lambda args: load_hippo(args.min_class, args.hippo_max_per_class, args.seed),
    "lymphnode": lambda args: load_lymphnode(args.min_class, args.ln_max_per_class, args.seed),
}


def paired_vs(results: pd.DataFrame, a: str, b: str, metric: str = "accuracy"):
    ok = results[results["status"] == "ok"]
    rows = []
    for dataset, g in ok.groupby("dataset"):
        wide = g.pivot_table(index="seed", columns="method", values=metric)
        if a not in wide.columns or b not in wide.columns:
            continue
        diff = wide[a] - wide[b]
        n = int(diff.notna().sum())
        mean = float(diff.mean())
        p = None
        if n >= 2:
            from scipy.stats import ttest_rel

            stat = ttest_rel(wide[a].to_numpy(), wide[b].to_numpy(), nan_policy="omit")
            p = float(stat.pvalue)
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "a": a,
                "b": b,
                "mean_diff_a_minus_b": mean,
                "n_repeats": n,
                "p_paired_t": p,
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="pancreas,hippo,lymphnode")
    p.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--min-class", type=int, default=50)
    p.add_argument("--hippo-max-per-class", type=int, default=0)
    p.add_argument("--ln-max-per-class", type=int, default=400)
    p.add_argument("--out", default="artifacts/celltypist_benchmark")
    args = p.parse_args()
    if args.hippo_max_per_class == 0:
        args.hippo_max_per_class = None
    if args.ln_max_per_class == 0:
        args.ln_max_per_class = None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    meta = {"datasets": {}, "methods": list(args.methods), "repeats": args.repeats}

    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if name not in LOADERS:
            raise ValueError(f"Unknown dataset {name!r}")
        print(f"\n==== {name} ====")
        adata = LOADERS[name](args)
        types = adata.obs["cell_type"].astype(str).value_counts().to_dict()
        meta["datasets"][name] = {"n": int(adata.n_obs), "n_vars": int(adata.n_vars), "types": types}
        print(f"n={adata.n_obs} genes={adata.n_vars} types={len(types)}")
        results = run_benchmark(
            adata,
            label_key="cell_type",
            methods=list(args.methods),
            test_size=args.test_size,
            n_top_genes=min(args.n_top_genes, adata.n_vars),
            n_repeats=args.repeats,
            seed=args.seed,
            epochs=args.epochs,
            n_hidden=args.n_hidden,
            verbose=True,
        )
        results.insert(0, "dataset", name)
        results.to_csv(out / f"{name}.csv", index=False)
        summarize(results).to_csv(out / f"{name}_summary.csv", index=False)
        frames.append(results)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(out / "metrics.csv", index=False)
    summary = summarize(all_rows)
    # per-dataset summary
    parts = []
    for dataset, g in all_rows.groupby("dataset"):
        s = summarize(g)
        s.insert(0, "dataset", dataset)
        parts.append(s)
    by_ds = pd.concat(parts, ignore_index=True)
    by_ds.to_csv(out / "summary_by_dataset.csv", index=False)

    comparisons = []
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "brier", "ece"):
        if metric not in all_rows.columns:
            continue
        comparisons.extend(paired_vs(all_rows, "scldl_interpretable", "celltypist", metric))
        comparisons.extend(paired_vs(all_rows, "celltypist", "logistic", metric))
        comparisons.extend(paired_vs(all_rows, "scldl_interpretable", "logistic", metric))
    (out / "paired_tests.json").write_text(json.dumps(comparisons, indent=2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n==== paired scldl_interpretable - celltypist (accuracy) ====")
    for row in comparisons:
        if row["metric"] == "accuracy" and row["a"] == "scldl_interpretable" and row["b"] == "celltypist":
            print(row)
    print(f"\nwrote {out}")
    return all_rows


if __name__ == "__main__":
    main()
