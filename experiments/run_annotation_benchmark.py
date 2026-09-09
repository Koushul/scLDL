"""Hold-out or cross-dataset annotation benchmark."""

import argparse

import scanpy as sc

from scLDL.benchmark import ALL_METHODS, CORE_METHODS, run_benchmark, summarize


def main():
    parser = argparse.ArgumentParser(description="Benchmark scLDL against annotation baselines")
    parser.add_argument("--adata", required=True, help="Reference/single h5ad")
    parser.add_argument("--query", default=None, help="Optional query h5ad (cross-dataset)")
    parser.add_argument("--label", default="cell_type")
    parser.add_argument("--methods", nargs="*", default=None, help="Method names; default is the core suite")
    parser.add_argument("--all-methods", action="store_true")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-rate", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--out", default="reports/annotation_benchmark.csv")
    args = parser.parse_args()

    methods = list(ALL_METHODS) if args.all_methods else args.methods
    adata = sc.read_h5ad(args.adata)
    query = sc.read_h5ad(args.query) if args.query else None
    results = run_benchmark(
        adata,
        label_key=args.label,
        methods=methods,
        query=query,
        test_size=args.test_size,
        n_top_genes=args.n_top_genes,
        n_repeats=args.repeats,
        seed=args.seed,
        noise_rate=args.noise_rate,
        epochs=args.epochs,
        n_hidden=args.n_hidden,
    )
    summary = summarize(results)
    print("\nSummary")
    print(summary.to_string(index=False))
    out = args.out
    if out:
        import os

        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        results.to_csv(out, index=False)
        summary_path = out.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {out} and {summary_path}")
    return results


if __name__ == "__main__":
    main()
