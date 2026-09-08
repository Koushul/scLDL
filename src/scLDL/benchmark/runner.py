from __future__ import annotations

import numpy as np
import pandas as pd

from scLDL.benchmark.methods import ALL_METHODS, CORE_METHODS, build_method, method_available, run_method
from scLDL.benchmark.protocol import prepare_holdout, prepare_reference_query


def run_benchmark(
    adata,
    label_key: str = "cell_type",
    methods: list[str] | None = None,
    query=None,
    test_size: float = 0.2,
    n_top_genes: int = 2000,
    n_repeats: int = 1,
    seed: int = 0,
    noise_rate: float = 0.0,
    epochs: int = 40,
    n_hidden: int = 128,
    batch_size: int = 64,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare annotation methods on a hold-out split or a query dataset.

    HVGs and the feature scaler are fit on the reference/train split only.
    """
    methods = list(methods) if methods is not None else list(CORE_METHODS)
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    rows = []
    repeats = 1 if query is not None else n_repeats
    for r in range(repeats):
        split_seed = seed + r
        if query is None:
            data = prepare_holdout(
                adata,
                label_key=label_key,
                test_size=test_size,
                n_top_genes=n_top_genes,
                seed=split_seed,
                noise_rate=noise_rate,
            )
            protocol = "holdout"
        else:
            data = prepare_reference_query(adata, query, label_key=label_key, n_top_genes=n_top_genes)
            if noise_rate > 0:
                from scLDL.benchmark.protocol import flip_labels

                data.y_train = flip_labels(data.y_train, noise_rate, np.random.default_rng(split_seed))
                data.classes = np.unique(data.y_train)
            protocol = "cross_dataset"

        if verbose:
            print(
                f"[{protocol} seed={split_seed}] train={data.X_train_log.shape[0]} "
                f"test={data.X_test_log.shape[0]} genes={data.X_train_log.shape[1]} "
                f"classes={len(data.classes)} overlap={data.n_overlap}"
            )

        for name in methods:
            skip = method_available(name, data)
            if skip:
                rows.append(_skipped(name, skip, r, split_seed, protocol, noise_rate))
                if verbose:
                    print(f"  skip {name}: {skip}")
                continue
            method = build_method(name, epochs=epochs, n_hidden=n_hidden, batch_size=batch_size)
            try:
                metrics = run_method(method, data)
            except Exception as exc:
                rows.append(_skipped(name, str(exc), r, split_seed, protocol, noise_rate))
                if verbose:
                    print(f"  fail {name}: {exc}")
                continue
            metrics.update(
                {
                    "repeat": r,
                    "seed": split_seed,
                    "protocol": protocol,
                    "noise_rate": noise_rate,
                    "gene_overlap": data.n_overlap,
                }
            )
            rows.append(metrics)
            if verbose:
                print(
                    f"  {name:22s} acc={metrics['accuracy']:.3f} "
                    f"f1={metrics['macro_f1']:.3f} brier={metrics['brier']:.3f} "
                    f"[{metrics['fit_seconds']:.1f}s]"
                )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"] == "ok"]
    if ok.empty:
        return ok
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "brier",
        "ece",
        "mean_confidence",
        "mean_entropy",
        "fit_seconds",
        "predict_seconds",
    ]
    present = [m for m in metrics if m in ok.columns]
    grouped = ok.groupby("method")[present].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.reset_index().sort_values("accuracy_mean", ascending=False)


def _skipped(name, error, repeat, seed, protocol, noise_rate):
    return {
        "method": name,
        "status": "skipped",
        "error": error,
        "repeat": repeat,
        "seed": seed,
        "protocol": protocol,
        "noise_rate": noise_rate,
    }
