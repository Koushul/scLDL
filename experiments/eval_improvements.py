"""Compare gene-space LDL vs interpretable reference-mapping on types, states, and batch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from anndata import AnnData
from sklearn.model_selection import train_test_split

from scLDL.pipeline import AnnotationPipeline

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "improvement_eval"


def _blob(n=240, n_genes=40, n_classes=4, seed=0, scale=3.2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    means = rng.normal(size=(n_classes, n_genes)) * scale
    x = means[y] + rng.normal(scale=0.3, size=(n, n_genes))
    ad = AnnData(x.astype(np.float32))
    ad.obs["cell_type"] = np.array([f"type_{i}" for i in y])
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    ad.obs_names = [f"c{i}" for i in range(n)]
    return ad


def _states(n=300, seed=1):
    rng = np.random.default_rng(seed)
    t = rng.beta(1.4, 1.4, size=n) * 2
    y = np.where(t < 0.65, 0, np.where(t < 1.35, 1, 2))
    x = rng.normal(scale=0.25, size=(n, 20)).astype(np.float32)
    x[:, 0:2] += ((2 - t) * 2.0)[:, None]
    x[:, 2:4] += ((1 - np.abs(t - 1)) * 2.2)[:, None]
    x[:, 4:6] += (t * 2.0)[:, None]
    ad = AnnData(x)
    ad.obs["cell_state"] = np.array(["early", "mid", "late"])[y]
    ad.var_names = [f"g{i}" for i in range(20)]
    ad.obs_names = [f"s{i}" for i in range(n)]
    return ad, {"early": ["g0", "g1"], "mid": ["g2", "g3"], "late": ["g4", "g5"]}


def _fit_eval(model, task, train, test, label_key, n_top_genes=80, n_pcs=30, epochs=30, **kwargs):
    pipe = AnnotationPipeline(
        model=model,
        task=task,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        n_hidden=64,
        epochs=epochs,
        batch_size=32,
        verbose=False,
        **kwargs,
    )
    pipe.fit(train, label_key=label_key)
    return pipe.evaluate(test)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    types = _blob()
    tr, te = train_test_split(np.arange(types.n_obs), test_size=0.3, random_state=0, stratify=types.obs["cell_type"])
    train, test = types[tr].copy(), types[te].copy()
    for model in ["mlp", "concentration", "interpretable"]:
        m = _fit_eval(model, "type", train, test, "cell_type")
        rows.append({"setting": "cell_type_holdout", "model": model, **{k: m[k] for k in ("accuracy", "macro_f1")}})

    shifted = test.copy()
    shifted.X = np.asarray(shifted.X) * 0.35 + 5.5
    for model, correct in [("mlp", "none"), ("concentration", "none"), ("interpretable", "auto")]:
        kwargs = {"query_correct": correct} if model == "interpretable" else {}
        m = _fit_eval(model, "type", train, shifted, "cell_type", **kwargs)
        rows.append({"setting": "cell_type_batch_shift", "model": model, **{k: m[k] for k in ("accuracy", "macro_f1")}})

    states, markers = _states()
    tr, te = train_test_split(np.arange(states.n_obs), test_size=0.3, random_state=1, stratify=states.obs["cell_state"])
    strain, stest = states[tr].copy(), states[te].copy()
    m_old = _fit_eval("concentration", "type", strain, stest, "cell_state")
    rows.append({"setting": "cell_state", "model": "concentration", "accuracy": m_old["accuracy"], "macro_f1": m_old["macro_f1"], "marker_spearman_mean": None})
    m_new = _fit_eval(
        "interpretable",
        "state",
        strain,
        stest,
        "cell_state",
        markers=markers,
        lineage_edges=[(0, 1), (1, 2)],
    )
    rows.append(
        {
            "setting": "cell_state",
            "model": "interpretable",
            "accuracy": m_new["accuracy"],
            "macro_f1": m_new["macro_f1"],
            "marker_spearman_mean": m_new.get("marker_spearman_mean"),
            "mean_illegal_mass": m_new.get("mean_illegal_mass"),
        }
    )

    try:
        import scanpy as sc

        pbmc = sc.datasets.pbmc68k_reduced()
        pbmc.obs["cell_type"] = pbmc.obs["bulk_labels"].astype(str)
        idx = np.arange(pbmc.n_obs)
        tr, te = train_test_split(idx, test_size=0.2, random_state=2, stratify=pbmc.obs["cell_type"])
        for model in ["mlp", "interpretable"]:
            m = _fit_eval(
                model,
                "type",
                pbmc[tr].copy(),
                pbmc[te].copy(),
                "cell_type",
                n_top_genes=500,
                n_pcs=40,
                epochs=40,
            )
            rows.append({"setting": "pbmc68k_types", "model": model, **{k: m[k] for k in ("accuracy", "macro_f1")}})
    except Exception as exc:
        rows.append({"setting": "pbmc68k_types", "model": "skipped", "error": str(exc)})

    (OUT / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
