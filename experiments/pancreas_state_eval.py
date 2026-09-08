"""Compare hard-label ConcentrationLE vs state-aware ConcentrationLE on pancreas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from scLDL.data import looks_like_counts, to_dense
from scLDL.metrics import score_annotations
from scLDL.models import ConcentrationLE, StateConcentrationLE
from scLDL.state_targets import (
    blend_targets,
    knn_smooth_labels,
    marker_targets,
    pancreas_lineage_edges,
)

ORDER = ["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine", "Beta", "Alpha", "Delta", "Epsilon"]
MARKERS = {
    "Ductal": ["Sox9", "Anxa2", "Krt19", "Spp1"],
    "Ngn3 low EP": ["Neurog3", "Bicc1", "Sox9"],
    "Ngn3 high EP": ["Neurog3", "Fev", "Neurod1"],
    "Pre-endocrine": ["Fev", "Pax4", "Pax6", "Neurod1"],
    "Beta": ["Ins1", "Ins2", "Pdx1", "Nkx6-1"],
    "Alpha": ["Gcg", "Arx", "Irx2"],
    "Delta": ["Sst", "Hhex"],
    "Epsilon": ["Ghrl"],
}
AXIS = {
    "ductal": ["Sox9", "Krt19", "Spp1"],
    "ngn3": ["Neurog3"],
    "fev": ["Fev", "Neurod1"],
    "beta": ["Ins1", "Ins2"],
    "alpha": ["Gcg"],
    "delta": ["Sst"],
    "eps": ["Ghrl"],
}


def _prepare():
    ad = scv.datasets.pancreas()
    ad.obs["cell_state"] = pd.Categorical(ad.obs["clusters"].astype(str), categories=ORDER, ordered=True)
    if looks_like_counts(ad.X):
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    for name, genes in AXIS.items():
        sc.tl.score_genes(ad, [g for g in genes if g in ad.var_names], score_name=f"score_{name}")
    extra = sorted({g for gs in MARKERS.values() for g in gs if g in ad.var_names})
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, subset=False)
    keep = ad.var["highly_variable"].copy()
    keep.loc[extra] = True
    ad_f = ad[:, keep].copy()
    X = np.nan_to_num(to_dense(ad_f.X), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(ad.obs["cell_state"].astype(str))
    idx = {c: i for i, c in enumerate(ORDER)}
    y_int = np.array([idx[v] for v in y])
    onehot = np.eye(len(ORDER), dtype=np.float32)[y_int]
    return ad, ad_f, X, y, y_int, onehot


def _summarize(name, proba, u, y_te, ad_te, classes):
    metrics, _ = score_annotations(y_te, proba, classes)
    entropy = (-proba * np.log(np.clip(proba, 1e-8, 1))).sum(1)
    rows = []
    p_map = {c: proba[:, i] for i, c in enumerate(classes)}
    for st in ORDER:
        m = y_te == st
        if m.sum() == 0:
            continue
        rec = {
            "method": name,
            "state": st,
            "n": int(m.sum()),
            "P_self": float(p_map[st][m].mean()),
            "entropy": float(entropy[m].mean()),
            "u": float(u[m].mean()),
        }
        for i, c in enumerate(classes):
            rec[f"P_{c}"] = float(proba[m, i].mean())
        rows.append(rec)
    pairs = [
        ("score_ductal", "Ductal"),
        ("score_ngn3", "Ngn3 high EP"),
        ("score_fev", "Pre-endocrine"),
        ("score_beta", "Beta"),
        ("score_alpha", "Alpha"),
        ("score_delta", "Delta"),
        ("score_eps", "Epsilon"),
    ]
    corrs = {}
    for score, state in pairs:
        r, _ = spearmanr(np.asarray(ad_te.obs[score]), p_map[state])
        corrs[f"{state}"] = float(r)
    m_low = y_te == "Ngn3 low EP"
    mix = {
        "ngn3low_P_ductal": float(p_map["Ductal"][m_low].mean()) if m_low.any() else np.nan,
        "ngn3low_P_ngn3low": float(p_map["Ngn3 low EP"][m_low].mean()) if m_low.any() else np.nan,
    }
    return metrics, pd.DataFrame(rows), corrs, mix


def _print_target_entropy(title, soft, y_tr):
    print(title)
    ent = (-soft * np.log(np.clip(soft, 1e-8, 1))).sum(1)
    for st in ORDER:
        m = y_tr == st
        print(f"  {st:16s} mean H={ent[m].mean():.3f}  P_self={soft[m, ORDER.index(st)].mean():.3f}")


def _fit_state(X_tr, soft, neighbor_p):
    model = StateConcentrationLE(
        n_features=X_tr.shape[1],
        n_outputs=8,
        n_hidden=256,
        epochs=50,
        batch_size=128,
        prior=0.2,
        lineage_weight=0.45,
        manifold_weight=0.12,
        kl_weight=0.3,
        vacuity_weight=0.15,
        mixup_alpha=0.3,
        lineage_edges=pancreas_lineage_edges(),
        verbose=True,
    )
    model.fit(X_tr, soft, neighbor_p=neighbor_p)
    return model


def main():
    ad, ad_f, X, y, y_int, onehot = _prepare()
    classes = np.array(ORDER)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0, stratify=y_int)
    marker_tr = marker_targets(X[tr], ad_f.var_names, MARKERS, ORDER, temperature=0.65)
    graph_rbf, P_rbf = knn_smooth_labels(
        X[tr], onehot[tr], n_neighbors=20, alpha=0.75, n_iter=20, method="global_rbf", return_graph=True
    )
    graph_prob, P_prob = knn_smooth_labels(
        X[tr],
        onehot[tr],
        n_neighbors=30,
        alpha=0.75,
        n_iter=20,
        method="adaptive",
        n_pcs=40,
        return_graph=True,
    )
    soft_rbf = blend_targets((onehot[tr], 0.25), (graph_rbf, 0.45), (marker_tr, 0.30))
    soft_prob = blend_targets((onehot[tr], 0.25), (graph_prob, 0.45), (marker_tr, 0.30))

    _print_target_entropy("\nSoft targets (global RBF kNN):", soft_rbf, y[tr])
    _print_target_entropy("\nSoft targets (adaptive probabilistic):", soft_prob, y[tr])

    print("\nTraining baseline ConcentrationLE on one-hots...")
    base = ConcentrationLE(
        n_features=X.shape[1], n_outputs=8, n_hidden=256, epochs=40, batch_size=128, verbose=True
    )
    base.fit(X[tr], onehot[tr])
    p0, u0 = base.predict_evidence(X[te])

    print("\nTraining StateConcentrationLE with global RBF neighbors...")
    state_rbf = _fit_state(X[tr], soft_rbf, P_rbf)
    p1, u1 = state_rbf.predict_evidence(X[te])

    print("\nTraining StateConcentrationLE with probabilistic neighbors...")
    state_prob = _fit_state(X[tr], soft_prob, P_prob)
    p2, u2 = state_prob.predict_evidence(X[te])

    ad_te = ad[te]
    y_te = y[te]
    runs = [
        ("ConcentrationLE", p0, u0),
        ("State + RBF kNN", p1, u1),
        ("State + probabilistic kNN", p2, u2),
    ]
    print("\n===== HELD-OUT vs cluster labels =====")
    for name, p, u in runs:
        metrics, by_state, corrs, mix = _summarize(name, p, u, y_te, ad_te, classes)
        print(f"\n{name}")
        print(
            f"  acc={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}  "
            f"brier={metrics['brier']:.3f}  ece={metrics['ece']:.3f}"
        )
        print(f"  Ngn3-low mix: P(Ductal)={mix['ngn3low_P_ductal']:.3f}  P(Ngn3 low)={mix['ngn3low_P_ngn3low']:.3f}")
        print("  marker Spearman:", {k: round(v, 3) for k, v in corrs.items()})
        cols = ["state", "P_self", "entropy", "u", "P_Ductal", "P_Ngn3 low EP", "P_Beta", "P_Alpha", "P_Delta", "P_Epsilon"]
        print(by_state[cols].round(3).to_string(index=False))

    print("\n===== Hormone tracking inside Alpha/Beta (held-out) =====")
    for name, p, _ in runs:
        p_map = {c: p[:, i] for i, c in enumerate(ORDER)}
        for st, score, pred in [("Beta", "score_beta", "Beta"), ("Alpha", "score_alpha", "Alpha")]:
            m = y_te == st
            r, _ = spearmanr(np.asarray(ad_te.obs[score])[m], p_map[pred][m])
            print(f"{name:28s} {st:6s} {score} vs P({pred}) r={r:.3f}")


if __name__ == "__main__":
    main()
