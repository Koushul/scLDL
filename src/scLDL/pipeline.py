from __future__ import annotations

import numpy as np

from scLDL.data import align_matrix, labels_to_onehot, log1p_normalize, looks_like_counts, preprocess_reference, to_dense
from scLDL.device import resolve_device
from scLDL.embedding import ReferenceEmbedding, mnn_map_supervised
from scLDL.interpret import (
    adaptive_blend,
    dissonance,
    entropy,
    graph_refine,
    illegal_mass,
    marker_spearman,
    project_lineage,
    residual_mass,
    top2_pairs,
)
from scLDL.metrics import classification_metrics, distribution_metrics
from scLDL.models import ANNOTATION_MODELS
from scLDL.neighbors import query_label_transfer
from scLDL.spatial import try_spatial_xy
from scLDL.spatial_smooth import spatial_refine
from scLDL.state_targets import blend_targets, knn_smooth_labels, marker_targets


class AnnotationPipeline:
    """Train on a labeled reference and annotate a query dataset.

    ``task="type"`` keeps predictions peaked for cell-type accuracy.
    ``task="state"`` uses marker programs, probabilistic neighbors, and a
    lineage prior so mixed / transitional cells keep mass on nearby states.

        Query cells are mapped into the reference PCA space (optional MNN
        correction, with a type-restricted second pass) so annotation does not
        require sharing a technical batch.

        After blending, ``graph_refine="on"`` smooths the simplex on the query
        kNN graph so nearby cells share mass while peaked cells stay put.
        When the query has coordinates, ``spatial="auto"`` then snaps isolated
        speckles without washing out layer interiors.
    """

    def __init__(
        self,
        model: str = "scldl",
        task: str = "type",
        n_top_genes: int = 2000,
        n_pcs: int = 50,
        n_neighbors: int = 30,
        n_hidden: int = 256,
        epochs: int = 40,
        batch_size: int = 128,
        lr: float = 1e-3,
        device=None,
        verbose: bool = True,
        markers: dict | None = None,
        lineage_edges=None,
        query_correct: str = "auto",
        spatial: str = "auto",
        graph_refine: str = "on",
        supervised_mnn: str = "on",
        **model_kwargs,
    ):
        if model not in ANNOTATION_MODELS:
            raise ValueError(f"Unknown annotation model {model!r}. Choose from {sorted(ANNOTATION_MODELS)}")
        if task not in {"type", "state"}:
            raise ValueError("task must be 'type' or 'state'")
        if spatial not in {"auto", "on", "off"}:
            raise ValueError("spatial must be 'auto', 'on', or 'off'")
        if graph_refine not in {"auto", "on", "off"}:
            raise ValueError("graph_refine must be 'auto', 'on', or 'off'")
        if supervised_mnn not in {"auto", "on", "off"}:
            raise ValueError("supervised_mnn must be 'auto', 'on', or 'off'")
        self.model_name = model
        self.task = task
        self.n_top_genes = n_top_genes
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = resolve_device(device)
        self.verbose = verbose
        self.markers = markers
        self.lineage_edges = lineage_edges
        self.query_correct = query_correct
        self.spatial = spatial
        self.graph_refine = graph_refine
        self.supervised_mnn = supervised_mnn
        self.model_kwargs = model_kwargs
        self.estimator_ = None
        self.embed_ = None
        self.var_names_ = None
        self.classes_ = None
        self.label_key_ = None
        self.Y_ref_ = None
        self.uses_embedding_ = model in {"scldl", "interpretable", "state_concentration"}
        self._ref_had_batches = False
        self.last_spatial_ = "off"
        self.last_graph_refine_ = "off"
        self.last_supervised_mnn_ = False

    def _is_scldl(self):
        return self.model_name in {"scldl", "interpretable"}

    def _marker_matrix(self, expr, gene_names):
        if not self.markers or self.classes_ is None:
            return None
        return marker_targets(expr, gene_names, self.markers, list(self.classes_), temperature=0.7)

    def fit(self, adata, label_key: str = "cell_type", batch_key: str | None = None):
        if label_key not in adata.obs:
            raise KeyError(f"{label_key!r} not found in adata.obs")
        extra = None
        if self.markers:
            extra = sorted({g for gs in self.markers.values() for g in gs})
        ref = preprocess_reference(adata, n_top_genes=self.n_top_genes, always_include=extra)
        x_log = to_dense(ref.X)
        y_onehot, _, encoder = labels_to_onehot(ref.obs[label_key].values)
        self.var_names_ = np.asarray(ref.var_names.astype(str))
        self.classes_ = encoder.classes_
        self.label_key_ = label_key
        batches = np.asarray(ref.obs[batch_key].astype(str)) if batch_key and batch_key in ref.obs else None
        self._ref_had_batches = batches is not None and len(np.unique(batches)) > 1

        if self.uses_embedding_:
            self.embed_ = ReferenceEmbedding(n_pcs=self.n_pcs, n_neighbors=self.n_neighbors)
            self.embed_.fit(x_log, batches=batches)
            x_model = self.embed_.ref_z_
        else:
            self.embed_ = None
            x_model = x_log

        concepts = self._marker_matrix(x_log, self.var_names_) if self.task == "state" else None
        if self.task == "state":
            graph, neighbor_p = knn_smooth_labels(
                x_model,
                y_onehot,
                n_neighbors=self.n_neighbors,
                alpha=0.75,
                n_iter=15,
                method="adaptive",
                n_pcs=None,
                return_graph=True,
            )
            parts = [(y_onehot, 0.30), (graph, 0.40)]
            if concepts is not None:
                parts.append((concepts, 0.30))
            targets = blend_targets(*parts)
        else:
            neighbor_p = None
            targets = y_onehot

        cls = ANNOTATION_MODELS[self.model_name]
        params = dict(
            n_features=x_model.shape[1],
            n_outputs=len(self.classes_),
            n_hidden=self.n_hidden,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device,
            verbose=self.verbose,
        )
        if self._is_scldl():
            params.update(
                n_concepts=0 if concepts is None else concepts.shape[1],
                lineage_edges=self.lineage_edges if self.task == "state" else None,
                illegal_weight=0.25 if self.task == "state" and self.lineage_edges else 0.0,
                marker_kl_weight=0.12 if self.task == "state" and concepts is not None else 0.0,
                peak_weight=0.08 if self.task == "type" else 0.0,
                mixup_alpha=0.0 if self.task == "type" else 0.3,
                lineage_weight=0.35 if self.task == "state" and self.lineage_edges else 0.0,
                manifold_weight=0.10 if self.task == "state" else 0.0,
                vacuity_weight=0.08 if self.task == "state" else 0.0,
                kl_weight=0.2 if self.task == "state" else 0.05,
            )
        elif self.model_name == "state_concentration":
            params.update(lineage_edges=self.lineage_edges if self.task == "state" else None)
            if self.task == "type":
                params.update(mixup_alpha=0.0, lineage_weight=0.0, manifold_weight=0.0, vacuity_weight=0.0)
        params.update(self.model_kwargs)
        extra_fit = {}
        if self._is_scldl():
            extra_fit["concepts"] = concepts
            extra_fit["neighbor_p"] = neighbor_p
        elif self.model_name == "state_concentration":
            extra_fit["neighbor_p"] = neighbor_p
        self.estimator_ = cls(**params)
        self.estimator_.fit(x_model, targets, **{k: v for k, v in extra_fit.items() if v is not None or k == "concepts"})
        self.Y_ref_ = y_onehot
        self.x_log_ref_ = x_log
        return self

    def _check_fitted(self):
        if self.estimator_ is None:
            raise RuntimeError("Call fit() before predict().")

    def _prepare_query(self, adata):
        X = adata.X
        if looks_like_counts(X):
            X = log1p_normalize(X)
        x, n_overlap = align_matrix(X, adata.var_names, self.var_names_)
        if self.verbose:
            print(f"Aligned query genes: {n_overlap}/{len(self.var_names_)} overlap")
        return x

    def _infer(self, adata):
        self._check_fitted()
        x = self._prepare_query(adata)
        parts = self._predict_parts(x)
        dist = self._apply_graph(parts, adata)
        dist = self._apply_spatial(dist, adata, parts["x_model"])
        return parts, dist

    def _use_supervised_mnn(self, correct):
        if self.supervised_mnn == "off" or not self.uses_embedding_ or self.Y_ref_ is None:
            return False
        return correct == "mnn"

    def _use_graph_refine(self):
        if self.graph_refine == "off":
            return False
        if self.graph_refine == "on":
            return True
        return self._is_scldl()

    def _model_inputs(self, x_log):
        if self.embed_ is None:
            self.last_supervised_mnn_ = False
            return x_log
        correct = self.query_correct
        if correct == "auto" and self._ref_had_batches:
            correct = "mnn"
        z = self.embed_.transform(x_log, correct=correct)
        if self._use_supervised_mnn(getattr(self.embed_, "last_correct_", "none")):
            knn_p = query_label_transfer(z, self.embed_.ref_z_, self.Y_ref_, n_neighbors=self.n_neighbors)
            z = mnn_map_supervised(
                z,
                self.embed_.ref_z_,
                knn_p.argmax(axis=1),
                self.Y_ref_.argmax(axis=1),
                n_neighbors=self.n_neighbors,
            )
            self.last_supervised_mnn_ = True
        else:
            self.last_supervised_mnn_ = False
        return z

    def _apply_graph(self, parts, adata):
        dist = parts["blended"]
        if not self._use_graph_refine():
            self.last_graph_refine_ = "off"
            return dist
        xy = try_spatial_xy(adata) if self.spatial != "off" else None
        n_iter = 2 if self.task == "type" else 3
        mix = 0.55 if self.task == "type" else 0.65
        dist = graph_refine(
            dist,
            parts["x_model"],
            vacuity=parts["vacuity"],
            xy=xy,
            n_neighbors=min(15, self.n_neighbors),
            n_iter=n_iter,
            mix=mix,
        )
        parts["blended"] = dist
        self.last_graph_refine_ = "on"
        return dist

    def _predict_parts(self, x_log):
        x_model = self._model_inputs(x_log)
        concepts = self._marker_matrix(x_log, self.var_names_) if self.task == "state" else None
        if self._is_scldl():
            model_p, vacuity, _ = self.estimator_.predict_evidence(x_model, concepts=concepts)
        elif hasattr(self.estimator_, "predict_evidence"):
            out = self.estimator_.predict_evidence(x_model)
            model_p, vacuity = out[0], out[1]
        else:
            model_p = self.estimator_.predict(x_model)
            vacuity = 1.0 - model_p.max(axis=1)
        knn_p = None
        if self.embed_ is not None and self.Y_ref_ is not None:
            knn_p = query_label_transfer(
                x_model, self.embed_.ref_z_, self.Y_ref_, n_neighbors=self.n_neighbors
            )
        blended = adaptive_blend(
            model_p,
            knn_p=knn_p,
            marker_p=concepts,
            task=self.task,
            batch_corrected=getattr(self.embed_, "last_correct_", "none") in {"mnn", "center"},
        )
        if self.task == "state" and self.lineage_edges:
            blended = project_lineage(blended, self.lineage_edges, strength=0.3)
        return {
            "model": model_p,
            "knn": knn_p,
            "marker": concepts,
            "blended": blended,
            "vacuity": np.asarray(vacuity, dtype=np.float32).ravel(),
            "x_model": x_model,
        }

    def _apply_spatial(self, dist, adata, z):
        xy = try_spatial_xy(adata)
        if self.spatial == "off":
            self.last_spatial_ = "off"
            return dist
        if xy is None:
            if self.spatial == "on":
                raise KeyError("spatial='on' requires coordinates on the query AnnData")
            self.last_spatial_ = "off"
            return dist
        self.last_spatial_ = "on"
        k = 8 if self.task == "type" else 12
        n_iter = 2 if self.task == "type" else 4
        return spatial_refine(dist, xy, z=z, n_neighbors=k, n_iter=n_iter, task=self.task)

    def predict_distribution(self, adata):
        _, dist = self._infer(adata)
        return dist

    def annotate(self, adata, obsm_key: str = "X_scldl", obs_key: str = "scldl_pred", copy: bool = False):
        ad = adata.copy() if copy else adata
        parts, dist = self._infer(ad)
        expr = parts["blended"]
        ad.obsm[obsm_key] = dist
        ad.obsm["X_scldl_model"] = parts["model"]
        if parts["knn"] is not None:
            ad.obsm["X_scldl_knn"] = parts["knn"]
        if parts["marker"] is not None:
            ad.obsm["X_scldl_marker"] = parts["marker"]
            ad.obsm["X_scldl_residual"] = residual_mass(parts["model"], parts["marker"], parts["knn"])
        ad.obs[obs_key] = self.classes_[dist.argmax(axis=1)]
        if self.last_spatial_ == "on":
            ad.obsm["X_scldl_expr"] = expr
            ad.obs["scldl_pred_expr"] = self.classes_[expr.argmax(axis=1)]
        ad.obs["scldl_uncertainty"] = parts["vacuity"]
        ad.obs["scldl_entropy"] = entropy(dist)
        ad.obs["scldl_dissonance"] = dissonance(dist)
        pairs, mass = top2_pairs(dist, self.classes_)
        ad.obs["scldl_pair"] = pairs
        ad.obs["scldl_p1"] = mass[:, 0]
        ad.obs["scldl_p2"] = mass[:, 1]
        if self.lineage_edges:
            ad.obs["scldl_illegal"] = illegal_mass(dist, self.lineage_edges)
        return ad

    def evaluate(self, adata, label_key: str | None = None):
        key = label_key or self.label_key_
        if key not in adata.obs:
            raise KeyError(f"{key!r} not found in adata.obs")
        parts, dist = self._infer(adata)
        y_true = np.asarray(adata.obs[key].astype(str))
        y_pred = self.classes_[dist.argmax(axis=1)]
        metrics = classification_metrics(y_true, y_pred)
        try:
            y_onehot, _, _ = labels_to_onehot(y_true, classes=self.classes_)
            metrics.update(distribution_metrics(y_onehot, dist))
        except ValueError:
            pass
        metrics["mean_entropy"] = float(np.mean(entropy(dist)))
        metrics["mean_dissonance"] = float(np.mean(dissonance(dist)))
        if self.lineage_edges:
            metrics["mean_illegal_mass"] = float(np.mean(illegal_mass(dist, self.lineage_edges)))
        if parts["marker"] is not None:
            rs = marker_spearman(parts["marker"], dist)
            metrics["marker_spearman_mean"] = float(np.nanmean(rs))
        return metrics
