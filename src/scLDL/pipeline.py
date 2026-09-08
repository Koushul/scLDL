from __future__ import annotations

import numpy as np

from scLDL.data import align_to_genes, labels_to_onehot, looks_like_counts, preprocess_reference, to_dense
from scLDL.device import resolve_device
from scLDL.metrics import classification_metrics, distribution_metrics
from scLDL.models import ANNOTATION_MODELS


class AnnotationPipeline:
    """Train on a labeled reference AnnData and annotate a query dataset.

    Only models that map expression X -> label distribution are supported
    (mlp, lible, concentration, hybrid). LEVI-style enhancers need labels at
    inference and are excluded on purpose.
    """

    def __init__(
        self,
        model: str = "concentration",
        n_top_genes: int = 2000,
        n_hidden: int = 256,
        epochs: int = 40,
        batch_size: int = 128,
        lr: float = 1e-3,
        device=None,
        verbose: bool = True,
        **model_kwargs,
    ):
        if model not in ANNOTATION_MODELS:
            raise ValueError(f"Unknown annotation model {model!r}. Choose from {sorted(ANNOTATION_MODELS)}")
        self.model_name = model
        self.n_top_genes = n_top_genes
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = resolve_device(device)
        self.verbose = verbose
        self.model_kwargs = model_kwargs
        self.estimator_ = None
        self.var_names_ = None
        self.classes_ = None
        self.label_key_ = None
        self.log_normalized_ = False

    def fit(self, adata, label_key: str = "cell_type"):
        if label_key not in adata.obs:
            raise KeyError(f"{label_key!r} not found in adata.obs")
        self.log_normalized_ = looks_like_counts(adata.X)
        ref = preprocess_reference(adata, n_top_genes=self.n_top_genes)
        x = to_dense(ref.X)
        y_onehot, _, encoder = labels_to_onehot(ref.obs[label_key].values)
        self.var_names_ = np.asarray(ref.var_names.astype(str))
        self.classes_ = encoder.classes_
        self.label_key_ = label_key
        cls = ANNOTATION_MODELS[self.model_name]
        params = dict(
            n_features=x.shape[1],
            n_outputs=len(self.classes_),
            n_hidden=self.n_hidden,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device,
            verbose=self.verbose,
        )
        params.update(self.model_kwargs)
        self.estimator_ = cls(**params)
        self.estimator_.fit(x, y_onehot)
        return self

    def _check_fitted(self):
        if self.estimator_ is None:
            raise RuntimeError("Call fit() before predict().")

    def _prepare_query(self, adata):
        import scanpy as sc

        ad = adata.copy()
        if self.log_normalized_ and looks_like_counts(ad.X):
            sc.pp.normalize_total(ad, target_sum=1e4)
            sc.pp.log1p(ad)
        query, n_overlap = align_to_genes(ad, self.var_names_)
        if self.verbose:
            print(f"Aligned query genes: {n_overlap}/{len(self.var_names_)} overlap")
        return query, to_dense(query.X)

    def predict_distribution(self, adata):
        self._check_fitted()
        _, x = self._prepare_query(adata)
        return self.estimator_.predict(x)

    def annotate(self, adata, obsm_key: str = "X_scldl", obs_key: str = "scldl_pred", copy: bool = False):
        self._check_fitted()
        ad = adata.copy() if copy else adata
        _, x = self._prepare_query(ad)
        dist = self.estimator_.predict(x)
        ad.obsm[obsm_key] = dist
        ad.obs[obs_key] = self.classes_[dist.argmax(axis=1)]
        if hasattr(self.estimator_, "predict_evidence"):
            _, uncertainty = self.estimator_.predict_evidence(x)
            ad.obs["scldl_uncertainty"] = uncertainty
        return ad

    def evaluate(self, adata, label_key: str | None = None):
        key = label_key or self.label_key_
        if key not in adata.obs:
            raise KeyError(f"{key!r} not found in adata.obs")
        dist = self.predict_distribution(adata)
        y_true = np.asarray(adata.obs[key].astype(str))
        y_pred = self.classes_[dist.argmax(axis=1)]
        metrics = classification_metrics(y_true, y_pred)
        try:
            y_onehot, _, _ = labels_to_onehot(y_true, classes=self.classes_)
            metrics.update(distribution_metrics(y_onehot, dist))
        except ValueError:
            pass
        return metrics
