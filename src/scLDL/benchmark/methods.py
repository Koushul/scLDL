from __future__ import annotations

import time

import numpy as np

from scLDL.benchmark.protocol import BenchmarkData
from scLDL.metrics import score_annotations


def expand_proba(proba, model_classes, classes):
    proba = np.asarray(proba, dtype=np.float64)
    out = np.zeros((proba.shape[0], len(classes)), dtype=np.float64)
    index = {str(c): i for i, c in enumerate(classes)}
    for j, c in enumerate(model_classes):
        i = index.get(str(c))
        if i is not None:
            out[:, i] = proba[:, j]
    row = out.sum(axis=1, keepdims=True)
    missing = row[:, 0] <= 0
    if np.any(missing):
        out[missing] = 1.0 / len(classes)
        row = out.sum(axis=1, keepdims=True)
    return out / row


def _softmax(x):
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Majority:
    name = "majority"
    features = "scaled"

    def fit(self, X, y, classes):
        self.classes_ = np.asarray(classes)
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=np.float64)
        if counts.sum() == 0:
            counts = np.ones_like(counts)
        self.prior_ = counts / counts.sum()
        return self

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))


class Logistic:
    name = "logistic"
    features = "scaled"

    def fit(self, X, y, classes):
        from sklearn.linear_model import LogisticRegression

        self.classes_ = np.asarray(classes)
        self.model_ = LogisticRegression(max_iter=2000, solver="lbfgs")
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        return expand_proba(self.model_.predict_proba(X), self.model_.classes_, self.classes_)


class LinearSVM:
    name = "svm"
    features = "scaled"

    def fit(self, X, y, classes):
        from sklearn.svm import LinearSVC

        self.classes_ = np.asarray(classes)
        self.model_ = LinearSVC()
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        scores = self.model_.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
            model_classes = self.model_.classes_
        else:
            model_classes = self.model_.classes_
        return expand_proba(_softmax(scores), model_classes, self.classes_)


class KNN:
    name = "knn"
    features = "scaled"

    def __init__(self, n_neighbors=15):
        self.n_neighbors = n_neighbors

    def fit(self, X, y, classes):
        from sklearn.neighbors import KNeighborsClassifier

        self.classes_ = np.asarray(classes)
        k = max(1, min(self.n_neighbors, len(X) - 1))
        self.model_ = KNeighborsClassifier(n_neighbors=k, weights="distance")
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        return expand_proba(self.model_.predict_proba(X), self.model_.classes_, self.classes_)


class PCAKNN:
    name = "pca_knn"
    features = "scaled"

    def __init__(self, n_components=50, n_neighbors=15):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit(self, X, y, classes):
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline

        self.classes_ = np.asarray(classes)
        n_comp = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        k = max(1, min(self.n_neighbors, len(X) - 1))
        self.model_ = Pipeline(
            [
                ("pca", PCA(n_components=n_comp, random_state=0)),
                ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
            ]
        )
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        knn = self.model_.named_steps["knn"]
        return expand_proba(self.model_.predict_proba(X), knn.classes_, self.classes_)


class ScanpyIngest:
    name = "scanpy_ingest"
    features = "scaled"

    def __init__(self, n_neighbors=15, n_pcs=50):
        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs

    def fit(self, X, y, classes):
        import anndata as ad
        import scanpy as sc

        self.classes_ = np.asarray(classes)
        self.label_key_ = "label"
        n_pcs = max(1, min(self.n_pcs, X.shape[1] - 1, X.shape[0] - 1))
        n_neighbors = max(2, min(self.n_neighbors, X.shape[0] - 1))
        ref = ad.AnnData(X)
        ref.obs[self.label_key_] = np.asarray(y).astype(str)
        sc.tl.pca(ref, n_comps=n_pcs)
        sc.pp.neighbors(ref, n_neighbors=n_neighbors, n_pcs=n_pcs)
        self.ref_ = ref
        return self

    def predict_proba(self, X):
        import anndata as ad
        import scanpy as sc

        query = ad.AnnData(X)
        sc.tl.ingest(query, self.ref_, obs=self.label_key_, embedding_method="pca")
        pred = np.asarray(query.obs[self.label_key_].astype(str))
        onehot = np.zeros((len(pred), len(self.classes_)), dtype=np.float64)
        index = {c: i for i, c in enumerate(self.classes_)}
        for i, p in enumerate(pred):
            j = index.get(p)
            if j is not None:
                onehot[i, j] = 1.0
            else:
                onehot[i] = 1.0 / len(self.classes_)
        return onehot


class SCLDLMethod:
    features = "log"

    def __init__(self, model: str, **kwargs):
        self.name = f"scldl_{model}"
        self.model_name = model
        self.kwargs = kwargs

    def fit(self, X, y, classes):
        from scLDL.models import ANNOTATION_MODELS

        self.classes_ = np.asarray(classes)
        cls = ANNOTATION_MODELS[self.model_name]
        onehot = np.zeros((len(y), len(self.classes_)), dtype=np.float32)
        index = {c: i for i, c in enumerate(self.classes_)}
        for i, label in enumerate(y):
            onehot[i, index[str(label)]] = 1.0
        params = dict(
            n_features=X.shape[1],
            n_outputs=len(self.classes_),
            verbose=False,
        )
        params.update(self.kwargs)
        self.estimator_ = cls(**params)
        self.estimator_.fit(X, onehot)
        return self

    def predict_proba(self, X):
        return expand_proba(self.estimator_.predict(X), self.classes_, self.classes_)


class CellTypistMethod:
    name = "celltypist"
    features = "log"

    def fit(self, X, y, classes):
        import celltypist

        self.classes_ = np.asarray(classes)
        self.model_ = celltypist.train(X, np.asarray(y).astype(str), use_SGD=False, feature_selection=False, check_expression=False)
        return self

    def predict_proba(self, X):
        import anndata as ad
        import celltypist

        query = ad.AnnData(X)
        pred = celltypist.annotate(query, model=self.model_, majority_voting=False)
        probs = pred.probability_matrix.reindex(columns=list(self.classes_), fill_value=0.0).to_numpy(dtype=np.float64)
        return expand_proba(probs, self.classes_, self.classes_)


class SCANVIMethod:
    name = "scanvi"
    features = "counts"

    def __init__(self, max_epochs=50, n_latent=10):
        self.max_epochs = max_epochs
        self.n_latent = n_latent

    def fit(self, X, y, classes):
        import anndata as ad
        import scvi

        self.classes_ = np.asarray(classes)
        unlabeled = "__unlabeled__"
        ref = ad.AnnData(X)
        ref.obs["label"] = np.asarray(y).astype(str)
        scvi.model.SCANVI.setup_anndata(ref, labels_key="label", unlabeled_category=unlabeled)
        self.model_ = scvi.model.SCANVI(ref, n_latent=self.n_latent)
        self.model_.train(max_epochs=self.max_epochs, early_stopping=True)
        return self

    def predict_proba(self, X):
        import anndata as ad

        query = ad.AnnData(X)
        pred = self.model_.predict(query, soft=True)
        if hasattr(pred, "to_numpy"):
            pred = pred.to_numpy()
        model_classes = getattr(self.model_, "adata", None)
        labels = self.model_.adata_manager.get_state_registry("labels").categorical_mapping
        labels = [c for c in labels if c != "__unlabeled__"]
        return expand_proba(np.asarray(pred), labels, self.classes_)


CORE_METHODS = ["majority", "logistic", "svm", "knn", "pca_knn", "scanpy_ingest", "scldl_mlp", "scldl_concentration"]
ALL_METHODS = CORE_METHODS + ["scldl_hybrid", "scldl_lible", "scldl_interpretable", "celltypist", "scanvi"]


def build_method(name: str, epochs: int = 40, n_hidden: int = 128, batch_size: int = 64):
    if name == "majority":
        return Majority()
    if name == "logistic":
        return Logistic()
    if name == "svm":
        return LinearSVM()
    if name == "knn":
        return KNN()
    if name == "pca_knn":
        return PCAKNN()
    if name == "scanpy_ingest":
        return ScanpyIngest()
    if name == "scldl_mlp":
        return SCLDLMethod("mlp", n_hidden=n_hidden, epochs=epochs, batch_size=batch_size)
    if name == "scldl_concentration":
        return SCLDLMethod("concentration", n_hidden=n_hidden, epochs=epochs, batch_size=batch_size)
    if name == "scldl_interpretable":
        return SCLDLMethod(
            "interpretable",
            n_hidden=n_hidden,
            epochs=epochs,
            batch_size=batch_size,
            mixup_alpha=0.0,
            peak_weight=0.15,
            lineage_weight=0.0,
            manifold_weight=0.0,
            vacuity_weight=0.0,
        )
    if name == "scldl_hybrid":
        return SCLDLMethod("hybrid", n_hidden=n_hidden, epochs=epochs, batch_size=batch_size, alpha=0.01)
    if name == "scldl_lible":
        return SCLDLMethod("lible", n_hidden=n_hidden, epochs=epochs, batch_size=batch_size)
    if name == "celltypist":
        return CellTypistMethod()
    if name == "scanvi":
        return SCANVIMethod(max_epochs=max(20, epochs))
    raise ValueError(f"Unknown method {name!r}. Choose from {ALL_METHODS}")


def method_available(name: str, data: BenchmarkData) -> str | None:
    if name == "celltypist":
        try:
            import celltypist  # noqa: F401
        except ImportError:
            return "celltypist is not installed"
    if name == "scanvi":
        try:
            import scvi  # noqa: F401
        except ImportError:
            return "scvi-tools is not installed"
        if data.X_train_counts is None:
            return "scANVI needs raw counts; the input looks already normalized"
    return None


def features_for(method, data: BenchmarkData):
    kind = getattr(method, "features", "scaled")
    if kind == "log":
        return data.X_train_log, data.X_test_log
    if kind == "counts":
        if data.X_train_counts is None:
            raise RuntimeError("counts features are unavailable")
        return data.X_train_counts, data.X_test_counts
    return data.X_train_scaled, data.X_test_scaled


def run_method(method, data: BenchmarkData) -> dict:
    x_train, x_test = features_for(method, data)
    t0 = time.perf_counter()
    method.fit(x_train, data.y_train, data.classes)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = method.predict_proba(x_test)
    pred_s = time.perf_counter() - t1
    metrics, _ = score_annotations(data.y_test, proba, data.classes)
    metrics.update(
        {
            "method": method.name,
            "fit_seconds": float(fit_s),
            "predict_seconds": float(pred_s),
            "n_features": int(x_train.shape[1]),
            "n_train": int(x_train.shape[0]),
            "n_classes": int(len(data.classes)),
            "status": "ok",
            "error": "",
        }
    )
    return metrics
