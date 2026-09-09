# scLDL

Single-cell **label distribution learning**: predict a distribution over cell types instead of a single hard label.

This repo was cleaned up so the next step is a real annotation pipeline, not more disconnected MNIST scripts.

## Install

```bash
pip install -e ".[dev]"
```

Python 3.10+ and PyTorch are required.

## Annotation pipeline

Use models that map expression `X` to a label distribution **without** needing labels at inference:

```python
from scLDL import AnnotationPipeline
import scanpy as sc

ref = sc.read_h5ad("reference.h5ad")
query = sc.read_h5ad("query.h5ad")

pipe = AnnotationPipeline(n_top_genes=2000, epochs=40)
pipe.fit(ref, label_key="cell_type")

query = pipe.annotate(query)
print(query.obs["scldl_pred"].head())
print(query.obsm["X_scldl"][:5])
print(pipe.evaluate(query, label_key="cell_type"))
```

`model` can be:

| Name | Class | Role |
|---|---|---|
| `scldl` | `InterpretableLE` | Default scLDL: reference PCA, optional MNN, type/state LDL; spatial queries also get neighborhood graph refine |
| `interpretable` | `InterpretableLE` | Alias for `scldl` |
| `mlp` | `MLPBaseline` | Softmax classifier baseline |
| `concentration` | `ConcentrationLE` | Dirichlet / evidential head (gene space) |
| `state_concentration` | `StateConcentrationLE` | State-aware concentration model |
| `hybrid` | `HybridLEVI` | VAE + evidential head |
| `lible` | `LIBLE` | Label information bottleneck, `X` only |

Evidential models also write `query.obs["scldl_uncertainty"]`.

## Label enhancement (not annotation)

`LEVI` and `ImprovedLEVI` encode `q(z | x, l)`. They **need logical labels at predict time** and should not be used to annotate unlabeled query cells.

```python
from scLDL.models import LEVI

model = LEVI(n_features=n_genes, n_outputs=n_types)
model.fit(X_train, L_onehot)
soft = model.predict(X_train, L_onehot)
```

## Layout

```
src/scLDL/
  pipeline.py      # AnnotationPipeline
  benchmark/       # hold-out / cross-dataset annotation comparison
  data.py          # AnnData preprocess + gene alignment
  metrics.py       # accuracy / F1, calibration, and LDL distances
  models/          # trainers with .fit / .predict
tests/             # pytest
experiments/       # longer research scripts
docs/notes/        # paper notes and old design docs
```

## Annotation benchmark

Compare scLDL models to simple baselines and existing Python annotation tools on the **same genes and split**. HVGs and the scaler are fit on the training/reference cells only.

| Method | What it is |
|---|---|
| `majority` | Predict the training class frequencies |
| `logistic` | Multinomial logistic regression |
| `svm` | Linear SVM (softmax of decision scores) |
| `knn` | Distance-weighted kNN on scaled HVGs |
| `pca_knn` | PCA then kNN (Seurat/scanpy-style transfer) |
| `scanpy_ingest` | `scanpy.tl.ingest` label transfer |
| `scldl_mlp` / `scldl_concentration` / `scldl_hybrid` / `scldl_lible` | This package |
| `celltypist` | Optional (`pip install -e ".[bench]"`) |
| `scanvi` | Optional (`pip install -e ".[scanvi]"`); skipped unless raw counts are present |

```python
from scLDL import run_benchmark, summarize
import scanpy as sc

adata = sc.read_h5ad("reference.h5ad")
results = run_benchmark(adata, label_key="cell_type", n_repeats=3)
print(summarize(results))
```

CLI:

```bash
python experiments/run_annotation_benchmark.py --adata reference.h5ad --label cell_type --repeats 3 --out reports/annotation_benchmark.csv
python experiments/run_annotation_benchmark.py --adata ref.h5ad --query query.h5ad --label cell_type
python experiments/run_annotation_benchmark.py --adata reference.h5ad --noise-rate 0.2 --repeats 3
```

Reported metrics: accuracy, balanced accuracy, macro-F1, log loss, Brier, ECE, plus LDL distances against one-hot test labels. Neural/CellTypist methods use log-normalized HVGs; sklearn/ingest methods use the same genes after a train-fit `StandardScaler`. `scanpy_ingest` is skipped if Scanpy's ingest stack cannot import (for example a JAX/ml_dtypes mismatch); `pca_knn` is the same idea without that dependency.

## Metrics

- Classification: accuracy, balanced accuracy, macro-F1
- Probabilities: log loss, Brier, expected calibration error
- Distributions (when you have a true simplex): Chebyshev, Clark, Canberra, cosine, intersection, KL, MSE

Toy LDL recovery: `python experiments/benchmark_toy.py`.

Annotation comparison: `python experiments/run_annotation_benchmark.py --adata your.h5ad`.

## What was removed

Broken or unreferenced pieces from the previous dump: DiffLEVI (CARD code was never in the repo), LESC, fictional `LabelEnhancerTrainer` / `scDataset` / `ConcentrationLDL` modules, Streamlit MNIST mixup apps, and duplicate training scripts that imported missing files.

## Next work

1. Run `run_annotation_benchmark.py` on a public PBMC/tonsil dataset and keep the CSV in `reports/`.
2. Optional: graph smoothing of predicted distributions; Negative Binomial reconstruction for RNA.
3. Do not bring back diffusion or LESC until those dependencies live in this repo and have tests.
