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

pipe = AnnotationPipeline(model="concentration", n_top_genes=2000, epochs=40)
pipe.fit(ref, label_key="cell_type")

query = pipe.annotate(query)
print(query.obs["scldl_pred"].head())
print(query.obsm["X_scldl"][:5])
print(pipe.evaluate(query, label_key="cell_type"))
```

`model` can be:

| Name | Class | Role |
|---|---|---|
| `mlp` | `MLPBaseline` | Softmax classifier baseline |
| `concentration` | `ConcentrationLE` | Dirichlet / evidential head (default) |
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
  data.py          # AnnData preprocess + gene alignment
  metrics.py       # accuracy / F1 and LDL distances
  models/          # trainers with .fit / .predict
tests/             # pytest
experiments/       # longer research scripts
docs/notes/        # paper notes and old design docs
```

## Metrics

- Classification: accuracy, macro-F1 on argmax labels
- Distributions (when you have a true simplex): Chebyshev, Clark, Canberra, cosine, intersection, KL, MSE

The toy recovery experiment is `python experiments/benchmark_toy.py`.

## What was removed

Broken or unreferenced pieces from the previous dump: DiffLEVI (CARD code was never in the repo), LESC, fictional `LabelEnhancerTrainer` / `scDataset` / `ConcentrationLDL` modules, Streamlit MNIST mixup apps, and duplicate training scripts that imported missing files.

## Next work

1. Train `AnnotationPipeline` on a public reference (e.g. tonsil or PBMC) and score a held-out query, including cross-dataset gene alignment.
2. Compare against label transfer / scANVI, not only an internal MLP.
3. Optional: graph smoothing of predicted distributions on the kNN graph; Negative Binomial reconstruction for RNA.
4. Do not bring back diffusion or LESC until those dependencies live in this repo and have tests.
