# Experiments

Scripts for longer runs. They are not the unit-test suite.

- `benchmark_toy.py` — recover known label distributions from hard labels.
- `run_annotation_benchmark.py` — hold-out or cross-dataset comparison of scLDL vs logistic/SVM/kNN/scanpy ingest.
- `eval_celltypist_benchmark.py` — hold-out CellTypist vs scLDL (plus logistic / PCA-kNN) on pancreas, hippocampus RCTD singlets, and lymph-node Slide-seqV2.
- `pancreas_state_eval.py` — ConcentrationLE vs state-aware model on endocrinogenesis mixtures.
- `spatial_refmap_eval.py` — annotate Visium HD and Slide-seqV2 with default scLDL; compare expression-only vs spatial refine.
- `eval_slideseq_label_noise.py` — corrupt labels on hippocampus RCTD singlets or mouse lymph-node Slide-seqV2 (`--dataset hippo|lymphnode`); `--variant compare` tests `label_smooth` off vs on.
