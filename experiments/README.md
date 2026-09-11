# Experiments

Scripts for longer runs. They are not the unit-test suite.

- `benchmark_toy.py` — recover known label distributions from hard labels.
- `run_annotation_benchmark.py` — hold-out or cross-dataset comparison of scLDL vs logistic/SVM/kNN/scanpy ingest.
- `pancreas_state_eval.py` — ConcentrationLE vs state-aware model on endocrinogenesis mixtures.
- `spatial_refmap_eval.py` — annotate Visium HD and Slide-seqV2 with default scLDL; compare expression-only vs spatial refine.
- `eval_slideseq_label_noise.py` — corrupt an increasing fraction of Slide-seqV2 RCTD singlet labels and score OOF discovery / correction.
