# Experiments

Scripts for longer runs. They are not the unit-test suite.

- `benchmark_toy.py` — recover known label distributions from hard labels (the intended LDL check).

Single-cell training should go through `AnnotationPipeline` in `src/scLDL/pipeline.py` rather than one-off copies of model training loops.
