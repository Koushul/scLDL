# Implementation Plan - KL Divergence Schedulers

The user wants to add "learning rate schedulers for the KL divergence term". In the context of VAEs and Information Bottleneck, this refers to **KL Annealing** (scheduling the $\beta$ hyperparameter).

## Proposed Changes

### `src/scLDL/models/trainer.py`

1.  **Define Scheduler Classes**:
    *   `BetaScheduler`: Base class.
    *   `LinearBetaScheduler`: Linearly increases $\beta$ from `start` to `stop` over `n_cycles` or epochs.
    *   `CyclicBetaScheduler`: Cyclical annealing (e.g., for fixing "KL vanishing").

2.  **Update `LabelEnhancerTrainer`**:
    *   Update `__init__` to accept a `beta_scheduler` object or configuration.
    *   Update `train` loop to call `scheduler.step()` and update `self.beta`.
    *   Log the current `beta` value.

## Verification Plan

### Automated Tests
*   Run a dummy training loop and verify `beta` changes over epochs as expected.
