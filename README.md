# Genetic Optimization Engine (GOE) - frozen

GOE was a small genetic-algorithm library (2024). It is no longer developed.

In November 2026 a survey of existing tools (Optuna, SMAC3, Nevergrad, pymoo, pycma, DEAP, PyGAD, Ax, Ray Tune
and others) showed that the optimisation algorithms themselves are well covered, and that at budgets of 100-200
noisy evaluations a plain GA loses to TPE or Bayesian optimisation. What is missing is the layer around the
optimiser: a file-driven ask/tell command line for evaluations that run outside Python (for example Slurm jobs),
deterministic resume, an append-only history, and a protocol for noisy scores (re-evaluation with new seeds,
mean / n / lower confidence bound).

That layer is a new project: **nofluke** (`~/Programming/nofluke`, GitHub `Horef/nofluke`).

The code here is kept as it was: `GeneticOptimizationEngine.py` (the `GOE` class), `fitness_wrappers.py`,
`projections.py`, `multiprocessing_helpers.py` and `testing.py`.
