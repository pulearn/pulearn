# Benchmarks

This directory holds contributor-facing benchmark scaffolds until the full
benchmark harness lands.

Use `benchmarks/templates/benchmark_entry_template.py.tmpl` as the starting
point for adding a new learner to benchmark tables and smoke runs.

---

## Reproducibility quickstart

`pulearn.benchmarks` ships a lightweight experiment configuration schema and
artifact-persistence layer — **no heavy external framework required**.

### 1 — Define an experiment configuration

```python
from pulearn.benchmarks import ExperimentConfig

cfg = ExperimentConfig(
    dataset="synthetic",  # dataset identifier
    model="elkanoto",     # algorithm identifier
    metric="f1",          # primary evaluation metric
    seed=42,              # master random seed
    pi=0.3,               # positive-class prior
    c=0.5,                # labeling propensity
    n_samples=500,
    tags=["baseline"],    # optional free-form labels
)
cfg.validate()            # raises ValueError on invalid inputs
```

### 2 — Run the benchmark

```python
from pulearn.benchmarks import BenchmarkRunner
from pulearn import ElkanotoPuClassifier
from sklearn.linear_model import LogisticRegression

def build():
    return ElkanotoPuClassifier(
        estimator=LogisticRegression(max_iter=200, random_state=cfg.seed),
        hold_out_ratio=0.1,
        random_state=cfg.seed,
    )

runner = BenchmarkRunner(random_state=cfg.seed)
runner.run({"elkanoto": build}, n_samples=cfg.n_samples, pi=cfg.pi, c=cfg.c)
print(runner.to_markdown())
```

### 3 — Save run artifacts

```python
run_dir = runner.save_run(cfg, results_dir="results")
print("Artifacts saved to:", run_dir)
```

Or using the standalone function:

```python
from pulearn.benchmarks import save_run_artifacts

run_dir = save_run_artifacts(runner, cfg, results_dir="results")
```

### 4 — Load and inspect saved artifacts

```python
from pulearn.benchmarks import load_run_artifacts

arts = load_run_artifacts(run_dir)
print(arts["config"])        # ExperimentConfig instance
print(arts["metadata"])      # environment / version info dict
print(arts["results"])       # list of result dicts (from results.csv)
```

---

## `results/` directory layout

Each call to `save_run_artifacts` (or `runner.save_run`) creates one
subdirectory:

```
results/
    {run_id}/
        config.json     ← ExperimentConfig fields (JSON)
        metadata.json   ← RunMetadata: timestamp, python/numpy/sklearn/
        │                  pulearn versions, random_state, test_size
        results.csv     ← BenchmarkResult rows (one row per estimator)
        summary.json    ← combined document: run_id + config + metadata +
                           results list
```

The `run_id` is auto-generated as
`{YYYYMMDD_HHMMSS}_{dataset}_{model}_{seed}` (UTC timestamp) or can be
overridden via the `run_id=` keyword argument.

### `config.json` fields

| Field        | Type      | Description                                    |
|-------------|-----------|------------------------------------------------|
| `dataset`   | `str`     | Dataset identifier                             |
| `model`     | `str`     | Algorithm identifier                           |
| `metric`    | `str`     | Primary metric (`f1`, `roc_auc`, …)            |
| `seed`      | `int`     | Master random seed                             |
| `n_samples` | `int`     | Number of samples (synthetic datasets)         |
| `n_features`| `int`     | Number of features (synthetic datasets)        |
| `pi`        | `float`   | Positive-class prior in (0, 1)                 |
| `c`         | `float`   | Labeling propensity in (0, 1]                  |
| `test_size` | `float`   | Held-out fraction in (0, 1)                    |
| `tags`      | `list`    | Optional free-form labels                      |

### `metadata.json` fields

| Field             | Description                              |
|-------------------|------------------------------------------|
| `timestamp`       | ISO 8601 UTC creation time               |
| `python_version`  | `sys.version` string                     |
| `pulearn_version` | Installed pulearn version                |
| `numpy_version`   | Installed NumPy version                  |
| `sklearn_version` | Installed scikit-learn version           |
| `random_state`    | Master seed (string ``"null"`` if None)  |
| `test_size`       | Held-out fraction                        |

