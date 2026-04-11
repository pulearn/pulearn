"""Benchmark utilities for pulearn.

Provides synthetic PU data generators, lightweight real-dataset loaders,
a deterministic benchmark runner that emits markdown/CSV tables, and a
minimal experiment configuration schema with artifact persistence.

Real dataset loaders (no external download required):

* :func:`load_pu_breast_cancer` – UCI Breast Cancer Wisconsin
  (569 samples, 30 features, BSD-3-Clause via scikit-learn).
* :func:`load_pu_wine` – UCI Wine Recognition
  (178 samples, 13 features, BSD-3-Clause via scikit-learn).
* :func:`load_pu_digits` – UCI Optical Recognition of Handwritten Digits
  (1797 samples, 64 features, BSD-3-Clause via scikit-learn).

Typical usage::

    from pulearn.benchmarks import (
        make_pu_dataset,
        BenchmarkRunner,
        ExperimentConfig,
        save_run_artifacts,
    )
    X, y_true, y_pu = make_pu_dataset(n_samples=500, pi=0.3, c=0.5,
                                       random_state=0)
    cfg = ExperimentConfig(
        dataset="synthetic", model="elkanoto", metric="f1", seed=42
    )
    cfg.validate()
    runner = BenchmarkRunner(random_state=cfg.seed)
    runner.run(estimator_builders={"my_estimator": lambda: ...})
    runner.to_csv("results.csv")
    print(runner.to_markdown())
    run_dir = runner.save_run(cfg, results_dir="results")

"""

from .datasets import (  # noqa: F401
    PUDatasetMetadata,
    load_pu_breast_cancer,
    load_pu_digits,
    load_pu_wine,
    make_pu_blobs,
    make_pu_dataset,
)
from .experiment import (  # noqa: F401
    ExperimentConfig,
    load_run_artifacts,
    save_run_artifacts,
)
from .runner import (  # noqa: F401
    BenchmarkResult,
    BenchmarkRunner,
    RunMetadata,
)
