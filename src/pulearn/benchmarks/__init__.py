"""Benchmark utilities for pulearn.

Provides synthetic PU data generators, lightweight real-dataset loaders,
and a deterministic benchmark runner that emits markdown/CSV tables.

Typical usage::

    from pulearn.benchmarks import make_pu_dataset, BenchmarkRunner
    X, y_true, y_pu = make_pu_dataset(n_samples=500, pi=0.3, c=0.5,
                                       random_state=0)
    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders={"my_estimator": lambda: ...}
    )
    runner.to_csv("results.csv")
    print(runner.to_markdown())

"""

from .datasets import (  # noqa: F401
    load_pu_breast_cancer,
    make_pu_blobs,
    make_pu_dataset,
)
from .runner import (  # noqa: F401
    BenchmarkResult,
    BenchmarkRunner,
)
