"""Deterministic benchmark runner for PU learning algorithms.

The runner executes a configurable set of PU estimators against one or more
datasets and produces a results table that can be serialised to CSV or
formatted as a Markdown table.

Typical usage::

    from sklearn.linear_model import LogisticRegression
    from pulearn import ElkanotoPuClassifier
    from pulearn.benchmarks import BenchmarkRunner, make_pu_dataset

    def build_elkanoto():
        return ElkanotoPuClassifier(
            estimator=LogisticRegression(), hold_out_ratio=0.1
        )

    runner = BenchmarkRunner(random_state=42)
    runner.run(
        estimator_builders={"elkanoto": build_elkanoto},
        n_samples=300,
        pi=0.3,
        c=0.5,
    )
    print(runner.to_markdown())
    runner.to_csv("results.csv")

"""

from __future__ import annotations

import csv
import io
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from pulearn.benchmarks.datasets import make_pu_dataset

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result record for a single estimator / dataset combination.

    Attributes
    ----------
    name : str
        Human-readable algorithm name.
    dataset : str
        Name of the dataset used.
    pi : float
        True class prior used for generation.
    c : float
        Labeling propensity used for generation.
    n_samples : int
        Total samples in the benchmark run.
    f1 : float
        F1 score on the held-out test set (``y_true``-based).
    roc_auc : float
        ROC-AUC on the held-out test set.
    fit_time_s : float
        Wall-clock fit time in seconds.
    predict_time_s : float
        Wall-clock predict time in seconds.
    error : str or None
        Non-empty when the run failed; contains the exception message.

    """

    name: str
    dataset: str
    pi: float
    c: float
    n_samples: int
    f1: float = float("nan")
    roc_auc: float = float("nan")
    fit_time_s: float = float("nan")
    predict_time_s: float = float("nan")
    error: Optional[str] = None

    def as_dict(self) -> dict:
        """Return a plain dictionary suitable for CSV/JSON serialisation."""
        return {
            "name": self.name,
            "dataset": self.dataset,
            "pi": self.pi,
            "c": self.c,
            "n_samples": self.n_samples,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "fit_time_s": self.fit_time_s,
            "predict_time_s": self.predict_time_s,
            "error": self.error or "",
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_FLOAT_FMT = "{:.4f}"
_CSV_FIELDNAMES = [
    "name",
    "dataset",
    "pi",
    "c",
    "n_samples",
    "f1",
    "roc_auc",
    "fit_time_s",
    "predict_time_s",
    "error",
]


class BenchmarkRunner:
    """Deterministic benchmark runner for PU learning algorithms.

    Parameters
    ----------
    random_state : int or None, default 42
        Master seed.  All data generation and train/test splits derive their
        seeds from this value so results are fully reproducible.
    test_size : float, default 0.3
        Fraction of data held out for evaluation.

    """

    def __init__(
        self,
        random_state: Optional[int] = 42,
        test_size: float = 0.3,
    ) -> None:
        """Initialise a BenchmarkRunner."""
        self.random_state = random_state
        self.test_size = test_size
        self._results: List[BenchmarkResult] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def results(self) -> List[BenchmarkResult]:
        """Return the accumulated list of :class:`BenchmarkResult` objects."""
        return list(self._results)

    def run(
        self,
        estimator_builders: Dict[str, Callable],
        *,
        n_samples: int = 500,
        n_features: int = 20,
        n_informative: int = 5,
        pi: float = 0.3,
        c: float = 0.5,
        corruption: float = 0.0,
        class_sep: float = 1.0,
        dataset_name: str = "synthetic",
        X: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        y_pu: Optional[np.ndarray] = None,
    ) -> "BenchmarkRunner":
        """Run all estimators on a single dataset configuration.

        If ``X``, ``y_true``, and ``y_pu`` are all provided the pre-built
        arrays are used directly (real-dataset path).  Otherwise a synthetic
        dataset is generated from the remaining keyword arguments.  Providing
        only a subset of the three arrays raises a *ValueError*.

        Parameters
        ----------
        estimator_builders : dict
            Mapping of *name* → *zero-argument callable* that returns a
            fresh unfitted estimator.
        n_samples, n_features, n_informative : int
            Passed to :func:`~pulearn.benchmarks.make_pu_dataset` when
            generating synthetic data.
        pi, c, corruption, class_sep : float
            Dataset generation parameters.
        dataset_name : str, default "synthetic"
            Label stored in each :class:`BenchmarkResult`.
        X, y_true, y_pu : ndarray or None
            Pre-built arrays (skip generation when all three are supplied).
            Must all be supplied together or all omitted.

        Returns
        -------
        self : BenchmarkRunner
            Allows method chaining.

        Raises
        ------
        ValueError
            If exactly one or two of ``X``, ``y_true``, ``y_pu`` are
            provided (partial pre-built data).

        """
        provided = sum(a is not None for a in (X, y_true, y_pu))
        if provided not in (0, 3):
            raise ValueError(
                "X, y_true, and y_pu must all be provided together or all "
                "omitted.  Got {:d}/3 arrays.".format(provided)
            )
        if provided == 0:
            X, y_true, y_pu = make_pu_dataset(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                pi=pi,
                c=c,
                corruption=corruption,
                class_sep=class_sep,
                random_state=self.random_state,
            )
            actual_pi = float(y_true.mean())
        else:
            actual_pi = float(y_true.mean())

        X_train, X_test, y_true_train, y_true_test, y_pu_train, y_pu_test = (
            self._split(X, y_true, y_pu)
        )

        for name, builder in estimator_builders.items():
            result = self._run_one(
                name=name,
                dataset=dataset_name,
                pi=actual_pi,
                c=c,
                n_samples=len(y_true),
                builder=builder,
                X_train=X_train,
                X_test=X_test,
                y_pu_train=y_pu_train,
                y_true_test=y_true_test,
            )
            self._results.append(result)

        return self

    def to_csv(self, path: str) -> None:
        """Write results to *path* as a CSV file.

        Parameters
        ----------
        path : str
            File path to write.

        """
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()
            for r in self._results:
                row = r.as_dict()
                # Format floats for readability.
                for k in ("f1", "roc_auc", "fit_time_s", "predict_time_s"):
                    v = row[k]
                    if isinstance(v, float) and not np.isnan(v):
                        row[k] = _FLOAT_FMT.format(v)
                writer.writerow(row)

    def to_csv_string(self) -> str:
        """Return results as a CSV-formatted string."""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for r in self._results:
            row = r.as_dict()
            for k in ("f1", "roc_auc", "fit_time_s", "predict_time_s"):
                v = row[k]
                if isinstance(v, float) and not np.isnan(v):
                    row[k] = _FLOAT_FMT.format(v)
            writer.writerow(row)
        return buf.getvalue()

    def to_markdown(self) -> str:
        """Return results as a GitHub-flavoured Markdown table string.

        Returns
        -------
        str
            Markdown table with one row per :class:`BenchmarkResult`.

        """
        headers = [
            "Algorithm",
            "Dataset",
            "pi",
            "c",
            "n",
            "F1",
            "ROC-AUC",
            "Fit (s)",
            "Predict (s)",
            "Error",
        ]
        rows: List[Sequence[str]] = []
        for r in self._results:

            def _fmt(v: float) -> str:
                if np.isnan(v):
                    return "—"
                return _FLOAT_FMT.format(v)

            rows.append(
                [
                    r.name,
                    r.dataset,
                    _FLOAT_FMT.format(r.pi),
                    _FLOAT_FMT.format(r.c),
                    str(r.n_samples),
                    _fmt(r.f1),
                    _fmt(r.roc_auc),
                    _fmt(r.fit_time_s),
                    _fmt(r.predict_time_s),
                    r.error or "",
                ]
            )

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        def _pad(cells: Sequence[str]) -> str:
            return (
                "| "
                + " | ".join(
                    str(c).ljust(col_widths[i]) for i, c in enumerate(cells)
                )
                + " |"
            )

        separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
        lines = [_pad(headers), separator] + [_pad(r) for r in rows]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pu: np.ndarray,
    ):
        """Deterministic train/test split based on *random_state*."""
        from sklearn.model_selection import train_test_split

        return train_test_split(
            X,
            y_true,
            y_pu,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_pu,
        )

    def _run_one(
        self,
        *,
        name: str,
        dataset: str,
        pi: float,
        c: float,
        n_samples: int,
        builder: Callable,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_pu_train: np.ndarray,
        y_true_test: np.ndarray,
    ) -> BenchmarkResult:
        """Fit one estimator and collect metrics, catching all exceptions."""
        try:
            from sklearn.metrics import f1_score, roc_auc_score

            estimator = builder()

            t0 = time.perf_counter()
            estimator.fit(X_train, y_pu_train)
            fit_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X_test)
                # Guard against single-column output (degenerate training).
                if proba.ndim == 2 and proba.shape[1] > 1:
                    y_score = proba[:, 1]
                else:
                    y_score = proba.ravel()
                y_pred = (y_score >= 0.5).astype(int)
            elif hasattr(estimator, "decision_function"):
                y_score = estimator.decision_function(X_test)
                y_pred = (y_score >= 0.0).astype(int)
            else:
                y_pred = np.where(estimator.predict(X_test) == 1, 1, 0)
                y_score = y_pred.astype(float)
            predict_time = time.perf_counter() - t1

            # Guard against degenerate predictions.
            unique_pred = np.unique(y_pred)
            unique_true = np.unique(y_true_test)
            if len(unique_pred) < 2 or len(unique_true) < 2:
                warnings.warn(
                    "[{}] Degenerate predictions; "
                    "F1/AUC may be undefined.".format(name),
                    stacklevel=2,
                )

            f1 = float(f1_score(y_true_test, y_pred, zero_division=0))
            try:
                auc = float(roc_auc_score(y_true_test, y_score))
            except ValueError:
                auc = float("nan")

            return BenchmarkResult(
                name=name,
                dataset=dataset,
                pi=pi,
                c=c,
                n_samples=n_samples,
                f1=f1,
                roc_auc=auc,
                fit_time_s=fit_time,
                predict_time_s=predict_time,
                error=None,
            )

        except Exception as exc:
            return BenchmarkResult(
                name=name,
                dataset=dataset,
                pi=pi,
                c=c,
                n_samples=n_samples,
                error=str(exc),
            )
