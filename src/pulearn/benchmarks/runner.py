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
import datetime
import io
import sys
import time
import warnings
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from pulearn.benchmarks.experiment import ExperimentConfig

import numpy as np
import sklearn

from pulearn.benchmarks.datasets import make_pu_dataset

try:
    _PULEARN_VERSION: str = version("pulearn")
except PackageNotFoundError:
    _PULEARN_VERSION = "unknown"

# ---------------------------------------------------------------------------
# Metadata container
# ---------------------------------------------------------------------------


@dataclass
class RunMetadata:
    """Environment and configuration metadata for a benchmark run.

    Captured once when :class:`BenchmarkRunner` is instantiated so that
    results can be reproduced and compared across environments.

    Attributes
    ----------
    timestamp : str
        ISO 8601 UTC timestamp of runner creation (e.g.
        ``"2024-01-01T12:00:00Z"``).
    python_version : str
        Full Python version string (``sys.version``).
    pulearn_version : str
        Installed pulearn package version.
    numpy_version : str
        Installed NumPy version.
    sklearn_version : str
        Installed scikit-learn version.
    random_state : int or None
        Master seed passed to the runner.
    test_size : float
        Held-out fraction passed to the runner.

    """

    timestamp: str
    python_version: str
    pulearn_version: str
    numpy_version: str
    sklearn_version: str
    random_state: Optional[int]
    test_size: float

    def as_dict(self) -> dict:
        """Return a plain dictionary suitable for serialisation."""
        return {
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "pulearn_version": self.pulearn_version,
            "numpy_version": self.numpy_version,
            "sklearn_version": self.sklearn_version,
            "random_state": (
                "null" if self.random_state is None else str(self.random_state)
            ),
            "test_size": self.test_size,
        }

    def to_markdown(self) -> str:
        """Return a Markdown-formatted metadata block.

        Returns
        -------
        str
            Bullet-list block suitable for embedding above a results table.

        """
        lines = ["**Benchmark metadata**", ""]
        for key, value in self.as_dict().items():
            lines.append("- **{}**: {}".format(key, value))
        return "\n".join(lines)


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
    extra_metrics : dict
        Optional additional metrics computed by caller-supplied metric
        scorers.  Used by publishing layers such as the leaderboard exporter
        to keep benchmark policy outside the generic runner.
    problem_type : str
        Benchmark problem family.  Defaults to ``"PU_SCAR"``.
    warnings : list of str
        Non-fatal warnings captured while running this benchmark row.
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
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    problem_type: str = "PU_SCAR"
    warnings: List[str] = field(default_factory=list)
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
            "extra_metrics": dict(self.extra_metrics),
            "problem_type": self.problem_type,
            "warnings": list(self.warnings),
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
    "problem_type",
    "warnings",
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
        self._metadata = RunMetadata(
            timestamp=datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            python_version=sys.version,
            pulearn_version=_PULEARN_VERSION,
            numpy_version=np.__version__,
            sklearn_version=sklearn.__version__,
            random_state=random_state,
            test_size=test_size,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def results(self) -> List[BenchmarkResult]:
        """Return the accumulated list of :class:`BenchmarkResult` objects."""
        return list(self._results)

    @property
    def metadata(self) -> RunMetadata:
        """Return the :class:`RunMetadata` captured at runner creation."""
        return self._metadata

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
        problem_type: str = "PU_SCAR",
        metric_scorers: Optional[Dict[str, Callable]] = None,
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
        problem_type : str, default "PU_SCAR"
            Problem-family label stored in each result row.
        metric_scorers : dict or None, default None
            Optional mapping of metric name to callable.  Each callable is
            invoked with keyword arguments ``y_true``, ``y_pu``, ``y_pred``,
            ``y_score``, ``pi`` and ``c`` for the held-out split.  Metric
            ``ValueError`` exceptions are captured as row warnings and stored
            as ``NaN`` so one unsupported metric does not fail the entire run.

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
        actual_c = self._labeling_propensity(y_true, y_pu, fallback=c)

        X_train, X_test, y_true_train, y_true_test, y_pu_train, y_pu_test = (
            self._split(X, y_true, y_pu)
        )
        scorers = {} if metric_scorers is None else dict(metric_scorers)

        for name, builder in estimator_builders.items():
            result = self._run_one(
                name=name,
                dataset=dataset_name,
                pi=actual_pi,
                c=actual_c,
                n_samples=len(y_true),
                problem_type=problem_type,
                builder=builder,
                X_train=X_train,
                X_test=X_test,
                y_pu_train=y_pu_train,
                y_pu_test=y_pu_test,
                y_true_test=y_true_test,
                metric_scorers=scorers,
            )
            self._results.append(result)

        return self

    def save_run(
        self,
        config: "ExperimentConfig",
        *,
        results_dir: str = "results",
        run_id: Optional[str] = None,
    ) -> str:
        """Persist run artifacts to ``{results_dir}/{run_id}/``.

        Convenience wrapper around
        :func:`~pulearn.benchmarks.experiment.save_run_artifacts`.

        Parameters
        ----------
        config : ExperimentConfig
            Experiment configuration; validated before saving.
        results_dir : str, default ``"results"``
            Parent directory for all run artifacts.
        run_id : str or None, default None
            Override the auto-generated run ID.

        Returns
        -------
        str
            Absolute path to the created run directory.

        """
        from pulearn.benchmarks.experiment import save_run_artifacts

        return save_run_artifacts(
            self,
            config,
            results_dir=results_dir,
            run_id=run_id,
        )

    def to_csv(self, path: str) -> None:
        """Write results to *path* as a CSV file.

        Parameters
        ----------
        path : str
            File path to write.

        """
        fieldnames = self._csv_fieldnames()
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._results:
                writer.writerow(self._csv_row(r))

    def to_csv_string(self) -> str:
        """Return results as a CSV-formatted string."""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=self._csv_fieldnames())
        writer.writeheader()
        for r in self._results:
            writer.writerow(self._csv_row(r))
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

    def _csv_fieldnames(self) -> List[str]:
        """Return CSV field names including any extra metric columns."""
        metric_names = sorted(
            {
                metric_name
                for result in self._results
                for metric_name in result.extra_metrics
            }
        )
        return list(_CSV_FIELDNAMES) + metric_names

    def _csv_row(self, result: BenchmarkResult) -> dict:
        """Return a flattened CSV row for a benchmark result."""
        row = result.as_dict()
        extra_metrics = row.pop("extra_metrics")
        row["warnings"] = " | ".join(row["warnings"])
        row.update(extra_metrics)
        for key, value in list(row.items()):
            if isinstance(value, float) and not np.isnan(value):
                row[key] = _FLOAT_FMT.format(value)
        return row

    @staticmethod
    def _labeling_propensity(
        y_true: np.ndarray,
        y_pu: np.ndarray,
        *,
        fallback: float,
    ) -> float:
        """Return realised labeling propensity from truth and PU labels."""
        y_true_arr = np.asarray(y_true)
        y_pu_arr = np.asarray(y_pu)
        pos_mask = y_true_arr == 1
        n_pos = int(np.sum(pos_mask))
        if n_pos == 0:
            return float(fallback)
        n_labeled_pos = int(np.sum((y_pu_arr == 1) & pos_mask))
        return float(n_labeled_pos / n_pos)

    def _run_one(
        self,
        *,
        name: str,
        dataset: str,
        pi: float,
        c: float,
        n_samples: int,
        problem_type: str,
        builder: Callable,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_pu_train: np.ndarray,
        y_pu_test: np.ndarray,
        y_true_test: np.ndarray,
        metric_scorers: Dict[str, Callable],
    ) -> BenchmarkResult:
        """Fit one estimator and collect metrics, catching all exceptions."""
        result_warnings: List[str] = []

        def _metric_or_nan(label: str, func: Callable, **kwargs):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    value = float(func(**kwargs))
                except ValueError as exc:
                    result_warnings.append("{}: {}".format(label, exc))
                    return float("nan")
            for warning in caught:
                result_warnings.append("{}: {}".format(label, warning.message))
            return value

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
                message = (
                    "[{}] Degenerate predictions; F1/AUC may be undefined."
                ).format(name)
                result_warnings.append(message)
                warnings.warn(message, stacklevel=2)

            f1 = float(f1_score(y_true_test, y_pred, zero_division=0))
            try:
                auc = float(roc_auc_score(y_true_test, y_score))
            except ValueError:
                auc = float("nan")

            extra_metrics = {}
            metric_kwargs = {
                "y_true": y_true_test,
                "y_pu": y_pu_test,
                "y_pred": y_pred,
                "y_score": y_score,
                "pi": pi,
                "c": c,
            }
            for metric_name, scorer in metric_scorers.items():
                extra_metrics[metric_name] = _metric_or_nan(
                    metric_name,
                    scorer,
                    **metric_kwargs,
                )

            return BenchmarkResult(
                name=name,
                dataset=dataset,
                pi=pi,
                c=c,
                n_samples=n_samples,
                problem_type=problem_type,
                f1=f1,
                roc_auc=auc,
                fit_time_s=fit_time,
                predict_time_s=predict_time,
                extra_metrics=extra_metrics,
                warnings=result_warnings,
                error=None,
            )

        except Exception as exc:
            return BenchmarkResult(
                name=name,
                dataset=dataset,
                pi=pi,
                c=c,
                n_samples=n_samples,
                problem_type=problem_type,
                warnings=result_warnings,
                error=str(exc),
            )
