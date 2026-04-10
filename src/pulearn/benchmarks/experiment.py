"""Experiment configuration schema and artifact persistence for pulearn.

This module provides a lightweight, framework-free mechanism for capturing
experiment parameters, logging run metadata, and persisting artifacts in a
structured ``results/`` directory layout.

Directory structure produced by :func:`save_run_artifacts`::

    results/
        {run_id}/
            config.json    ← ExperimentConfig fields
            metadata.json  ← RunMetadata (environment / versions)
            results.csv    ← BenchmarkResult rows
            summary.json   ← combined config + metadata + results

Typical usage::

    from pulearn.benchmarks import (
        BenchmarkRunner,
        ExperimentConfig,
        save_run_artifacts,
    )
    from sklearn.linear_model import LogisticRegression
    from pulearn import ElkanotoPuClassifier

    cfg = ExperimentConfig(
        dataset="synthetic",
        model="elkanoto",
        metric="f1",
        seed=42,
        pi=0.3,
        c=0.5,
    )
    cfg.validate()  # raises ValueError on bad inputs

    def build():
        return ElkanotoPuClassifier(
            estimator=LogisticRegression(max_iter=200, random_state=cfg.seed),
            hold_out_ratio=0.1,
            random_state=cfg.seed,
        )

    runner = BenchmarkRunner(random_state=cfg.seed)
    runner.run({"elkanoto": build}, n_samples=300, pi=cfg.pi, c=cfg.c)

    run_dir = save_run_artifacts(runner, cfg, results_dir="results")
    print("Artifacts saved to:", run_dir)

"""

from __future__ import annotations

import csv
import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from pulearn.benchmarks.runner import BenchmarkRunner

__all__ = [
    "ExperimentConfig",
    "save_run_artifacts",
    "load_run_artifacts",
]

# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

_VALID_METRICS = frozenset(
    ["f1", "roc_auc", "accuracy", "precision", "recall", "f1_macro"]
)


@dataclass
class ExperimentConfig:
    """Lightweight experiment configuration schema.

    Captures the minimal set of parameters required to reproduce a PU
    learning experiment.  All fields are plain Python types (no external
    framework required).

    Attributes
    ----------
    dataset : str
        Dataset identifier (e.g. ``"synthetic"``, ``"breast_cancer"``).
    model : str
        Algorithm / estimator identifier (e.g. ``"elkanoto"``).
    metric : str
        Primary evaluation metric key used to assess results.  Must be one
        of: ``"f1"``, ``"roc_auc"``, ``"accuracy"``, ``"precision"``,
        ``"recall"``, ``"f1_macro"``.
    seed : int
        Master random seed.  All data generation, splits, and estimator
        random states should derive from this value.
    n_samples : int, default 500
        Total number of samples to generate (synthetic datasets only).
    n_features : int, default 20
        Number of features (synthetic datasets only).
    pi : float, default 0.3
        True positive-class prior in **(0, 1)**.
    c : float, default 0.5
        Labeling propensity in **(0, 1]**.
    test_size : float, default 0.3
        Fraction of data held out for evaluation; must be in **(0, 1)**.
    tags : list of str, default empty
        Optional free-form labels for grouping or filtering runs.

    Examples
    --------
    >>> cfg = ExperimentConfig(dataset="synthetic", model="elkanoto",
    ...                        metric="f1", seed=42)
    >>> cfg.validate()
    >>> cfg.dataset
    'synthetic'
    >>> cfg.seed
    42

    """

    dataset: str
    model: str
    metric: str
    seed: int
    n_samples: int = 500
    n_features: int = 20
    pi: float = 0.3
    c: float = 0.5
    test_size: float = 0.3
    tags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate all configuration fields.

        Raises
        ------
        ValueError
            On any invalid field value.

        Examples
        --------
        >>> ExperimentConfig(dataset="d", model="m", metric="f1",
        ...                  seed=0).validate()
        >>> ExperimentConfig(dataset="", model="m", metric="f1",
        ...                  seed=0).validate()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: ...

        """
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError("'dataset' must be a non-empty string.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("'model' must be a non-empty string.")
        if self.metric not in _VALID_METRICS:
            raise ValueError(
                "'metric' must be one of {}; got {!r}.".format(
                    sorted(_VALID_METRICS), self.metric
                )
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(
                "'seed' must be a non-negative integer; got {!r}.".format(
                    self.seed
                )
            )
        if self.seed < 0:
            raise ValueError(
                "'seed' must be >= 0; got {:d}.".format(self.seed)
            )
        if not isinstance(self.n_samples, int) or self.n_samples < 1:
            raise ValueError(
                "'n_samples' must be a positive integer; got {!r}.".format(
                    self.n_samples
                )
            )
        if not isinstance(self.n_features, int) or self.n_features < 1:
            raise ValueError(
                "'n_features' must be a positive integer; got {!r}.".format(
                    self.n_features
                )
            )
        if not (0.0 < self.pi < 1.0):
            raise ValueError(
                "'pi' must be in (0, 1); got {!r}.".format(self.pi)
            )
        if not (0.0 < self.c <= 1.0):
            raise ValueError(
                "'c' must be in (0, 1]; got {!r}.".format(self.c)
            )
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(
                "'test_size' must be in (0, 1); got {!r}.".format(
                    self.test_size
                )
            )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary of all fields.

        Examples
        --------
        >>> cfg = ExperimentConfig(dataset="d", model="m", metric="f1",
        ...                        seed=1)
        >>> d = cfg.as_dict()
        >>> d["dataset"]
        'd'
        >>> d["seed"]
        1

        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Construct an :class:`ExperimentConfig` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Must contain all required fields (``dataset``, ``model``,
            ``metric``, ``seed``).  Optional fields fall back to class
            defaults when absent.

        Returns
        -------
        ExperimentConfig

        Examples
        --------
        >>> d = {"dataset": "d", "model": "m", "metric": "f1", "seed": 7}
        >>> cfg = ExperimentConfig.from_dict(d)
        >>> cfg.seed
        7
        >>> cfg.pi  # default
        0.3

        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_json(self, indent: int = 2) -> str:
        """Return a pretty-printed JSON string of the configuration.

        Examples
        --------
        >>> import json
        >>> cfg = ExperimentConfig(dataset="d", model="m", metric="f1",
        ...                        seed=0)
        >>> obj = json.loads(cfg.to_json())
        >>> obj["dataset"]
        'd'

        """
        return json.dumps(self.as_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentConfig":
        """Construct an :class:`ExperimentConfig` from a JSON string.

        Examples
        --------
        >>> import json
        >>> cfg = ExperimentConfig(dataset="d", model="m", metric="f1",
        ...                        seed=0)
        >>> cfg2 = ExperimentConfig.from_json(cfg.to_json())
        >>> cfg2 == cfg
        True

        """
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Artifact persistence helpers
# ---------------------------------------------------------------------------


def _make_run_id(config: ExperimentConfig) -> str:
    """Generate a deterministic, human-readable run ID.

    Format: ``{YYYYMMDD_HHMMSS}_{dataset}_{model}_{seed}``

    The timestamp component is UTC so IDs are comparable across time zones.
    """
    ts = (
        datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y%m%d_%H%M%S")
    )
    # Replace characters unsafe for directory names with underscores.
    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)

    return "{}_{}_{}_{}".format(
        ts,
        _safe(config.dataset),
        _safe(config.model),
        config.seed,
    )


def save_run_artifacts(
    runner: "BenchmarkRunner",
    config: ExperimentConfig,
    *,
    results_dir: str = "results",
    run_id: Optional[str] = None,
) -> str:
    """Persist all run artifacts to ``{results_dir}/{run_id}/``.

    Files created:

    * ``config.json``   — :class:`ExperimentConfig` fields.
    * ``metadata.json`` — :class:`~pulearn.benchmarks.RunMetadata` (env,
      versions).
    * ``results.csv``   — one row per
      :class:`~pulearn.benchmarks.BenchmarkResult`.
    * ``summary.json``  — combined document containing all three above.

    Parameters
    ----------
    runner : BenchmarkRunner
        Runner *after* :meth:`~BenchmarkRunner.run` has been called.
    config : ExperimentConfig
        Experiment configuration; validated before saving.
    results_dir : str, default ``"results"``
        Parent directory for all run artifacts.
    run_id : str or None, default None
        Override the auto-generated run ID.  When ``None`` a timestamp-
        and config-based ID is generated via :func:`_make_run_id`.

    Returns
    -------
    str
        Absolute path to the created run directory.

    Raises
    ------
    ValueError
        If *config* fails validation.

    Examples
    --------
    >>> import tempfile, os
    >>> from pulearn.benchmarks import BenchmarkRunner
    >>> from pulearn.benchmarks.experiment import (
    ...     ExperimentConfig, save_run_artifacts, load_run_artifacts
    ... )
    >>> cfg = ExperimentConfig(dataset="synthetic", model="dummy",
    ...                        metric="f1", seed=0)
    >>> runner = BenchmarkRunner(random_state=0)
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     run_dir = save_run_artifacts(
    ...         runner, cfg, results_dir=tmpdir, run_id="test_run"
    ...     )
    ...     files = sorted(os.listdir(run_dir))
    ...     "config.json" in files
    True

    """
    config.validate()

    if run_id is None:
        run_id = _make_run_id(config)

    run_dir = os.path.abspath(os.path.join(results_dir, run_id))
    os.makedirs(run_dir, exist_ok=True)

    # ---- config.json ---------------------------------------------------
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(config.to_json())

    # ---- metadata.json -------------------------------------------------
    metadata_dict = runner.metadata.as_dict()
    metadata_path = os.path.join(run_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata_dict, fh, indent=2)

    # ---- results.csv ---------------------------------------------------
    fieldnames = [
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
    results_path = os.path.join(run_dir, "results.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in runner.results:
            writer.writerow(r.as_dict())

    # ---- summary.json --------------------------------------------------
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "config": config.as_dict(),
        "metadata": metadata_dict,
        "results": [r.as_dict() for r in runner.results],
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return run_dir


def load_run_artifacts(run_dir: str) -> Dict[str, Any]:
    """Load artifacts from a previously saved run directory.

    Parameters
    ----------
    run_dir : str
        Path to a run directory created by :func:`save_run_artifacts`.

    Returns
    -------
    dict with keys:

    * ``"run_id"``   — basename of *run_dir*.
    * ``"config"``   — :class:`ExperimentConfig` reconstructed from
      ``config.json``.
    * ``"metadata"`` — plain dict loaded from ``metadata.json``.
    * ``"results"``  — list of dicts loaded from ``results.csv``.

    Raises
    ------
    FileNotFoundError
        If any of the four required artifact files is missing.

    Examples
    --------
    >>> import tempfile
    >>> from pulearn.benchmarks import BenchmarkRunner
    >>> from pulearn.benchmarks.experiment import (
    ...     ExperimentConfig, save_run_artifacts, load_run_artifacts
    ... )
    >>> cfg = ExperimentConfig(dataset="synthetic", model="dummy",
    ...                        metric="f1", seed=0)
    >>> runner = BenchmarkRunner(random_state=0)
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     run_dir = save_run_artifacts(
    ...         runner, cfg, results_dir=tmpdir, run_id="test_run"
    ...     )
    ...     arts = load_run_artifacts(run_dir)
    ...     arts["config"].dataset
    'synthetic'

    """
    run_dir = os.path.abspath(run_dir)
    required = ("config.json", "metadata.json", "results.csv", "summary.json")
    for fname in required:
        p = os.path.join(run_dir, fname)
        if not os.path.isfile(p):
            raise FileNotFoundError(
                "Expected artifact file not found: {}".format(p)
            )

    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        config = ExperimentConfig.from_json(fh.read())

    with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as fh:
        metadata: Dict[str, Any] = json.load(fh)

    results: List[Dict[str, Any]] = []
    with open(
        os.path.join(run_dir, "results.csv"), newline="", encoding="utf-8"
    ) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            results.append(dict(row))

    return {
        "run_id": os.path.basename(run_dir),
        "config": config,
        "metadata": metadata,
        "results": results,
    }
