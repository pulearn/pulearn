"""Tests for pulearn.benchmarks.experiment.

Covers:
- ExperimentConfig schema validation
- Serialisation round-trips (as_dict / from_dict / to_json / from_json)
- Artifact persistence (save_run_artifacts / load_run_artifacts)
- Reproducibility: same config + seed => same results
- BenchmarkRunner.save_run convenience wrapper
"""

import csv
import json
import os

import pytest
from sklearn.linear_model import LogisticRegression

from pulearn import ElkanotoPuClassifier
from pulearn.benchmarks import (
    BenchmarkRunner,
    ExperimentConfig,
    load_run_artifacts,
    save_run_artifacts,
)
from pulearn.benchmarks.experiment import _make_run_id

# Artifact files expected in every run directory.
_ARTIFACT_FILES = (
    "config.json",
    "metadata.json",
    "results.csv",
    "summary.json",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_elkanoto(seed=0):
    def _build():
        return ElkanotoPuClassifier(
            estimator=LogisticRegression(max_iter=200, random_state=seed),
            hold_out_ratio=0.1,
            random_state=seed,
        )

    return _build


def _minimal_config(**overrides):
    defaults = {
        "dataset": "synthetic",
        "model": "elkanoto",
        "metric": "f1",
        "seed": 42,
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ---------------------------------------------------------------------------
# ExperimentConfig — field defaults and basic construction
# ---------------------------------------------------------------------------


def test_experiment_config_defaults():
    cfg = _minimal_config()
    assert cfg.dataset == "synthetic"
    assert cfg.model == "elkanoto"
    assert cfg.metric == "f1"
    assert cfg.seed == 42
    assert cfg.n_samples == 500
    assert cfg.n_features == 20
    assert cfg.pi == pytest.approx(0.3)
    assert cfg.c == pytest.approx(0.5)
    assert cfg.test_size == pytest.approx(0.3)
    assert cfg.tags == []


def test_experiment_config_custom_fields():
    cfg = ExperimentConfig(
        dataset="breast_cancer",
        model="bagging_pu",
        metric="roc_auc",
        seed=7,
        n_samples=300,
        pi=0.4,
        c=0.6,
        test_size=0.2,
        tags=["v1", "experiment"],
    )
    assert cfg.dataset == "breast_cancer"
    assert cfg.seed == 7
    assert cfg.tags == ["v1", "experiment"]


# ---------------------------------------------------------------------------
# ExperimentConfig — validate()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"dataset": ""}, "dataset"),
        ({"dataset": "   "}, "dataset"),
        ({"model": ""}, "model"),
        ({"metric": "bad_metric"}, "metric"),
        ({"seed": -1}, "seed"),
        ({"seed": 3.5}, "seed"),
        ({"seed": True}, "seed"),
        # bool is a subclass of int — must be explicitly rejected
        ({"n_samples": True}, "n_samples"),
        ({"n_features": True}, "n_features"),
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": -10}, "n_samples"),
        ({"n_features": 0}, "n_features"),
        # non-numeric float fields must raise ValueError, not TypeError
        ({"pi": "0.3"}, "pi"),
        ({"pi": True}, "pi"),
        ({"pi": 0.0}, "pi"),
        ({"pi": 1.0}, "pi"),
        ({"pi": -0.1}, "pi"),
        ({"c": "0.5"}, "c"),
        ({"c": True}, "c"),
        ({"c": 0.0}, "c"),
        ({"c": 1.5}, "c"),
        ({"test_size": "0.3"}, "test_size"),
        ({"test_size": True}, "test_size"),
        ({"test_size": 0.0}, "test_size"),
        ({"test_size": 1.0}, "test_size"),
        # tags must be a list of strings
        ({"tags": "not-a-list"}, "tags"),
        ({"tags": [1, 2]}, r"tags\[0\]"),
    ],
)
def test_experiment_config_validate_rejects(overrides, match):
    cfg = _minimal_config(**overrides)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_experiment_config_validate_c_equal_one_allowed():
    cfg = _minimal_config(c=1.0)
    cfg.validate()  # c=1.0 is valid (propensity = 1 = fully observed)


@pytest.mark.parametrize(
    "metric",
    ["f1", "roc_auc", "accuracy", "precision", "recall", "f1_macro"],
)
def test_experiment_config_validate_all_valid_metrics(metric):
    cfg = _minimal_config(metric=metric)
    cfg.validate()  # must not raise


def test_experiment_config_validate_passes_on_valid():
    cfg = ExperimentConfig(
        dataset="d",
        model="m",
        metric="f1",
        seed=0,
        n_samples=10,
        n_features=5,
        pi=0.3,
        c=0.5,
        test_size=0.2,
    )
    cfg.validate()  # no exception expected


def test_experiment_config_validate_valid_tags():
    cfg = _minimal_config(tags=["run1", "baseline"])
    cfg.validate()  # valid list of strings — must not raise


def test_experiment_config_validate_empty_tags():
    cfg = _minimal_config(tags=[])
    cfg.validate()  # empty list is valid


# ---------------------------------------------------------------------------
# ExperimentConfig — serialisation round-trips
# ---------------------------------------------------------------------------


def test_as_dict_round_trip():
    cfg = _minimal_config()
    d = cfg.as_dict()
    assert d["dataset"] == "synthetic"
    assert d["seed"] == 42
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2 == cfg


def test_from_dict_ignores_unknown_keys():
    d = {"dataset": "d", "model": "m", "metric": "f1", "seed": 1, "extra": "x"}
    cfg = ExperimentConfig.from_dict(d)
    assert cfg.dataset == "d"
    assert not hasattr(cfg, "extra")


def test_to_json_is_valid_json():
    cfg = _minimal_config()
    s = cfg.to_json()
    obj = json.loads(s)
    assert obj["dataset"] == "synthetic"
    assert obj["seed"] == 42


def test_from_json_round_trip():
    cfg = _minimal_config(tags=["a", "b"])
    cfg2 = ExperimentConfig.from_json(cfg.to_json())
    assert cfg2 == cfg


def test_as_dict_contains_all_fields():
    cfg = _minimal_config()
    d = cfg.as_dict()
    for field in (
        "dataset",
        "model",
        "metric",
        "seed",
        "n_samples",
        "n_features",
        "pi",
        "c",
        "test_size",
        "tags",
    ):
        assert field in d, f"Missing field '{field}' in as_dict()"


# ---------------------------------------------------------------------------
# _make_run_id
# ---------------------------------------------------------------------------


def test_make_run_id_contains_dataset_model_seed():
    cfg = _minimal_config(dataset="breast_cancer", model="elkanoto", seed=7)
    run_id = _make_run_id(cfg)
    assert "breast_cancer" in run_id
    assert "elkanoto" in run_id
    assert "7" in run_id


def test_make_run_id_replaces_special_chars():
    cfg = _minimal_config(dataset="my dataset", model="my/model")
    run_id = _make_run_id(cfg)
    assert " " not in run_id
    assert "/" not in run_id


# ---------------------------------------------------------------------------
# save_run_artifacts / load_run_artifacts
# ---------------------------------------------------------------------------


def test_save_run_artifacts_creates_expected_files(tmp_path):
    cfg = _minimal_config()
    runner = BenchmarkRunner(random_state=cfg.seed)
    run_dir = save_run_artifacts(
        runner, cfg, results_dir=str(tmp_path), run_id="test_run"
    )
    for fname in _ARTIFACT_FILES:
        assert os.path.isfile(os.path.join(run_dir, fname)), (
            f"Expected {fname} in {run_dir}"
        )


def test_save_run_artifacts_returns_absolute_path(tmp_path):
    cfg = _minimal_config()
    runner = BenchmarkRunner(random_state=cfg.seed)
    run_dir = save_run_artifacts(
        runner, cfg, results_dir=str(tmp_path), run_id="abs_test"
    )
    assert os.path.isabs(run_dir)


def test_save_run_artifacts_config_json_content(tmp_path):
    cfg = _minimal_config(dataset="wine", seed=99)
    runner = BenchmarkRunner(random_state=cfg.seed)
    run_dir = save_run_artifacts(
        runner, cfg, results_dir=str(tmp_path), run_id="cfg_test"
    )
    with open(os.path.join(run_dir, "config.json")) as fh:
        obj = json.load(fh)
    assert obj["dataset"] == "wine"
    assert obj["seed"] == 99


def test_save_run_artifacts_metadata_json_has_version_fields(tmp_path):
    cfg = _minimal_config()
    runner = BenchmarkRunner(random_state=cfg.seed)
    run_dir = save_run_artifacts(
        runner, cfg, results_dir=str(tmp_path), run_id="meta_test"
    )
    with open(os.path.join(run_dir, "metadata.json")) as fh:
        obj = json.load(fh)
    for key in (
        "timestamp",
        "python_version",
        "pulearn_version",
        "numpy_version",
        "sklearn_version",
        "random_state",
        "test_size",
    ):
        assert key in obj, f"metadata.json missing key '{key}'"


def test_save_run_artifacts_results_csv_has_header(tmp_path):
    cfg = _minimal_config()
    runner = BenchmarkRunner(random_state=cfg.seed)
    run_dir = save_run_artifacts(
        runner, cfg, results_dir=str(tmp_path), run_id="csv_test"
    )
    with open(os.path.join(run_dir, "results.csv"), newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
    for col in ("name", "dataset", "f1", "roc_auc", "error"):
        assert col in fieldnames, f"results.csv missing column '{col}'"


def test_save_run_artifacts_results_csv_rows_match_runner(tmp_path):
    cfg = _minimal_config()
    runner = BenchmarkRunner(random_state=cfg.seed)
    runner.run(
        {"elkanoto": _build_elkanoto(cfg.seed)},
        n_samples=100,
        pi=cfg.pi,
        c=cfg.c,
    )
    run_dir = save_run_artifacts(
        runner, cfg, results_dir=str(tmp_path), run_id="rows_test"
    )
    with open(os.path.join(run_dir, "results.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(runner.results)


def test_save_run_artifacts_summary_json_has_all_sections(tmp_path):
    cfg = _minimal_config()
    runner = BenchmarkRunner(random_state=cfg.seed)
    run_dir = save_run_artifacts(
        runner, cfg, results_dir=str(tmp_path), run_id="summary_test"
    )
    with open(os.path.join(run_dir, "summary.json")) as fh:
        obj = json.load(fh)
    assert "run_id" in obj
    assert "config" in obj
    assert "metadata" in obj
    assert "results" in obj
    assert obj["run_id"] == "summary_test"


def test_save_run_artifacts_auto_run_id(tmp_path):
    cfg = _minimal_config()
    runner = BenchmarkRunner(random_state=cfg.seed)
    run_dir = save_run_artifacts(runner, cfg, results_dir=str(tmp_path))
    # Auto-generated run_id should produce a valid directory.
    assert os.path.isdir(run_dir)
    basename = os.path.basename(run_dir)
    # Should contain the dataset and model names.
    assert "synthetic" in basename
    assert "elkanoto" in basename


def test_save_run_artifacts_rejects_invalid_config(tmp_path):
    cfg = ExperimentConfig(dataset="", model="m", metric="f1", seed=0)
    runner = BenchmarkRunner(random_state=0)
    with pytest.raises(ValueError, match="dataset"):
        save_run_artifacts(runner, cfg, results_dir=str(tmp_path))


@pytest.mark.parametrize(
    "bad_run_id",
    [
        "",           # empty
        ".",          # current dir
        "..",         # parent dir traversal
        "../escape",  # path traversal
        "/absolute",  # absolute path
        "a/b",        # slash-separated sub-path
    ],
)
def test_save_run_artifacts_rejects_unsafe_run_id(tmp_path, bad_run_id):
    cfg = _minimal_config()
    runner = BenchmarkRunner(random_state=cfg.seed)
    with pytest.raises(ValueError, match="run_id"):
        save_run_artifacts(
            runner, cfg, results_dir=str(tmp_path), run_id=bad_run_id
        )


def test_save_run_artifacts_warns_on_seed_mismatch(tmp_path):
    cfg = _minimal_config(seed=42)
    # Create runner with a different random_state.
    runner = BenchmarkRunner(random_state=99)
    with pytest.warns(UserWarning, match="random_state"):
        save_run_artifacts(
            runner, cfg, results_dir=str(tmp_path), run_id="seed_mismatch"
        )


def test_save_run_artifacts_warns_on_test_size_mismatch(tmp_path):
    cfg = _minimal_config()  # test_size=0.3 by default
    runner = BenchmarkRunner(random_state=cfg.seed, test_size=0.2)
    with pytest.warns(UserWarning, match="test_size"):
        save_run_artifacts(
            runner, cfg, results_dir=str(tmp_path), run_id="ts_mismatch"
        )


def test_save_run_artifacts_no_warning_when_consistent(tmp_path):
    cfg = _minimal_config(seed=42)
    runner = BenchmarkRunner(random_state=42)
    # Should not emit any UserWarning when seed and test_size agree.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        save_run_artifacts(
            runner, cfg, results_dir=str(tmp_path), run_id="consistent"
        )


# ---------------------------------------------------------------------------
# load_run_artifacts
# ---------------------------------------------------------------------------


def test_load_run_artifacts_round_trip(tmp_path):
    cfg = _minimal_config(dataset="digits", model="bagging", seed=3)
    runner = BenchmarkRunner(random_state=cfg.seed)
    run_dir = save_run_artifacts(
        runner, cfg, results_dir=str(tmp_path), run_id="load_test"
    )
    arts = load_run_artifacts(run_dir)
    assert arts["run_id"] == "load_test"
    assert isinstance(arts["config"], ExperimentConfig)
    assert arts["config"].dataset == "digits"
    assert arts["config"].seed == 3
    assert isinstance(arts["metadata"], dict)
    assert isinstance(arts["results"], list)


def test_load_run_artifacts_missing_file_raises(tmp_path):
    cfg = _minimal_config()
    runner = BenchmarkRunner(random_state=cfg.seed)
    run_dir = save_run_artifacts(
        runner, cfg, results_dir=str(tmp_path), run_id="missing_test"
    )
    os.remove(os.path.join(run_dir, "config.json"))
    with pytest.raises(FileNotFoundError):
        load_run_artifacts(run_dir)


def test_load_run_artifacts_config_equality(tmp_path):
    cfg = _minimal_config(pi=0.4, c=0.7, tags=["x"])
    runner = BenchmarkRunner(random_state=cfg.seed)
    run_dir = save_run_artifacts(
        runner, cfg, results_dir=str(tmp_path), run_id="eq_test"
    )
    arts = load_run_artifacts(run_dir)
    assert arts["config"] == cfg


# ---------------------------------------------------------------------------
# BenchmarkRunner.save_run convenience wrapper
# ---------------------------------------------------------------------------


def test_runner_save_run_creates_artifacts(tmp_path):
    cfg = _minimal_config()
    runner = BenchmarkRunner(random_state=cfg.seed)
    runner.run(
        {"elkanoto": _build_elkanoto(cfg.seed)},
        n_samples=100,
        pi=cfg.pi,
        c=cfg.c,
    )
    run_dir = runner.save_run(
        cfg, results_dir=str(tmp_path), run_id="save_run_test"
    )
    assert os.path.isdir(run_dir)
    for fname in _ARTIFACT_FILES:
        assert os.path.isfile(os.path.join(run_dir, fname))


def test_runner_save_run_returns_string(tmp_path):
    cfg = _minimal_config()
    runner = BenchmarkRunner(random_state=cfg.seed)
    result = runner.save_run(cfg, results_dir=str(tmp_path), run_id="str_test")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Reproducibility: same config/seed => same outputs
# ---------------------------------------------------------------------------


def test_reproducibility_same_config_same_results(tmp_path):
    """Two runners with identical config/seed must yield identical F1/AUC."""
    cfg = _minimal_config(seed=0)

    r1 = BenchmarkRunner(random_state=cfg.seed)
    r1.run(
        {"elkanoto": _build_elkanoto(cfg.seed)},
        n_samples=150,
        pi=cfg.pi,
        c=cfg.c,
    )

    r2 = BenchmarkRunner(random_state=cfg.seed)
    r2.run(
        {"elkanoto": _build_elkanoto(cfg.seed)},
        n_samples=150,
        pi=cfg.pi,
        c=cfg.c,
    )

    for res1, res2 in zip(r1.results, r2.results):
        assert res1.f1 == pytest.approx(res2.f1, abs=1e-9)


def test_reproducibility_different_seeds_may_differ():
    """Different seeds should generally produce different results."""
    import numpy as np

    cfg_a = _minimal_config(seed=0)
    cfg_b = _minimal_config(seed=999)

    r1 = BenchmarkRunner(random_state=cfg_a.seed)
    r1.run(
        {"elkanoto": _build_elkanoto(cfg_a.seed)},
        n_samples=200,
        pi=cfg_a.pi,
        c=cfg_a.c,
    )

    r2 = BenchmarkRunner(random_state=cfg_b.seed)
    r2.run(
        {"elkanoto": _build_elkanoto(cfg_b.seed)},
        n_samples=200,
        pi=cfg_b.pi,
        c=cfg_b.c,
    )

    # At least one metric should differ (this is virtually always true with
    # different seeds but we assert they are not both NaN, i.e. runs finished).
    assert not (np.isnan(r1.results[0].f1) and np.isnan(r2.results[0].f1))


def test_reproducibility_saved_artifacts_agree(tmp_path):
    """Artifacts from two identical runs must have equal results lists."""
    cfg = _minimal_config(seed=1)

    for run_id in ("run_a", "run_b"):
        runner = BenchmarkRunner(random_state=cfg.seed)
        runner.run(
            {"elkanoto": _build_elkanoto(cfg.seed)},
            n_samples=100,
            pi=cfg.pi,
            c=cfg.c,
        )
        save_run_artifacts(
            runner, cfg, results_dir=str(tmp_path), run_id=run_id
        )

    arts_a = load_run_artifacts(str(tmp_path / "run_a"))
    arts_b = load_run_artifacts(str(tmp_path / "run_b"))

    assert len(arts_a["results"]) == len(arts_b["results"])
    for row_a, row_b in zip(arts_a["results"], arts_b["results"]):
        assert row_a["f1"] == row_b["f1"]


# ---------------------------------------------------------------------------
# Public API: importable from pulearn.benchmarks
# ---------------------------------------------------------------------------


def test_experiment_config_importable_from_benchmarks():
    from pulearn.benchmarks import ExperimentConfig as _EC

    assert _EC is ExperimentConfig


def test_save_run_artifacts_importable_from_benchmarks():
    from pulearn.benchmarks import save_run_artifacts as _sra

    assert _sra is save_run_artifacts


def test_load_run_artifacts_importable_from_benchmarks():
    from pulearn.benchmarks import load_run_artifacts as _lra

    assert _lra is load_run_artifacts
