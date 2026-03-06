import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from pulearn import ElkanotoPuClassifier
from pulearn.registry import (
    PUAlgorithmSpec,
    _build_registry,
    _resolve_scaffold_templates,
    get_algorithm_registry,
    get_algorithm_spec,
    get_new_algorithm_checklist,
    get_scaffold_templates,
    list_registered_algorithms,
    validate_algorithm_spec,
)
from tests.contract_checks import assert_base_pu_estimator_contract


class _ClassifierOnlyEstimator(ClassifierMixin, BaseEstimator):
    pass


class _NonClassifierEstimator(BaseEstimator):
    pass


class _DuplicateKeyEstimator(ElkanotoPuClassifier):
    pass


def test_registry_exposes_known_algorithms():
    registered = list_registered_algorithms()
    assert registered == (
        "elkanoto",
        "weighted_elkanoto",
        "bagging",
        "nnpu",
        "positive_naive_bayes",
        "weighted_naive_bayes",
        "positive_tan",
        "weighted_tan",
    )

    spec = get_algorithm_spec("elkanoto")
    assert spec.estimator_cls is ElkanotoPuClassifier
    assert spec.assumption == "SCAR"
    assert spec.uses_base_contract


def test_registry_metadata_validates_for_all_entries():
    registry = get_algorithm_registry()
    assert set(registry) == set(list_registered_algorithms())
    for spec in registry.values():
        validate_algorithm_spec(spec)


def test_registry_unknown_key_lists_available_algorithms():
    try:
        get_algorithm_spec("does-not-exist")
    except KeyError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected unknown registry key to fail.")
    assert "elkanoto" in message
    assert "weighted_tan" in message


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (
            PUAlgorithmSpec(
                key="",
                estimator_cls=ElkanotoPuClassifier,
                family="family",
                assumption="SCAR",
                summary="summary",
                docs_reference="docs",
                test_reference="tests",
                benchmark_reference="benchmarks",
                contract_reference="contracts",
            ),
            "non-empty key",
        ),
        (
            PUAlgorithmSpec(
                key="bad-summary",
                estimator_cls=ElkanotoPuClassifier,
                family="family",
                assumption="SCAR",
                summary="",
                docs_reference="docs",
                test_reference="tests",
                benchmark_reference="benchmarks",
                contract_reference="contracts",
            ),
            "non-empty summary",
        ),
        (
            PUAlgorithmSpec(
                key="bad-family",
                estimator_cls=ElkanotoPuClassifier,
                family="",
                assumption="SCAR",
                summary="summary",
                docs_reference="docs",
                test_reference="tests",
                benchmark_reference="benchmarks",
                contract_reference="contracts",
            ),
            "non-empty family",
        ),
        (
            PUAlgorithmSpec(
                key="bad-assumption",
                estimator_cls=ElkanotoPuClassifier,
                family="family",
                assumption="MNAR",
                summary="summary",
                docs_reference="docs",
                test_reference="tests",
                benchmark_reference="benchmarks",
                contract_reference="contracts",
            ),
            "unsupported assumption",
        ),
        (
            PUAlgorithmSpec(
                key="not-classifier",
                estimator_cls=_NonClassifierEstimator,
                family="family",
                assumption="SCAR",
                summary="summary",
                docs_reference="docs",
                test_reference="tests",
                benchmark_reference="benchmarks",
                contract_reference="contracts",
            ),
            "must reference a classifier",
        ),
        (
            PUAlgorithmSpec(
                key="missing-base-contract",
                estimator_cls=_ClassifierOnlyEstimator,
                family="family",
                assumption="SCAR",
                summary="summary",
                docs_reference="docs",
                test_reference="tests",
                benchmark_reference="benchmarks",
                contract_reference="contracts",
            ),
            "does not inherit from BasePUClassifier",
        ),
        (
            PUAlgorithmSpec(
                key="missing-docs-ref",
                estimator_cls=ElkanotoPuClassifier,
                family="family",
                assumption="SCAR",
                summary="summary",
                docs_reference="",
                test_reference="tests",
                benchmark_reference="benchmarks",
                contract_reference="contracts",
            ),
            "docs_reference",
        ),
        (
            PUAlgorithmSpec(
                key="not-a-type",
                estimator_cls="not-a-class",
                family="family",
                assumption="SCAR",
                summary="summary",
                docs_reference="docs",
                test_reference="tests",
                benchmark_reference="benchmarks",
                contract_reference="contracts",
            ),
            "estimator_cls as a class/type",
        ),
    ],
)
def test_registry_validation_rejects_incomplete_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        validate_algorithm_spec(spec)


def test_new_algorithm_checklist_covers_required_workstreams():
    checklist = get_new_algorithm_checklist()
    assert len(checklist) == 6
    assert "registry.py" in checklist[0]
    assert "test_new_algorithm_template.py.tmpl" in checklist[1]
    assert "test_api_contract_template.py.tmpl" in checklist[2]
    assert "new_algorithm_doc_stub.md" in checklist[3]
    assert "benchmark_entry_template.py.tmpl" in checklist[4]
    assert "README.rst" in checklist[5]


def test_scaffold_templates_exist_in_repository():
    for template_path in get_scaffold_templates().values():
        assert template_path.is_absolute()
        assert template_path.exists()


def test_resolve_scaffold_templates_raises_outside_repo_checkout(tmp_path):
    with pytest.raises(FileNotFoundError, match="repository checkout"):
        _resolve_scaffold_templates(tmp_path)


def test_registry_spec_exposes_estimator_name():
    spec = get_algorithm_spec("elkanoto")
    assert spec.estimator_name == "ElkanotoPuClassifier"


def test_registry_spec_handles_non_type_estimator_name():
    spec = PUAlgorithmSpec(
        key="opaque",
        estimator_cls="not-a-class",
        family="family",
        assumption="SCAR",
        summary="summary",
        docs_reference="docs",
        test_reference="tests",
        benchmark_reference="benchmarks",
        contract_reference="contracts",
    )
    assert spec.estimator_name == "'not-a-class'"


def test_elkanoto_validated_against_shared_contract_helper():
    X = np.array(
        [
            [2.0, 1.0],
            [2.1, 1.2],
            [1.9, 0.8],
            [-1.5, -1.1],
            [-1.7, -0.9],
            [-1.2, -1.4],
            [0.4, 0.3],
            [0.3, -0.1],
        ]
    )
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    estimator = ElkanotoPuClassifier(
        estimator=LogisticRegression(solver="liblinear"),
        hold_out_ratio=0.25,
        random_state=0,
    )

    assert_base_pu_estimator_contract(
        estimator,
        X,
        y,
        allow_out_of_bounds=True,
    )


def test_duplicate_registry_key_detection():
    duplicate = (
        PUAlgorithmSpec(
            key="duplicate",
            estimator_cls=ElkanotoPuClassifier,
            family="family",
            assumption="SCAR",
            summary="summary",
            docs_reference="docs",
            test_reference="tests",
            benchmark_reference="benchmarks",
            contract_reference="contracts",
        ),
        PUAlgorithmSpec(
            key="duplicate",
            estimator_cls=_DuplicateKeyEstimator,
            family="family",
            assumption="SCAR",
            summary="summary",
            docs_reference="docs",
            test_reference="tests",
            benchmark_reference="benchmarks",
            contract_reference="contracts",
        ),
    )

    with pytest.raises(ValueError, match="Duplicate PU algorithm key"):
        _build_registry(duplicate)
