"""Registry and contributor scaffolding for PU algorithms."""

from dataclasses import dataclass
from pathlib import Path

from sklearn.base import ClassifierMixin

from pulearn.bagging import BaggingPuClassifier
from pulearn.base import BasePUClassifier
from pulearn.bayesian_pu import (
    PositiveNaiveBayesClassifier,
    PositiveTANClassifier,
    WeightedNaiveBayesClassifier,
    WeightedTANClassifier,
)
from pulearn.elkanoto import (
    ElkanotoPuClassifier,
    WeightedElkanotoPuClassifier,
)
from pulearn.nnpu import NNPUClassifier

_ALLOWED_ASSUMPTIONS = {"SCAR", "SAR", "SCAR/SAR"}

_NEW_ALGORITHM_CHECKLIST = (
    "Register the estimator metadata in `src/pulearn/registry.py`.",
    "Add fit/predict regression coverage using "
    "`tests/templates/test_new_algorithm_template.py.tmpl`.",
    "Run shared API contract checks using "
    "`tests/templates/test_api_contract_template.py.tmpl`.",
    "Document usage and caveats with "
    "`doc/templates/new_algorithm_doc_stub.md`.",
    "Add a benchmark placeholder or runnable entry from "
    "`benchmarks/templates/benchmark_entry_template.py.tmpl`.",
    "Update top-level documentation references in `README.rst` / "
    "`src/pulearn/documentation.md` as needed.",
)

_SCAFFOLD_TEMPLATES = {
    "checklist": "doc/new_algorithm_checklist.md",
    "docs": "doc/templates/new_algorithm_doc_stub.md",
    "tests": "tests/templates/test_new_algorithm_template.py.tmpl",
    "api_contract": "tests/templates/test_api_contract_template.py.tmpl",
    "benchmark": "benchmarks/templates/benchmark_entry_template.py.tmpl",
}


@dataclass(frozen=True)
class PUAlgorithmSpec:
    """Metadata describing a registered PU algorithm."""

    key: str
    estimator_cls: type
    family: str
    assumption: str
    summary: str
    docs_reference: str
    test_reference: str
    benchmark_reference: str
    contract_reference: str
    requires_predict_proba: bool = True
    shared_label_normalization: bool = True
    uses_base_contract: bool = True

    @property
    def estimator_name(self):
        """Return the estimator class name."""
        if isinstance(self.estimator_cls, type):
            return self.estimator_cls.__name__
        return repr(self.estimator_cls)


def validate_algorithm_spec(spec):
    """Raise an informative error if registry metadata is incomplete."""
    if not isinstance(spec.key, str) or not spec.key.strip():
        raise ValueError("Registry entries must define a non-empty key.")
    if not isinstance(spec.summary, str) or not spec.summary.strip():
        raise ValueError(
            "Registry entry {!r} must define a non-empty summary.".format(
                spec.key
            )
        )
    if not isinstance(spec.family, str) or not spec.family.strip():
        raise ValueError(
            "Registry entry {!r} must define a non-empty family.".format(
                spec.key
            )
        )
    if spec.assumption not in _ALLOWED_ASSUMPTIONS:
        raise ValueError(
            "Registry entry {!r} uses unsupported assumption {!r}. "
            "Expected one of {}.".format(
                spec.key,
                spec.assumption,
                sorted(_ALLOWED_ASSUMPTIONS),
            )
        )
    if not isinstance(spec.estimator_cls, type):
        raise ValueError(
            "Registry entry {!r} must define estimator_cls as a class/type.".format(
                spec.key
            )
        )
    if not issubclass(spec.estimator_cls, ClassifierMixin):
        raise ValueError(
            "Registry entry {!r} must reference a classifier.".format(spec.key)
        )
    if spec.uses_base_contract and not issubclass(
        spec.estimator_cls, BasePUClassifier
    ):
        raise ValueError(
            "Registry entry {!r} opts into the shared PU contract but {!r} "
            "does not inherit from BasePUClassifier.".format(
                spec.key,
                spec.estimator_name,
            )
        )

    required_refs = {
        "docs_reference": spec.docs_reference,
        "test_reference": spec.test_reference,
        "benchmark_reference": spec.benchmark_reference,
        "contract_reference": spec.contract_reference,
    }
    for field_name, value in required_refs.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "Registry entry {!r} must define {}.".format(
                    spec.key,
                    field_name,
                )
            )


def get_new_algorithm_checklist():
    """Return the contributor checklist for adding a new algorithm."""
    return _NEW_ALGORITHM_CHECKLIST


def get_scaffold_templates():
    """Return absolute scaffold paths from a repository checkout."""
    return _resolve_scaffold_templates(_project_root())


def _project_root():
    """Return the repository root inferred from the package layout."""
    return Path(__file__).resolve().parents[2]


def _resolve_scaffold_templates(repo_root):
    """Resolve scaffold paths and fail clearly outside a repo checkout."""
    root = Path(repo_root).resolve()
    resolved = {}
    missing = []
    for key, relative_path in _SCAFFOLD_TEMPLATES.items():
        path = root / relative_path
        if not path.exists():
            missing.append(relative_path)
        resolved[key] = path
    if missing:
        raise FileNotFoundError(
            "Scaffold templates are only available from a pulearn repository "
            "checkout. Missing files relative to {!r}: {}.".format(
                str(root),
                ", ".join(missing),
            )
        )
    return resolved


def _registry_entries():
    """Build the static PU algorithm registry."""
    entries = (
        PUAlgorithmSpec(
            key="elkanoto",
            estimator_cls=ElkanotoPuClassifier,
            family="elkanoto",
            assumption="SCAR",
            summary="Classic Elkan-Noto hold-out calibration wrapper.",
            docs_reference="src/pulearn/documentation.md",
            test_reference="tests/test_elkanoto.py",
            benchmark_reference="benchmarks/templates/"
            "benchmark_entry_template.py.tmpl",
            contract_reference="tests/contract_checks.py",
        ),
        PUAlgorithmSpec(
            key="weighted_elkanoto",
            estimator_cls=WeightedElkanotoPuClassifier,
            family="elkanoto",
            assumption="SCAR",
            summary="Weighted Elkan-Noto variant with labeled/unlabeled "
            "counts.",
            docs_reference="src/pulearn/documentation.md",
            test_reference="tests/test_elkanoto.py",
            benchmark_reference="benchmarks/templates/"
            "benchmark_entry_template.py.tmpl",
            contract_reference="tests/contract_checks.py",
        ),
        PUAlgorithmSpec(
            key="bagging",
            estimator_cls=BaggingPuClassifier,
            family="bagging",
            assumption="SCAR",
            summary="Bagging-based PU ensemble wrapper.",
            docs_reference="src/pulearn/documentation.md",
            test_reference="tests/test_bagging.py",
            benchmark_reference="benchmarks/templates/"
            "benchmark_entry_template.py.tmpl",
            contract_reference="tests/contract_checks.py",
            uses_base_contract=False,
        ),
        PUAlgorithmSpec(
            key="nnpu",
            estimator_cls=NNPUClassifier,
            family="risk-estimator",
            assumption="SCAR",
            summary="Linear nnPU/uPU learner with non-negative risk "
            "correction.",
            docs_reference="src/pulearn/documentation.md",
            test_reference="tests/test_nnpu.py",
            benchmark_reference="benchmarks/templates/"
            "benchmark_entry_template.py.tmpl",
            contract_reference="tests/contract_checks.py",
        ),
        PUAlgorithmSpec(
            key="positive_naive_bayes",
            estimator_cls=PositiveNaiveBayesClassifier,
            family="bayesian",
            assumption="SCAR",
            summary="Positive-only naive Bayes baseline for discretized "
            "features.",
            docs_reference="src/pulearn/documentation.md",
            test_reference="tests/test_bayesian_pu.py",
            benchmark_reference="benchmarks/templates/"
            "benchmark_entry_template.py.tmpl",
            contract_reference="tests/contract_checks.py",
        ),
        PUAlgorithmSpec(
            key="weighted_naive_bayes",
            estimator_cls=WeightedNaiveBayesClassifier,
            family="bayesian",
            assumption="SCAR",
            summary="Mutual-information weighted naive Bayes PU learner.",
            docs_reference="src/pulearn/documentation.md",
            test_reference="tests/test_bayesian_pu.py",
            benchmark_reference="benchmarks/templates/"
            "benchmark_entry_template.py.tmpl",
            contract_reference="tests/contract_checks.py",
        ),
        PUAlgorithmSpec(
            key="positive_tan",
            estimator_cls=PositiveTANClassifier,
            family="bayesian",
            assumption="SCAR",
            summary="Positive-only tree-augmented naive Bayes PU learner.",
            docs_reference="src/pulearn/documentation.md",
            test_reference="tests/test_bayesian_pu.py",
            benchmark_reference="benchmarks/templates/"
            "benchmark_entry_template.py.tmpl",
            contract_reference="tests/contract_checks.py",
        ),
        PUAlgorithmSpec(
            key="weighted_tan",
            estimator_cls=WeightedTANClassifier,
            family="bayesian",
            assumption="SCAR",
            summary="Weighted tree-augmented naive Bayes PU learner.",
            docs_reference="src/pulearn/documentation.md",
            test_reference="tests/test_bayesian_pu.py",
            benchmark_reference="benchmarks/templates/"
            "benchmark_entry_template.py.tmpl",
            contract_reference="tests/contract_checks.py",
        ),
    )

    return _build_registry(entries)


def _build_registry(entries):
    """Validate and index registry entries by their unique key."""
    registry = {}
    for spec in entries:
        validate_algorithm_spec(spec)
        if spec.key in registry:
            raise ValueError(
                "Duplicate PU algorithm key detected: {!r}.".format(spec.key)
            )
        registry[spec.key] = spec
    return registry


_ALGORITHM_REGISTRY = _registry_entries()


def get_algorithm_registry():
    """Return a copy of the algorithm registry keyed by short name."""
    return dict(_ALGORITHM_REGISTRY)


def list_registered_algorithms():
    """Return the registered algorithm keys in deterministic order."""
    return tuple(_ALGORITHM_REGISTRY)


def get_algorithm_spec(key):
    """Return the registry entry for a known algorithm key."""
    try:
        return _ALGORITHM_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(
            "Unknown PU algorithm {!r}. Available keys: {}.".format(
                key,
                ", ".join(list_registered_algorithms()),
            )
        ) from exc
