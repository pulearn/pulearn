# AGENTS.md

Guidance for coding agents working in this repository.

## Scope

- This is the `pulearn` Python package (positive-unlabeled learning tools).
- Main code lives in `src/pulearn/`.
- Tests live in `tests/`.
- Examples live in `examples/`.
- Docs source and build scripts live in `doc/`.

## Repository layout

```
src/pulearn/
  __init__.py          # public re-exports (see "Public API" below)
  base.py              # BasePUClassifier, label utilities, shared validators
  bagging.py           # BaggingPuClassifier
  bayesian_pu.py       # PositiveNaiveBayesClassifier, Weighted/TAN variants
  elkanoto.py          # ElkanotoPuClassifier, WeightedElkanotoPuClassifier
  metrics.py           # PU evaluation metrics, make_pu_scorer, calibration
  model_selection.py   # PUStratifiedKFold, PUCrossValidator, pu_train_test_split
  nnpu.py              # NNPUClassifier (non-negative PU risk)
  registry.py          # PUAlgorithmSpec, algorithm registry helpers
  priors/              # Class-prior estimation (BasePriorEstimator + 3 estimators,
  │                    #   diagnostics, sensitivity analysis, bootstrap CIs)
  propensity/          # Labeling-propensity estimation (5 estimators,
                       #   bootstrap CIs, SAR hooks, scar_sanity_check)

tests/
  contract_checks.py           # shared helper: assert_base_pu_estimator_contract
  test_api_foundations.py      # base contract / validation helper tests
  test_bagging.py
  test_bayesian_pu.py
  test_contract_checks.py
  test_elkanoto.py / test_elkanoto_edge_cases.py
  test_metrics.py / test_metrics_robustness.py / test_pu_metrics_corrected.py
  test_nnpu.py
  test_prior_diagnostics.py / test_prior_sensitivity.py / test_priors.py
  test_propensity.py / test_propensity_diagnostics.py / test_propensity_sar.py
  test_pu_curves.py / test_pu_detectors.py
  test_pu_model_selection.py
  test_registry.py / test_version.py
  templates/                   # scaffold templates for new algorithms
    test_new_algorithm_template.py.tmpl
    test_api_contract_template.py.tmpl

doc/
  new_algorithm_checklist.md   # required steps when adding a new learner
  templates/                   # doc stub template for new algorithms
```

## Tooling and workflow

- Use repo-specific Git/GitHub MCP tools for git and GitHub operations.
- Keep changes focused; avoid opportunistic refactors.
- Do not edit generated or local artifact directories unless explicitly asked
  (`build/`, `dist/`, `.coverage`, `cov.xml`, `pulearn.egg-info/`).

## Environment setup

Use the same install flow as CI:

```bash
python -m pip install --upgrade pip
python -m pip install --only-binary=all numpy pandas scikit-learn
python -m pip install -e . -r tests/requirements.txt
```

## Validation commands

Run targeted tests first, then full test suite before finalizing:

```bash
python -m pytest tests/test_<area>.py
python -m pytest
```

> **Note:** pytest is configured with `--doctest-modules` (see `pyproject.toml`).
> All public docstrings with `>>>` examples are executed as doctests.
> Keep inline examples runnable, or mark them with `# doctest: +SKIP`.

Format/lint with repository tools when relevant:

```bash
ruff format .
ruff check . --fix
pre-commit run --all-files
```

## Code conventions

- Follow existing style and API shape in touched modules.
- Respect Ruff settings in `pyproject.toml` (79-char line length, enabled lint families).
- Preserve scikit-learn estimator conventions (`fit` returns `self`,
  learned attributes end with `_`).
- Keep docstrings concise and consistent with surrounding code.

### PU label conventions (public API)

The canonical internal label scheme is **`1` = labeled positive,
`0` = unlabeled**. The public API accepts any of `{1, 0}`, `{1, -1}`,
or `{True, False}` and normalizes inputs immediately on entry.

- Use `pulearn.normalize_pu_labels(y)` (or the alias `normalize_pu_y(y)`)
  to convert external labels to canonical form at API boundaries.
- `pulearn.pu_label_masks(y)` returns `(pos_mask, unl_mask)` for the
  canonical-form array.
- `unlabeled` input values of `0`, `-1`, and `False` are all accepted and
  normalized to `0`; `1` and `True` are normalized to `1`.
- Both helpers reject empty arrays with a `ValueError`.

### Shared validation helpers (`src/pulearn/base.py`)

Use these consistently instead of writing ad-hoc checks:

| Helper                                                         | Purpose                                 |
| -------------------------------------------------------------- | --------------------------------------- |
| `validate_non_empty_1d_array(arr, name=...)`                   | ensures 1-D, non-empty                  |
| `validate_same_sample_count(a, b, lhs_name=..., rhs_name=...)` | length match                            |
| `validate_required_pu_labels(y)`                               | at least one positive and one unlabeled |
| `validate_pu_fit_inputs(X, y)`                                 | composite: calls all three above        |

All estimator `fit()` methods call `validate_pu_fit_inputs` as their first
step.

### `BasePUClassifier`

New PU classifiers should subclass `BasePUClassifier` in `base.py`:

- Inherits `fit`-returns-`self`, `classes_` exposure, and `predict_proba`
  shape/validity enforcement.
- `_validate_predict_proba_output(proba)` checks shape `(n, 2)`, finite
  values, non-negative values; if `allow_out_of_bounds=False` also requires
  values in `[0, 1]` and rows summing to `1 ± 1e-6`.

### Algorithm registry

Every new learner must be registered before wiring docs, benchmarks, or
tests. Use the registry API to stay discoverable:

```python
from pulearn import (
    get_algorithm_registry,
    get_algorithm_spec,
    list_registered_algorithms,
)
```

Consult `doc/new_algorithm_checklist.md` (or
`pulearn.get_new_algorithm_checklist()`) for the required workflow.
Scaffold templates are in `tests/templates/` and `doc/templates/`.

### `pulearn.metrics`

- All metric functions validate PU labels via the shared validators in
  `base.py` and normalize to canonical form before computing.
- `pi` (positive class prior) is validated by `_validate_pi`: rejects
  `bool`, `None`, `str`, or non-finite values and raises `ValueError`;
  emits `UserWarning` when `pi < 0.02` or `pi > 0.98`.
- `make_pu_scorer(metric_name, pi=...)` validates `pi` eagerly at
  construction time (rejects `None`, `bool`, `NaN`, `inf`, out-of-range).
  Uses `response_method="predict_proba"/"predict"` (not the deprecated
  `needs_proba` kwarg) for sklearn ≥ 1.4.2 compatibility.

### `pulearn.priors`

- All estimators subclass `BasePriorEstimator` with `fit(X, y)` /
  `estimate(X, y)` / `estimate()` (no args, returns stored `result_`).
- Stored attributes after fitting: `result_` (`PriorEstimateResult`),
  `pi_`, `positive_label_rate_`, `metadata_`.
- Bootstrap CIs via `estimator.bootstrap(X, y, n_resamples=..., confidence_level=..., random_state=...)`.
- Diagnostics: `diagnose_prior_estimator`, `analyze_prior_sensitivity`,
  `summarize_prior_stability`, `plot_prior_sensitivity` (requires
  matplotlib).

### `pulearn.propensity`

- All score-based estimators subclass `BasePropensityEstimator` with
  `fit(y_pu, s_proba=...)` / `estimate(y_pu, s_proba=...)` / `estimate()`
  (no args, returns stored `result_`).
- `CrossValidatedPropensityEstimator` uses `fit(y_pu, X=...)` and a base
  sklearn estimator.
- Stored attributes: `result_` (`PropensityEstimateResult`), `c_`,
  `metadata_`.
- Bootstrap CIs: `estimator.bootstrap(y_pu, s_proba=..., ...)`;
  `result_.confidence_interval` and `confidence_interval_` are attached.
- `scar_sanity_check(y_pu, s_proba=..., X=..., ...)` checks whether the
  SCAR assumption is plausible.
- Experimental SAR helpers (`ExperimentalSarHook`, `predict_sar_propensity`,
  `compute_inverse_propensity_weights`) emit a `UserWarning` on every call
  because semantics are still unstable. Do not stabilize them without an
  explicit issue/milestone.

### `pulearn.model_selection`

- `PUStratifiedKFold` wraps `StratifiedKFold` and stratifies by the PU
  label.
- `PUCrossValidator` is compatible with `cross_validate` / `GridSearchCV`;
  emits a `UserWarning` and falls back to `KFold` when labeled-positive
  count < `n_splits`.
- `pu_train_test_split` validates that the training split always contains
  at least one labeled positive.

## Tests and behavior changes

- Any bug fix should include or update a regression test.
- Any behavior/API change should update tests and relevant docs (`README.rst`
  and/or `doc/` as needed).
- Prefer small, deterministic tests.
- Use `tests/contract_checks.py::assert_base_pu_estimator_contract` to
  verify that any new `BasePUClassifier` subclass satisfies the shared
  estimator contract.
- When adding a new algorithm, copy the scaffold templates from
  `tests/templates/` and `doc/templates/` as starting points.
- Docstring examples are run as doctests (`--doctest-modules`). Keep them
  self-contained and runnable, or annotate with `# doctest: +SKIP`.

## Commit hygiene

- Commit only files relevant to the task.
- Include a clear, imperative commit message.

## Local overrides (optional, untracked)

- If `LOCAL_AGENTS.md` exists at repo root, treat it as additive local
  instructions.
- On conflicts, security and repository policy rules in tracked docs take
  precedence.
- `LOCAL_AGENTS.md` may refine local workflow and tool routing.
- Never commit machine-specific paths, personal tokens, or local MCP server
  names into tracked docs.
