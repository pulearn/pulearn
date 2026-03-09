# Copilot Instructions

This repository contains the `pulearn` Python package — tools for
positive-unlabeled (PU) learning built on top of scikit-learn.

## Repository layout

- `src/pulearn/` — main package source code
- `tests/` — test suite (pytest)
- `examples/` — usage examples
- `doc/` — documentation source and build scripts
- `benchmarks/` — performance benchmarks
- `notebooks/` — Jupyter notebooks

## Environment setup

```bash
python -m pip install --upgrade pip
python -m pip install --only-binary=all numpy pandas scikit-learn
python -m pip install -e . -r tests/requirements.txt
```

## Validation commands

Run targeted tests first, then the full suite:

```bash
python -m pytest tests/test_<area>.py
python -m pytest
```

Format and lint:

```bash
ruff format .
ruff check . --fix
pre-commit run --all-files
```

## Code conventions

- Follow existing style and API shape in touched modules.
- Respect Ruff settings in `pyproject.toml` (79-char line length).
- Preserve scikit-learn estimator conventions: `fit` returns `self`,
  learned attributes end with `_`.
- Keep docstrings concise and consistent with surrounding code.
- PU label handling: the public API normalizes labels to `{1, 0}`.
  Preserve any module-specific expectations and update tests/docs if
  behavior changes; use `pulearn.normalize_pu_labels(...)` when in
  doubt.
- PU labels (public API): positive = `1` / `True`; unlabeled
  ∈ `{0, -1, False}` and are internally normalized to `0`.

## Tests and behavior changes

- Any bug fix should include or update a regression test.
- Any behavior/API change should update tests and relevant docs
  (`README.rst` and/or `doc/` as needed).
- Prefer small, deterministic tests.

## Commit hygiene

- Commit only files relevant to the task.
- Do not edit generated or local artifact directories (`build/`,
  `dist/`, `.coverage`, `cov.xml`, `pulearn.egg-info/`).
- Include a clear, imperative commit message.
