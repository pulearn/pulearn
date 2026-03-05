# AGENTS.md

Guidance for coding agents working in this repository.

## Scope

- This is the `pulearn` Python package (positive-unlabeled learning tools).
- Main code lives in `src/pulearn/`.
- Tests live in `tests/`.
- Examples live in `examples/`.
- Docs source and build scripts live in `doc/`.

## Tooling and workflow

- Use repo-specific Git/GitHub MCP tools for git and GitHub operations.
- Keep changes focused; avoid opportunistic refactors.
- Do not edit generated or local artifact directories unless explicitly asked (`build/`, `dist/`, `.coverage`, `cov.xml`, `pulearn.egg-info/`).

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

Format/lint with repository tools when relevant:

```bash
ruff format .
ruff check . --fix
pre-commit run --all-files
```

## Code conventions

- Follow existing style and API shape in touched modules.
- Respect Ruff settings in `pyproject.toml` (79-char line length, enabled lint families).
- Preserve scikit-learn estimator conventions (`fit` returns `self`, learned attributes end with `_`).
- Keep docstrings concise and consistent with surrounding code.
- PU label conventions differ by algorithm; preserve each module's expected label scheme and update tests/docs if behavior changes.

## Tests and behavior changes

- Any bug fix should include or update a regression test.
- Any behavior/API change should update tests and relevant docs (`README.rst` and/or `doc/` as needed).
- Prefer small, deterministic tests.

## Commit hygiene

- Commit only files relevant to the task.
- Include a clear, imperative commit message.

## Local overrides (optional, untracked)

- If `LOCAL_AGENTS.md` exists at repo root, treat it as additive local instructions.
- On conflicts, security and repository policy rules in tracked docs take precedence.
- `LOCAL_AGENTS.md` may refine local workflow and tool routing.
- Never commit machine-specific paths, personal tokens, or local MCP server names into tracked docs.
