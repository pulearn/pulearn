# Benchmark Gating Policy

This document describes the automated benchmark gates used in CI and the
nightly scheduled workflow, including runtime budgets, pass/fail criteria,
and troubleshooting guidance.

______________________________________________________________________

## Overview

Two benchmark automation paths exist:

| Path               | Workflow file                             | Trigger                        | Budget         |
| ------------------ | ----------------------------------------- | ------------------------------ | -------------- |
| **Smoke (fast)**   | `.github/workflows/benchmark-smoke.yml`   | Every PR / push to `master`    | < 5 min total  |
| **Nightly (full)** | `.github/workflows/benchmark-nightly.yml` | Nightly cron + manual dispatch | < 30 min total |

______________________________________________________________________

## Smoke Gate (CI — Pull Requests)

### What runs

The smoke gate runs on every pull request and push to `master`. It
exercises the benchmark harness with a single, lightweight estimator on a
small synthetic dataset:

- **Estimator:** `ElkanotoPuClassifier` (Logistic Regression base)
- **Dataset:** synthetic, `n_samples=300`, `pi=0.3`, `c=0.5`
- **Python version:** 3.11 only

The gate also runs the full benchmark unit test suites
(`tests/test_benchmark_datasets.py`, `tests/test_benchmark_runner.py`) and
the dedicated smoke-gate fixture tests (`tests/test_benchmark_smoke_gate.py`).

### Pass/fail criteria

The smoke gate **fails** if any of the following occur:

1. Any benchmark unit test fails.
2. The inline benchmark script raises an exception.
3. `result.error` is not `None` for any result.
4. `result.f1 < 0` (non-negative F1 required).

### Runtime budget

| Step                          | Expected wall-clock time |
| ----------------------------- | ------------------------ |
| `pytest` benchmark unit tests | < 30 s                   |
| Inline smoke benchmark script | < 60 s                   |
| **Total smoke gate**          | **< 5 min**              |

### Artifacts

On failure the smoke workflow uploads `benchmark_smoke_results.csv` as a
GitHub Actions artifact named `benchmark-smoke-results`. Open the failing
run in the GitHub Actions UI and download this artifact to inspect the raw
results.

______________________________________________________________________

## Nightly Full Benchmark

### What runs

The nightly workflow runs at **02:00 UTC every day** and can also be
triggered manually via the GitHub Actions `workflow_dispatch` UI.

- **Estimators:** `ElkanotoPuClassifier`, `WeightedElkanotoPuClassifier`,
  `BaggingPuClassifier`
- **Datasets:** synthetic (`n_samples=2000`) and real breast-cancer dataset
- **Python versions:** 3.11 and 3.12

### Pass/fail criteria

The nightly run **fails** if any result has `result.error is not None`. An
exit code of 1 is returned in that case, causing the job to fail.

### Runtime budget

| Step                                              | Expected wall-clock time    |
| ------------------------------------------------- | --------------------------- |
| `pytest` benchmark unit tests                     | < 60 s                      |
| Full benchmark script (synthetic + breast cancer) | < 10 min per Python version |
| **Total nightly gate**                            | **< 30 min**                |

### Artifacts

Every nightly run (success or failure) uploads `benchmark_results.csv` as a
GitHub Actions artifact named
`benchmark-results-py<python-version>` (one artifact per matrix entry).
These artifacts are retained for 14 days (GitHub default).

You can download artifacts from the nightly run's summary page under
**Artifacts**.

______________________________________________________________________

## Troubleshooting

### "Benchmark error: …" in the smoke log

The inline smoke script prints the error message returned by
`result.error`. Steps to debug:

1. Download the `benchmark-smoke-results` artifact from the failed CI run.
2. Open `benchmark_smoke_results.csv` and find rows where `error` is
   non-empty.
3. Reproduce locally:
   ```bash
   python -m pytest tests/test_benchmark_smoke_gate.py -v --no-cov
   ```
4. Run the same inline script from `.github/workflows/benchmark-smoke.yml`
   locally.

### "F1 must be non-negative" failure

This indicates the estimator returned predictions that caused an undefined
or negative F1. Common causes:

- All predictions are the same class (degenerate predictor).
- Numerical instability in the base estimator (try increasing `max_iter`).

Run `tests/test_benchmark_smoke_gate.py::test_smoke_gate_f1_non_negative`
locally with verbose output.

### "Non-deterministic F1" failure

If the determinism test fails, a `random_state` is not being propagated
through the estimator or dataset generator. Check:

- That `BenchmarkRunner(random_state=…)` is passed through.
- That your estimator's `random_state` parameter is set.

### Nightly artifacts missing

If the nightly job finishes but no artifact appears, the benchmark script
likely crashed before writing `benchmark_results.csv`. Check the job log
for the traceback; the script always writes the CSV before calling
`sys.exit(1)` on partial failures.

### Manually re-running the nightly workflow

Go to **Actions → Benchmark Nightly → Run workflow** and optionally
override `n_samples`, `pi`, or `c`.

______________________________________________________________________

## Extending the Gates

To add a new estimator to the **smoke** gate, add a builder function to
`.github/workflows/benchmark-smoke.yml` and to the
`tests/test_benchmark_smoke_gate.py` `_SMOKE_BUILDERS` dict.

To add a new estimator to the **nightly** gate, add a builder to
`.github/workflows/benchmark-nightly.yml`.

See `benchmarks/templates/benchmark_entry_template.py.tmpl` for the
canonical scaffold for a new benchmark entry.
