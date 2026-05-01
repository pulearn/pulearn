# Backwards Compatibility and Deprecation Policy

This document defines `pulearn`'s backward-compatibility commitments, the
deprecation process for removing or changing public API, labeling conventions
for breaking changes in commit messages and release notes, and guidance for
contributors and users planning upgrades.

______________________________________________________________________

## Overview

`pulearn` follows a structured deprecation lifecycle to give users adequate
warning before any breaking change takes effect.  The goals are:

- Users can upgrade without unexpected breakage within a documented grace
  period.
- Deprecation warnings are actionable: they tell you *what* is changing and
  *how* to update your code.
- Contributors know the required steps before landing a breaking change.

______________________________________________________________________

## Versioning Scheme

`pulearn` uses [Semantic Versioning](https://semver.org/) (**SemVer**):

```
MAJOR.MINOR.PATCH
```

| Component | Incremented when …                                                       |
| --------- | ------------------------------------------------------------------------ |
| `MAJOR`   | Incompatible API changes that **cannot** be guarded by a deprecation     |
| `MINOR`   | New backwards-compatible features, or deprecation of existing API        |
| `PATCH`   | Backwards-compatible bug fixes                                           |

A `0.x` series may contain breaking changes in `MINOR` releases.  Once
`1.0.0` is released the guarantees below are binding.

______________________________________________________________________

## Compatibility Guarantee

For any **stable** public symbol (class, function, parameter, attribute):

1. **No surprise removal.** A public API cannot be removed without first
   going through at least one full `MINOR` release as deprecated.
2. **No silent behaviour change.** If the semantics of a stable API change
   in a non-trivial way, the old behaviour must be preserved (and a
   deprecation warning emitted) for at least one `MINOR` release cycle.
3. **Experimental APIs are excluded.** Symbols explicitly documented as
   *experimental* or annotated `# experimental` may change or be removed in
   any `MINOR` release without a deprecation cycle.  They are announced in
   release notes under **Experimental Changes**.

______________________________________________________________________

## What Is Considered Public API

The following are considered **public API** and subject to the guarantee
above:

- All names exported in `pulearn.__init__` (i.e. importable as
  `from pulearn import <name>`).
- All public methods and attributes of `BasePUClassifier` subclasses
  (`fit`, `predict`, `predict_proba`, `classes_`, learned `*_` attrs).
- All public functions and classes in `pulearn.metrics`,
  `pulearn.model_selection`, `pulearn.priors`, and `pulearn.propensity`.
- The `PUAlgorithmSpec` fields exposed by `pulearn.registry`.

The following are **not** considered stable public API:

- Private helpers prefixed with `_`.
- Anything explicitly marked `# experimental` in source or docs.
- Internal constants and module-level implementation details not mentioned
  in the documentation.
- Test utilities in `tests/`.

______________________________________________________________________

## Deprecation Lifecycle

```
Release N   ──► Deprecation warning added (old API still works)
Release N+1 ──► Warning continues; migration guide published
Release N+2 ──► Old API removed (MINOR bump or MAJOR if widespread)
```

In practice:

| Stage              | What happens                                                              |
| ------------------ | ------------------------------------------------------------------------- |
| **Announce**       | A `DeprecationWarning` is added to the old code path.  The warning message includes the first version that will remove the API and the migration path. |
| **Grace period**   | At least **one full `MINOR` release** passes.  The old code path continues to work. |
| **Removal**        | The symbol or behaviour is removed.  The removal is listed in the release notes under **Breaking Changes / Removals**. |

### DeprecationWarning convention

Use `warnings.warn` with `stacklevel=2` so the warning points at the
caller's code rather than the library internals:

```python
import warnings

def old_function(x):
    warnings.warn(
        "old_function() is deprecated since pulearn 0.4 and will be "
        "removed in 0.6. Use new_function() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return new_function(x)
```

The message **must** include:

1. The name of the deprecated symbol.
2. The version in which it was deprecated.
3. The planned removal version (or "a future release" if not yet fixed).
4. The recommended replacement or migration path.

______________________________________________________________________

## Labeling Conventions

### Commit messages

Breaking changes and deprecations **must** be flagged in the commit message
subject line and body:

| Tag                 | Usage                                                          |
| ------------------- | -------------------------------------------------------------- |
| `[deprecate]`       | Adds a `DeprecationWarning`; old behaviour still works         |
| `[breaking]`        | Removes or changes a previously stable API                     |
| `[experimental]`    | Introduces or modifies an explicitly experimental API          |

Examples:

```
feat(metrics): [deprecate] mark estimate_label_frequency_c() as deprecated

Deprecated since 0.4; will be removed in 0.6.
Use MeanPositivePropensityEstimator instead.
```

```
refactor(base): [breaking] remove deprecated hold_out_frac parameter

hold_out_frac was deprecated in 0.3 (replaced by hold_out_ratio).
Removed in 0.5 as announced.
```

### GitHub labels

Pull requests that introduce deprecation or breaking changes should carry
the corresponding GitHub label:

| Label               | Meaning                                      |
| ------------------- | -------------------------------------------- |
| `breaking-change`   | The PR removes or alters stable API          |
| `deprecation`       | The PR adds a deprecation warning            |
| `experimental`      | The PR changes experimental-only API         |

______________________________________________________________________

## Release Notes Requirements

Every release note (see `doc/release_notes_template.md`) **must** include
the following sections when applicable:

- **Deprecations** — new `DeprecationWarning`s added in this release, with
  the planned removal version and migration path.
- **Breaking Changes / Removals** — APIs removed or semantically changed in
  this release.
- **Upgrade Notes** — step-by-step instructions for users who need to
  migrate.

If a section has no entries, it may be omitted or marked `*None.*`.

______________________________________________________________________

## Contributor Guidance

Before landing a PR that touches public API:

1. **Check** whether the change is additive (new optional parameter, new
   class) or breaking (parameter removal, semantic change).
2. **Add a deprecation warning** for any behaviour being removed or changed,
   following the convention above.
3. **Update** the release notes draft with a `[Deprecations]` or
   `[Breaking Changes]` entry.
4. **Update** affected docstrings with a `.. deprecated::` RST directive
   (or Markdown equivalent) citing the deprecation version and replacement.
5. **Add or update** a regression test that asserts the `DeprecationWarning`
   is raised when the old code path is used.

Example docstring deprecation note:

```python
def old_function(x):
    """Compute something.

    .. deprecated:: 0.4
       Use :func:`new_function` instead.
       Will be removed in 0.6.
    """
```

______________________________________________________________________

## Exceptions and Fast-Track Removals

Security vulnerabilities and critical correctness bugs may justify a
**fast-track removal** without the full grace period.  In such cases:

- The rationale is documented in the release notes under
  **Security / Critical Fixes**.
- A `MAJOR` version bump is used if the change is widespread.
- The GitHub issue tracking the vulnerability is linked in the release notes.

______________________________________________________________________

## Related Documents

- [Release Notes Template](release_notes_template.md) — the standard template
  for every `pulearn` release.
- [New Algorithm Checklist](new_algorithm_checklist.md) — required steps when
  adding a learner, including API stability notes.
- [Benchmark Gating Policy](benchmark_gating_policy.md) — CI and benchmark
  pass/fail criteria.
