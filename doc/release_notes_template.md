# Release Notes Template

Copy this template when drafting release notes for a new `pulearn` version.
Fill in each section; omit or mark `*None.*` any section with no entries.

Replace `X.Y.Z` with the actual version number throughout.

______________________________________________________________________

## pulearn X.Y.Z — YYYY-MM-DD

### Highlights

*One-paragraph summary of the most important changes in this release.*

______________________________________________________________________

### New Features

- **Module / class:** Short description. Example:
  `pulearn.priors.ScarEMPriorEstimator` now supports bootstrap confidence
  intervals via `.bootstrap(...)`.

### Improvements

- **Module / function:** What improved and why.

### Bug Fixes

- **Issue #NNN:** Short description of the bug and the fix. Link to the
  GitHub issue or PR.

______________________________________________________________________

### Deprecations

*List every new `DeprecationWarning` introduced in this release.*

| Deprecated symbol | Deprecated since | Planned removal | Replacement / migration       |
| ----------------- | ---------------- | --------------- | ----------------------------- |
| `old_function()`  | `X.Y.Z`          | `X.Y+2.0`       | Use `new_function()` instead. |

For each entry, include the full migration path:

```python
# Before (deprecated)
from pulearn import old_function

result = old_function(x)

# After
from pulearn import new_function

result = new_function(x)
```

*None.* ← replace with entries or leave this marker

______________________________________________________________________

### Breaking Changes / Removals

*List every API removed or semantically changed in this release.*

> **Note:** Breaking changes to previously stable API require a `MAJOR`
> version bump unless the symbol was already marked deprecated or
> experimental.

| Symbol / behaviour | Changed since | What changed                                 | Migration                                |
| ------------------ | ------------- | -------------------------------------------- | ---------------------------------------- |
| `OldClass.param`   | `X.Y.Z`       | Parameter removed (deprecated in `X.Y-2.0`). | Pass the new kwarg `new_param=` instead. |

*None.* ← replace with entries or leave this marker

______________________________________________________________________

### Upgrade Notes

*Step-by-step instructions for upgrading from the previous release.*

1. **Rename / replace deprecated calls** listed in the Deprecations section
   above.
2. **Check** that any removed parameters are no longer passed.
3. **Re-run** the test suite with `python -m pytest` to catch regressions.

If this release has no deprecations or breaking changes, you can upgrade by
running:

```bash
pip install --upgrade pulearn
```

______________________________________________________________________

### Experimental Changes

*Changes to APIs explicitly marked experimental. These may occur without a
full deprecation cycle.*

| Symbol                | Change                          |
| --------------------- | ------------------------------- |
| `ExperimentalSarHook` | New parameter `clip_min` added. |

*None.* ← replace with entries or leave this marker

______________________________________________________________________

### Dependency and Environment Changes

- Minimum Python version: `3.x`
- Minimum scikit-learn version: `x.y`
- New required dependency: *none* / `package>=x.y`
- Dropped dependency: *none* / `package`

______________________________________________________________________

### Security / Critical Fixes

*Fast-track removals or fixes for security vulnerabilities or critical
correctness bugs. Link to the GitHub advisory or issue.*

*None.* ← replace with entries or leave this marker

______________________________________________________________________

### Contributors

Thank you to everyone who contributed to this release:

- @username — description of contribution

______________________________________________________________________

*See [Backwards Compatibility and Deprecation Policy](compatibility_policy.md)
for the full policy and deprecation lifecycle.*
