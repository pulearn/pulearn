# New Algorithm Checklist

Use this checklist whenever you add a PU learner to `pulearn`.

1. Register the estimator in `src/pulearn/registry.py`.
2. Add focused regression coverage using
   `tests/templates/test_new_algorithm_template.py.tmpl`.
3. Run the shared BasePUClassifier contract helper from
   `tests/templates/test_api_contract_template.py.tmpl` when the learner
   uses the common API foundations.
4. Add a user-facing docs section from
   `doc/templates/new_algorithm_doc_stub.md`.
5. Add a benchmark placeholder or runnable benchmark entry using
   `benchmarks/templates/benchmark_entry_template.py.tmpl`.
6. Update `README.rst` or `src/pulearn/documentation.md` if the new learner
   changes how users choose or configure algorithms.

Required metadata for each registry entry:

- Stable registry key
- Estimator class
- Method family
- Assumption (`SCAR`, `SAR`, or `SCAR/SAR`)
- Short summary
- Primary docs reference
- Primary test reference
- Benchmark reference
- Shared contract reference

Recommended review before opening a PR:

- Confirm label handling follows the package-wide normalization policy.
- Confirm `fit` returns `self` and exposes `classes_` after fitting.
- Confirm `predict_proba` shape and numeric behavior are documented.
- Call out any deviations from the shared base contract in the registry entry.
- If any existing public API is altered or removed, follow the deprecation
  steps in `doc/compatibility_policy.md` before merging.
