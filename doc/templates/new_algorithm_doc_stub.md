## `<Algorithm Name>`

Short description:

- What problem the learner solves
- Assumptions (`SCAR`, `SAR`, or experimental)
- When a user should prefer it over existing methods

Suggested sections:

1. One-paragraph overview.
2. Key parameters and defaults.
3. Label convention expectations.
4. Minimal usage example.
5. Failure modes or caveats.
6. Benchmark note or comparison target.

Example skeleton:

```python
from pulearn import <EstimatorClass>

clf = <EstimatorClass>(<constructor arguments>)
clf.fit(X_train, y_pu)
scores = clf.predict_proba(X_test)
```
