# AI Agent Journal

## 2026-04-08 — Bug Fix Marathon

Completed a full code review and bug fix session for asgl.

### Bugs Fixed (11 of 12)

**Critical (3/3 fixed):**
- `#163` — Logistic regression objective formula was mathematically incorrect
- `#506` — `_wpca_1` did not handle sparse matrices
- `#519` — `_wpca_pct` failed for sparse X when `variability_pct == 1`

**High (5/5 fixed):**
- `#382` — Classifier validation allowed multi-output y (added explicit `y.ndim > 1` check)
- `#620` — `_wsparse_pca` n_comp missing `+1` offset (also added `max(1, n_comp)` guard)
- `#529-534` — Dense `_wpca_pct` passed float to `PCA(n_components=)` (refactored to use searchsorted)
- `#448` — `__sklearn_tags__()` now sets both `estimator_type` AND semantic `*_tags` objects
- `#437` — Mitigated by #382 fix

**Medium (3/4 fixed):**
- `#446` — `target_tags.multi_output` flipped to True
- `#459` — `binary_only` invalid tag removed, `_more_tags()` simplified
- `#599` — Redundant centering in `_wsparse_pca` removed

**Known Limitation:**
- `#568,600` — `np.var(X, axis=0)` densifies sparse matrices (not fixed, documented as known limitation)

### Test Results
- All 111 tests pass
- 12 warnings (CVXPY reshape FutureWarning — benign, sklearn default order changing)
- 1 RuntimeWarning in PLS weight technique (expected behavior, not a bug)

### Files Modified
- `asgl/skmodels.py` — core fixes
- `tests/test_skmodels.py` — updated expected logit coefficients, updated `_more_tags` assertions
- `tests/test_skmodels_sparse.py` — updated expected logit coefficients, updated `_more_tags` assertions
- `AGENTS.md` — bug dashboard updated with "Fixed" dates

### Notes
- Bug #163 logistic formula fix required flattening both y and pred to 1D for universal element-wise multiply across single and multi-output cases
- Bug #448 (`__sklearn_tags__`) required keeping `_more_tags()` as a minimal override for backward test compatibility
- Tests updated to reflect the correct (fixed) logistic regression coefficients rather than the buggy ones