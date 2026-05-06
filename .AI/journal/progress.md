# AI Agent Journal

## 2026-04-26 — PCA Optimization and Edge Case Fix

### Optimization: `_wpca_pct` Performance

Optimized the dense branch of `_wpca_pct` in `asgl/skmodels.py`. 
- **Change**: Switched from `svd_solver="arpack"` to `svd_solver="full"`.
- **Reason**: The standard LAPACK SVD solver is much faster for computing almost all components of dense matrices than the iterative ARPACK solver.
- **Parity**: Exact numerical parity is maintained by keeping the manual `np.searchsorted` logic for selecting `n_comp`.

### Bug Fix: PCA Edge Cases

Fixed `ValueError` crashes in `_wpca_pct` for small datasets where `min(X.shape) == 1`.
- **Dense**: Removed the `max_comp = np.min(X.shape) - 1` cap. Now uses `n_components=None` which correctly handles single-feature/single-sample cases.
- **Sparse**: Added a check for `np.min(X.shape) > 1`. If `min == 1`, it now densifies and uses the optimized dense logic, bypassing ARPACK's "k < min(n, p)" limitation.

### Verification
- Full `pytest` suite passed (68 tests).
- Dedicated edge case verification script passed for both dense and sparse single-feature scenarios.

---

## 2026-04-08 — Second Code Review + Sparse Fixes

### Independent Code Review (4 oracle agents)

Found additional bugs beyond the first session:
- `_wpca_pct` sparse branch missing n_comp upper bound
- `_wpls_1`/`_wpls_pct`/`_wsparse_pca` crash on sparse (PLSRegression/SparsePCA require dense)
- `_obtain_beta` `intercept_var = 0` when `fit_intercept=False` → AttributeError (NOT YET FIXED)

### Sparse Matrix Research

Created `.AI/docs/sparse_matrix_handling.md` with comprehensive findings. Key insight: **DO NOT demean sparse — it densifies completely**.

PLSRegression and SparsePCA fundamentally require dense input. Attempted implicit centering NIPALS approach but found it too complex. **Decision**: throw clear ValueError with `X.toarray()` instructions instead.

### Sparse Fix Implementation

**Implemented:**
- `_wpca_pct` sparse: added `n_comp = min(n_comp, max_comp)` upper bound
- `_wpls_1`: added `ValueError` with sparse instructions
- `_wpls_pct`: added `ValueError` with sparse instructions
- `_wsparse_pca`: added `ValueError` with sparse instructions

**Also fixed:**
- cvxpy FutureWarning: added `order="F"` to `cp.reshape` calls (12 warnings → 0)

### Test Results
- **111 passed**, 2 RuntimeWarnings (informative, not bugs)
- 0 FutureWarnings (fixed)

---

## 2026-04-08 — Bug Fix Marathon (First Session)

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
- 0 FutureWarnings (cvxpy reshape order fixed)
- 2 RuntimeWarnings (variability_pct exceeds achievable PLS variance — expected, informative)
- Updated warning messages for clarity: `variability_pct=X was requested, but PLS attains a maximum of Y`

### Files Modified
- `asgl/skmodels.py` — core fixes
- `tests/test_skmodels.py` — updated expected logit coefficients, updated `_more_tags` assertions
- `tests/test_skmodels_sparse.py` — updated expected logit coefficients, updated `_more_tags` assertions
- `AGENTS.md` — bug dashboard updated with "Fixed" dates

### Notes
- Bug #163 logistic formula fix required flattening both y and pred to 1D for universal element-wise multiply across single and multi-output cases
- Bug #448 (`__sklearn_tags__`) required keeping `_more_tags()` as a minimal override for backward test compatibility
- Tests updated to reflect the correct (fixed) logistic regression coefficients rather than the buggy ones