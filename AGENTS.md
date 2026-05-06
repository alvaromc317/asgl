# AGENTS.md

## AI Agent Workflow

Before starting any work, AI agents must:

1. Read this `AGENTS.md` at the repo root for context and constraints
2. Check `.AI/journal/` for progress updates and prior work
3. Check `.AI/plans/` for any active plans or work in progress
4. Check `.AI/docs/` for supplementary documentation

When work is finished:
- **Always** update progress in `.AI/journal/`
- For major changes (e.g., bug fixes), update the bug dashboard below

## Package Overview

Single-package Python library for penalized regression (linear, quantile, logistic).
Main entry point: `from asgl import Regressor` (class in `asgl/regressor.py`).

## Developer Commands

```bash
pytest                    # run all tests (from repo root)
pip install -e .         # dev install in editable mode
```

## Dependencies

- Python >= 3.10
- cvxpy >= 1.5.0, numpy >= 1.20.0, scikit-learn >= 1.6, scipy >= 1.1, pytest >= 7.1.2

## Architecture Notes

- `asgl/__init__.py` exports `Regressor`, `BaseModel`, `AdaptiveWeights`, and `ArrayOrSparse`.
- Core logic is split into multiple modules:
  - `asgl/constants.py`: Centralized constants and custom types.
  - `asgl/utils.py`: Shared utility functions like `_get_group_info`.
  - `asgl/base_model.py`: `BaseModel` (base class for penalized models).
  - `asgl/adaptive_weights.py`: `AdaptiveWeights` (logic for weight estimation).
  - `asgl/regressor.py`: `Regressor` (the primary sklearn-compatible estimator).
- `Regressor` is sklearn-compatible — works with `GridSearchCV`, `RandomizedSearchCV`, `cross_val_predict`.
- Group-indexed penalizations (gl, sgl, agl, asgl) require passing `group_index` array to `fit()`.

## Critical CVXPY Rules (from .jules/)

**Do not wrap `lambda1` or weights in `cp.Parameter`.** The `Regressor` instantiates a new `cp.Problem` per `fit` call, so `cp.Parameter` offers no caching benefit but triggers expensive DPP checks on every rebuild. Use raw floats/numpy arrays instead.

**Validate `canon_backend` via the allowlist** (`ALLOWED_CANON_BACKENDS = {"CPP", "SCIPY", "COO"}`) in `_check_attributes` before passing to cvxpy. Do not blindly pass user strings to backend dispatch.

## Testing Notes

- `pytest.ini` sets `testpaths = tests`.
- Tests use synthetic data fixtures (`make_regression`, `make_classification`) and real CSV files in `tests/` (`data.csv`, `data_logit.csv`).
- Some tests are slow (e.g., solver fallback, sparse matrix tests).

## Bug Dashboard

| Severity | Location | Description | Status |
|----------|----------|-------------|--------|
| Critical | `asgl/base_model.py` | Logistic regression objective formula was mathematically incorrect. | Fixed (2026-04-08) |
| Critical | `asgl/adaptive_weights.py` | `_wpca_1` did not handle sparse matrices. | Fixed (2026-04-08) |
| Critical | `asgl/adaptive_weights.py` | `_wpca_pct` failed for sparse X when `variability_pct == 1`. | Fixed (2026-04-08) |
| High | `asgl/base_model.py` | `predict()` failed with multi-output y for logit. | Mitigated by #382 fix (2026-04-08) |
| High | `asgl/base_model.py` | Classifier validation allowed multi-output y. | Fixed (2026-04-08) |
| High | `asgl/adaptive_weights.py` | Dense `_wpca_pct` passed float to `PCA(n_components=)`. | Fixed (2026-04-08) |
| High | `asgl/adaptive_weights.py` | Inconsistent `n_comp` calculation across weight methods. | Fixed (2026-04-08) |
| High | `asgl/base_model.py` | `__sklearn_tags__()` did not set semantic `*_tags` objects. | Fixed (2026-04-08) |
| Medium | `asgl/base_model.py` | `target_tags.multi_output` was hardcoded `False`. | Fixed (2026-04-08) |
| Medium | `asgl/base_model.py` | `_more_tags()` returned invalid `binary_only` tag. | Fixed (2026-04-08) |
| Medium | `asgl/adaptive_weights.py` | `_wsparse_pca` had redundant centering. | Fixed (2026-04-08) |
| High | `asgl/adaptive_weights.py` | `_wpca_pct` sparse branch missing `n_comp = min(n_comp, max_comp)` upper bound. | Fixed (2026-04-08) |
| High | `asgl/adaptive_weights.py` | `_wpls_1` throws no error for sparse but PLSRegression requires dense. | Fixed (2026-04-08) |
| High | `asgl/adaptive_weights.py` | `_wpls_pct` throws no error for sparse but PLSRegression + np.var require dense. | Fixed (2026-04-08) |
| High | `asgl/adaptive_weights.py` | `_wpca_pct` dense branch was inefficient (arpack) and crashed on min(X.shape)==1. | Fixed (2026-04-26) |
| High | `asgl/adaptive_weights.py` | `_wpca_pct` sparse branch crashed on min(X.shape)==1. | Fixed (2026-04-26) |
| High | `asgl/adaptive_weights.py` | `_wsparse_pca` throws no error for sparse but SparsePCA requires dense. | Fixed (2026-04-08) |

## Sparse Matrix Handling

**DO NOT demean sparse matrices** — it densifies them, defeating the purpose of sparse storage.

- `PCA(svd_solver='arpack')` supports sparse natively — use for `pca_1` and `pca_pct` weight techniques
- `PLSRegression` and `SparsePCA` require dense input — `_wpls_1`, `_wpls_pct`, `_wsparse_pca` raise `ValueError` with densification instructions for sparse input
- `np.var(X, axis=0)` raises `AxisError` on sparse — but the methods using it now raise a clear sparse error first
- See `.AI/docs/sparse_matrix_handling.md` and `.AI/plans/sparse_matrix_plan.md`

## Warnings

- **cvxpy FutureWarning** (12 occurrences, now fixed): `cp.reshape` without explicit `order` — use `order="F"` when reshaping
- **RuntimeWarning** (2 occurrences): `variability_pct` exceeds achievable PLS/SparsePCA explained variance — informative, not a bug

## Repository Map

A full codemap is available at `codemap.md` in the project root.

Before working on any task, read `codemap.md` to understand:
- Project architecture and entry points
- Directory responsibilities and design patterns
- Data flow and integration points between modules

For deep work on a specific folder, also read that folder's `codemap.md`.
