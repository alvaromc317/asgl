# Contributors

## Project Author and Maintainer

**Álvaro Méndez Civieta** ([@alvaromc317](https://github.com/alvaromc317))

Original author of `asgl` and all core penalized regression algorithms, including:
- The adaptive sparse group lasso formulations (linear, quantile, and logistic regression)
- All penalization methods: lasso, ridge, group lasso, sparse group lasso, and their adaptive variants
- The weight-estimation framework (PCA, PLS, sparse PCA, unpenalized, lasso, ridge weight techniques)
- The scikit-learn compatible `Regressor` API
- The original `asgl/skmodels.py` monolith from which all current modules derive

Research papers underlying the implementation:
- [Adaptive Sparse Group Lasso in Quantile Regression](https://link.springer.com/article/10.1007/s11634-020-00413-8)
- [Adaptive Sparse Group Lasso in Logistic Regression](https://arxiv.org/abs/2111.00472)

---

## Contributors

### zeyuz35 — PR #14 (v2.2.0)

Contributed a major refactor and feature expansion:

**Architecture**
- Split `asgl/skmodels.py` into `base_model.py`, `adaptive_weights.py`,
  `regressor.py`, `constants.py`, and `utils.py`

**New features**
- `scipy.sparse` input support throughout `Regressor.fit()` and all weight methods
- Multivariate Y (multi-output) regression for `lm` and `qr` models
- Solver fallback list: `solver` parameter now accepts a list of solver names
- `verbose` and `canon_backend` constructor parameters with allowlist validation

**Performance improvements**
- Vectorized group weight computation (O(N·G) → O(N log N) via `np.bincount`)
- PLS weight calculation without model refitting
- Single-pass sparse PCA with manual variance truncation
- Quantile loss reformulated as a residual-splitting LP

**Bug fixes**
- Removed module-level `warnings.filterwarnings` (was mutating global state on import)
- Fixed logistic regression target validation (bitwise-OR bug in fallback check)
- Fixed state leakage in `fit_weights` (fitted attributes now use trailing underscore)
- Sparse-aware `decision_function` (uses `X @ coef_` for sparse input)

**Tests**
- Expanded test suite from 24 to 96 test functions across 6 test files
- Added coverage for sparse input, multi-output Y, state leakage, and solver fallback
