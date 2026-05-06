# Contributors

## Project Author and Maintainer

**Álvaro Méndez Civieta** ([@alvaromc317](https://github.com/alvaromc317))

Original author of `asgl` and all core penalized regression algorithms, including:
- The adaptive sparse group lasso formulations (linear, quantile, and logistic regression)
- All penalization methods: lasso, ridge, group lasso, sparse group lasso, and their adaptive variants
- The weight-estimation framework (PCA, PLS, sparse PCA, unpenalized, lasso, ridge weight techniques)
- The CVXPY-based convex optimization formulation underlying all solvers
- The scikit-learn compatible `Regressor` API
- The original `asgl/skmodels.py` monolith from which all current modules derive

Research papers underlying the implementation:
- [Adaptive Sparse Group Lasso in Quantile Regression](https://link.springer.com/article/10.1007/s11634-020-00413-8)
- [Adaptive Sparse Group Lasso in Logistic Regression](https://arxiv.org/abs/2111.00472)

---

## Contributors

### zeyuz35 — PR #14 (v2.2.0)

Refactored `asgl/skmodels.py` into a modular layout (`base_model.py`,
`adaptive_weights.py`, `regressor.py`, `constants.py`, `utils.py`) and added:
- `scipy.sparse` input support, multi-output Y regression, and solver fallback lists
- `verbose` and `canon_backend` constructor parameters
- Performance improvements: vectorized group weights, PLS without refitting, single-pass sparse PCA
- Bug fixes: global `warnings` side-effect, logistic target validation, state leakage in `fit_weights`
- Expanded test suite from 24 to 111 test functions across 6 test files
