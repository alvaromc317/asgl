# asgl <img src="figures/logo.png" align="right" height="140" alt="asgl logo" />


[![PyPI version](https://img.shields.io/badge/version-2.2.0-blue.svg)](https://pypi.org/project/asgl/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![Downloads](https://pepy.tech/badge/asgl)](https://pepy.tech/project/asgl)
[![Downloads/month](https://pepy.tech/badge/asgl/month)](https://pepy.tech/project/asgl)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


## Introduction

`asgl` fits penalized regression models for high-dimensional variable selection.
It supports linear (`lm`), quantile (`qr`), and logistic (`logit`) regression,
with a rich menu of penalizations — from plain Lasso to Adaptive Sparse Group
Lasso (ASGL) — all through a single scikit-learn compatible `Regressor` class.

The package is especially useful when:
- Variables have a known group structure (gene pathways, dummy-variable families, …)
- You need simultaneous group- and individual-level sparsity
- You want adaptive weights to improve oracle properties
- Your design matrix `X` is a `scipy.sparse` matrix

Based on:
- [Adaptive Sparse Group Lasso in Quantile Regression](https://link.springer.com/article/10.1007/s11634-020-00413-8)
- [`asgl`: A Python Package for Penalized Linear and Quantile Regression](https://arxiv.org/abs/2111.00472)

---

## Features

| Feature | Details |
|---------|---------|
| **Models** | Linear (`lm`), quantile (`qr`), logistic binary classification (`logit`) |
| **Penalizations** | `lasso`, `ridge`, `gl`, `sgl`, `alasso`, `aridge`, `agl`, `asgl`, or `None` |
| **Sparse input** |Both dense and `scipy.sparse` matrices accepted.  |
| **Multi-output Y** | `lm` and `qr` accept a 2D `y` matrix for simultaneous multi-response fitting |
| **Solver fallback** | `solver` accepts a list; falls back through installed CVXPY solvers automatically |
| **Adaptive weights** | 8 built-in weight techniques: `pca_pct`, `pca_1`, `pls_pct`, `pls_1`, `lasso`, `ridge`, `unpenalized`, `sparse_pca` |
| **sklearn API** | Full `fit` / `predict` / `score` / `GridSearchCV` / `cross_val_predict` support |

---

## Installation

```bash
pip install asgl
```

**Requirements:** Python >= 3.10, cvxpy >= 1.5.0, numpy >= 1.20.0, scikit-learn >= 1.6, scipy >= 1.1

To run the test suite after installation:

```bash
pytest
```

---

## Quickstart

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from asgl import Regressor

X, y = make_regression(n_samples=500, n_features=50, n_informative=20,
                       noise=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Regressor(model='lm', penalization='lasso', lambda1=0.1)
model.fit(X_train, y_train)

print(model.coef_)
print(mean_squared_error(y_test, model.predict(X_test)))
```

---

## API Reference

### `Regressor`

```python
from asgl import Regressor

Regressor(
    model='lm',                  # 'lm' | 'qr' | 'logit'
    penalization='lasso',        # see Penalizations table below, or None
    quantile=0.5,                # quantile level (qr only)
    fit_intercept=True,
    lambda1=0.1,                 # penalization strength
    alpha=0.5,                   # lasso/group-lasso tradeoff for sgl/asgl
    solver='default',            # str or list[str] — CVXPY solver(s)
    canon_backend='CPP',         # 'CPP' | 'SCIPY' | 'COO'
    verbose=False,
    weight_technique='pca_pct',  # adaptive weight method (adaptive penalties only)
    individual_power_weight=1,
    group_power_weight=1,
    variability_pct=0.9,
    lambda1_weights=0.1,
    spca_alpha=1e-5,
    spca_ridge_alpha=1e-2,
    individual_weights=None,     # override weight estimation with custom array
    group_weights=None,
    tol=1e-3,
    weight_tol=1e-4,
)
```

**Penalizations**

| `penalization` | Type | Group structure required |
|---------------|------|--------------------------|
| `None` | Unpenalized | No |
| `'lasso'` | Individual | No |
| `'ridge'` | Individual | No |
| `'gl'` | Group | Yes |
| `'sgl'` | Individual + Group | Yes |
| `'alasso'` | Adaptive individual | No |
| `'aridge'` | Adaptive individual | No |
| `'agl'` | Adaptive group | Yes |
| `'asgl'` | Adaptive individual + Group | Yes |

**Key methods**

| Method | Description |
|--------|-------------|
| `fit(X, y, group_index=None)` | Fit the model |
| `predict(X)` | Predict (regression output or class labels for logit) |
| `predict_proba(X)` | Class probabilities (logit only) |
| `decision_function(X)` | Raw linear scores |
| `score(X, y)` | R² (regression) or accuracy (classifier) |

**Fitted attributes**

| Attribute | Description |
|-----------|-------------|
| `coef_` | `(n_features,)` or `(n_features, n_outputs)` coefficient array |
| `intercept_` | Intercept scalar |
| `n_features_in_` | Number of features seen during fit |
| `solver_stats_` | Dict with solver name, iterations, timing |

---

## Examples

### 1 — Quantile regression with Adaptive Sparse Group Lasso + cross-validation

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from asgl import Regressor

X, y = make_regression(n_samples=1000, n_features=50, n_informative=25,
                       noise=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

group_index = np.repeat(np.arange(1, 11), 5)   # 10 groups of 5 features each

model = Regressor(model='qr', penalization='asgl', quantile=0.5)

param_grid = {
    'lambda1': [1e-3, 1e-2, 1e-1, 1.0],
    'alpha':   [0.0, 0.5, 1.0],
}
cv = RandomizedSearchCV(model, param_grid, scoring='neg_median_absolute_error',
                        n_iter=12, cv=5)
cv.fit(X_train, y_train, **{'group_index': group_index})
print(cv.best_params_)
print(cv.score(X_test, y_test))
```

### 2 — Sparse input (`scipy.sparse`)

```python
import scipy.sparse as sp
from sklearn.datasets import make_regression
from asgl import Regressor

X_dense, y = make_regression(n_samples=500, n_features=200, n_informative=30,
                              random_state=0)
X = sp.random(500, 200, density=0.05, format='csr')  # or your real sparse matrix

model = Regressor(model='lm', penalization='lasso', lambda1=0.05)
model.fit(X, y)
print(f"Non-zero coefficients: {(model.coef_ != 0).sum()}")
```

### 3 — Multi-output regression

```python
import numpy as np
from sklearn.datasets import make_regression
from asgl import Regressor

X, y_1d = make_regression(n_samples=300, n_features=30, n_informative=10,
                           noise=3, random_state=7)
y = np.column_stack([y_1d, y_1d * 0.5 + np.random.randn(300) * 2])  # 2 outputs

group_index = np.repeat(np.arange(1, 6), 6)   # 5 groups

model = Regressor(model='lm', penalization='gl', lambda1=0.1)
model.fit(X, y, group_index=group_index)
print(model.coef_.shape)   # (n_features, 2)
```

### 4 — Solver fallback

```python
from asgl import Regressor

# Try CLARABEL first, then SCS, then let cvxpy choose
model = Regressor(model='lm', penalization='lasso',
                  solver=['CLARABEL', 'SCS', 'default'])
model.fit(X_train, y_train)
print(model.solver_stats_['solver_name'])
```

### 5 — Logistic regression with custom decision threshold

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score
from asgl import Regressor

X, y = make_classification(n_samples=1000, n_features=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = Regressor(model='logit', penalization='ridge')
proba_cv = cross_val_predict(model, X_train, y_train, method='predict_proba', cv=5)

# Find threshold that maximises CV accuracy
thresholds = np.linspace(0.01, 0.99, 99)
best_thr = thresholds[np.argmax(
    [accuracy_score(y_train, (proba_cv[:, 1] >= t).astype(int)) for t in thresholds]
)]

model.fit(X_train, y_train)
test_preds = (model.predict_proba(X_test)[:, 1] >= best_thr).astype(int)
print(f"Test accuracy: {accuracy_score(y_test, test_preds):.3f}")
```

---

## Citation

If you use `asgl` in a scientific publication, please cite:

```bibtex
@article{mendez2022adaptive,
  title   = {Adaptive sparse group lasso in quantile regression},
  author  = {M{\'e}ndez-Civieta, {\'A}lvaro and Aguilera-Morillo, M Carmen and Lillo, Rosa E},
  journal = {Advances in Data Analysis and Classification},
  year    = {2021},
  doi     = {10.1007/s11634-020-00413-8}
}
```

[Full paper](https://link.springer.com/article/10.1007/s11634-020-00413-8) |
[Package paper](https://arxiv.org/abs/2111.00472) |
[Towards Data Science walkthrough](https://towardsdatascience.com/sparse-group-lasso-in-python-255e379ab892)

---

## Contributions

Contributions are welcome! Please open an issue to discuss ideas or submit a pull request.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a full list of contributors.

### Acknowledgments

v2.2.0 incorporates a major contribution from
[zeyuz35](https://github.com/zeyuz35):
sparse matrix support, multi-output Y regression, solver fallbacks,
performance improvements (vectorized group weights, PLS optimization),
and an expanded test suite.
See [CONTRIBUTORS.md](CONTRIBUTORS.md) for details.

---

## What's new?

### 2.2.0
- Sparse matrix (`scipy.sparse`) input support throughout
- Multivariate Y (multi-output) for `lm` and `qr` models
- `solver` accepts a list of names with automatic fallback
- New parameters: `verbose`, `canon_backend`
- Performance: vectorized group weights, PLS without refitting
- Internal refactor: `skmodels.py` → 5 focused modules
- Test suite: 24 → 96 test functions
- Requires Python >= 3.10

### 2.1.4
- scikit-learn estimator tag compliance
- Quantile loss optimized via residual-splitting LP

### 2.1.3
- Logistic model rewritten: `predict_proba`, `decision_function` added
- `logit_proba` and `logit_raw` model types removed

### 2.1.0
- Ridge and adaptive ridge penalizations added (`'ridge'`, `'aridge'`)

### 2.0.0
- `Regressor` class introduced with full scikit-learn compatibility

---

## License

GPL-3.0 — open source, modifications must be redistributed under the same license.
See [LICENSE](LICENSE) for full text.
