# Sparse Matrix Handling in ASGL

## Critical Principle: DO NOT Demedian Sparse Matrices

Subtracting the mean from a sparse matrix **densifies it completely**:

```python
X_sparse = sparse.random(100, 50, density=0.1, format='csr')
X_centered = X_sparse - X_sparse.mean(axis=0)  # DENSIFIES!
```

---

## Weight Technique Compatibility

| Weight Technique | Sparse Input | Dense Input | Notes |
|-----------------|-------------|-------------|-------|
| `pca_1` | ✅ Works | ✅ Works | PCA with arpack solver |
| `pca_pct` | ✅ Works | ✅ Works | PCA with arpack, searchsorted |
| `pls_1` | ❌ Error | ✅ Works | PLSRegression requires centering |
| `pls_pct` | ❌ Error | ✅ Works | PLSRegression requires centering |
| `sparse_pca` | ❌ Error | ✅ Works | SparsePCA requires dense |
| `unpenalized` | ✅ Works | ✅ Works | |
| `lasso` | ✅ Works | ✅ Works | |
| `ridge` | ✅ Works | ✅ Works | |

---

## Why PLS and SparsePCA Don't Support Sparse

### PLSRegression

`PLSRegression` internally uses mean centering which densifies sparse matrices:

```python
# sklearn/cross_decomposition/_pls.py
def _center_scale_xy(X, y, scale=True):
    x_mean = X.mean(axis=0)
    X -= x_mean  # <-- This densifies sparse!
```

The `scale=False` parameter only disables scaling, not centering. There is no way to disable centering in sklearn's PLSRegression.

### SparsePCA

`SparsePCA.fit_transform()` requires dense input:

```python
SparsePCA().fit_transform(X_sparse)
# TypeError: Sparse data was passed for X, but dense data is required.
```

---

## What To Do With Sparse Input

If you have sparse input and want to use PLS or SparsePCA weight techniques:

### Option 1: Convert to Dense

```python
from scipy import sparse
from asgl import Regressor

X_sparse = sparse.load_npz('my_sparse_matrix.npz')
X_dense = X_sparse.toarray()  # Warning: may use a lot of memory

model = Regressor(weight_technique='pls_1')
model.fit(X_dense, y)
```

### Option 2: Use a Different Weight Technique

For sparse input, use a sparse-compatible weight technique:

```python
model = Regressor(weight_technique='pca_1')  # Works with sparse!
model.fit(X_sparse, y)
```

---

## PCA with Sparse Matrices

`PCA(svd_solver='arpack')` supports sparse matrices natively without densification:

```python
from scipy import sparse
from sklearn.decomposition import PCA

X_sparse = sparse.random(100, 50, density=0.1, format='csr')

# Works with sparse, no densification
pca = PCA(n_components=5, svd_solver='arpack')
pca.fit(X_sparse)
```

### Key Constraints for PCA with Sparse

1. **n_components must be an integer** — fractions (e.g., 0.9) only work with dense
2. **Only 'arpack' or 'covariance_eigh' solvers work** — 'auto' picks one of these for sparse
3. **Variability percentage**: For `pca_pct` with sparse, use `variability_pct` as a fraction (e.g., 0.9) — the code uses searchsorted to find the correct integer n_components

---

## Variance Computation

`np.var(X_sparse, axis=0)` raises `AxisError` on sparse matrices.

For dense matrices, the weight methods use `np.var(X, axis=0)` which works fine.

No sparse-aware variance computation is needed since the methods that use variance (`pls_pct`, `wsparse_pca`) already throw an error for sparse input.

---

## Summary

- **DO NOT demean sparse matrices** — it densifies completely
- **PLS and SparsePCA weight techniques do not support sparse** — they throw a clear error with instructions
- **Use `pca_1` or `pca_pct`** for sparse input — these work natively with arpack solver
- **Convert to dense** if you specifically need PLS or SparsePCA behavior