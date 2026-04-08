# Sparse Matrix Handling Plan (Simplified)

## Context

After research (2026-04-08), supporting sparse matrices for PLS and SparsePCA weight techniques was found to be impractical:
- `PLSRegression` requires mandatory mean centering which densifies sparse matrices
- `SparsePCA.fit_transform()` requires dense input
- Implementing implicit centering NIPALS is complex and only works for n_comp=1
- The deflation step in NIPALS for multiple components fundamentally cannot work with sparse

**Decision**: Throw a clear error message when sparse input is detected in `_wpls_1`, `_wpls_pct`, and `_wsparse_pca`, with instructions on how to densify.

---

## Changes Required

### 1. `_wpca_pct` (lines 546-557) — Fix n_comp Upper Bound

**Fix**: Add `n_comp = min(n_comp, max_comp)` to sparse branch.

```python
if sparse.issparse(X):
    max_comp = np.min(X.shape) - 1
    pca = PCA(n_components=max_comp, svd_solver="arpack")
    t = pca.fit_transform(X)
    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = (
        np.searchsorted(explained_variance_ratio_cumsum, self.variability_pct)
        + 1
    )
    # FIX: Add upper bound
    n_comp = min(n_comp, max_comp)
    t = t[:, :n_comp]
    p = pca.components_[:n_comp].T
```

### 2. `_wpls_1` (lines 590-597) — Add Sparse Error

```python
def _wpls_1(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
    """
    Weights based on the first partial least squares component.
    """
    if sparse.issparse(X):
        raise ValueError(
            "weight_technique='pls_1' does not support sparse matrices. "
            "PLSRegression requires mean centering which would densify the matrix. "
            "Convert to dense with X = X.toarray() before fitting."
        )
    pls = PLSRegression(n_components=1, scale=False)
    pls.fit(X, y)
    tmp_weight = np.abs(pls.x_rotations_).ravel()
    return tmp_weight
```

### 3. `_wpls_pct` (lines 599-630) — Add Sparse Error + Fix np.var

```python
def _wpls_pct(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
    """
    Weights based on partial least squares.
    """
    if sparse.issparse(X):
        raise ValueError(
            "weight_technique='pls_pct' does not support sparse matrices. "
            "PLSRegression requires mean centering which would densify the matrix. "
            "Convert to dense with X = X.toarray() before fitting."
        )
    total_variance_in_x = np.sum(np.var(X, axis=0))  # Now safe since X is dense
    # ... rest unchanged
```

### 4. `_wsparse_pca` (lines 632-678) — Add Sparse Error + Fix np.var

```python
def _wsparse_pca(self, X: ArrayOrSparse, y: ArrayOrSparse) -> np.ndarray:
    """
    Weights based on sparse principal component analysis.
    """
    if sparse.issparse(X):
        raise ValueError(
            "weight_technique='sparse_pca' does not support sparse matrices. "
            "SparsePCA requires dense input. "
            "Convert to dense with X = X.toarray() before fitting."
        )
    # ... rest unchanged, np.var is now safe since X is dense
```

---

## Summary of Changes

| Method | Change | Reason |
|--------|--------|--------|
| `_wpca_1` | None | Already works with sparse via arpack |
| `_wpca_pct` | Add `n_comp = min(n_comp, max_comp)` | Missing upper bound |
| `_wpls_1` | Add sparse error | PLSRegression requires centering |
| `_wpls_pct` | Add sparse error | PLSRegression requires centering |
| `_wsparse_pca` | Add sparse error | SparsePCA requires dense |
| `_wpls_pct` np.var | Now safe | Only reached with dense input |
| `_wsparse_pca` np.var | Now safe | Only reached with dense input |

---

## Weight Technique Compatibility

| Weight Technique | Sparse Input | Dense Input |
|-----------------|-------------|-------------|
| `pca_1` | ✅ Works | ✅ Works |
| `pca_pct` | ✅ Works (fixed) | ✅ Works |
| `pls_1` | ❌ Error + instructions | ✅ Works |
| `pls_pct` | ❌ Error + instructions | ✅ Works |
| `sparse_pca` | ❌ Error + instructions | ✅ Works |
| `unpenalized` | ✅ Works | ✅ Works |
| `lasso` | ✅ Works | ✅ Works |
| `ridge` | ✅ Works | ✅ Works |

---

## Implementation Order

1. Fix `_wpca_pct` n_comp upper bound (sparse path)
2. Add sparse error to `_wpls_1`
3. Add sparse error to `_wpls_pct`
4. Add sparse error to `_wsparse_pca`
5. Update `.AI/docs/sparse_matrix_handling.md`
6. Update `AGENTS.md` bug dashboard
7. Run tests