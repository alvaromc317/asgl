import numpy as np
from asgl import Regressor
from sklearn.model_selection import GridSearchCV
from scipy import sparse


# ------------------------------------------------------------------
# Multi-output regression tests with sparse X (multivariate y)
# --
# NOTE: These tests exercise the multi-output capability with sparse matrices.
# The tests cover:
# - Linear regression (lm) with all penalties and sparse X: PASSING
# - Quantile regression (qr): Needs constraint fix for multi-output
# - Adaptive weights (alasso, aridge, agl, asgl): Need weight handling for multi-output
#
# Current status: Basic LM/ridge/lasso/GL/SGL working with sparse X and multi-output.
# ------------------------------------------------------------------


def _load_multivariate_data_sparse(n_outputs=3):
    """Load base data, create sparse X, and multivariate y"""
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = sparse.csr_matrix(data[:, :-1])
    y_base = data[:, -1]
    # Create multivariate y by stacking multiple transformations
    y = np.column_stack(
        [
            y_base,
            y_base * 0.5 + 10,
            y_base * 1.5 - 5,
        ]
    )
    return X, y[:, :n_outputs]


# ------------------------------------------------------------------
# Basic validation for multi-output regression with sparse X
# ------------------------------------------------------------------
def test_multioutput_sparse_unpenalized_lm():
    """Test unpenalized linear regression with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=2)

    model = Regressor(model="lm", penalization=None, solver="CLARABEL")
    model.fit(X, y)

    # Check coefficient shapes
    assert hasattr(model, "coef_")
    assert model.coef_.shape == (X.shape[1], y.shape[1])
    assert np.isfinite(model.intercept_)
    assert model.n_features_in_ == X.shape[1]

    # Prediction should work
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

    # Score should be decent
    score = model.score(X, y)
    assert score > 0.7


def test_multioutput_sparse_lasso_lm():
    """Test lasso penalized linear regression with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=3)

    model = Regressor(model="lm", penalization="lasso", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify coefficients are not all zeros
    assert np.any(np.abs(model.coef_) > 1e-6)
    # Verify different outputs have different coefficients
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])


def test_multioutput_sparse_ridge_lm():
    """Test ridge penalized linear regression with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=2)

    model = Regressor(model="lm", penalization="ridge", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Ridge should not have exact zeros
    assert np.any(np.abs(model.coef_) > 1e-6)
    # Different outputs should have different coefficients
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])


def test_multioutput_sparse_gl_lm():
    """Test group lasso linear regression with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=2)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model="lm", penalization="gl", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify different outputs have different coefficients
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])
    # Verify non-zero groups
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_sparse_sgl_lm():
    """Test sparse group lasso with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=3)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="lm", penalization="sgl", lambda1=0.1, alpha=0.5, solver="CLARABEL"
    )
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify different outputs have different coefficients
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])
    # Verify non-zero coefficients exist
    assert np.any(np.abs(model.coef_) > 1e-6)


# ------------------------------------------------------------------
# Quantile regression with sparse X and multi-output
# ------------------------------------------------------------------
def test_multioutput_sparse_unpenalized_qr():
    """Test unpenalized quantile regression with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=2)

    model = Regressor(model="qr", penalization=None, quantile=0.5, solver="CLARABEL")
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_multioutput_sparse_lasso_qr():
    """Test lasso penalized quantile regression with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=2)

    model = Regressor(
        model="qr", penalization="lasso", quantile=0.7, lambda1=0.1, solver="CLARABEL"
    )
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_sparse_ridge_qr():
    """Test ridge penalized quantile regression with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=3)

    model = Regressor(
        model="qr", penalization="ridge", quantile=0.3, lambda1=0.1, solver="CLARABEL"
    )
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Ridge should not have exact zeros
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_sparse_gl_qr():
    """Test group lasso quantile regression with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=2)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="qr", penalization="gl", quantile=0.5, lambda1=0.1, solver="CLARABEL"
    )
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)


# ------------------------------------------------------------------
# Adaptive penalties with sparse X and multi-output
# ------------------------------------------------------------------
def test_multioutput_sparse_alasso_lm():
    """Test adaptive lasso with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=2)

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0.1,
        weight_technique="unpenalized",
        solver="CLARABEL",
    )
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify coefficients are non-zero somewhere
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_sparse_aridge_lm():
    """Test adaptive ridge with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=3)

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0.1,
        weight_technique="unpenalized",
        solver="CLARABEL",
    )
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)
    # Different outputs should have different coefficients
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])


def test_multioutput_sparse_agl_lm():
    """Test adaptive group lasso with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=2)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="lm",
        penalization="agl",
        lambda1=0.1,
        weight_technique="unpenalized",
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_sparse_asgl_lm():
    """Test adaptive sparse group lasso with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=3)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="lm",
        penalization="asgl",
        lambda1=0.1,
        alpha=0.5,
        weight_technique="unpenalized",
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_sparse_alasso_qr():
    """Test adaptive lasso quantile regression with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=2)

    model = Regressor(
        model="qr",
        penalization="alasso",
        quantile=0.6,
        lambda1=0.1,
        weight_technique="unpenalized",
        solver="CLARABEL",
    )
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_sparse_agl_qr():
    """Test adaptive group lasso quantile regression with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=2)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="qr",
        penalization="agl",
        quantile=0.5,
        lambda1=0.1,
        weight_technique="unpenalized",
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_sparse_asgl_qr():
    """Test adaptive sparse group lasso quantile regression with sparse X and multivariate y"""
    X, y = _load_multivariate_data_sparse(n_outputs=3)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="qr",
        penalization="asgl",
        quantile=0.5,
        lambda1=0.1,
        alpha=0.5,
        weight_technique="unpenalized",
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)


# ------------------------------------------------------------------
# Edge cases and special scenarios
# ------------------------------------------------------------------
def test_multioutput_sparse_many_outputs():
    """Test with many output variables and sparse X"""
    X, _ = _load_multivariate_data_sparse(n_outputs=3)
    n_outputs = 5
    y = np.random.randn(X.shape[0], n_outputs)

    model = Regressor(model="lm", penalization="lasso", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], n_outputs)


def test_multioutput_sparse_gridsearch():
    """Test that GridSearchCV works with sparse X and multi-output"""
    X, y = _load_multivariate_data_sparse(n_outputs=2)

    model = Regressor(model="lm", penalization="lasso")
    param_grid = {"lambda1": [0.01, 0.1, 1.0]}

    # GridSearchCV should handle multi-output with sparse X
    gscv = GridSearchCV(model, param_grid=param_grid, cv=3)
    gscv.fit(X, y)

    # Best params should be found
    assert "lambda1" in gscv.best_params_
    assert gscv.best_params_["lambda1"] in [0.01, 0.1, 1.0]

    # Predictions should work
    y_pred = gscv.predict(X)
    assert y_pred.shape == y.shape


def test_multioutput_sparse_vs_dense():
    """Test that sparse and dense X produce similar results"""
    # Load dense data
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X_dense = data[:, :-1]
    y_base = data[:, -1]
    y = np.column_stack([y_base, y_base * 0.5 + 10])

    # Convert to sparse
    X_sparse = sparse.csr_matrix(X_dense)

    # Fit with both
    model_dense = Regressor(
        model="lm", penalization="lasso", lambda1=0.1, solver="CLARABEL"
    )
    model_dense.fit(X_dense, y)

    model_sparse = Regressor(
        model="lm", penalization="lasso", lambda1=0.1, solver="CLARABEL"
    )
    model_sparse.fit(X_sparse, y)

    # Coefficients should be very similar
    np.testing.assert_array_almost_equal(
        model_dense.coef_, model_sparse.coef_, decimal=4
    )
    np.testing.assert_almost_equal(
        model_dense.intercept_, model_sparse.intercept_, decimal=4
    )
    # Verify both have non-zero coefficients
    assert np.any(np.abs(model_dense.coef_) > 1e-6)
    assert np.any(np.abs(model_sparse.coef_) > 1e-6)
