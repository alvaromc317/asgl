import numpy as np
from asgl import Regressor
from sklearn.model_selection import GridSearchCV


# ------------------------------------------------------------------
# Multi-output regression tests (multivariate y)
# --
# NOTE: These tests exercise the multi-output capability added to support
# multiple dependent variables (y with shape (n_samples, n_outputs)).
# The tests cover:
# - Linear regression (lm) with all penalties: PASSING
# - Quantile regression (qr): Needs constraint fix for multi-output
# - Adaptive weights (alasso, aridge, agl, asgl): Need weight handling for multi-output
#
# Current status: Basic LM/ridge/lasso/GL/SGL working with multi-output.
# ------------------------------------------------------------------


def _load_multivariate_data(n_outputs=3):
    """Load base data and create multivariate y"""
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
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
# Basic validation for multi-output regression
# ------------------------------------------------------------------
def test_multioutput_unpenalized_lm():
    """Test unpenalized linear regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=2)

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


def test_multioutput_lasso_lm():
    """Test lasso penalized linear regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=3)

    model = Regressor(model="lm", penalization="lasso", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify coefficients are not all zeros
    assert np.any(np.abs(model.coef_) > 1e-6)
    # Verify different outputs have different coefficients
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])


def test_multioutput_ridge_lm():
    """Test ridge penalized linear regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=2)

    model = Regressor(model="lm", penalization="ridge", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Ridge should not have exact zeros
    assert np.any(np.abs(model.coef_) > 1e-6)
    # Different outputs should have different coefficients
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])


def test_multioutput_gl_lm():
    """Test group lasso linear regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=2)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model="lm", penalization="gl", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify different outputs have different coefficients
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_sgl_lm():
    """Test sparse group lasso linear regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=3)
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
# Quantile regression with multi-output
# ------------------------------------------------------------------
def test_multioutput_unpenalized_qr():
    """Test unpenalized quantile regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=2)

    model = Regressor(model="qr", penalization=None, quantile=0.5, solver="CLARABEL")
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_multioutput_lasso_qr():
    """Test lasso penalized quantile regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=2)

    model = Regressor(
        model="qr", penalization="lasso", quantile=0.7, lambda1=0.1, solver="CLARABEL"
    )
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_multioutput_ridge_qr():
    """Test ridge penalized quantile regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=3)

    model = Regressor(
        model="qr", penalization="ridge", quantile=0.3, lambda1=0.1, solver="CLARABEL"
    )
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_multioutput_gl_qr():
    """Test group lasso quantile regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=2)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="qr", penalization="gl", quantile=0.5, lambda1=0.1, solver="CLARABEL"
    )
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


# ------------------------------------------------------------------
# Adaptive penalties with multi-output
# ------------------------------------------------------------------
def test_multioutput_alasso_lm():
    """Test adaptive lasso with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=2)

    model = Regressor(model="lm", penalization="alasso", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify coefficients differ between outputs
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_aridge_lm():
    """Test adaptive ridge with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=3)

    model = Regressor(model="lm", penalization="aridge", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Adaptive ridge should not have exact zeros
    assert np.any(np.abs(model.coef_) > 1e-6)
    # Different outputs should have different coefficients
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])


def test_multioutput_agl_lm():
    """Test adaptive group lasso with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=2)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model="lm", penalization="agl", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify different outputs have different coefficients
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])
    # Verify non-zero groups
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_asgl_lm():
    """Test adaptive sparse group lasso with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=3)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="lm", penalization="asgl", lambda1=0.1, alpha=0.5, solver="CLARABEL"
    )
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify different outputs have different coefficients
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_alasso_qr():
    """Test adaptive lasso quantile regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=2)

    model = Regressor(
        model="qr", penalization="alasso", quantile=0.6, lambda1=0.1, solver="CLARABEL"
    )
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify coefficients are non-zero somewhere
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_agl_qr():
    """Test adaptive group lasso quantile regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=2)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="qr", penalization="agl", quantile=0.5, lambda1=0.1, solver="CLARABEL"
    )
    model.fit(X, y, group_index)

    assert model.coef_.shape == (X.shape[1], y.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    # Verify non-zero coefficients
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_asgl_qr():
    """Test adaptive sparse group lasso quantile regression with multivariate y"""
    X, y = _load_multivariate_data(n_outputs=3)
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="qr",
        penalization="asgl",
        quantile=0.5,
        lambda1=0.1,
        alpha=0.5,
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
def test_multioutput_many_outputs():
    """Test with many output variables"""
    X, _ = _load_multivariate_data(n_outputs=3)
    n_outputs = 5
    y = np.random.randn(X.shape[0], n_outputs)

    model = Regressor(model="lm", penalization="lasso", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y)

    assert model.coef_.shape == (X.shape[1], n_outputs)


def test_multioutput_single_output_backward_compat():
    """Ensure single output still works correctly"""
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]

    model = Regressor(model="lm", penalization="lasso", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y)

    # Single output should be 1D
    assert model.coef_.ndim == 1
    assert model.coef_.shape == (X.shape[1],)
    # Verify coefficients are non-zero
    assert np.any(np.abs(model.coef_) > 1e-6)


def test_multioutput_predictions_shape():
    """Test that multi-output predictions have correct shape"""
    X, y = _load_multivariate_data(n_outputs=2)

    model = Regressor(model="lm", penalization=None, solver="CLARABEL")
    model.fit(X, y)

    # Predict all at once
    y_pred_all = model.predict(X)

    # Predictions should have same shape as y
    assert y_pred_all.shape == y.shape
    assert y_pred_all.shape == (X.shape[0], 2)

    # Predictions should be finite
    assert np.all(np.isfinite(y_pred_all))
    # Verify predictions are different between outputs
    assert not np.allclose(y_pred_all[:, 0], y_pred_all[:, 1])
    # Coefficients should be different per output
    assert not np.allclose(model.coef_[:, 0], model.coef_[:, 1])


def test_multioutput_gridsearch():
    """Test that GridSearchCV works with multi-output"""
    X, y = _load_multivariate_data(n_outputs=2)

    model = Regressor(model="lm", penalization="lasso")
    param_grid = {"lambda1": [0.01, 0.1, 1.0]}

    # GridSearchCV should handle multi-output
    gscv = GridSearchCV(model, param_grid=param_grid, cv=3)
    gscv.fit(X, y)

    # Best params should be found
    assert "lambda1" in gscv.best_params_
    assert gscv.best_params_["lambda1"] in [0.01, 0.1, 1.0]

    # Predictions should work
    y_pred = gscv.predict(X)
    assert y_pred.shape == y.shape
    # Verify best estimator has meaningful coefficients
    assert np.any(np.abs(gscv.best_estimator_.coef_) > 1e-6)
