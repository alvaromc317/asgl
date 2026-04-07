import pytest
import numpy as np
from asgl import Regressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy import sparse


# ------------------------------------------------------------------
# Basic validation of constructor arguments
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_kwargs",
    [
        dict(model="foo"),  # unsupported model
        dict(penalization="foo"),  # unsupported penalty
        dict(lambda1=-0.1),  # negative λ
        dict(alpha=1.5),  # alpha outside [0, 1]
    ],
)
def test_bad_constructor_arguments_raises(bad_kwargs):
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    reg = Regressor(**bad_kwargs)
    with pytest.raises(ValueError):
        reg.fit(X, y)


# ------------------------------------------------------------------
# Regressor vs Classifier automatic tag / estimator type
# ------------------------------------------------------------------
def test_estimator_type_tags():
    reg = Regressor(model="lm")
    clf = Regressor(model="logit")
    assert reg._estimator_type == "regressor"
    assert clf._estimator_type == "classifier"
    tags = clf._more_tags()
    assert tags["requires_y"] is True
    assert tags["allow_nan"] is False


# ------------------------------------------------------------------
# Fit, predict & score for ordinary linear models
# ------------------------------------------------------------------
@pytest.mark.parametrize("penalty", [None, "lasso", "ridge"])
def test_linear_regression_basic_behaviour(penalty):
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    reg = Regressor(model="lm", penalization=penalty, lambda1=0.1, tol=1e-4)
    reg.fit(X, y)
    # Fitted attributes ------------------------------------------------
    assert hasattr(reg, "coef_")
    assert reg.coef_.shape == (X.shape[1],)
    assert np.isfinite(reg.intercept_)
    assert reg.n_features_in_ == X.shape[1]
    # Prediction API ---------------------------------------------------
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
    assert reg.score(X, y) > 0.8


# ------------------------------------------------------------------
# Classifier path: decision_function / predict_proba / score
# ------------------------------------------------------------------
@pytest.mark.parametrize("penalty", [None, "lasso", "ridge"])
def test_logistic_classifier_api(penalty):
    data = np.loadtxt("data_logit.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1].astype("int")
    clf = Regressor(model="logit", penalization=penalty, lambda1=0.2, solver="SCS")
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    y_hat = clf.predict(X)
    assert set(np.unique(y_hat)) <= {0, 1}

    # Accuracy on training data should be high for the easy toy set
    acc = clf.score(X, y)
    assert acc >= 0.8


# TEST UNPENALIZED --------------------------------------------------------


def test_unpenalized_lm():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]

    model = Regressor(model="lm", penalization=None, solver="CLARABEL")
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Unpenalized lm failure",
    )
    np.testing.assert_array_almost_equal(
        model.intercept_,
        np.array([7.2923]),
        decimal=3,
        err_msg="Unpenalized lm failure",
    )


def test_unpenalized_qr():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]

    model = Regressor(model="qr", penalization=None, quantile=0.8, solver="CLARABEL")
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                25.92647702,
                15.32296485,
                26.18622489,
                59.82066746,
                99.17547549,
                17.72659176,
                11.82218774,
                34.74489955,
                58.59336065,
                65.68631127,
            ]
        ),
        decimal=3,
        err_msg="Unpenalized qr failure for quantile 0.8",
    )

    model = Regressor(model="qr", penalization=None, quantile=0.2, solver="CLARABEL")
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                24.53442726,
                15.60297538,
                24.87555318,
                58.24387077,
                99.36779578,
                15.09292701,
                12.15865,
                33.94239404,
                63.53956298,
                66.47009116,
            ]
        ),
        decimal=3,
        err_msg="Unpenalized qr failure for quantile 0.2",
    )


def test_unpenalized_logit():
    data = np.loadtxt("data_logit.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1].astype("int")

    model = Regressor(model="logit", penalization=None, solver="SCS")
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                -0.63590658,
                2.0383636,
                0.60518489,
                11.52344861,
                -0.55281497,
                -25.8225801,
                -12.00634409,
                2.66287183,
                2.72357986,
                -10.73596602,
            ]
        ),
        decimal=3,
        err_msg="Unpenalized logit failure",
    )


# TEST LASSO PENALIZATION ---------------------------------------------------------------------------------------------


def test_lasso_lm():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]

    model = Regressor(model="lm", penalization="lasso", lambda1=0, solver="CLARABEL")
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Lasso lm failure for lambda=0",
    )

    model = Regressor(model="lm", penalization="lasso", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.25821113,
                15.00872081,
                25.33148317,
                56.132575,
                99.34021902,
                15.39857671,
                10.35704837,
                34.86962766,
                61.42098122,
                66.24809898,
            ]
        ),
        decimal=3,
        err_msg="Lasso lm failure for lambda=0.1",
    )


def test_lasso_qr():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]

    model = Regressor(
        model="qr",
        penalization="lasso",
        quantile=0.8,
        lambda1=0,
        solver="CLARABEL",
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                25.92647702,
                15.32296485,
                26.18622489,
                59.82066746,
                99.17547549,
                17.72659176,
                11.82218774,
                34.74489955,
                58.59336065,
                65.68631127,
            ]
        ),
        decimal=3,
        err_msg="Lasso qr failure for quantile 0.8 and lambda1=0",
    )

    model = Regressor(
        model="qr",
        penalization="lasso",
        quantile=0.8,
        lambda1=0.1,
        solver="CLARABEL",
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                0.0,
                0.0,
                0.0,
                27.66071305,
                90.01397685,
                0.0,
                0.0,
                21.77214275,
                45.01300237,
                36.34016822,
            ]
        ),
        decimal=3,
        err_msg="Lasso qr failure for quantile 0.8 and lambda1=0.1",
    )

    model = Regressor(
        model="qr",
        penalization="lasso",
        quantile=0.2,
        lambda1=0.1,
        solver="CLARABEL",
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                0.0,
                2.37522451,
                0.0,
                22.15449983,
                77.79554315,
                0.0,
                0.0,
                38.65124062,
                30.55826802,
                29.50093582,
            ]
        ),
        decimal=3,
        err_msg="Lasso qr failure for quantile 0.2 and lambda1=0.1",
    )


# TEST RIDGE PENALIZATION -----------------------------------------------


def test_ridge_lm():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]

    model = Regressor(model="lm", penalization="ridge", lambda1=0, solver="CLARABEL")
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Ridge lm failure for lambda=0",
    )

    model = Regressor(model="lm", penalization="ridge", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                18.46772718,
                15.2438716,
                25.06222985,
                50.31818813,
                89.9845044,
                14.03017725,
                8.6725918,
                33.33322947,
                55.33924112,
                58.02248048,
            ]
        ),
        decimal=3,
        err_msg="Ridge lm failure for lambda=0.1",
    )


# TEST GROUP LASSO PENALIZATION ---------------------------------------------


def test_gl_lm():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model="lm", penalization="gl", lambda1=0, solver="CLARABEL")
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Group lasso lm failure for lambda=0",
    )

    model = Regressor(model="lm", penalization="gl", lambda1=0.1, solver="CLARABEL")
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.27677831,
                15.0266414,
                25.33723793,
                56.16210945,
                99.29269691,
                15.45069947,
                10.37091253,
                34.87509388,
                61.40852042,
                66.24043667,
            ]
        ),
        decimal=3,
        err_msg="Group lasso lm failure for lambda=0.1",
    )


def test_gl_qr():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="qr",
        penalization="gl",
        quantile=0.8,
        lambda1=0,
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                25.92647702,
                15.32296485,
                26.18622489,
                59.82066746,
                99.17547549,
                17.72659176,
                11.82218774,
                34.74489955,
                58.59336065,
                65.68631127,
            ]
        ),
        decimal=3,
        err_msg="Group lasso qr failure for quantile 0.8 and lambda1=0",
    )

    model = Regressor(
        model="qr",
        penalization="gl",
        quantile=0.8,
        lambda1=0.1,
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                1.33241462e01,
                3.66710686e01,
                4.56237828e00,
                0.00000000e00,
                0.00000000e00,
                1.02229130e-04,
                1.07166280e-04,
            ]
        ),
        decimal=3,
        err_msg="Group lasso qr failure for quantile 0.8 and lambda1=0.1",
    )

    model = Regressor(
        model="qr",
        penalization="gl",
        quantile=0.2,
        lambda1=0.1,
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                0.0,
                0.65037989,
                0.62171602,
                27.42813623,
                55.53152069,
                7.38334327,
                0.0,
                34.14211386,
                21.23025218,
                30.17858448,
            ]
        ),
        decimal=3,
        err_msg="Group lasso qr failure for quantile 0.2 and lambda1=0.1",
    )


# TEST SPARSE GROUP LASSO PENALIZATION ----------------------------------------


def test_sgl_lm():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model="lm", penalization="sgl", lambda1=0, solver="CLARABEL")
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Sparse group lasso lm failure for lambda=0",
    )

    model = Regressor(
        model="lm", penalization="sgl", lambda1=0.1, alpha=0, solver="CLARABEL"
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.27677831,
                15.0266414,
                25.33723793,
                56.16210945,
                99.29269691,
                15.45069947,
                10.37091253,
                34.87509388,
                61.40852042,
                66.24043667,
            ]
        ),
        decimal=3,
        err_msg="Sparse group lasso lm failure for lambda=0.1 and alpha=0",
    )

    model = Regressor(
        model="lm", penalization="sgl", lambda1=0.1, alpha=1, solver="CLARABEL"
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.25821113,
                15.00872081,
                25.33148317,
                56.132575,
                99.34021902,
                15.39857671,
                10.35704837,
                34.86962766,
                61.42098122,
                66.24809898,
            ]
        ),
        decimal=3,
        err_msg="Sparse group lasso lm failure for lambda=0.1 and alpha=1",
    )

    model = Regressor(
        model="lm",
        penalization="sgl",
        lambda1=0.1,
        alpha=0.5,
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.26750237,
                15.01768931,
                25.33436154,
                56.14735784,
                99.3164447,
                15.42465486,
                10.36398602,
                34.8723578,
                61.41474779,
                66.24426926,
            ]
        ),
        decimal=3,
        err_msg="Sparse group lasso lm failure for lambda=0.1 and alpha=0.5",
    )


def test_sgl_qr():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="qr",
        penalization="sgl",
        quantile=0.8,
        lambda1=0,
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                25.92647702,
                15.32296485,
                26.18622489,
                59.82066746,
                99.17547549,
                17.72659176,
                11.82218774,
                34.74489955,
                58.59336065,
                65.68631127,
            ]
        ),
        decimal=3,
        err_msg="Sparse group lasso qr failure for quantile 0.8 and lambda1=0",
    )

    model = Regressor(
        model="qr",
        penalization="sgl",
        quantile=0.8,
        lambda1=0.1,
        alpha=0,
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                1.33241462e01,
                3.66710686e01,
                4.56237828e00,
                0.00000000e00,
                0.00000000e00,
                1.02229130e-04,
                1.07166280e-04,
            ]
        ),
        decimal=3,
        err_msg="Sparse group lasso qr failure for quantile 0.8, lambda1=0.1 and alpha=0",
    )

    model = Regressor(
        model="qr",
        penalization="sgl",
        quantile=0.8,
        lambda1=0.1,
        alpha=1,
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                0.0,
                0.0,
                0.0,
                27.66071305,
                90.01397685,
                0.0,
                0.0,
                21.77214275,
                45.01300237,
                36.34016822,
            ]
        ),
        decimal=3,
        err_msg="Sparse group lasso qr failure for quantile 0.8, lambda1=0.1 and alpha=1",
    )

    model = Regressor(
        model="qr",
        penalization="sgl",
        quantile=0.8,
        lambda1=0.1,
        alpha=0.5,
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                0.0,
                0.0,
                0.0,
                21.95907657,
                81.32544812,
                2.84475096,
                0.0,
                20.76154792,
                35.57092954,
                36.05204591,
            ]
        ),
        decimal=3,
        err_msg="Sparse group lasso qr failure for quantile 0.8, lambda1=0.1 and alpha=0.5",
    )

    model = Regressor(
        model="qr",
        penalization="sgl",
        quantile=0.2,
        lambda1=0.1,
        alpha=0.5,
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                0.00000000e00,
                3.97542269e-04,
                6.63474565e-04,
                2.09861368e01,
                6.62803756e01,
                3.97923687e-01,
                0.00000000e00,
                3.84701424e01,
                2.45480120e01,
                3.01299190e01,
            ]
        ),
        decimal=3,
        err_msg="Sparse group lasso qr failure for quantile 0.2, lambda1=0.1 and alpha=0.5",
    )


# ADAPTIVE LASSO ----------------------------------------------------


def test_alasso_lm():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0,
        weight_technique="unpenalized",
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Adaptive lasso lm failure for lambda=0",
    )

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0.1,
        individual_weights=[0] * 10,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Adaptive lasso lm failure for lambda=0.1 and weights=0",
    )

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0.1,
        individual_power_weight=0,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.25821113,
                15.00872081,
                25.33148317,
                56.132575,
                99.34021902,
                15.39857671,
                10.35704837,
                34.86962766,
                61.42098122,
                66.24809898,
            ]
        ),
        decimal=3,
        err_msg="Adaptive lasso lm failure for lambda=0.1 and power_weight=0",
    )

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41482334,
                15.0351054,
                25.42670541,
                56.26480226,
                99.31477567,
                15.48593938,
                10.47706149,
                34.8790109,
                61.46425705,
                66.32666869,
            ]
        ),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="unpenalized" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0.1,
        weight_technique="unpenalized",
        variability_pct=1,
        individual_power_weight=1.2,
        solver="CLARABEL",
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41482334,
                15.0351054,
                25.42670541,
                56.26480226,
                99.31477567,
                15.48593938,
                10.47706149,
                34.8790109,
                61.46425705,
                66.32666869,
            ]
        ),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="unpenalized", variability_pct=1 and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41482334,
                15.0351054,
                25.42670541,
                56.26480226,
                99.31477567,
                15.48593938,
                10.47706149,
                34.8790109,
                61.46425705,
                66.32666869,
            ]
        ),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="pca_1" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41482334,
                15.0351054,
                25.42670541,
                56.26480226,
                99.31477567,
                15.48593938,
                10.47706149,
                34.8790109,
                61.46425705,
                66.32666869,
            ]
        ),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="pls_1" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41482334,
                15.0351054,
                25.42670541,
                56.26480226,
                99.31477567,
                15.48593938,
                10.47706149,
                34.8790109,
                61.46425705,
                66.32666869,
            ]
        ),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="pls_pct" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0.1,
        weight_technique="pca_pct",
        individual_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=0.9,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.40759085,
                15.02505376,
                25.42714541,
                56.26154095,
                99.31716436,
                15.48349891,
                10.46980939,
                34.88025905,
                61.46483173,
                66.32724564,
            ]
        ),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="pca_pct" (sparse), variability_pct=0.9 and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41482334,
                15.0351054,
                25.42670541,
                56.26480226,
                99.31477567,
                15.48593938,
                10.47706149,
                34.8790109,
                61.46425705,
                66.32666869,
            ]
        ),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="unpenalized" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="alasso",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        lambda1_weights=10,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41482334,
                15.0351054,
                25.42670541,
                56.26480226,
                99.31477567,
                15.48593938,
                10.47706149,
                34.8790109,
                61.46425705,
                66.32666869,
            ]
        ),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="lasso", power_weight=1.2 and lasso_weights=10',
    )


def test_alasso_qr():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]

    model = Regressor(
        model="qr",
        penalization="alasso",
        quantile=0.8,
        weight_technique="unpenalized",
        lambda1=0.1,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.14542310,
                17.42545415,
                24.36700662,
                56.76984471,
                99.05520685,
                15.34657501,
                9.81232822,
                35.04760081,
                58.23243936,
                64.60533355,
            ]
        ),
        decimal=3,
        err_msg='Adaptive lasso qr failure for quantile 0.8, weight_technique="unpenalized" and lambda1=0.1',
    )

    model = Regressor(
        model="qr",
        penalization="alasso",
        quantile=0.2,
        weight_technique="unpenalized",
        lambda1=0.1,
        solver="CLARABEL",
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                22.67705224,
                14.58882783,
                25.99692488,
                56.41496513,
                98.34592946,
                15.34213707,
                10.65080076,
                33.47053520,
                62.47406969,
                64.89297292,
            ]
        ),
        decimal=3,
        err_msg='Adaptive lasso qr failure for quantile 0.2, weight_technique="unpenalized" and lambda1=0.1',
    )


# ADAPTIVE RIDGE -----------------------------------------------------


def test_aridge_lm():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0,
        weight_technique="unpenalized",
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Adaptive ridge lm failure for lambda=0",
    )

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0.1,
        individual_weights=[0] * 10,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Adaptive ridge lm failure for lambda=0.1 and weights=0",
    )

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0.1,
        individual_power_weight=0,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                18.46853439,
                15.24387691,
                25.06232772,
                50.31922196,
                89.98619455,
                14.03041184,
                8.67287096,
                33.33355092,
                55.34033474,
                58.02393396,
            ]
        ),
        decimal=3,
        err_msg="Adaptive ridge lm failure for lambda=0.1 and power_weight=0",
    )

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41410188,
                15.0347823,
                25.42636878,
                56.26431388,
                99.31535436,
                15.48547789,
                10.47592391,
                34.8791498,
                61.46432291,
                66.32665609,
            ]
        ),
        decimal=3,
        err_msg='Adaptive ridge lm failure for lambda=0.1, weight_technique="unpenalized" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0.1,
        weight_technique="unpenalized",
        variability_pct=1,
        individual_power_weight=1.2,
        solver="CLARABEL",
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41410188,
                15.0347823,
                25.42636878,
                56.26431388,
                99.31535436,
                15.48547789,
                10.47592391,
                34.8791498,
                61.46432291,
                66.32665609,
            ]
        ),
        decimal=3,
        err_msg='Adaptive ridge lm failure for lambda=0.1, weight_technique="unpenalized", variability_pct=1 and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41410188,
                15.0347823,
                25.42636878,
                56.26431388,
                99.31535436,
                15.48547789,
                10.47592391,
                34.8791498,
                61.46432291,
                66.32665609,
            ]
        ),
        decimal=3,
        err_msg='Adaptive ridge lm failure for lambda=0.1, weight_technique="pca_1" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41410188,
                15.0347823,
                25.42636878,
                56.26431388,
                99.31535436,
                15.48547789,
                10.47592391,
                34.8791498,
                61.46432291,
                66.32665609,
            ]
        ),
        decimal=3,
        err_msg='Adaptive ridge lm failure for lambda=0.1, weight_technique="pls_1" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        solver="CLARABEL",
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41410179,
                15.03478239,
                25.42636872,
                56.26431383,
                99.31535441,
                15.48547795,
                10.4759237,
                34.8791498,
                61.46432289,
                66.32665609,
            ]
        ),
        decimal=3,
        err_msg='Adaptive ridge lm failure for lambda=0.1, weight_technique="pls_pct" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0.1,
        weight_technique="pca_pct",
        individual_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=0.9,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.38402864,
                14.96457767,
                25.43043672,
                56.25291803,
                99.32471105,
                15.47816076,
                10.44635765,
                34.88416976,
                61.4689987,
                66.33217868,
            ]
        ),
        decimal=3,
        err_msg='Adaptive ridge lm failure for lambda=0.1, weight_technique="pca_pct" (sparse), variability_pct=0.9 and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41410188,
                15.0347823,
                25.42636878,
                56.26431388,
                99.31535436,
                15.48547789,
                10.47592391,
                34.8791498,
                61.46432291,
                66.32665609,
            ]
        ),
        decimal=3,
        err_msg='Adaptive ridge lm failure for lambda=0.1, weight_technique="unpenalized" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="aridge",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        lambda1_weights=10,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41410188,
                15.0347823,
                25.42636878,
                56.26431388,
                99.31535436,
                15.48547789,
                10.47592391,
                34.8791498,
                61.46432291,
                66.32665609,
            ]
        ),
        decimal=3,
        err_msg='Adaptive ridge lm failure for lambda=0.1, weight_technique="lasso", power_weight=1.2 and lasso_weights=10',
    )


# ADAPTIVE GROUP LASSO ------------------------------------------------------


def test_agl_lm():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="lm",
        penalization="agl",
        lambda1=0,
        weight_technique="unpenalized",
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Adaptive group lasso lm failure for lambda=0",
    )

    model = Regressor(
        model="lm",
        penalization="agl",
        lambda1=0.1,
        group_weights=[0] * 5,
        solver="CLARABEL",
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Adaptive group lasso lm failure for lambda=0.1 and weights=0",
    )

    model = Regressor(
        model="lm",
        penalization="agl",
        lambda1=0.1,
        group_power_weight=0,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.27677831,
                15.0266414,
                25.33723793,
                56.16210945,
                99.29269691,
                15.45069947,
                10.37091253,
                34.87509388,
                61.40852042,
                66.24043667,
            ]
        ),
        decimal=3,
        err_msg="Adaptive group lasso lm failure for lambda=0.1 and power_weight=0",
    )

    model = Regressor(
        model="lm",
        penalization="agl",
        lambda1=0.1,
        weight_technique="unpenalized",
        group_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41482334,
                15.03676818,
                25.42670541,
                56.26480226,
                99.31477567,
                15.48802755,
                10.47706149,
                34.8790109,
                61.46425705,
                66.32666869,
            ]
        ),
        decimal=3,
        err_msg='Adaptive group lasso lm failure for lambda=0.1, weight_technique="unpenalized" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="agl",
        lambda1=0.1,
        weight_technique="pca_pct",
        group_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=0.9,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.40811348,
                15.03787478,
                25.42532851,
                56.26226056,
                99.31646661,
                15.48644655,
                10.46922814,
                34.87989905,
                61.46374758,
                66.32666011,
            ]
        ),
        decimal=3,
        err_msg='Adaptive group lasso lm failure for lambda=0.1, weight_technique="pca_pct" (sparse), variability_pct=0.9 and power_weight=1.2',
    )


def test_agl_qr():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="qr",
        penalization="agl",
        quantile=0.8,
        weight_technique="unpenalized",
        lambda1=0.1,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                22.82847025,
                18.05969993,
                25.70884251,
                57.69999125,
                98.25142994,
                16.16643666,
                10.84128479,
                36.52330689,
                58.58619985,
                65.37050165,
            ]
        ),
        decimal=3,
        err_msg='Adaptive group lasso qr failure for quantile 0.8, weight_technique="unpenalized" and lambda1=0.1',
    )

    model = Regressor(
        model="qr",
        penalization="agl",
        quantile=0.2,
        weight_technique="unpenalized",
        lambda1=0.1,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.68744830,
                15.89748766,
                25.07758790,
                57.27744038,
                98.13974353,
                15.09267232,
                11.68729906,
                33.34240393,
                62.44319720,
                65.29524738,
            ]
        ),
        decimal=3,
        err_msg='Adaptive group lasso qr failure for quantile 0.2, weight_technique="unpenalized" and lambda1=0.1',
    )


# ADAPTIVE SPARSE GROUP LASSO --------------------------------------------


def test_asgl_lm():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="lm",
        penalization="asgl",
        lambda1=0,
        weight_technique="unpenalized",
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Adaptive sparse group lasso lm failure for lambda=0",
    )

    model = Regressor(
        model="lm",
        penalization="asgl",
        lambda1=0.1,
        group_weights=[0] * 5,
        individual_weights=[0] * 10,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41982223,
                15.03683211,
                25.42968171,
                56.26839201,
                99.31178417,
                15.48907319,
                10.48258919,
                34.87868221,
                61.46433177,
                66.32752383,
            ]
        ),
        decimal=3,
        err_msg="Adaptive sparse group lasso lm failure for lambda=0.1 and weights=0",
    )

    model = Regressor(
        model="lm",
        penalization="asgl",
        lambda1=0.1,
        alpha=0.5,
        group_power_weight=0,
        individual_power_weight=0,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.26751759,
                15.01769123,
                25.33437106,
                56.14736993,
                99.31644424,
                15.42466129,
                10.36399787,
                34.87235843,
                61.41475275,
                66.24427758,
            ]
        ),
        decimal=3,
        err_msg="Adaptive sparse group lasso lm failure for lambda=0.1, alpha=0.5 and power_weight=0",
    )

    model = Regressor(
        model="lm",
        penalization="asgl",
        lambda1=0.1,
        weight_technique="unpenalized",
        individual_power_weight=1.2,
        group_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.41514751,
                15.03593681,
                25.42672403,
                56.26528742,
                99.31437680,
                15.48698344,
                10.47672225,
                34.87943567,
                61.46372079,
                66.32657473,
            ]
        ),
        decimal=3,
        err_msg='Adaptive sparse group lasso lm failure for lambda=0.1, weight_technique="unpenalized" and power_weight=1.2',
    )

    model = Regressor(
        model="lm",
        penalization="asgl",
        lambda1=0.1,
        weight_technique="pca_pct",
        individual_power_weight=1.2,
        group_power_weight=1.2,
        solver="CLARABEL",
        variability_pct=0.9,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.40785211,
                15.03146437,
                25.42623686,
                56.26190065,
                99.31681556,
                15.48497266,
                10.46951872,
                34.8800792,
                61.46428965,
                66.32695275,
            ]
        ),
        decimal=3,
        err_msg='Adaptive sparse group lasso lm failure for lambda=0.1, weight_technique="pca_pct" (sparse), variability_pct=0.9 and power_weight=1.2',
    )


def test_asgl_qr():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="qr",
        penalization="asgl",
        quantile=0.8,
        weight_technique="unpenalized",
        lambda1=0.1,
        alpha=0.5,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                22.82846508,
                18.05970421,
                25.70884349,
                57.69999000,
                98.25142749,
                16.16644023,
                10.84128032,
                36.52330761,
                58.58620251,
                65.37050311,
            ]
        ),
        decimal=3,
        err_msg='Adaptive sparse group lasso qr failure for quantile 0.8, weight_technique="unpenalized", lambda1=0.1 and alpha=0.5',
    )

    model = Regressor(
        model="qr",
        penalization="asgl",
        quantile=0.2,
        weight_technique="unpenalized",
        lambda1=0.1,
        alpha=0.5,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array(
            [
                23.68743307,
                15.89746205,
                25.07760707,
                57.27742573,
                98.13975082,
                15.09265787,
                11.68727502,
                33.34238536,
                62.44320700,
                65.29526732,
            ]
        ),
        decimal=3,
        err_msg='Adaptive sparse group lasso qr failure for quantile 0.2, weight_technique="unpenalized", lambda1=0.1 and alpha=0.5',
    )


# ERROR HANDLING -------------------------------------------------------------


def test_errors():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="qr",
        penalization="gl",
        quantile=0.2,
        lambda1=0.1,
        solver="CLARABEL",
        variability_pct=1,
    )
    with pytest.raises(
        ValueError,
        match="The penalization provided requires fitting the model with a group_index parameter but no group_index was detected.",
    ):
        model.fit(X, y)


def test_negative_group_index_raises_error():
    # Generate dummy data
    X = np.random.rand(10, 5)
    X = sparse.csr_matrix(X)
    y = np.random.rand(10)

    # Create a group_index with a negative value
    group_index = np.array([1, 1, 2, 2, -1])

    model = Regressor(model="lm", penalization="gl", lambda1=0.1, variability_pct=1)

    with pytest.raises(
        ValueError,
        match="group_index must be a positive integer array. Negative values detected",
    ):
        model.fit(X, y, group_index=group_index)


# SKLEARN COMPATIBILITY -----------------------------------------------------


def test_predict():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(
        model="lm",
        penalization="asgl",
        lambda1=0.1,
        solver="CLARABEL",
        variability_pct=1,
    )
    model.fit(X, y, group_index)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    np.testing.assert_almost_equal(
        mse,
        np.float64(63.11994260430959),
        decimal=3,
        err_msg="Failed prediction and / or metric computation",
    )


def test_grid_search():
    data = np.loadtxt("data.csv", delimiter=",", dtype=float)
    X = data[:, :-1]
    X = sparse.csr_matrix(X)
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model="lm", penalization="asgl", solver="CLARABEL")
    param_grid = {
        "lambda1": [1e-3, 1e-2, 10],
        "alpha": [0, 0.5, 1],
        "weight_technique": ["unpenalized"],
    }
    gscv = GridSearchCV(model, param_grid=param_grid)
    gscv.fit(X, y, **{"group_index": group_index})
    expected_output = {
        "alpha": 1,
        "lambda1": 0.01,
        "weight_technique": "unpenalized",
    }

    # Assert that the dictionary contains the expected key-value pairs
    for key, value in gscv.best_params_.items():
        assert expected_output.get(key) == value, (
            f"Expected {key} to be {value}, but got {expected_output.get(key)}"
        )


if __name__ == "__main__":
    pytest.main()
