from unittest.mock import MagicMock
import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor

from optuna.integration import OptunaSearchCV
from optuna.distributions import FloatDistribution

from asgl.sklearn import (
    LinearRegression,
    QuantileRegression,
    Lasso,
    QuantileLasso,
    AdaptiveLasso,
    QuantileAdaptiveLasso,
    GroupLasso,
    QuantileGroupLasso,
    SparseGroupLasso,
    QuantileSparseGroupLasso,
    AdaptiveSparseGroupLasso,
    QuantileAdaptiveSparseGroupLasso,
)


allmodels = [
    LinearRegression(),
    QuantileRegression(),
    Lasso(),
    QuantileLasso(),
    AdaptiveLasso(),
    QuantileAdaptiveLasso(),
    GroupLasso(),
    QuantileGroupLasso(),
    SparseGroupLasso(),
    QuantileSparseGroupLasso(),
    AdaptiveSparseGroupLasso(),
    QuantileAdaptiveSparseGroupLasso(),
]
adaptive_models = [
    AdaptiveLasso(),
    QuantileAdaptiveLasso(),
    AdaptiveSparseGroupLasso(),
    QuantileAdaptiveSparseGroupLasso(),
]

group_models = [
    GroupLasso(),
    QuantileGroupLasso(),
    SparseGroupLasso(),
    QuantileSparseGroupLasso(),
    AdaptiveSparseGroupLasso(),
    QuantileAdaptiveSparseGroupLasso(),
]

not_group_models = [
    LinearRegression(),
    QuantileRegression(),
    Lasso(),
    QuantileLasso(),
    AdaptiveLasso(),
    QuantileAdaptiveLasso(),
]


@pytest.mark.parametrize("estimator", allmodels)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_model_fit_predict(estimator, fit_intercept) -> None:
    X, y = make_regression(n_samples=30, n_features=30)
    estimator.set_params(fit_intercept=fit_intercept)
    estimator.fit(X, y)
    estimator.predict(X)


@pytest.mark.parametrize("estimator", allmodels)
def test_model_fit_predict_pandas(estimator) -> None:
    X, y = make_regression(n_samples=30, n_features=30)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    estimator.fit(X, y)
    estimator.predict(X)


@pytest.mark.parametrize("estimator", allmodels)
def test_multioutput_model_fit_predict(estimator) -> None:
    X, y = make_regression(n_targets=2)
    print(X.shape)
    print(y.shape)
    group_index = np.random.randint(low=0, high=5, size=X.shape[1])
    fit_params = {"group_index": group_index}
    estimator = MultiOutputRegressor(estimator)
    estimator.fit(X, y, fit_params)
    estimator.predict(X)


@pytest.mark.parametrize("estimator", group_models)
def test_group_model_fit_predict(estimator) -> None:
    X, y = make_regression(n_samples=30, n_features=30)
    group_index = np.random.randint(low=0, high=5, size=X.shape[1])
    estimator.fit(X, y, group_index=group_index)
    estimator.predict(X)


@pytest.mark.parametrize("estimator", group_models)
def test_group_model_cross_validate(estimator) -> None:
    X, y = make_regression(n_samples=30, n_features=30)
    group_index = np.random.randint(low=0, high=5, size=X.shape[1])
    estimator.fit(X, y, group_index=group_index)
    fit_params = {"group_index": group_index}
    cross_validate(estimator, X, y, fit_params=fit_params)


@pytest.mark.parametrize("estimator", [AdaptiveLasso()])
def test_model_cross_validate(estimator) -> None:
    X, y = make_regression(n_samples=30, n_features=30)
    cross_validate(estimator, X, y)


@pytest.mark.parametrize("estimator", not_group_models)
def test_model_pipeline_fit_predict(estimator) -> None:
    X, y = make_regression(n_samples=30, n_features=30)
    pipe = Pipeline([("selector", SelectFromModel(Ridge())), ("estimator", estimator)])
    pipe.fit(X, y)


@pytest.mark.parametrize("estimator", not_group_models)
def test_model_pipeline_cross_validate(estimator) -> None:
    X, y = make_regression(n_samples=30, n_features=30)
    pipe = Pipeline([("selector", SelectFromModel(Ridge())), ("estimator", estimator)])
    cross_validate(pipe, X, y)


@pytest.mark.parametrize("estimator", [AdaptiveLasso()])
@pytest.mark.parametrize("param_grid", [{"alpha": [0.01, 0.1, 1.0]}])
def test_model_pipeline_gridsearchcv_fit_predict(estimator, param_grid) -> None:
    X, y = make_regression(n_samples=30, n_features=30)
    gscv = GridSearchCV(estimator, param_grid=param_grid)
    gscv.fit(X, y)
    best_estimator_ = gscv.best_estimator_
    best_estimator_.fit(X, y)


@pytest.mark.parametrize("estimator", [AdaptiveLasso()])
@pytest.mark.parametrize(
    "param_distributions", [{"alpha": FloatDistribution(0.01, 0.11, step=0.05)}]
)
def test_model_pipeline_optunasearchcv_fit_predict(
    estimator, param_distributions
) -> None:
    X, y = make_regression(n_samples=30, n_features=30)
    opcv = OptunaSearchCV(estimator, param_distributions=param_distributions)
    opcv.fit(X, y)
    best_estimator_ = opcv.best_estimator_
    best_estimator_.fit(X, y)
    best_estimator_.predict(X)


@pytest.mark.parametrize(
    "estimator",
    [
        QuantileSparseGroupLasso(),
        QuantileAdaptiveSparseGroupLasso(),
    ],
)
@pytest.mark.parametrize(
    "param_grid",
    [
        {
            "alpha": [1e-5, 1e-3],
            "l1_ratio": [1e-5],
            "tau": [0.50],
        }
    ],
)
def test_model_pipeline_gridsearchcv_cross_validate(estimator, param_grid) -> None:
    X, y = make_regression(n_samples=15, n_features=15)
    gscv = GridSearchCV(estimator, param_grid=param_grid)
    cross_validate(gscv, X, y)


@pytest.mark.parametrize("alpha", [1e-10, 1e-5, 1e-0, 0, 1e1])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_compare_lasso(alpha, fit_intercept) -> None:
    X, y = make_regression(n_samples=30, n_features=30)
    sklasso = sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=fit_intercept)
    lasso = asgl.sklearn.linear_model.Lasso(
        alpha=alpha, fit_intercept=fit_intercept
    )
    sklasso.fit(X, y)
    lasso.fit(X, y)

    y_sk_predict = sklasso.predict(X)
    y_predict = lasso.predict(X)

    corrcoef = np.corrcoef(y_sk_predict, y_predict)
    assert corrcoef[0][1] >= 0.90


@pytest.mark.parametrize("estimator", allmodels)
@pytest.mark.parametrize(
    "weight_calculator",
    [
        None,
        "pca",
        "pcar",
        "pls",
        "pls_percent",
        "unpenalized",
        "unpenalized_lm",
        "unpenalized_qr",
        "lasso",
        "spca",
    ],
)
def test_model_fit_predict_weitht_calculator(estimator, weight_calculator) -> None:
    X, y = make_regression(n_samples=15, n_features=15)
    setattr(estimator, "weight_calculator", weight_calculator)
    group_index = np.random.randint(low=0, high=5, size=X.shape[1])
    estimator.fit(X, y, group_index=group_index)
    estimator.predict(X)


"""
@pytest.mark.parametrize("estimator", allmodels)
@pytest.mark.parametrize(
    "solver",
    [
        "default",
        "ECOS",
        "OSQP",
        "SCS",
    ],
)
def test_model_fit_solver(estimator, solver) -> None:
    X, y = make_regression(n_samples=30, n_features=30)
    estimator.solver = solver
    estimator.fit(X, y)
"""
