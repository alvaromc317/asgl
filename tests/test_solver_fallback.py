import pytest
import numpy as np
import cvxpy as cp
from asgl import Regressor
from sklearn.datasets import make_regression


def generate_data(n_samples=100, n_features=10, noise=0.1, random_state=42):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    return X, y


def test_solver_fallback_with_invalid_solver():
    X, y = generate_data()

    # "ASD" is an invalid solver name
    solvers = ["ASD", "OSQP", "SCS", "CLARABEL"]

    model = Regressor(model="lm", penalization="lasso", lambda1=0.1, solver=solvers)

    # We expect a warning about ASD failing
    with pytest.warns(RuntimeWarning, match="Solver ASD failed. Trying next solver."):
        model.fit(X, y)

    assert model.is_fitted_
    # Ensure the used solver is one of the valid ones, not ASD
    assert model.solver_stats_["solver_name"] in ["OSQP", "SCS", "CLARABEL"]
    assert model.solver_stats_["solver_name"] != "ASD"


def test_solver_fallback_all_requested_fail():
    X, y = generate_data()

    # Only provide an invalid solver
    solvers = ["ASD"]

    model = Regressor(model="lm", penalization="lasso", lambda1=0.1, solver=solvers)

    with pytest.warns(RuntimeWarning) as record:
        model.fit(X, y)

    assert model.is_fitted_
    # It should have fallen back to an installed solver
    assert model.solver_stats_["solver_name"] in cp.installed_solvers()

    # Check for the sequence of warnings
    warnings_list = [str(w.message) for w in record]

    # 1. ASD failed
    assert any("Solver ASD failed. Trying next solver." in w for w in warnings_list)

    # 2. All requested failed, trying remaining
    assert any(
        "Requested solver(s) ['ASD'] failed. Trying remaining installed solvers" in w
        for w in warnings_list
    )

    # 3. Successfully solved with fallback
    assert any("Successfully solved with fallback solver" in w for w in warnings_list)
