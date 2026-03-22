import warnings
from asgl.skmodels import Regressor
import numpy as np
import cvxpy as cp

X = np.random.randn(10, 2)
y = np.random.randn(10)
group_index = [1, 2]

# Setup to show all warnings
warnings.simplefilter("always")

print("Checking global state before fitting...")
with warnings.catch_warnings(record=True) as w:
    x = cp.Variable()
    p = cp.Parameter(value=2.0)
    prob = cp.Problem(cp.Minimize(p * cp.square(x)))
    try:
        prob.solve()
    except Exception:
        pass

    print("Global warnings emitted:", [warn.message for warn in w])

print("\nFitting model...")
model = Regressor(model="lm", penalization="asgl", lambda1=0.1)
with warnings.catch_warnings(record=True) as w:
    model.fit(X, y, group_index=group_index)
    print("Fitting warnings emitted:", [warn.message for warn in w])

print("\nChecking global state after fitting...")
with warnings.catch_warnings(record=True) as w:
    x = cp.Variable()
    p = cp.Parameter(value=2.0)
    prob = cp.Problem(cp.Minimize(p * cp.square(x)))
    try:
        prob.solve()
    except Exception:
        pass

    print("Global warnings emitted:", [warn.message for warn in w])
