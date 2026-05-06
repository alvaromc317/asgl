import numpy as np
from asgl import Regressor
from sklearn.datasets import make_regression
import pytest

def test_group_weights_leakage():
    # 1. Setup first dataset
    X1, y1 = make_regression(n_samples=10, n_features=4, noise=0.1, random_state=42)
    group_index = [0, 0, 1, 1]

    # Check group weights leakage (agl)
    model = Regressor(model='lm', penalization='agl', weight_technique='pca_pct', lambda1=0.1, solver='CLARABEL')

    model.fit(X1, y1, group_index=group_index)
    weights1 = model.group_weights_

    # 2. Setup second dataset (very different scale)
    X2, y2 = make_regression(n_samples=10, n_features=4, noise=0.1, random_state=43)
    X2 = X2 * 100

    model.fit(X2, y2, group_index=group_index)
    weights2 = model.group_weights_

    assert not np.allclose(weights1, weights2), "Group weights leaked across fit calls"

def test_individual_weights_leakage():
    # 1. Setup first dataset
    X1, y1 = make_regression(n_samples=10, n_features=4, noise=0.1, random_state=42)

    # Check individual weights leakage (alasso)
    model = Regressor(model='lm', penalization='alasso', weight_technique='pca_pct', lambda1=0.1, solver='CLARABEL')

    model.fit(X1, y1)
    iweights1 = model.individual_weights_

    # 2. Setup second dataset (very different scale)
    X2, y2 = make_regression(n_samples=10, n_features=4, noise=0.1, random_state=43)
    X2 = X2 * 100

    model.fit(X2, y2)
    iweights2 = model.individual_weights_

    assert not np.allclose(iweights1, iweights2), "Individual weights leaked across fit calls"
