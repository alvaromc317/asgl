import sklearn
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression, load_diabetes
import numpy as np
from numpy import geomspace
X,y = make_regression(n_samples=30, n_features=10, n_informative=5)
X,y = load_diabetes(return_X_y=True)

from asgl import AdaptiveLasso
estimator = AdaptiveLasso()
param_grid = {
            "alpha": geomspace(5e-4, 5e-1, num=5),
        }

# cross validation
cross_validate(estimator, X,y, n_jobs=-1, cv=2)
gscv = GridSearchCV(estimator, param_grid=param_grid, n_jobs=-1, error_score="raise", cv=2)
# double cross validation
cross_validate(gscv, X,y, cv=2)
groups = np.random.randint(0,4, size=y.shape)
cv = LeaveOneGroupOut()
cross_validate(estimator, X,y, n_jobs=-1, cv=cv, groups=groups)
