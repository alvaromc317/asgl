import numpy as np
from sklearn.cross_decomposition import PLSRegression
X, y = np.random.rand(100, 10), np.random.rand(100)
pls1 = PLSRegression(n_components=9, scale=False).fit(X, y)
pls2 = PLSRegression(n_components=3, scale=False).fit(X, y)
print("Coefs equal:", np.allclose(pls1.coef_, pls2.coef_))
# Actually PLS coefs for n_components=3 are NOT the first 3 columns of n_components=9
# PLS coefs are computed for the whole model.
print("Shape 9:", pls1.coef_.shape)
print("Shape 3:", pls2.coef_.shape)
# They are both (10, 1)!
# Are they equal?
print("Equal:", np.allclose(pls1.coef_, pls2.coef_))
