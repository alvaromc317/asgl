import numpy as np
import asgl
from sklearn.datasets import load_boston
import time

# Import test data #

boston = load_boston()
x = boston.data
y = boston.target
group_index = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5])

if __name__ == '__main__':
    start_time = time.time()

    weights = asgl.WEIGHTS(penalization='asgl_lasso', tau=0.5, weight_technique='pca_1',
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3])
    l, g = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(model='lm', penalization='asgl_gl', tau=0.5, weight_technique='pca_pct',
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3], variability_pct=0.8)
    l1, g1 = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(model='qr', penalization='asgl_gl', tau=0.5, weight_technique='pca_pct',
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3], variability_pct=0.8)
    l1b, g1b = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(penalization='asgl', tau=0.5, weight_technique='pls_1',
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3])
    l2, g2 = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(penalization='asgl_lasso', tau=0.5, weight_technique='sparse_pca',
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3], variability_pct=0.9,
                           spca_alpha=0.1, spca_ridge_alpha=1e-6)
    l3, g3 = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(penalization='asgl', tau=0.5, weight_technique='unpenalized',
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3])
    l5, g5 = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(penalization='asgl', tau=0.5, weight_technique='unpenalized',
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3])
    l6, g6 = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(model='lm', penalization='alasso', tau=0.5, weight_technique='lasso', lambda1_weights=1e-2,
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3])
    l7, g7 = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(model='qr', penalization='alasso', tau=0.5, weight_technique='lasso', lambda1_weights=1e-2,
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3])
    l9, g9 = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(penalization='agl', tau=0.5, weight_technique='lasso', lambda1_weights=1e-2,
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3])
    l8, g8 = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(penalization='asgl', tau=0.5, weight_technique='pls_pct',
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3], variability_pct=0.9)
    l4, g4 = weights.fit(x, y, group_index=group_index)

    asgl_model = asgl.ASGL(model='qr', penalization='asgl_lasso', intercept=True, tol=1e-5, lambda1=[0.01, 0.1, 1, 10],
                           alpha=[0.1, 0.5, 0.9], lasso_weights=l4, gl_weights=g4, tau=0.5)
    asgl_model.fit(x, y, group_index=group_index)
    asgl_fit2 = asgl_model.coef_

    print(f'Finished with no error. Execution time: {time.time() - start_time}')
