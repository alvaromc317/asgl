import numpy as np
import asgl
import time
from sklearn.datasets import make_regression

# Import test data #
x, y = make_regression(n_samples=100, n_features=30, random_state=42)
group_index = np.random.randint(1, 6, 30)

if __name__ == '__main__':
    start_time = time.time()

    weights = asgl.WEIGHTS(model='lm', penalization='alasso', weight_technique='pca_1',
                           lasso_power_weight=[0.8, 1, 1.2])
    l, g = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(model='lm', penalization='agl', tau=0.5, weight_technique='pca_pct',
                           gl_power_weight=[0.9, 1.1, 1.3], variability_pct=0.8)
    l1, g1 = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(model='qr', penalization='alasso', tau=0.5, weight_technique='pca_pct',
                           lasso_power_weight=[0.8, 1, 1.2], variability_pct=0.8)
    l1b, g1b = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(model='qr', penalization='asgl', tau=0.5, weight_technique='pls_1',
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3, 1.5])
    l2, g2 = weights.fit(x, y, group_index=group_index)


    weights = asgl.WEIGHTS(model='lm', penalization='asgl', tau=0.5, weight_technique='unpenalized',
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3])
    l5, g5 = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(model='qr', penalization='asgl', tau=0.5, weight_technique='lasso', lambda1_weights=1e-2,
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3])
    l6, g6 = weights.fit(x, y, group_index=group_index)

    weights = asgl.WEIGHTS(model='lm', penalization='alasso', tau=0.5, weight_technique='lasso', lambda1_weights=1e-3,
                           lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.9, 1.1, 1.3])
    l7, g7 = weights.fit(x, y, group_index=group_index)

    print(f'Finished with no error. Execution time: {time.time() - start_time}')
