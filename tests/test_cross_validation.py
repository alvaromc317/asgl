import numpy as np
import asgl
from sklearn.datasets import make_regression
import time

# Import test data #

# Import test data #
x, y = make_regression(n_samples=100, n_features=30, random_state=42)
group_index = np.random.randint(1, 6, 30)

def equal_array(array0, array1, tol=1e-6):
    n0 = array0.shape[0]
    n1 = array1.shape[0]
    if n0 != n1:
        f'Shapes do not match. Received: {n0} and {n1}'
    else:
        dif = array0 - array1
        if np.abs(np.mean(dif)) > tol:
            f'Arrays are not equal. Mean difference is {np.mean(dif)}'


if __name__ == '__main__':
    start_time = time.time()
    # UNPENALIZED TEST
    cross_validation_class = asgl.CV(model='lm', penalization=None, nfolds=5, error_type='MSE', random_state=2)

    cv1 = cross_validation_class.cross_validation(x, y, group_index)
    cv2 = cross_validation_class.cross_validation(x, y, group_index)
    equal_array(cv1, cv2)

    # LASSO penalization
    cross_validation_class = asgl.CV(model='qr', penalization='lasso', lambda1=[0.01, 0.1, 1, 10], parallel=True,
                                     tau=0.5, nfolds=6, error_type='QRE', random_state=1)
    cross_validation_class.cross_validation(x, y, group_index)

    # Group LASSO
    cross_validation_class = asgl.CV(model='lm', penalization='gl', lambda1=[0.01, 0.1, 1, 10], parallel=True,
                                     nfolds=7, error_type='MAE', random_state=2)
    cross_validation_class.cross_validation(x, y, group_index)

    # Sparse Group LASSO
    cross_validation_class = asgl.CV(model='qr', penalization='sgl', lambda1=[0.01, 0.1, 1, 10], parallel=True,
                                     tau=0.02, nfolds=7, error_type='QRE', random_state=3, alpha=[0.1, 0.5, 0.9])
    cross_validation_class.cross_validation(x, y, group_index)

    print(f'Entering adaptive formulations.  Execution time: {time.time() - start_time}')

    # Adaptive lasso
    cross_validation_class = asgl.CV(model='qr', penalization='alasso', lambda1=[0.01, 0.1, 1, 10], tau=0.5,
                                     parallel=True, num_cores=3, weight_technique='lasso',
                                     lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.8, 1, 1.2],
                                     variability_pct=0.7, nfolds=3, error_type='QRE', random_state=99,
                                     lambda1_weights=1e-3)
    cross_validation_class.cross_validation(x, y, group_index)

    # Adaptive group lasso
    cross_validation_class = asgl.CV(model='qr', penalization='agl', lambda1=[0.01, 0.1, 1, 10], tau=0.5,
                                     parallel=True, num_cores=9, weight_technique='pca_pct',
                                     gl_power_weight=[0.8, 1, 1.2], variability_pct=0.7, nfolds=3, error_type='QRE',
                                     random_state=99)
    cross_validation_class.cross_validation(x, y, group_index)

    # Adaptive sparse group lasso
    cross_validation_class = asgl.CV(model='qr', penalization='asgl', lambda1=[0.01, 0.1, 1, 10], tau=0.5,
                                     alpha=[0.1, 0.5, 0.9], parallel=True, num_cores=9, weight_technique='pca_1',
                                     lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.8, 1, 1.2],
                                     variability_pct=0.7, nfolds=3, error_type='QRE', random_state=99)
    cross_validation_class.cross_validation(x, y, group_index)

    cross_validation_class = asgl.CV(model='lm', penalization='asgl', lambda1=[0.01, 0.1, 1, 10], tau=0.5,
                                     alpha=[0.1, 0.5, 0.9], parallel=True, num_cores=9, weight_technique='pls_1',
                                     lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.8, 1, 1.2],
                                     variability_pct=0.7, nfolds=3, error_type='MSE', random_state=99)
    cross_validation_class.cross_validation(x, y, group_index)

    cross_validation_class = asgl.CV(model='qr', penalization='asgl', lambda1=[0.01, 0.1, 1, 10], tau=0.5,
                                     alpha=[0.1, 0.5, 0.9], parallel=True, num_cores=9, weight_technique='sparse_pca',
                                     lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.8, 1, 1.2],
                                     variability_pct=0.7, nfolds=3, error_type='QRE', random_state=99, spca_alpha=1e-1,
                                     spca_ridge_alpha=1e-5)
    cross_validation_class.cross_validation(x, y, group_index)

    cross_validation_class = asgl.TVT(model='qr', penalization='asgl', lambda1=[0.01, 0.1, 1, 10], tau=0.5,
                                      alpha=[0.1, 0.5, 0.9], parallel=True, num_cores=9, weight_technique='lasso',
                                      lasso_power_weight=[0.8, 1, 1.2], gl_power_weight=[0.8, 1, 1.2],
                                      error_type='QRE', random_state=99, train_pct=0.6,
                                      validate_pct=0.2)
    cross_validation_class.train_validate_test(x, y, group_index)

    print(f'Finished with no error. Execution time: {time.time() - start_time}')
