import numpy as np
import asgl
from sklearn.datasets import make_regression
import time


# Import test data #
x, y = make_regression(n_samples=100, n_features=30, random_state=42)
group_index = np.random.randint(1, 6, 30)

# TEST MODELS #########################################################################################################

if __name__ == '__main__':
    start_time = time.time()
    # Unpenalized model
    asgl_model = asgl.ASGL(model='lm', penalization=None, intercept=True, lambda1=None, alpha=None, lasso_weights=None,
                           gl_weights=None)
    asgl_model.fit(x, y)
    lm = asgl_model.coef_

    asgl_model = asgl.ASGL(model='qr', penalization=None, intercept=True, tau=0.5)
    asgl_model.fit(x, y)
    qr = asgl_model.coef_

    # LASSO model
    asgl_model = asgl.ASGL(model='lm', penalization='lasso', intercept=True, tol=1e-5, lambda1=[0.01, 0.1, 1, 10])
    asgl_model.fit(x, y)
    lasso_fit = asgl_model.coef_

    asgl_model = asgl.ASGL(model='lm', penalization='lasso', intercept=True, tol=1e-5, lambda1=[0.01, 0.1, 1, 10],
                           parallel=True)
    asgl_model.fit(x, y)
    lasso_parallel = asgl_model.coef_

    asgl_model = asgl.ASGL(model='qr', penalization='lasso', intercept=True, tol=1e-5,
                           lambda1=[0.01, 0.1, 1, 10], tau=0.25)
    asgl_model.fit(x, y)
    lasso_fit_qr = asgl_model.coef_

    asgl_model = asgl.ASGL(model='qr', penalization='lasso', intercept=True, tol=1e-5,
                           lambda1=[0.01, 0.1, 1, 10], tau=0.25,
                           parallel=True)
    asgl_model.fit(x, y)
    lasso_parallel_qr = asgl_model.coef_

    asgl_model.retrieve_parameters_value(2)

    # Group LASSO model
    asgl_model = asgl.ASGL(model='lm', penalization='gl', intercept=True, tol=1e-5, lambda1=[0.01, 0.1, 1, 10])
    asgl_model.fit(x, y, group_index=group_index)
    gl_fit = asgl_model.coef_

    asgl_model = asgl.ASGL(model='qr', penalization='gl', intercept=True, tol=1e-5, lambda1=[0.01, 0.1, 1, 10], tau=0.5,
                           parallel=True, num_cores=3)
    asgl_model.fit(x, y, group_index=group_index)
    gl_fit_parallel = asgl_model.coef_

    asgl_model.retrieve_parameters_value(2)

    # Sparse group LASSO model
    asgl_model = asgl.ASGL(model='lm', penalization='sgl', intercept=True, tol=1e-5, lambda1=[0.01, 0.1, 1, 10],
                           alpha=[0.1, 0.5, 0.9], parallel=True, num_cores=2)
    asgl_model.fit(x, y, group_index=group_index)
    sgl_fit_parallel = asgl_model.coef_

    asgl_model = asgl.ASGL(model='qr', penalization='sgl', intercept=True, tol=1e-5, lambda1=[0.01, 0.1, 1, 10],
                           alpha=[0.1, 0.5, 0.9], tau=0.2)
    asgl_model.fit(x, y, group_index=group_index)
    sgl_fit = asgl_model.coef_

    asgl_model.retrieve_parameters_value(7)

    # Adaptive lasso
    alasso_model = asgl.ASGL(model='qr', penalization='alasso', lambda1=[0.01, 0.1, 1, 10], parallel=True, num_cores=2,
                             lasso_weights=[0.5] * 30, tau=0.25)
    alasso_model.fit(x, y, group_index=group_index)

    # Adaptive group lasso
    agl_model = asgl.ASGL(model='lm', penalization='agl', lambda1=[0.01, 0.1, 1, 10], parallel=True, num_cores=2,
                          gl_weights=[1.2] * 5)
    agl_model.fit(x, y, group_index=group_index)

    # Adaptive sparse group lasso
    asgl_model = asgl.ASGL(model='lm', penalization='asgl', intercept=True, tol=1e-5, lambda1=[0.01, 0.1, 1, 10],
                           alpha=[0.1, 0.5, 0.9], lasso_weights=[0.5] * 30, gl_weights=[1.2] * 5,
                           parallel=True, num_cores=3)
    asgl_model.fit(x, y, group_index=group_index)
    asgl_fit_parallel = asgl_model.coef_

    asgl_model = asgl.ASGL(model='qr', penalization='asgl', intercept=True, tol=1e-5, lambda1=[0.01, 0.1, 1, 10],
                           alpha=[0.1, 0.5, 0.9], lasso_weights=[0.5] * 30, gl_weights=[1.2] * 5, tau=0.5)
    asgl_model.fit(x, y, group_index=group_index)
    asgl_fit = asgl_model.coef_

    asgl_model.retrieve_parameters_value(7)

    # Test predict
    y_pred = asgl_model.predict(x)

    # Test error
    assert asgl.error_calculator(y_true=y, prediction_list=y_pred, error_type="MSE", tau=None)
    assert asgl.error_calculator(y_true=y, prediction_list=y_pred, error_type="MAE", tau=None)
    assert asgl.error_calculator(y_true=y, prediction_list=y_pred, error_type="MDAE", tau=None)
    assert asgl.error_calculator(y_true=y, prediction_list=y_pred, error_type="QRE", tau=0.2)

    print(f'Finished with no error. Execution time: {time.time() - start_time}')
