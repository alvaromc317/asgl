import pytest
import numpy as np
from asgl import Regressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


def test_prepare_data():
    model = Regressor()
    X = np.array([[1, 2], [3, 4]])
    X_prepared, m, group_index = model._prepare_data(X)
    assert X_prepared.shape == (2, 3)  # Adding intercept term
    assert m == 3
    assert group_index is None


# TEST UNPENALIZED ----------------------------------------------------------------------------------------------------


def test_unpenalized_lm():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]

    model = Regressor(model='lm', penalization=None)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29234411, 23.41982223, 15.03683211, 25.42968171, 56.26839201, 99.31178417,
                  15.48907319, 10.48258919, 34.87868221, 61.46433177, 66.32752383]),
        decimal=3,
        err_msg='Unpenalized lm failure')


def test_unpenalized_qr():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]

    model = Regressor(model='qr', penalization=None, quantile=0.8)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([13.23231242, 25.92647702, 15.32296485, 26.18622489, 59.82066746,
                  99.17547549, 17.72659176, 11.82218774, 34.74489955, 58.59336065,
                  65.68631127]),
        decimal=3,
        err_msg='Unpenalized qr failure for quantile 0.8')

    model = Regressor(model='qr', penalization=None, quantile=0.2)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([0.72185286, 24.53442726, 15.60297538, 24.87555318, 58.24387077,
                  99.36779578, 15.09292701, 12.15865, 33.94239404, 63.53956298,
                  66.47009116]),
        decimal=3,
        err_msg='Unpenalized qr failure for quantile 0.2')


# TEST LASSO PENALIZATION ---------------------------------------------------------------------------------------------


def test_lasso_lm():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]

    model = Regressor(model='lm', penalization='lasso', lambda1=0)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29234411, 23.41982223, 15.03683211, 25.42968171, 56.26839201, 99.31178417,
                  15.48907319, 10.48258919, 34.87868221, 61.46433177, 66.32752383]),
        decimal=3,
        err_msg='Lasso lm failure for lambda=0')

    model = Regressor(model='lm', penalization='lasso', lambda1=0.1)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.30445498, 23.25821113, 15.00872081, 25.33148317, 56.132575,
                  99.34021902, 15.39857671, 10.35704837, 34.86962766, 61.42098122,
                  66.24809898]),
        decimal=3,
        err_msg='Lasso lm failure for lambda=0.1')


def test_lasso_qr():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]

    model = Regressor(model='qr', penalization='lasso', quantile=0.8, lambda1=0)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([13.23231242, 25.92647702, 15.32296485, 26.18622489, 59.82066746,
                  99.17547549, 17.72659176, 11.82218774, 34.74489955, 58.59336065,
                  65.68631127]),
        decimal=3,
        err_msg='Lasso qr failure for quantile 0.8 and lambda1=0')

    model = Regressor(model='qr', penalization='lasso', quantile=0.8, lambda1=0.1)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([51.93589522, 0., 0., 0., 27.66071305,
                  90.01397685, 0., 0., 21.77214275, 45.01300237,
                  36.34016822]),
        decimal=3,
        err_msg='Lasso qr failure for quantile 0.8 and lambda1=0.1')

    model = Regressor(model='qr', penalization='lasso', quantile=0.2, lambda1=0.1)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([-33.70360444, 0., 2.37522451, 0.,
                  22.15449983, 77.79554315, 0., 0.,
                  38.65124062, 30.55826802, 29.50093582]),
        decimal=3,
        err_msg='Lasso qr failure for quantile 0.2 and lambda1=0.1')


# TEST GROUP LASSO PENALIZATION ---------------------------------------------------------------------------------------


def test_gl_lm():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model='lm', penalization='gl', lambda1=0)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29234411, 23.41982223, 15.03683211, 25.42968171, 56.26839201, 99.31178417,
                  15.48907319, 10.48258919, 34.87868221, 61.46433177, 66.32752383]),
        decimal=3,
        err_msg='Group lasso lm failure for lambda=0')

    model = Regressor(model='lm', penalization='gl', lambda1=0.1)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.31191744, 23.27677831, 15.0266414, 25.33723793, 56.16210945,
                  99.29269691, 15.45069947, 10.37091253, 34.87509388, 61.40852042,
                  66.24043667]),
        decimal=3,
        err_msg='Group lasso lm failure for lambda=0.1')


def test_gl_qr():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model='qr', penalization='gl', quantile=0.8, lambda1=0)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([13.23231242, 25.92647702, 15.32296485, 26.18622489, 59.82066746,
                  99.17547549, 17.72659176, 11.82218774, 34.74489955, 58.59336065,
                  65.68631127]),
        decimal=3,
        err_msg='Group lasso qr failure for quantile 0.8 and lambda1=0')

    model = Regressor(model='qr', penalization='gl', quantile=0.8, lambda1=0.1)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([1.09642347e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                  1.33241462e+01, 3.66710686e+01, 4.56237828e+00, 0.00000000e+00,
                  0.00000000e+00, 1.02229130e-04, 1.07166280e-04]),
        decimal=3,
        err_msg='Group lasso qr failure for quantile 0.8 and lambda1=0.1')

    model = Regressor(model='qr', penalization='gl', quantile=0.2, lambda1=0.1)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([-50.74719872, 0., 0.65037989, 0.62171602,
                  27.42813623, 55.53152069, 7.38334327, 0.,
                  34.14211386, 21.23025218, 30.17858448]),
        decimal=3,
        err_msg='Group lasso qr failure for quantile 0.2 and lambda1=0.1')


# TEST SPARSE GROUP LASSO PENALIZATION --------------------------------------------------------------------------------


def test_sgl_lm():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model='lm', penalization='sgl', lambda1=0)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29234411, 23.41982223, 15.03683211, 25.42968171, 56.26839201, 99.31178417,
                  15.48907319, 10.48258919, 34.87868221, 61.46433177, 66.32752383]),
        decimal=3,
        err_msg='Sparse group lasso lm failure for lambda=0')

    model = Regressor(model='lm', penalization='sgl', lambda1=0.1, alpha=0)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.31191744, 23.27677831, 15.0266414, 25.33723793, 56.16210945,
                  99.29269691, 15.45069947, 10.37091253, 34.87509388, 61.40852042,
                  66.24043667]),
        decimal=3,
        err_msg='Sparse group lasso lm failure for lambda=0.1 and alpha=0')

    model = Regressor(model='lm', penalization='sgl', lambda1=0.1, alpha=1)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.30445498, 23.25821113, 15.00872081, 25.33148317, 56.132575,
                  99.34021902, 15.39857671, 10.35704837, 34.86962766, 61.42098122,
                  66.24809898]),
        decimal=3,
        err_msg='Sparse group lasso lm failure for lambda=0.1 and alpha=1')

    model = Regressor(model='lm', penalization='sgl', lambda1=0.1, alpha=0.5)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.30818763, 23.26750237, 15.01768931, 25.33436154, 56.14735784,
                  99.3164447, 15.42465486, 10.36398602, 34.8723578, 61.41474779,
                  66.24426926]),
        decimal=3,
        err_msg='Sparse group lasso lm failure for lambda=0.1 and alpha=0.5')


def test_sgl_qr():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model='qr', penalization='sgl', quantile=0.8, lambda1=0)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([13.23231242, 25.92647702, 15.32296485, 26.18622489, 59.82066746,
                  99.17547549, 17.72659176, 11.82218774, 34.74489955, 58.59336065,
                  65.68631127]),
        decimal=3,
        err_msg='Sparse group lasso qr failure for quantile 0.8 and lambda1=0')

    model = Regressor(model='qr', penalization='sgl', quantile=0.8, lambda1=0.1, alpha=0)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([1.09642347e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                  1.33241462e+01, 3.66710686e+01, 4.56237828e+00, 0.00000000e+00,
                  0.00000000e+00, 1.02229130e-04, 1.07166280e-04]),
        decimal=3,
        err_msg='Sparse group lasso qr failure for quantile 0.8, lambda1=0.1 and alpha=0')

    model = Regressor(model='qr', penalization='sgl', quantile=0.8, lambda1=0.1, alpha=1)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([51.93589522, 0., 0., 0., 27.66071305,
                  90.01397685, 0., 0., 21.77214275, 45.01300237,
                  36.34016822]),
        decimal=3,
        err_msg='Sparse group lasso qr failure for quantile 0.8, lambda1=0.1 and alpha=1')

    model = Regressor(model='qr', penalization='sgl', quantile=0.8, lambda1=0.1, alpha=0.5)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([61.42372486, 0., 0., 0., 21.95907657,
                  81.32544812, 2.84475096, 0., 20.76154792, 35.57092954,
                  36.05204591]),
        decimal=3,
        err_msg='Sparse group lasso qr failure for quantile 0.8, lambda1=0.1 and alpha=0.5')

    model = Regressor(model='qr', penalization='sgl', quantile=0.2, lambda1=0.1, alpha=0.5)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([-4.31305453e+01, 0.00000000e+00, 3.97542269e-04, 6.63474565e-04,
                  2.09861368e+01, 6.62803756e+01, 3.97923687e-01, 0.00000000e+00,
                  3.84701424e+01, 2.45480120e+01, 3.01299190e+01]),
        decimal=3,
        err_msg='Sparse group lasso qr failure for quantile 0.2, lambda1=0.1 and alpha=0.5')


# ADAPTIVE LASSO ------------------------------------------------------------------------------------------------------

def test_alasso_lm():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]

    model = Regressor(model='lm', penalization='alasso', lambda1=0)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29234411, 23.41982223, 15.03683211, 25.42968171, 56.26839201, 99.31178417,
                  15.48907319, 10.48258919, 34.87868221, 61.46433177, 66.32752383]),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0')

    model = Regressor(model='lm', penalization='alasso', lambda1=0.1, individual_weights=[0] * 10)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29234411, 23.41982223, 15.03683211, 25.42968171, 56.26839201, 99.31178417,
                  15.48907319, 10.48258919, 34.87868221, 61.46433177, 66.32752383]),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1 and weights=0')

    model = Regressor(model='lm', penalization='alasso', lambda1=0.1, individual_power_weight=0)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.30445498, 23.25821113, 15.00872081, 25.33148317, 56.132575,
                  99.34021902, 15.39857671, 10.35704837, 34.86962766, 61.42098122,
                  66.24809898]),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1 and power_weight=0')

    model = Regressor(model='lm', penalization='alasso', lambda1=0.1, weight_technique='pca_pct',
                      individual_power_weight=1.2)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29239354, 23.40759085, 15.02505376, 25.42714541, 56.26154095,
                  99.31716436, 15.48349891, 10.46980939, 34.88025905, 61.46483173,
                  66.32724564]),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="pca_pct" and power_weight=1.2')

    model = Regressor(model='lm', penalization='alasso', lambda1=0.1, weight_technique='pca_pct', variability_pct=0.1, individual_power_weight=1.2)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.33029558, 23.41947508, 15.12591301, 25.53676036, 56.26451599,
                  99.17162385, 15.56976165, 10.40965701, 34.91913924, 60.40617589,
                  66.2358729]),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="pca_pct", variability_pct=0.1 and power_weight=1.2')

    model = Regressor(model='lm', penalization='alasso', lambda1=0.1, weight_technique='pca_1',
                      individual_power_weight=1.2)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([9.40953544, 23.31744912, 19.7624, 31.42703026, 55.99456847,
                  91.4541783, 19.99217028, 6.34233369, 37.12892737, 2.06328706,
                  61.16029076]),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="pca_1" and power_weight=1.2')

    model = Regressor(model='lm', penalization='alasso', lambda1=0.1, weight_technique='pls_1',
                      individual_power_weight=1.2)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([8.18422468, 0., 15.8424663, 20.7085942,
                  47.99452202, 103.62678229, 10.07182228, 0.37932526,
                  36.87899711, 61.42687489, 65.81253078]),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="pls_1" and power_weight=1.2')

    model = Regressor(model='lm', penalization='alasso', lambda1=0.1, weight_technique='pls_pct',
                      individual_power_weight=1.2)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29261375, 23.41482334, 15.0351054, 25.42670541, 56.26480226,
                  99.31477567, 15.48593938, 10.47706149, 34.8790109, 61.46425705,
                  66.32666869]),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="pls_pct" and power_weight=1.2')

    model = Regressor(model='lm', penalization='alasso', lambda1=0.1, weight_technique='unpenalized',
                      individual_power_weight=1.2)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29261375, 23.41482334, 15.0351054, 25.42670541, 56.26480226,
                  99.31477567, 15.48593938, 10.47706149, 34.8790109, 61.46425705,
                  66.32666869]),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="unpenalized" and power_weight=1.2')

    model = Regressor(model='lm', penalization='alasso', lambda1=0.1, weight_technique='lasso',
                      individual_power_weight=1.2, lambda1_weights=10)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([8.12689376, 18.51362468, 16.00038822, 23.01204692,
                  52.6861621, 102.88112817, 14.34447744, 0.,
                  35.57177763, 61.03376887, 65.84327928]),
        decimal=3,
        err_msg='Adaptive lasso lm failure for lambda=0.1, weight_technique="lasso", power_weight=1.2 and lasso_weights=10')


def test_alasso_qr():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]

    model = Regressor(model='qr', penalization='alasso', quantile=0.8, weight_technique='pca_pct', lambda1=0.1)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([15.49701439, 21.98299416, 18.58883895, 24.00320703, 55.48866531,
                  99.26130871, 14.48554396, 9.7589368, 36.5871078, 58.41362123,
                  64.40034079]),
        decimal=3,
        err_msg='Adaptive lasso qr failure for quantile 0.8, weight_technique="pca_pct" and lambda1=0.1')

    model = Regressor(model='qr', penalization='alasso', quantile=0.2, weight_technique='pca_pct', lambda1=0.1)
    model.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([1.02568342, 23.06895887, 13.70817285, 25.1125269,
                  56.67751033, 100.14818519, 14.66378046, 9.83289852,
                  34.53294509, 62.71185356, 67.31032544]),
        decimal=3,
        err_msg='Adaptive lasso qr failure for quantile 0.2, weight_technique="pca_pct" and lambda1=0.1')


# ADAPTIVE GROUP LASSO ------------------------------------------------------------------------------------------------

def test_agl_lm():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model='lm', penalization='agl', lambda1=0)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29234411, 23.41982223, 15.03683211, 25.42968171, 56.26839201, 99.31178417,
                  15.48907319, 10.48258919, 34.87868221, 61.46433177, 66.32752383]),
        decimal=3,
        err_msg='Adaptive group lasso lm failure for lambda=0')

    model = Regressor(model='lm', penalization='agl', lambda1=0.1, group_weights=[0] * 5)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29234411, 23.41982223, 15.03683211, 25.42968171, 56.26839201, 99.31178417,
                  15.48907319, 10.48258919, 34.87868221, 61.46433177, 66.32752383]),
        decimal=3,
        err_msg='Adaptive group lasso lm failure for lambda=0.1 and weights=0')

    model = Regressor(model='lm', penalization='agl', lambda1=0.1, group_power_weight=0)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.31191744, 23.27677831, 15.0266414, 25.33723793, 56.16210945,
                  99.29269691, 15.45069947, 10.37091253, 34.87509388, 61.40852042,
                  66.24043667]),
        decimal=3,
        err_msg='Adaptive group lasso lm failure for lambda=0.1 and power_weight=0')

    model = Regressor(model='lm', penalization='agl', lambda1=0.1, weight_technique='pca_pct', group_power_weight=1.2)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29351925, 23.40811348, 15.03787478, 25.42532851, 56.26226056,
                  99.31646661, 15.48644655, 10.46922814, 34.87989905, 61.46374758,
                  66.32666011]),
        decimal=3,
        err_msg='Adaptive group lasso lm failure for lambda=0.1, weight_technique="pca_pct" and power_weight=1.2')


def test_agl_qr():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model='qr', penalization='agl', quantile=0.8, weight_technique='pca_pct', lambda1=0.1)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([14.86083072, 22.79424377, 18.96442717, 24.65673831, 55.84711541,
                  98.68308529, 14.67274364, 9.53221104, 35.33865081, 57.41022152,
                  64.67356877]),
        decimal=3,
        err_msg='Adaptive group lasso qr failure for quantile 0.8, weight_technique="pca_pct" and lambda1=0.1')

    model = Regressor(model='qr', penalization='agl', quantile=0.2, weight_technique='pca_pct', lambda1=0.1)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([0.56702454, 22.56319979, 14.50549661, 25.96562376, 56.27532891,
                  98.3527059, 15.28195246, 10.47099655, 33.49308075, 62.33930551,
                  64.95028568]),
        decimal=3,
        err_msg='Adaptive group lasso qr failure for quantile 0.2, weight_technique="pca_pct" and lambda1=0.1')


# ADAPTIVE SPARSE GROUP LASSO -----------------------------------------------------------------------------------------

def test_asgl_lm():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model='lm', penalization='asgl', lambda1=0)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29234411, 23.41982223, 15.03683211, 25.42968171, 56.26839201, 99.31178417,
                  15.48907319, 10.48258919, 34.87868221, 61.46433177, 66.32752383]),
        decimal=3,
        err_msg='Adaptive sparse group lasso lm failure for lambda=0')

    model = Regressor(model='lm', penalization='asgl', lambda1=0.1, group_weights=[0] * 5, individual_weights=[0] * 10)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29234411, 23.41982223, 15.03683211, 25.42968171, 56.26839201, 99.31178417,
                  15.48907319, 10.48258919, 34.87868221, 61.46433177, 66.32752383]),
        decimal=3,
        err_msg='Adaptive sparse group lasso lm failure for lambda=0.1 and weights=0')

    model = Regressor(model='lm', penalization='asgl', lambda1=0.1, alpha=0.5, group_power_weight=0,
                      individual_power_weight=0)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.32400563, 23.11530379, 14.9984229, 25.23922429, 56.02652827,
                  99.32094058, 15.36037545, 10.24549425, 34.86597551, 61.36516422,
                  66.16108843]),
        decimal=3,
        err_msg='Adaptive sparse group lasso lm failure for lambda=0.1, alpha=0.5 and power_weight=0')

    model = Regressor(model='lm', penalization='asgl', lambda1=0.1, weight_technique='pca_pct',
                      individual_power_weight=1.2, group_power_weight=1.2)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([7.29356871, 23.39588208, 15.02609669, 25.42279202, 56.25540948,
                  99.32184685, 15.48087228, 10.45644831, 34.88147587, 61.46424754,
                  66.3263819]),
        decimal=3,
        err_msg='Adaptive sparse group lasso lm failure for lambda=0.1, weight_technique="pca_pct" and power_weight=1.2')


def test_asgl_qr():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model='qr', penalization='asgl', quantile=0.8, weight_technique='pca_pct', lambda1=0.1, alpha=0.5)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([17.15145409, 17.06745707, 17.90240696, 22.84631114,
                  49.77249431, 102.7879899, 13.27559268, 2.58110347,
                  34.05590417, 63.55204528, 65.5195799]),
        decimal=3,
        err_msg='Adaptive sparse group lasso qr failure for quantile 0.8, weight_technique="pca_pct", lambda1=0.1 and alpha=0.5')

    model = Regressor(model='qr', penalization='asgl', quantile=0.2, weight_technique='pca_pct', lambda1=0.1, alpha=0.5)
    model.fit(X, y, group_index)
    np.testing.assert_array_almost_equal(
        model.coef_,
        np.array([-2.18684365, 17.73865638, 11.62360808, 23.16738009,
                  53.97848008, 102.2359526, 13.9172748, 6.2259722,
                  35.88996468, 60.71910699, 67.64206903]),
        decimal=3,
        err_msg='Adaptive sparse group lasso qr failure for quantile 0.2, weight_technique="pca_pct", lambda1=0.1 and alpha=0.5')


# ERROR HANDLING ------------------------------------------------------------------------------------------------------


def test_errors():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model='qr', penalization='gl', quantile=0.2, lambda1=0.1)
    with pytest.raises(ValueError,
                       match="The penalization provided requires fitting the model with a group_index parameter but no group_index was detected."):
        model.fit(X, y)

    model = Regressor(model='aaa', penalization='lasso', quantile=0.2, lambda1=0.1)
    with pytest.raises(ValueError,
                       match="Invalid value for model parameter."):
        model.fit(X, y)

    model = Regressor(model='lm', penalization='aaa', quantile=0.2, lambda1=0.1)
    with pytest.raises(ValueError,
                       match="Invalid value for penalization parameter."):
        model.fit(X, y)

    model = Regressor(model='lm', penalization='alasso', quantile=0.1, lambda1=0.1, weight_technique='aaa')
    with pytest.raises(AttributeError,
                       match="'Regressor' object has no attribute '_waaa'"):
        model.fit(X, y)

# SKLEARN COMPATIBILITY -----------------------------------------------------------------------------------------------


def test_predict():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model='lm', penalization='asgl', lambda1=0.1)
    model.fit(X, y, group_index)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    np.testing.assert_almost_equal(mse, np.float64(63.11994260430959),
                                   decimal=3,
                                   err_msg='Failed prediction and / or metric computation')

def test_grid_search():
    data = np.loadtxt('data.csv', delimiter=",", dtype=float)
    X = data[:, :-1]
    y = data[:, -1]
    group_index = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 5])

    model = Regressor(model='lm', penalization='asgl')
    param_grid = {'lambda1': [1e-3, 1e-2, 10], 'alpha': [0, 0.5, 1], 'weight_technique': ['pca_pct', 'unpenalized']}
    gscv = GridSearchCV(model, param_grid=param_grid)
    gscv.fit(X, y, **{'group_index': group_index})
    expected_output = {'alpha': 0, 'lambda1': 0.01, 'weight_technique': 'pca_pct'}

    # Assert that the dictionary contains the expected key-value pairs
    for key, value in gscv.best_params_.items():
        assert expected_output.get(key) == value, f"Expected {key} to be {value}, but got {expected_output.get(key)}"


if __name__ == "__main__":
    pytest.main()
