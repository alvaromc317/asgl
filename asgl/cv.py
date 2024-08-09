import logging
import sys

import numpy as np
from sklearn.model_selection import KFold, GroupKFold

from . import asgl
from . import weights

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CvGeneralClass(asgl.ASGL):
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=False, num_cores=None, solver=None, max_iters=500,
                 weight_technique='pca_pct', weight_tol=1e-4, lasso_power_weight=1, gl_power_weight=1,
                 variability_pct=0.9, lambda1_weights=1e-1, spca_alpha=1e-5, spca_ridge_alpha=1e-2, error_type='MSE',
                 random_state=None):
        # ASGL
        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, lasso_weights, gl_weights, parallel,
                         num_cores, solver, max_iters)
        # Adaptive weights
        self.weight_technique = weight_technique
        self.weight_tol = weight_tol
        self.lasso_power_weight = lasso_power_weight
        self.gl_power_weight = gl_power_weight
        self.variability_pct = variability_pct
        self.lambda1_weights = lambda1_weights
        self.spca_alpha = spca_alpha
        self.spca_ridge_alpha = spca_ridge_alpha
        # Relative to CV
        self.error_type = error_type
        self.random_state = random_state

    # FIT WEIGHT AND MODEL ############################################################################################

    def fit_weights_and_model(self, x, y, group_index=None):
        if (self.penalization is not None) and \
                (self.penalization in ['alasso', 'agl', 'asgl', 'asgl_lasso', 'asgl_gl']):
            # Compute weights
            weights_class = weights.WEIGHTS(model=self.model, penalization=self.penalization,
                                            tau=self.tau, weight_technique=self.weight_technique,
                                            lasso_power_weight=self.lasso_power_weight,
                                            gl_power_weight=self.gl_power_weight,
                                            variability_pct=self.variability_pct,
                                            lambda1_weights=self.lambda1_weights,
                                            spca_alpha=self.spca_alpha,
                                            spca_ridge_alpha=self.spca_ridge_alpha)
            self.lasso_weights, self.gl_weights = weights_class.fit(x=x, y=y, group_index=group_index)
        # Solve the regression model and obtain coefficients
        self.fit(x=x, y=y, group_index=group_index)


# CROSS VALIDATION CLASS ##############################################################################################


class CV(CvGeneralClass):
    """
    Parameters
    ----------
    model: str, default = 'lm'
        Model to be fit. Currently, accepts:
            - 'lm': linear regression models.
            - 'qr': quantile regression models.
    penalization: str or None, default = 'lasso'
        Penalization to use. Currently, accepts:
            - None: unpenalized model.
            - 'lasso': lasso model.
            - 'gl': group lasso model.
            - 'sgl': sparse group lasso model.
            - 'alasso': adaptive lasso model.
            - 'agl': adaptive group lasso model.
            - 'asgl': adaptive sparse group lasso model.
    intercept: bool, default=True,
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
    tol: float, default=1e-5
        The tolerance for a coefficient in the model to be considered as 0. Values smaller than ``tol`` are assumed to
        be 0.
    lambda1: float or array, defaul=0.1
        Controls the level of shrinkage applied on penalizations. Smaller values specify weaker regularization.
        If it is float, it solves the model for the specific value. If it is an array, it solves the model for all
        specified values in the array.
    alpha: float or array, default=0.5
        Tradeoff between lasso and group lasso in sgl and asgl penalizations. ``alpha=1`` enforces a lasso penalization
        while ``alpha=0`` enforces a group lasso penalization. If it is float, it solves the model for the specific
        value. If it is an array, it solves the model for all specified values in the array.
    tau: float, defaul=0.5
        quantile level in quantile regression models. Valid values are between 0 and 1. It only has effect if
        ``model='qr'``
    lasso_weights: array, lost of arrays or None, default=None
        An array containing the values of lasso weights in adaptive penalizations.
        If it is an array, it solves the model for the specific set of weights. If it is a list of arrays, it solves
        the model for all specified arrays in the list. Each array must have length equal to the number of variables.
    gl_weights: array, list of arrays or None default=None
        An array containing the values of group lasso weights in adaptive penalizations.
        If it is an array, it solves the model for the specific set of weights. If it is a list of arrays, it solves
        the model for all specified arrays in the list. Each array must have length equal to the number of groups.
    parallel: bool, default=False
        Should te models be solved in parallel or sequentially.
    num_cores: int or None, default=None
        If ``parallel=True``, ``num_cores`` indicates the number of cores to use in the execution. If it has value
        None, it takes the value of maximum number of cores -1
    solver: str, defaul='defaul'
        Solver to be used by CVXPY. Default uses optimal alternative depending on the problem.
    max_iters: int, default=500
        CVXPY parameter indicating the maximum number of iterations.
    weight_technique: str, default='pca_pct'
        Weight technique used to fit the adaptive weights. Currently, accepts:
            - pca_1: Builds the weights using the first component from PCA.
            - pca_pct: Builds the weights using as many components from PCAas required to achieve the `
            `variability_pct``.
            - pls_1: Builds the weights using the first component from PLS.
            - pls_pct:  Builds the weights using as many components from PLS as indicated to achieve the
            ``variability_pct``.
            - lasso: Builds the weights using the lasso model.
            - unpenalized: Builds the weights using the unpenalized model.
            - sparse_pca: Similar to 'pca_pct' but it builds the weights using sparse PCA components.
    weight_tol: float, default=1e-5
        Tolerance value used to avoid ZeroDivision errors when computing the weights.
    lasso_power_weight: float, default=1
        Power at which the lasso weights are risen. If it is float, it solves the model for the specific value.
        If it is an array, it solves the model for all specified values in the array.
    gl_power_weight: float, default=1
        Power at which the group lasso weights are risen. If it is float, it solves the model for the specific value.
        If it is an array, it solves the model for all specified values in the array.
    variability_pct: float, default=0.9
        Percentage of variability explained by pca, pls and sparse_pca components. It only has effect if
        `` weight_technique`` is one of the following: 'pca_pct', 'pls_pct', 'sparse_pca'.
    lambda1_weights: float, default=0.1
        The value of the parameter ``lambda1`` used to solve the lasso model if ``weight_technique='lasso'``
    spca_alpha: float, default=1e-5
        sparse PCA parameter. See sklearn implementation of sparse PCA for more details.
    spca_ridge_alpha: float, default=1e-2
        sparse PCA parameter. See sklearn implementation of sparse PCA for more details.
    error_type: str, default='MSE'
        Error metric to consider. Currently, accepts:
        - 'MSE': Mean Squared Error.
        - 'MAE': Mean Absolute Error.
        - 'MDAE': Median absolute error.
        - 'QRE': Quantile regression error
    random_state: int or None, default=None
        Pass an int for reproducible results across multiple function calls
    nfolds: int, default=3
        Number of cross-validation folds.
    """
    def __init__(self, model='lm', penalization='lasso', intercept=True, tol=1e-5, lambda1=0.1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=False, num_cores=None, solver=None, max_iters=500,
                 weight_technique='pca_pct', weight_tol=1e-4, lasso_power_weight=1, gl_power_weight=1,
                 variability_pct=0.9, lambda1_weights=0.1, spca_alpha=1e-5, spca_ridge_alpha=1e-2, error_type='MSE',
                 random_state=None, nfolds=3):
        # CvGeneralClass
        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, lasso_weights, gl_weights, parallel,
                         num_cores, solver, max_iters, weight_technique, weight_tol, lasso_power_weight,
                         gl_power_weight, variability_pct, lambda1_weights, spca_alpha, spca_ridge_alpha, error_type,
                         random_state)
        # Relative to cross validation / train validate / test
        self.nfolds = nfolds

    # SPLIT DATA METHODS ##############################################################################################

    def _cross_validation_split(self, nrows, split_index=None):
        """
        Split data based on kfold or group kfold cross validation
        :param nrows: number of rows in the dataset
        :param split_index: Group structure of observations used in GroupKfold. 
                            same length as nrows
        """
        if split_index is None:
            # Randomly generate index
            data_index = np.random.choice(nrows, nrows, replace=False)
            # Split data into k folds
            k_folds = KFold(n_splits=self.nfolds).split(data_index)
        else:
            data_index = np.arange(0, nrows)
            k_folds = GroupKFold(n_splits=self.nfolds).split(X=data_index, groups=split_index)
        # List containing zips of (train, test) indices
        response = [(data_index[train], data_index[validate]) for train, validate in list(k_folds)]
        return response

    # CROSS VALIDATION ################################################################################################

    def _one_step_cross_validation(self, x, y, group_index=None, zip_index=None):
        # Given a zip (one element from the cross_validation_split function) retrieve train / test splits
        train_index, test_index = zip_index
        x_train, x_test = (x[train_index, :], x[test_index, :])
        y_train, y_test = (y[train_index], y[test_index])
        self.fit_weights_and_model(x=x_train, y=y_train, group_index=group_index)
        predictions = self.predict(x_new=x_test)
        error = asgl.error_calculator(y_true=y_test, prediction_list=predictions, error_type=self.error_type,
                                      tau=self.tau)
        return error

    def cross_validation(self, x, y, group_index=None, split_index=None):
        error_list = []
        # Define random state if required
        if self.random_state is not None:
            np.random.seed(self.random_state)
        cv_index = self._cross_validation_split(nrows=x.shape[0], split_index=split_index)
        for zip_index in cv_index:
            error = self._one_step_cross_validation(x, y, group_index=group_index, zip_index=zip_index)
            error_list.append(error)
        return np.array(error_list).T  # Each row is a model, each column is a k fold split


class TVT(CvGeneralClass):
    """
    Parameters
    ----------
    model: str, default = 'lm'
        Model to be fit. Currently, accepts:
            - 'lm': linear regression models.
            - 'qr': quantile regression models.
    penalization: str or None, default = 'lasso'
        Penalization to use. Currently, accepts:
            - None: unpenalized model.
            - 'lasso': lasso model.
            - 'gl': group lasso model.
            - 'sgl': sparse group lasso model.
            - 'alasso': adaptive lasso model.
            - 'agl': adaptive group lasso model.
            - 'asgl': adaptive sparse group lasso model.
    intercept: bool, default=True,
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
    tol: float, default=1e-5
        The tolerance for a coefficient in the model to be considered as 0. Values smaller than ``tol`` are assumed to
        be 0.
    lambda1: float or array, defaul=0.1
        Controls the level of shrinkage applied on penalizations. Smaller values specify weaker regularization.
        If it is float, it solves the model for the specific value. If it is an array, it solves the model for all
        specified values in the array.
    alpha: float or array, default=0.5
        Tradeoff between lasso and group lasso in sgl and asgl penalizations. ``alpha=1`` enforces a lasso penalization
        while ``alpha=0`` enforces a group lasso penalization. If it is float, it solves the model for the specific
        value. If it is an array, it solves the model for all specified values in the array.
    tau: float, defaul=0.5
        quantile level in quantile regression models. Valid values are between 0 and 1. It only has effect if
        ``model='qr'``
    lasso_weights: array, lost of arrays or None, default=None
        An array containing the values of lasso weights in adaptive penalizations.
        If it is an array, it solves the model for the specific set of weights. If it is a list of arrays, it solves
        the model for all specified arrays in the list. Each array must have length equal to the number of variables.
    gl_weights: array, list of arrays or None default=None
        An array containing the values of group lasso weights in adaptive penalizations.
        If it is an array, it solves the model for the specific set of weights. If it is a list of arrays, it solves
        the model for all specified arrays in the list. Each array must have length equal to the number of groups.
    parallel: bool, default=False
        Should te models be solved in parallel or sequentially.
    num_cores: int or None, default=None
        If ``parallel=True``, ``num_cores`` indicates the number of cores to use in the execution. If it has value
        None, it takes the value of maximum number of cores -1
    solver: str, defaul='defaul'
        Solver to be used by CVXPY. Default uses optimal alternative depending on the problem.
    max_iters: int, default=500
        CVXPY parameter indicating the maximum number of iterations.
    weight_technique: str, default='pca_pct'
        Weight technique used to fit the adaptive weights. Currently, accepts:
            - pca_1: Builds the weights using the first component from PCA.
            - pca_pct: Builds the weights using as many components from PCAas required to achieve the `
            `variability_pct``.
            - pls_1: Builds the weights using the first component from PLS.
            - pls_pct:  Builds the weights using as many components from PLS as indicated to achieve the
            ``variability_pct``.
            - lasso: Builds the weights using the lasso model.
            - unpenalized: Builds the weights using the unpenalized model.
            - sparse_pca: Similar to 'pca_pct' but it builds the weights using sparse PCA components.
    weight_tol: float, default=1e-5
        Tolerance value used to avoid ZeroDivision errors when computing the weights.
    lasso_power_weight: float, default=1
        Power at which the lasso weights are risen. If it is float, it solves the model for the specific value.
        If it is an array, it solves the model for all specified values in the array.
    gl_power_weight: float, default=1
        Power at which the group lasso weights are risen. If it is float, it solves the model for the specific value.
        If it is an array, it solves the model for all specified values in the array.
    variability_pct: float, default=0.9
        Percentage of variability explained by pca, pls and sparse_pca components. It only has effect if
        `` weight_technique`` is one of the following: 'pca_pct', 'pls_pct', 'sparse_pca'.
    lambda1_weights: float, default=0.1
        The value of the parameter ``lambda1`` used to solve the lasso model if ``weight_technique='lasso'``
    spca_alpha: float, default=1e-5
        sparse PCA parameter. See sklearn implementation of sparse PCA for more details.
    spca_ridge_alpha: float, default=1e-2
        sparse PCA parameter. See sklearn implementation of sparse PCA for more details.
    error_type: str, default='MSE'
        Error metric to consider. Currently, accepts:
        - 'MSE': Mean Squared Error.
        - 'MAE': Mean Absolute Error.
        - 'MDAE': Median absolute error.
        - 'QRE': Quantile regression error
    random_state: int or None, default=None
        Pass an int for reproducible results across multiple function calls
    train_pct: float, default=0.6
        Percentage of observations used in training. ``train_pct`` must be smaller than 1, and the total sum of
        ``train_pct + validate_pct`` cannot be larger than 1. The remaining percentage  not used in train or
        validation will be used in testing.
    validate_pct: float, default=0.2
        Percentage of observations used in validation. ``validate_pct`` must be smaller than 1, and the total sum of
        ``train_pct + validate_pct`` cannot be larger than 1. The remaining percentage  not used in train or
        validation will be used in testing.
    train_size: int or None, default=None
        Number of observations used in training. ``train_size`` must be smaller than x.shape[0], and the total sum of
        ``train_size + validate_size`` cannot be larger than x.shape[0]. The remaining observations  not used in train
        or validation will be used in testing. This parameter takes precedence over ``train_pct``.
    validate_size: int or None, default=None
        Number of observations used in training. ``validate_size`` must be smaller than x.shape[0], and the total sum of
        ``train_size + validate_size`` cannot be larger than x.shape[0]. The remaining observations  not used in train
        or validation will be used in testing. This parameter takes precedence over ``validate_pct``.
    """
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=False, num_cores=None, solver=None, max_iters=500,
                 weight_technique='pca_pct', weight_tol=1e-4, lasso_power_weight=1, gl_power_weight=1,
                 variability_pct=0.9, lambda1_weights=1e-1, spca_alpha=1e-5, spca_ridge_alpha=1e-2, error_type='MSE',
                 random_state=None, train_pct=0.7, validate_pct=0.05, train_size=None, validate_size=None):

        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, lasso_weights, gl_weights, parallel,
                         num_cores, solver, max_iters, weight_technique, weight_tol, lasso_power_weight,
                         gl_power_weight, variability_pct, lambda1_weights, spca_alpha, spca_ridge_alpha, error_type,
                         random_state)
        self.train_pct = train_pct
        self.validate_pct = validate_pct
        self.train_size = train_size
        self.validate_size = validate_size

    # TRAIN VALIDATE TEST SPLIT #######################################################################################

    def _train_validate_test_split(self, nrows):
        """
        Splits data into train validate and test. Takes into account the random state for consistent repetitions
        :param nrows: Number of observations
        :return:  A three items object including train index, validate index and test index
        """
        # Randomly generate index
        data_index = np.random.choice(nrows, nrows, replace=False)
        if self.train_size is None:
            self.train_size = int(round(nrows * self.train_pct))
        if self.validate_size is None:
            self.validate_size = int(round(nrows * self.validate_pct))
        # List of 3 elements of size train_size, validate_size, remaining_size
        split_index = np.split(data_index, [self.train_size, self.train_size + self.validate_size])
        return split_index

    # TRAIN VALIDATE TEST #############################################################################################

    def _train_validate(self, x, y, group_index=None):
        """
        Split the data into train validate and test. Fit an adaptive lasso based model on the train split. Obtain
        predictions and compute error using validation set.
        :param x: data matrix x
        :param y: response vector y
        :param group_index: group index of predictors in data matrix x
        :return: Validate error and test index
        """
        # Define random state if required
        if self.random_state is not None:
            np.random.seed(self.random_state)
        # Split data
        split_index = self._train_validate_test_split(nrows=x.shape[0])
        x_train, x_validate, drop = [x[idx, :] for idx in split_index]
        y_train, y_validate, drop = [y[idx] for idx in split_index]
        test_index = split_index[2]
        # Solve models
        self.fit_weights_and_model(x=x_train, y=y_train, group_index=group_index)
        # Obtain predictions and compute validation error
        predictions = self.predict(x_new=x_validate)
        validate_error = asgl.error_calculator(y_true=y_validate, prediction_list=predictions,
                                               error_type=self.error_type, tau=self.tau)
        return validate_error, test_index

    def _tv_test(self, x, y, validate_error, test_index):
        """
        Given a validate error and a test index obtains the final test_error and stores the optimal parameter values and
        optimal betas
        :param x: data matrix x
        :param y: response vector y
        :param validate_error: Validation error computed in __train_validate()
        :param test_index: Test index
        :return: optimal_betas, optimal_parameters, test_error
        """
        # Select the minimum error
        minimum_error_idx = np.argmin(validate_error)
        # Select the parameters index associated to mininum error values
        optimal_parameters = self.retrieve_parameters_value(minimum_error_idx)
        # Minimum error model
        optimal_betas = self.coef_[minimum_error_idx]
        test_prediction = [self.predict(x_new=x[test_index, :])[minimum_error_idx]]
        test_error = asgl.error_calculator(y_true=y[test_index], prediction_list=test_prediction,
                                           error_type=self.error_type, tau=self.tau)[0]
        return optimal_betas, optimal_parameters, test_error

    def train_validate_test(self, x, y, group_index=None):
        """
        Runs functions __train_validate() and __tv_test(). Stores results in a dictionary
        :param x: data matrix x
        :param y: response vector y
        :param group_index: group index of predictors in data matrix x
        """
        validate_error, test_index = self._train_validate(x, y, group_index)
        optimal_betas, optimal_parameters, test_error = self._tv_test(x, y, validate_error, test_index)
        result = dict(
            optimal_betas=optimal_betas,
            optimal_parameters=optimal_parameters,
            test_error=test_error)
        return result


# TRAIN TEST SPLIT ###################################################################################################

def train_test_split(nrows, train_size=None, train_pct=0.7, random_state=None):
    """
    Splits data into train / test. Takes into account random_state for future consistency.
    """
    # Define random state if required
    if random_state is not None:
        np.random.seed(random_state)
    data_index = np.random.choice(nrows, nrows, replace=False)
    if train_size is None:
        train_size = int(round(nrows * train_pct))
    # Check that nrows is larger than train_size
    if nrows < train_size:
        logging.error(f'Train size is too large. Input number of rows:{nrows}, current train_size: {train_size}')
    # List of 2 elements of size train_size, remaining_size (test)
    split_index = np.split(data_index, [train_size])
    train_idx, test_idx = [elt for elt in split_index]
    return train_idx, test_idx

