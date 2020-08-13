import logging

import numpy as np
from sklearn.model_selection import KFold

from . import asgl
from . import weights

logger = logging.getLogger(__name__)


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
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=False, num_cores=None, solver=None, max_iters=500,
                 weight_technique='pca_pct', weight_tol=1e-4, lasso_power_weight=1, gl_power_weight=1,
                 variability_pct=0.9, lambda1_weights=1e-1, spca_alpha=1e-5, spca_ridge_alpha=1e-2, error_type='MSE',
                 random_state=None, nfolds=5):
        """
        Parameters:
            All the parameters from ASGL class
            All the parameters from WEIGHTS class
            error_type: error measurement to use. Accepts:
                'MSE': mean squared error
                'MAE': mean absolute error
                'MDAE': mean absolute deviation error
                'QRE': quantile regression error
            random_state: random state value in case reproducible data splits are required
            nfolds: number of folds in which the dataset should be split. Default value is 5
        """
        # ASGL
        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, lasso_weights, gl_weights, parallel,
                         num_cores, solver, max_iters, weight_technique, weight_tol, lasso_power_weight, gl_power_weight,
                         variability_pct, lambda1_weights, spca_alpha, spca_ridge_alpha, error_type, random_state)
        # Relative to cross validation / train validate / test
        self.nfolds = nfolds

    # SPLIT DATA METHODS ##############################################################################################

    def __cross_validation_split(self, nrows):
        # Randomly generate index
        data_index = np.random.choice(nrows, nrows, replace=False)
        # Split data into k folds
        k_folds = KFold(n_splits=self.nfolds).split(data_index)
        # List containing zips of (train, test) indices
        response = [(data_index[train], data_index[validate]) for train, validate in list(k_folds)]
        return response

    # CROSS VALIDATION ################################################################################################

    def __one_step_cross_validation(self, x, y, group_index=None, zip_index=None):
        # Given a zip (one element from the cross_validation_split function) retrieve train / test splits
        train_index, test_index = zip_index
        x_train, x_test = (x[train_index, :], x[test_index, :])
        y_train, y_test = (y[train_index], y[test_index])
        self.fit_weights_and_model(x=x_train, y=y_train, group_index=group_index)
        predictions = self.predict(x_new=x_test)
        error = asgl.error_calculator(y_true=y_test, prediction_list=predictions, error_type=self.error_type,
                                      tau=self.tau)
        return error

    def cross_validation(self, x, y, group_index=None):
        error_list = []
        # Define random state if required
        if self.random_state is not None:
            np.random.seed(self.random_state)
        cv_index = self.__cross_validation_split(nrows=x.shape[0])
        for zip_index in cv_index:
            error = self.__one_step_cross_validation(x, y, group_index=group_index, zip_index=zip_index)
            error_list.append(error)
        return np.array(error_list).T  # Each row is a model, each column is a k fold split


class TVT(CvGeneralClass):
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=False, num_cores=None, solver=None, max_iters=500,
                 weight_technique='pca_pct', weight_tol=1e-4, lasso_power_weight=1, gl_power_weight=1,
                 variability_pct=0.9, lambda1_weights=1e-1, spca_alpha=1e-5, spca_ridge_alpha=1e-2, error_type='MSE',
                 random_state=None, train_pct=0.05, validate_pct=0.05, train_size=None, validate_size=None):

        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, lasso_weights, gl_weights, parallel,
                         num_cores, solver, max_iters, weight_technique, weight_tol, lasso_power_weight, gl_power_weight,
                         variability_pct, lambda1_weights, spca_alpha, spca_ridge_alpha, error_type, random_state)
        # Relative to / train validate / test
        self.train_pct = train_pct
        self.validate_pct = validate_pct
        self.train_size = train_size
        self.validate_size = validate_size

    # TRAIN VALIDATE TEST SPLIT #######################################################################################

    def __train_validate_test_split(self, nrows):
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

    def __train_validate(self, x, y, group_index=None):
        # Define random state if required
        if self.random_state is not None:
            np.random.seed(self.random_state)
        # Split data
        split_index = self.__train_validate_test_split(nrows=x.shape[0])
        x_train, x_validate, x_test = [x[idx, :] for idx in split_index]
        y_train, y_validate, y_test = [y[idx] for idx in split_index]
        test_index = split_index[2]
        # Solve models
        self.fit_weights_and_model(x=x_train, y=y_train, group_index=group_index)
        # Obtain predictions and compute validation error
        predictions = self.predict(x_new=x_validate)
        validate_error = asgl.error_calculator(y_true=y_validate, prediction_list=predictions,
                                               error_type=self.error_type, tau=self.tau)
        return validate_error, test_index

    def __tv_test(self, x, y, validate_error, test_index):
        # Select the minimum error
        minimum_error_idx = np.argmin(validate_error)
        # Select the parameters index associated to mininum error values
        optimal_parameters_idx = self._retrieve_parameters_idx(minimum_error_idx)
        # Optimal model
        optimal_betas = self.coef_[minimum_error_idx]
        prediction = self.predict(x_new=x[test_index, :])
        test_error = asgl.error_calculator(y_true=y[test_index], prediction_list=prediction,
                                           error_type=self.error_type, tau=self.tau)
        return optimal_betas, optimal_parameters_idx, test_error

    def train_validate_test(self, x, y, group_index=None):
        validate_error, test_index = self.__train_validate(x, y, group_index)
        optimal_betas, optimal_parameters_idx, test_error = self.__tv_test(x, y, validate_error, test_index)
        return optimal_betas, optimal_parameters_idx, test_error


# TRAIN TEST SPLIT ###################################################################################################

def train_test_split(nrows, train_size=None, train_pct=0.7, random_state=None):
    # Define random state if required
    if random_state is not None:
        np.random.seed(random_state)
    data_index = np.random.choice(nrows, nrows, replace=False)
    if train_size is None:
        train_size = int(round(nrows * train_pct))
    # Check that nrows is larger than train_size
    if nrows < train_size:
        logger.error(f'Train size is too large. Input number of rows:{nrows}, current train_size: {train_size}')
    # List of 2 elements of size train_size, remaining_size (test)
    split_index = np.split(data_index, [train_size])
    train_idx, test_idx = [elt for elt in split_index]
    return train_idx, test_idx
