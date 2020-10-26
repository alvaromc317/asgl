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
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=False, num_cores=None, solver=None, max_iters=500,
                 weight_technique='pca_pct', weight_tol=1e-4, lasso_power_weight=1, gl_power_weight=1,
                 variability_pct=0.9, lambda1_weights=1e-1, spca_alpha=1e-5, spca_ridge_alpha=1e-2, error_type='MSE',
                 random_state=None, train_pct=0.05, validate_pct=0.05, train_size=None, validate_size=None):

        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, lasso_weights, gl_weights, parallel,
                         num_cores, solver, max_iters, weight_technique, weight_tol, lasso_power_weight,
                         gl_power_weight, variability_pct, lambda1_weights, spca_alpha, spca_ridge_alpha, error_type,
                         random_state)
        # Relative to / train validate / test
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
