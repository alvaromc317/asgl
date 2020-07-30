import logging

import numpy as np
from sklearn.model_selection import KFold

from . import asgl
from . import weights

logger = logging.getLogger(__name__)


class CvGeneralClass(asgl.ASGL):
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=None, alpha=None, tau=None,
                 l_weights=None, gl_weights=None, solver='ECOS', parallel=False, num_cores=None,
                 weight_technique='pca_pct', lasso_power_weight=None, gl_power_weight=None, variability_pct=0.8,
                 spca_alpha=None, spca_ridge_alpha=None, error_type='QRE', random_state=None):
        # ASGL
        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, l_weights, gl_weights, solver,
                         parallel, num_cores)
        # Adaptive weights
        self.weight_technique = weight_technique
        self.lasso_power_weight = lasso_power_weight
        self.gl_power_weight = gl_power_weight
        self.variability_pct = variability_pct
        self.spca_alpha = spca_alpha
        self.spca_ridge_alpha = spca_ridge_alpha
        # Relative to CV
        self.error_type = error_type
        self.random_state = random_state

    # FIT WEIGHT AND MODEL ############################################################################################

    def fit_weights_and_model(self, x, y, group_index=None):
        """
        Given the input specified in the class definition, computes adaptive weights (if required) and fits the model
        """
        if (self.penalization is not None) and ('asgl' in self.penalization):
            # Compute weights
            weights_class = weights.WEIGHTS(penalization=self.penalization, tau=self.tau,
                                            weight_technique=self.weight_technique,
                                            lasso_power_weight=self.lasso_power_weight,
                                            gl_power_weight=self.gl_power_weight,
                                            variability_pct=self.variability_pct, spca_alpha=self.spca_alpha,
                                            spca_ridge_alpha=self.spca_ridge_alpha)
            self.l_weights, self.gl_weights = weights_class.fit(x=x, y=y, group_index=group_index)
        # Solve the regression model and obtain coefficients
        self.fit(x=x, y=y, group_index=group_index)

    # OPTIMAL PARAMETER SEARCH RELATED ################################################################################

    def optimal_parameter_idx_to_value(self, optimal_parameters_idx):
        """
        optimal_parameters_idx: output from ASGL.retrieve_parameters_given_param_index() function
        Given the index of the optimal parameters, this function retrieves its value
        To be run only after cross validation or train validate test methods.
        """
        best_lambda = assign_value(self.lambda1, optimal_parameters_idx[0])
        best_alpha = assign_value(self.alpha, optimal_parameters_idx[1])
        best_lasso_power_weight = assign_value(self.lasso_power_weight, optimal_parameters_idx[2])
        best_gl_power_weight = assign_value(self.gl_power_weight, optimal_parameters_idx[3])
        return best_lambda, best_alpha, best_lasso_power_weight, best_gl_power_weight


class CV(CvGeneralClass):
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=None, alpha=None, tau=None,
                 l_weights=None, gl_weights=None, solver='ECOS', parallel=False, num_cores=None,
                 weight_technique='pca_pct', lasso_power_weight=None, gl_power_weight=None, variability_pct=0.8,
                 spca_alpha=None, spca_ridge_alpha=None, error_type='QRE', random_state=None, nfolds=5):
        # ASGL
        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, l_weights, gl_weights, solver,
                         parallel, num_cores, weight_technique, lasso_power_weight, gl_power_weight, variability_pct,
                         spca_alpha, spca_ridge_alpha, error_type, random_state)
        # Relative to cross validation / train validate / test
        self.nfolds = nfolds

    # SPLIT DATA METHODS ##############################################################################################

    def __cross_validation_split(self, nrow):
        # Randomly generate index
        data_index = np.random.choice(nrow, nrow, replace=False)
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

    def cross_validation(self, x, y, group_index):
        """
        Performs crosss validation on a gridsearch defined by the class input parameter values
        """
        error_list = []
        # Define random state if required
        if self.random_state is not None:
            np.random.seed(self.random_state)
        cv_index = self.__cross_validation_split(nrow=x.shape[0])
        for zip_index in cv_index:
            error = self.__one_step_cross_validation(x, y, group_index=group_index, zip_index=zip_index)
            error_list.append(error)
        return np.array(error_list).T  # Each row is a model, each column is a k fold split


class TVT(CvGeneralClass):
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=None, alpha=None, tau=None,
                 l_weights=None, gl_weights=None, solver='ECOS', parallel=False, num_cores=None,
                 weight_technique='pca_pct', lasso_power_weight=None, gl_power_weight=None, variability_pct=0.8,
                 spca_alpha=None, spca_ridge_alpha=None, error_type='QRE', random_state=None, train_pct=0.05,
                 validate_pct=0.05, train_size=None, validate_size=None, ):

        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, l_weights, gl_weights, solver,
                         parallel, num_cores, weight_technique, lasso_power_weight, gl_power_weight, variability_pct,
                         spca_alpha, spca_ridge_alpha, error_type, random_state)
        # Relative to / train validate / test
        self.train_pct = train_pct
        self.validate_pct = validate_pct
        self.train_size = train_size
        self.validate_size = validate_size

    # TRAIN VALIDATE TEST SPLIT #######################################################################################

    def __train_validate_test_split(self, nrow):
        # Randomly generate index
        data_index = np.random.choice(nrow, nrow, replace=False)
        if self.train_size is None:
            self.train_size = int(round(nrow * self.train_pct))
        if self.validate_size is None:
            self.validate_size = int(round(nrow * self.validate_pct))
        # List of 3 elements of size train_size, validate_size, remaining_size
        split_index = np.split(data_index, [self.train_size, self.train_size + self.validate_size])
        return split_index

    # TRAIN VALIDATE TEST #############################################################################################

    def __train_validate(self, x, y, group_index):
        # Define random state if required
        if self.random_state is not None:
            np.random.seed(self.random_state)
        # Split data
        split_index = self.__train_validate_test_split(nrow=x.shape[0])
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
        optimal_parameters_idx = self.retrieve_parameters_given_param_index(minimum_error_idx)
        # Optimal model
        optimal_betas = self.coef_[minimum_error_idx]
        prediction = self.predict(x_new=x[test_index, :])
        test_error = asgl.error_calculator(y_true=y[test_index], prediction_list=prediction,
                                           error_type=self.error_type, tau=self.tau)
        return optimal_betas, optimal_parameters_idx, test_error

    def train_validate_test(self, x, y, group_index):
        """
        Performs a train validation test analysis. Outputs the optimal_beta, the optimal_parameter index and the
        test error achieved.
        """
        validate_error, test_index = self.__train_validate(x, y, group_index)
        optimal_betas, optimal_parameters_idx, test_error = self.__tv_test(x, y, validate_error, test_index)
        return optimal_betas, optimal_parameters_idx, test_error


def assign_value(elt, idx):
    if idx is None:
        return None
    else:
        return elt[idx]
