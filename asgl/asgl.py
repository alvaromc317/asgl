import functools
import itertools
import logging
import multiprocessing as mp

import cvxpy
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class ASGL:
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=None, alpha=None, tau=None,
                 l_weights=None, gl_weights=None, solver='ECOS', parallel=False, num_cores=None):
        self.valid_models = ['lm', 'qr']
        self.valid_penalizations = ['lasso', 'gl', 'sgl', 'asgl', 'asgl_lasso', 'asgl_gl']
        self.model = model
        self.penalization = penalization
        self.intercept = intercept
        self.tol = tol
        self.lambda1 = lambda1
        self.alpha = alpha
        self.tau = tau
        self.l_weights = l_weights
        self.gl_weights = gl_weights
        self.solver = solver
        self.parallel = parallel
        self.num_cores = num_cores
        self.coef_ = None

    # Model checker related functions #################################################################################

    def __model_checker(self):
        """
        Checks if the input model is one of the valid options:
         - lm for linear models
         - qr for quantile regression models
        """
        if self.model in self.valid_models:
            return True
        else:
            logger.error(f'{self.model} is not a valid model. Valid models are {self.valid_models}')
            return False

    def __penalization_checker(self):
        """
        Checks if the penalization is one of the valid options:
         - lasso for lasso penalization
         - gl for group lasso penalization
         - sgl for sparse group lasso penalization
         - asgl for adaptive sparse group lasso penalization
         - asgl_lasso for an sparse group lasso with adaptive weights in the lasso part
         - asgl_gl for an sparse group lasso with adaptive weights in the group lasso part
        """
        if (self.penalization in self.valid_penalizations) or (self.penalization is None):
            return True
        else:
            logger.error(f'{self.penalization} is not a valid penalization. '
                  f'Valid penalizations are {self.valid_penalizations} or None')
            return False

    def __dtype_checker(self):
        """
        Checks if some of the inputs are in the correct format
        """
        response_1 = False
        response_2 = False
        if isinstance(self.intercept, bool):
            response_1 = True
        if isinstance(self.tol, np.float):
            response_2 = True
        response = response_1 and response_2
        return response

    def __input_checker(self):
        """
        Checks that every input parameter for the model solvers has the expected format
        """
        response_list = [self.__model_checker(), self.__penalization_checker(), self.__dtype_checker()]
        return False not in response_list

    # Preprocessing related functions #################################################################################

    def __preprocessing_lambda(self):
        """
        Processes the input lambda1 parameter and transforms it as required by the solver pacakge functions
        """
        n_lambda = None
        lambda_vector = None
        if self.penalization is not None:
            if isinstance(self.lambda1, (np.float, np.int)):
                lambda_vector = [self.lambda1]
            else:
                lambda_vector = self.lambda1
            n_lambda = len(lambda_vector)
        return n_lambda, lambda_vector

    def __preprocessing_alpha(self):
        """
        Processes the input alpha parameter from sgl and asgl penalizations and transforms it as required by the solver
        pacakge functions
        """
        n_alpha = None
        alpha_vector = None
        if 'sgl' in self.penalization:
            if self.alpha is not None:
                if isinstance(self.alpha, (np.float, np.int)):
                    alpha_vector = [self.alpha]
                else:
                    alpha_vector = self.alpha
                n_alpha = len(alpha_vector)
        return n_alpha, alpha_vector

    def __preprocessing_weights(self, weights):
        """
        Converts l_weights into a list of lists. Each list inside l_weights defines a set of weights for a model
        """
        n_weights = None
        weights_list = None
        if 'asgl' in self.penalization:
            if weights is not None:
                if isinstance(weights, list):
                    # If l_weights is a list of lists -> convert to list of arrays
                    if isinstance(weights[0], list):
                        weights_list = [np.asarray(elt) for elt in weights]
                    # If l_weights is a list of numbers -> store in a list
                    elif isinstance(weights[0], (np.float, np.int)):
                        weights_list = [np.asarray(weights)]
                    else:
                        # If it is a list of arrays, maintain this way
                        weights_list = weights
                # If l_weights is a ndarray -> store in a list and convert into list
                elif isinstance(weights, np.ndarray):
                    weights_list = [weights]
                if self.intercept:
                    weights_list = [np.insert(elt, 0, 0, axis=0) for elt in weights_list]
            n_weights = len(weights_list)
        return n_weights, weights_list

    def __preprocessing_itertools_param(self, lambda_vector, alpha_vector, l_weights_list, gl_weights_list):
        """
        Receives as input the results from preprocessing_lambda, preprocessing_alpha and preprocessing_weights
        Outputs an iterable list of parameter values "param"
        """
        if self.penalization in ['lasso', 'gl']:
            param = lambda_vector
        elif self.penalization == 'sgl':
            param = itertools.product(lambda_vector, alpha_vector)
        elif 'asgl' in self.penalization:
            param = itertools.product(lambda_vector, alpha_vector, l_weights_list, gl_weights_list)
        else:
            param = None
            logger.error(f'Error preprocessing input parameters')
        param = list(param)
        return param

    def __preprocessing(self):
        """
        Receives all the parameters of the models and creates tuples of the parameters to be executed in the penalized
        model solvers
        """
        # Run the input_checker to verify that the inputs have the correct format
        if self.__input_checker() is False:
            logger.error('incorrect input parameters')
            raise ValueError('incorrect input parameters')
        # Defines param as None for the unpenalized model
        if self.penalization is None:
            param = None
        else:
            # Reformat parameter vectors
            n_lambda, lambda_vector = self.__preprocessing_lambda()
            n_alpha, alpha_vector = self.__preprocessing_alpha()
            n_l_weights, l_weights_list = self.__preprocessing_weights(self.l_weights)
            n_gl_weights, gl_weights_list = self.__preprocessing_weights(self.gl_weights)
            # Store correctly formatted parameters in self
            self.lambda1 = lambda_vector
            self.alpha = alpha_vector
            self.l_weights = l_weights_list
            self.gl_weights = gl_weights_list
            param = self.__preprocessing_itertools_param(lambda_vector, alpha_vector, l_weights_list, gl_weights_list)
        return param

    # NUMBER OF PARAMETERS ############################################################################################

    def num_parameters(self):
        """
        retrieves the number of parameters to be considered in a model
        Output: tuple [num_models, n_lambda, n_alpha, n_l_weights, n_gl_weights] where
        - num_models: total number of models to be solved for the grid of parameters given
        - n_lambda: number of different lambda1 values
        - n_alpha: number of different alpha values
        - n_l_weights: number of different weights for the lasso part of the asgl (or asgl_lasso) penalizations
        - n_gl_weights: number of different weights for the lasso part of the asgl (or asgl_gl) penalizations
        """
        # Run the input_checker to verify that the inputs have the correct format
        if self.__input_checker() is False:
            logger.error('incorrect input parameters')
            raise ValueError('incorrect input parameters')
        if self.penalization is None:
            # See meaning of each element in the "else" result statement.
            result = [1, None, None, None, None]
        else:
            n_lambda, drop = self.__preprocessing_lambda()
            n_alpha, drop = self.__preprocessing_alpha()
            n_l_weights, drop = self.__preprocessing_weights(self.l_weights)
            n_gl_weights, drop = self.__preprocessing_weights(self.gl_weights)
            list_param = [n_lambda, n_alpha, n_l_weights, n_gl_weights]
            list_param_no_None = [elt for elt in list_param if elt]
            num_models = np.prod(list_param_no_None)
            result = [num_models, n_lambda, n_alpha, n_l_weights, n_gl_weights]
        return result

    def retrieve_parameters_given_param_index(self, param_index):
        """
        Given an index for the param iterable output from __preprocessing function, this function returns a tupple
        with the index of the value of each parameter.
        Example: Solving an adaptive sparse group lasso model with 5 values for lambda1, 4 values for alpha,
                 3 possible lasso weights and 3 possible group lasso weights yields in a grid search on
                 5*4*3*3=180 parameters.
                 Inputing param_index=120 (out of the 180 possible values)in this function will output the
                 lambda, alpha, and weights index for such value
        If the penalization under consideration does not include any of the required parameters (for example, if we are
        solving an sparse group lasso model, we do not consider adaptive weights), the output regarding the non used
        parameters are set to be None.
        """
        n_models, n_lambda, n_alpha, n_l_weights, n_gl_weights = self.num_parameters()
        if param_index > n_models:
            string = f'param_index should be smaller or equal than the number of models solved. n_models={n_models}, ' \
                     f'param_index={param_index}'
            logger.error(string)
            raise ValueError(string)
        # If penalization is None, all parameters are set to None
        if self.penalization is None:
            result = [None, None, None, None]
        # If penalization is lasso or gl, there is only one parameter, so param_index = position of that parameter
        elif self.penalization in ['lasso', 'gl']:
            result = [param_index, None, None, None]
        # If penalization is sgl, there are two parameters and two None
        elif self.penalization == 'sgl':
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_alpha))
            optimal_parameter_idx = np.where(parameter_matrix == param_index)
            result = [optimal_parameter_idx[0][0], optimal_parameter_idx[1][0], None, None]
        else:
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_alpha, n_l_weights, n_gl_weights))
            optimal_parameter_idx = np.where(parameter_matrix == param_index)
            result = [optimal_parameter_idx[0][0], optimal_parameter_idx[1][0],
                      optimal_parameter_idx[2][0], optimal_parameter_idx[3][0]]
        return result

    # SOLVERS #########################################################################################################

    def __quantile_function(self, x):
        """
        Quantile function required for quantile regression models.
        """
        return 0.5 * cvxpy.abs(x) + (self.tau - 0.5) * x

    def __num_beta_var_from_group_index(self, group_index):
        """
        Internal function used in group based penalizations (gl, sgl, asgl, asgl_lasso, asgl_gl)
        """
        group_sizes = []
        beta_var = []
        unique_group_index = np.unique(group_index)
        # Define the objective function
        for idx in unique_group_index:
            group_sizes.append(len(np.where(group_index == idx)[0]))
            beta_var.append(cvxpy.Variable(len(np.where(group_index == idx)[0])))
        return group_sizes, beta_var

    def unpenalized_solver(self, x, y):
        n, m = x.shape
        # If we want an intercept, it adds a column of ones to the matrix x
        if self.intercept:
            m = m + 1
            x = np.c_[np.ones(n), x]
        # Define the objective function
        beta_var = cvxpy.Variable(m)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - x @ beta_var)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self.__quantile_function(x=(y - x @ beta_var)))
        objective = cvxpy.Minimize(objective_function)
        problem = cvxpy.Problem(objective)
        # Solve the problem
        problem.solve(solver=getattr(cvxpy, self.solver))
        if problem.status in ["infeasible", "unbounded"]:
            logger.warning('Optimization problem status failure')
        beta_sol = beta_var.value
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return [beta_sol]

    def lasso(self, x, y, param):
        """
        Lasso penalized solver
        """
        n, m = x.shape
        # If we want an intercept, it adds a column of ones to the matrix x.
        # Init_pen controls when the penalization starts, this way the intercept is not penalized
        if self.intercept:
            m = m + 1
            x = np.c_[np.ones(n), x]
            init_pen = 1
        else:
            init_pen = 0
        # Define the objective function
        lambda_param = cvxpy.Parameter(nonneg=True)
        beta_var = cvxpy.Variable(m)
        lasso_penalization = lambda_param * cvxpy.norm(beta_var[init_pen:], 1)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - x @ beta_var)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self.__quantile_function(x=(y - x @ beta_var)))
        objective = cvxpy.Minimize(objective_function + lasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam in param:
            lambda_param.value = lam
            problem.solve(solver=getattr(cvxpy, self.solver))
            if problem.status in ["infeasible", "unbounded"]:
                logger.warning('Optimization problem status failure')
            beta_sol = beta_var.value
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        logger.debug('Function finished without errors')
        return beta_sol_list

    def gl(self, x, y, group_index, param):
        """
        Group lasso penalized solver
        """
        n = x.shape[0]
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self.__num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.intercept:
            # Adds an element (referring to the intercept) to group_index, group_sizes, num groups
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            x = np.c_[np.ones(n), x]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            # Compute model prediction for the intercept with no penalization
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(beta_var[i], 2)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - model_prediction)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self.__quantile_function(x=(y - model_prediction)))
        lambda_param = cvxpy.Parameter(nonneg=True)
        objective = cvxpy.Minimize(objective_function + (lambda_param * group_lasso_penalization))
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam in param:
            lambda_param.value = lam
            problem.solve(solver=getattr(cvxpy, self.solver))
            if problem.status in ["infeasible", "unbounded"]:
                logger.warning('Optimization problem status failure')
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def sgl(self, x, y, group_index, param):
        """
        Sparse group lasso penalized solver
        """
        n = x.shape[0]
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self.__num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        lasso_penalization = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.intercept:
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            x = np.c_[np.ones(n), x]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(beta_var[i], 2)
            lasso_penalization += cvxpy.norm(beta_var[i], 1)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - model_prediction)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self.__quantile_function(x=(y - model_prediction)))
        lasso_param = cvxpy.Parameter(nonneg=True)
        grp_lasso_param = cvxpy.Parameter(nonneg=True)
        objective = cvxpy.Minimize(objective_function +
                                   (grp_lasso_param * group_lasso_penalization) +
                                   (lasso_param * lasso_penalization))
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam, al in param:
            lasso_param.value = lam * al
            grp_lasso_param.value = lam * (1 - al)
            problem.solve(solver=getattr(cvxpy, self.solver))
            if problem.status in ["infeasible", "unbounded"]:
                logger.warning('Optimization problem status failure')
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def asgl(self, x, y, group_index, param):
        """
        adaptive sparse group lasso penalized solver
        """
        n, m = x.shape
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self.__num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        alasso_penalization = 0
        a_group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.intercept:
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            x = np.c_[np.ones(n), x]
            m = m + 1
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        l_weights_param = cvxpy.Parameter(m, nonneg=True)
        gl_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            a_group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * gl_weights_param[i] * cvxpy.norm(beta_var[i], 2)
            alasso_penalization += l_weights_param[np.where(group_index ==
                                                            unique_group_index[i])[0]].T @ cvxpy.abs(beta_var[i])
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - model_prediction)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self.__quantile_function(x=(y - model_prediction)))
        objective = cvxpy.Minimize(objective_function +
                                   a_group_lasso_penalization +
                                   alasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam, al, lw, glw in param:
            l_weights_param.value = lw * lam * al
            gl_weights_param.value = glw * lam * (1 - al)
            problem.solve(solver=getattr(cvxpy, self.solver))
            if problem.status in ["infeasible", "unbounded"]:
                logger.warning('Optimization problem status failure')
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    # PARALLEL CODE ###################################################################################################

    def parallel_execution(self, x, y, param, group_index=None):
        """
        Parallel implementation of the solvers
        """
        if self.num_cores is None:
            # If the number of cores is not defined by user, use all available minus 1
            num_chunks = mp.cpu_count() - 1
        else:
            num_chunks = np.min((self.num_cores, mp.cpu_count()))
        # Divide the list of parameter values into as many chunks as threads that we want to use in parallel
        tmp_group_index_chunks = np.array_split(range(len(param)), num_chunks)
        # If the number of parameters is shorter than the number of threads, delete the empty groups
        group_index_chunks = []
        for elt in tmp_group_index_chunks:
            if elt.size != 0:
                group_index_chunks.append(elt)
        # chunks is a list with as many elements as num_chunks
        # Each element of the list is another list of tuples of parameter values
        chunks = []
        for elt in group_index_chunks:
            chunks.append(param[elt[0]:(1 + elt[-1])])
        # Solve problem in parallel
        pool = mp.Pool(num_chunks)
        if self.penalization == 'lasso':
            global_results = pool.map(functools.partial(getattr(self, self.__get_solver_names()), x, y), chunks)
        else:
            global_results = pool.map(functools.partial(getattr(self, self.__get_solver_names()), x, y, group_index),
                                      chunks)
        pool.close()
        pool.join()
        # Re-build the output of the function
        beta_sol_list = []
        if len(param) < num_chunks:
            limit = len(param)
        else:
            limit = num_chunks
        for i in range(limit):
            beta_sol_list.extend(global_results[i])
        return beta_sol_list

    # FIT METHOD ######################################################################################################

    def __get_solver_names(self):
        if 'asgl' in self.penalization:
            return 'asgl'
        else:
            return self.penalization

    def fit(self, x, y, group_index=None):
        """
        Main function of the module. Given a model, penalization and parameter values specified in the class definition,
        this function solves the model and produces the regression coefficients
        """
        param = self.__preprocessing()
        if self.penalization is None:
            self.coef_ = self.unpenalized_solver(x=x, y=y)
        else:
            if self.parallel is False:
                if self.penalization == 'lasso':
                    self.coef_ = getattr(self, self.__get_solver_names())(x=x, y=y, param=param)
                else:
                    self.coef_ = getattr(self, self.__get_solver_names())(x=x, y=y, param=param,
                                                                          group_index=group_index)
            else:
                self.coef_ = self.parallel_execution(x=x, y=y, param=param, group_index=group_index)

    # PREDICTION METHOD ###############################################################################################

    def predict(self, x_new):
        """
        To be executed after fitting a model. Given a new dataset, this function produces predictions for that data
        considering the different model coefficients output provided by function fit
        """
        if self.intercept:
            x_new = np.c_[np.ones(x_new.shape[0]), x_new]
        if x_new.shape[1] != len(self.coef_[0]):
            logger.error('Model dimension and new data dimension does not match')
            raise ValueError('Model dimension and new data dimension does not match')
        # Store predictions in a list
        prediction_list = []
        for elt in self.coef_:
            prediction_list.append(np.dot(x_new, elt))
        return prediction_list


# ERROR CALCULATOR METHOD #############################################################################################

def quantile_function(y_true, y_pred, tau):
    """
    Quantile function required for error computation
    """
    return (1.0 / len(y_true)) * np.sum(0.5 * np.abs(y_true - y_pred) + (tau - 0.5) * (y_true - y_pred))


def error_calculator(y_true, prediction_list, error_type="MSE", tau=None):
    """
    Computes the error between the predicted value and the true value of the response variable
    """
    error_dict = dict(
        MSE=mean_squared_error,
        MAE=mean_absolute_error,
        MDAE=median_absolute_error,
        QRE=quantile_function)
    valid_error_types = error_dict.keys()
    # Check that the error_type is a valid error type considered
    if error_type not in valid_error_types:
        raise ValueError(f'invalid error type. Valid error types are {error_dict.keys()}')
    if y_true.shape[0] != len(prediction_list[0]):
        logger.error('Dimension of test data does not match dimension of prediction')
        raise ValueError('Dimension of test data does not match dimension of prediction')
    # For each prediction, store the error associated to that prediction in a list
    error_list = []
    if error_type == 'QRE':
        for y_pred in prediction_list:
            error_list.append(error_dict[error_type](y_true=y_true, y_pred=y_pred, tau=tau))
    else:
        for y_pred in prediction_list:
            error_list.append(error_dict[error_type](y_true=y_true, y_pred=y_pred))
    return error_list
