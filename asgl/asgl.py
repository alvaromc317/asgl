import sys
import functools
import itertools
import logging
import multiprocessing as mp

import cvxpy
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ASGL:
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=False, num_cores=None, solver=None, max_iters=500):
        """
        Parameters:
            model: model to be fit (accepts 'lm' or 'qr')
            penalization: penalization to use (accepts None, 'lasso', 'gl', 'sgl', 'asgl', 'asgl_lasso', 'asgl_gl',
                          alasso, agl)
            intercept: boolean, whether to fit the model including intercept or not
            tol:  tolerance for a coefficient in the model to be considered as 0
            lambda1: parameter value that controls the level of shrinkage applied on penalizations
            alpha: parameter value, tradeoff between lasso and group lasso in sgl penalization
            tau: quantile level in quantile regression models
            lasso_weights: lasso weights in adaptive penalizations
            gl_weights: group lasso weights in adaptive penalizations
            parallel: boolean, whether to execute the code in parallel or sequentially
            num_cores: if parallel is set to true, the number of cores to use in the execution. Default is (max - 1)
            solver: solver to be used by CVXPY. Default uses optimal alternative depending on the problem
            max_iters: CVXPY parameter. Default is 500

        Returns:
            This is a class definition so there is no return. Main method of this class is fit,  that has no return
            but outputs automatically to _coef.
            ASGL._coef stores a list of regression model coefficients.
        """
        self.valid_models = ['lm', 'qr']
        self.valid_penalizations = ['lasso', 'gl', 'sgl', 'alasso', 'agl', 'asgl', 'asgl_lasso', 'asgl_gl']
        self.model = model
        self.penalization = penalization
        self.intercept = intercept
        self.tol = tol
        self.lambda1 = lambda1
        self.alpha = alpha
        self.tau = tau
        self.lasso_weights = lasso_weights
        self.gl_weights = gl_weights
        self.parallel = parallel
        self.num_cores = num_cores
        self.max_iters = max_iters
        self.coef_ = None
        # Define solver param as a list of the three default solvers for CVXPY
        if solver is None:
            self.solver = ['ECOS', 'OSQP', 'SCS']
        elif not isinstance(solver, list):
            self.solver = [solver]
        else:
            self.solver = solver

    # Model checker related functions #################################################################################

    def _model_checker(self):
        """
        Checks if the input model is one of the valid options:
         - lm for linear models
         - qr for quantile regression models
        """
        if self.model in self.valid_models:
            return True
        else:
            logging.error(f'{self.model} is not a valid model. Valid models are {self.valid_models}')
            return False

    def _penalization_checker(self):
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
            logging.error(f'{self.penalization} is not a valid penalization. '
                          f'Valid penalizations are {self.valid_penalizations} or None')
            return False

    def _dtype_checker(self):
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

    def _input_checker(self):
        """
        Checks that every input parameter for the model solvers has the expected format
        """
        response_list = [self._model_checker(), self._penalization_checker(), self._dtype_checker()]
        return False not in response_list

    # Preprocessing related functions #################################################################################

    def _preprocessing_lambda(self):
        """
        Processes the input lambda1 parameter and transforms it as required by the solver package functions
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

    def _preprocessing_alpha(self):
        """
        Processes the input alpha parameter from sgl and asgl penalizations and transforms it as required by the solver
        package functions
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

    def _preprocessing_weights(self, weights):
        """
        Converts l_weights into a list of lists. Each list inside l_weights defines a set of weights for a model
        """
        n_weights = None
        weights_list = None
        if self.penalization in ['asgl', 'asgl_lasso', 'asgl_gl', 'alasso', 'agl']:
            if weights is not None:
                if isinstance(weights, list):
                    # If weights is a list of lists -> convert to list of arrays
                    if isinstance(weights[0], list):
                        weights_list = [np.asarray(elt) for elt in weights]
                    # If weights is a list of numbers -> store in a list
                    elif isinstance(weights[0], (np.float, np.int)):
                        weights_list = [np.asarray(weights)]
                    else:
                        # If it is a list of arrays, maintain this way
                        weights_list = weights
                # If weights is a ndarray -> store in a list and convert into list
                elif isinstance(weights, np.ndarray):
                    weights_list = [weights]
                if self.intercept:
                    weights_list = [np.insert(elt, 0, 0, axis=0) for elt in weights_list]
                n_weights = len(weights_list)
        return n_weights, weights_list

    def _preprocessing_itertools_param(self, lambda_vector, alpha_vector, lasso_weights_list, gl_weights_list):
        """
        Receives as input the results from preprocessing_lambda, preprocessing_alpha and preprocessing_weights
        Outputs an iterable list of parameter values "param"
        """
        if self.penalization in ['lasso', 'gl']:
            param = lambda_vector
        elif self.penalization == 'sgl':
            param = itertools.product(lambda_vector, alpha_vector)
        elif self.penalization == 'alasso':
            param = itertools.product(lambda_vector, lasso_weights_list)
        elif self.penalization == 'agl':
            param = itertools.product(lambda_vector, gl_weights_list)
        elif 'asgl' in self.penalization:
            param = itertools.product(lambda_vector, alpha_vector, lasso_weights_list, gl_weights_list)
        else:
            param = None
            logging.error(f'Error preprocessing input parameters')
        param = list(param)
        return param

    def _preprocessing(self):
        """
        Receives all the parameters of the models and creates tuples of the parameters to be executed in the penalized
        model solvers
        """
        # Run the input_checker to verify that the inputs have the correct format
        if self._input_checker() is False:
            logging.error('incorrect input parameters')
            raise ValueError('incorrect input parameters')
        # Defines param as None for the unpenalized model
        if self.penalization is None:
            param = None
        else:
            # Reformat parameter vectors
            n_lambda, lambda_vector = self._preprocessing_lambda()
            n_alpha, alpha_vector = self._preprocessing_alpha()
            n_lasso_weights, lasso_weights_list = self._preprocessing_weights(self.lasso_weights)
            n_gl_weights, gl_weights_list = self._preprocessing_weights(self.gl_weights)
            param = self._preprocessing_itertools_param(lambda_vector, alpha_vector, lasso_weights_list,
                                                        gl_weights_list)
        return param

    # CVXPY SOLVER RELATED OPTIONS ###################################################################################

    def _cvxpy_solver_options(self, solver):
        if solver == 'ECOS':
            solver_dict = dict(solver=solver,
                               max_iters=self.max_iters)
        elif solver == 'OSQP':
            solver_dict = dict(solver=solver,
                               max_iter=self.max_iters)
        else:
            solver_dict = dict(solver=solver)
        return solver_dict

    # SOLVERS #########################################################################################################

    def _quantile_function(self, x):
        """
        Quantile function required for quantile regression models.
        """
        return 0.5 * cvxpy.abs(x) + (self.tau - 0.5) * x

    def _num_beta_var_from_group_index(self, group_index):
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
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - x @ beta_var)))
        objective = cvxpy.Minimize(objective_function)
        problem = cvxpy.Problem(objective)
        # Solve the problem. Try first default CVXPY option, which is usually optimal for the problem. If a ValueError
        # arises, try the solvers provided as input to the method.
        try:
            problem.solve(warm_start=True)
        except (ValueError, cvxpy.error.SolverError):
            for elt in self.solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if 'optimal' in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning('Optimization problem status failure')
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
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - x @ beta_var)))
        objective = cvxpy.Minimize(objective_function + lasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam in param:
            lambda_param.value = lam
            # Solve the problem. Try first default CVXPY option, which is usually optimal for the problem. If a
            # ValueError arises, try the solvers provided as input to the method.
            try:
                problem.solve(warm_start=True)
            except (ValueError, cvxpy.error.SolverError):
                for elt in self.solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            beta_sol = beta_var.value
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        logging.debug('Function finished without errors')
        return beta_sol_list

    def gl(self, x, y, group_index, param):
        """
        Group lasso penalized solver
        """
        n = x.shape[0]
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
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
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - model_prediction)))
        lambda_param = cvxpy.Parameter(nonneg=True)
        objective = cvxpy.Minimize(objective_function + (lambda_param * group_lasso_penalization))
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam in param:
            lambda_param.value = lam
            # Solve the problem. Try first default CVXPY option, which is usually optimal for the problem. If a
            # ValueError arises, try the solvers provided as input to the method.
            try:
                problem.solve(warm_start=True)
            except (ValueError, cvxpy.error.SolverError):
                for elt in self.solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
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
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
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
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - model_prediction)))
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
            # Solve the problem. Try first default CVXPY option, which is usually optimal for the problem. If a
            # ValueError arises, try the solvers provided as input to the method.
            try:
                problem.solve(warm_start=True)
            except (ValueError, cvxpy.error.SolverError):
                for elt in self.solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def alasso(self, x, y, param):
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
        l_weights_param = cvxpy.Parameter(m, nonneg=True)
        beta_var = cvxpy.Variable(m)
        lasso_penalization = cvxpy.norm(l_weights_param[init_pen:].T @ cvxpy.abs(beta_var[init_pen:]), 1)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - x @ beta_var)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - x @ beta_var)))
        objective = cvxpy.Minimize(objective_function + lasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam, lw in param:
            l_weights_param.value = lam * lw
            # Solve the problem. Try first default CVXPY option, which is usually optimal for the problem. If a
            # ValueError arises, try the solvers provided as input to the method.
            try:
                problem.solve(warm_start=True)
            except (ValueError, cvxpy.error.SolverError):
                for elt in self.solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            beta_sol = beta_var.value
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        logging.debug('Function finished without errors')
        return beta_sol_list

    def agl(self, x, y, group_index, param):
        """
        Group lasso penalized solver
        """
        n = x.shape[0]
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
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
        gl_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * gl_weights_param[i] * cvxpy.norm(beta_var[i], 2)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - model_prediction)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - model_prediction)))
        objective = cvxpy.Minimize(objective_function + group_lasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam, gl in param:
            gl_weights_param.value = lam * gl
            # Solve the problem. Try first default CVXPY option, which is usually optimal for the problem. If a
            # ValueError arises, try the solvers provided as input to the method.
            try:
                problem.solve(warm_start=True)
            except (ValueError, cvxpy.error.SolverError):
                for elt in self.solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
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
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
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
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - model_prediction)))
        objective = cvxpy.Minimize(objective_function +
                                   a_group_lasso_penalization +
                                   alasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam, al, lw, glw in param:
            l_weights_param.value = lw * lam * al
            gl_weights_param.value = glw * lam * (1 - al)
            # Solve the problem. Try first default CVXPY option, which is usually optimal for the problem. If a
            # ValueError arises, try the solvers provided as input to the method.
            try:
                problem.solve(warm_start=True)
            except (ValueError, cvxpy.error.SolverError):
                for elt in self.solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    # PARALLEL CODE ###################################################################################################

    def _parallel_execution(self, x, y, param, group_index=None):
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
        if self.penalization in ['lasso', 'alasso']:
            global_results = pool.map(functools.partial(getattr(self, self._get_solver_names()), x, y), chunks)
        else:
            global_results = pool.map(functools.partial(getattr(self, self._get_solver_names()), x, y, group_index),
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

    def _get_solver_names(self):
        if 'asgl' in self.penalization:
            return 'asgl'
        else:
            return self.penalization

    def fit(self, x, y, group_index=None):
        """
        Main function of the module. Given a model, penalization and parameter values specified in the class definition,
        this function solves the model and produces the regression coefficients
        """
        param = self._preprocessing()
        group_index = np.asarray(group_index).astype(int)
        if self.penalization is None:
            self.coef_ = self.unpenalized_solver(x=x, y=y)
        else:
            if self.parallel is False:
                if self.penalization in ['lasso', 'alasso']:
                    self.coef_ = getattr(self, self._get_solver_names())(x=x, y=y, param=param)
                else:
                    self.coef_ = getattr(self, self._get_solver_names())(x=x, y=y, param=param,
                                                                         group_index=group_index)
            else:
                self.coef_ = self._parallel_execution(x=x, y=y, param=param, group_index=group_index)

    # PREDICTION METHOD ###############################################################################################

    def predict(self, x_new):
        """
        To be executed after fitting a model. Given a new dataset, this function produces predictions for that data
        considering the different model coefficients output provided by function fit
        """
        if self.intercept:
            x_new = np.c_[np.ones(x_new.shape[0]), x_new]
        if x_new.shape[1] != len(self.coef_[0]):
            logging.error('Model dimension and new data dimension does not match')
            raise ValueError('Model dimension and new data dimension does not match')
        # Store predictions in a list
        prediction_list = []
        for elt in self.coef_:
            prediction_list.append(np.dot(x_new, elt))
        return prediction_list

    # NUMBER OF PARAMETERS ############################################################################################

    def _num_parameters(self):
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
        if self._input_checker() is False:
            logging.error('incorrect input parameters')
            raise ValueError('incorrect input parameters')
        if self.penalization is None:
            # See meaning of each element in the "else" result statement.
            result = dict(num_models=1,
                          n_lambda=None,
                          n_alpha=None,
                          n_lasso_weights=None,
                          n_gl_weights=None)
        else:
            n_lambda, drop = self._preprocessing_lambda()
            n_alpha, drop = self._preprocessing_alpha()
            n_lasso_weights, drop = self._preprocessing_weights(self.lasso_weights)
            n_gl_weights, drop = self._preprocessing_weights(self.gl_weights)
            list_param = [n_lambda, n_alpha, n_lasso_weights, n_gl_weights]
            list_param_no_none = [elt for elt in list_param if elt is not None]
            num_models = np.prod(list_param_no_none)
            result = dict(num_models=num_models,
                          n_lambda=n_lambda,
                          n_alpha=n_alpha,
                          n_lasso_weights=n_lasso_weights,
                          n_gl_weights=n_gl_weights)
        return result

    def _retrieve_parameters_idx(self, param_index):
        """
        Given an index for the param iterable output from _preprocessing function, this function returns a tupple
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
        number_parameters = self._num_parameters()
        n_models, n_lambda, n_alpha, n_l_weights, n_gl_weights = [number_parameters[elt] for elt in number_parameters]
        if param_index > n_models:
            string = f'param_index should be smaller or equal than the number of models solved. n_models={n_models}, ' \
                     f'param_index={param_index}'
            logging.error(string)
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
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], parameter_idx[1][0], None, None]
        elif self.penalization == 'alasso':
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_l_weights))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], None, parameter_idx[1][0], None]
        elif self.penalization == 'agl':
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_gl_weights))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], None, None, parameter_idx[1][0]]
        else:
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_alpha, n_l_weights, n_gl_weights))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], parameter_idx[1][0],
                      parameter_idx[2][0], parameter_idx[3][0]]
        return result

    def retrieve_parameters_value(self, param_index):
        """
        Converts the index output from retrieve_parameters_idx into the actual numerical value of the parameters.
        Outputs None if the parameter was not used in the model formulation.
        To be executed after fit method.
        """
        param_index = self._retrieve_parameters_idx(param_index)
        result = [param[idx] if idx is not None else None for idx, param in
                  zip(param_index, [self.lambda1, self.alpha, self.lasso_weights, self.gl_weights])]
        result = dict(lambda1=result[0],
                      alpha=result[1],
                      lasso_weights=result[2],
                      gl_weights=result[3])
        return result


# ERROR CALCULATOR METHOD #############################################################################################

def _quantile_function(y_true, y_pred, tau):
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
        QRE=_quantile_function)
    valid_error_types = error_dict.keys()
    # Check that the error_type is a valid error type considered
    if error_type not in valid_error_types:
        raise ValueError(f'invalid error type. Valid error types are {error_dict.keys()}')
    if y_true.shape[0] != len(prediction_list[0]):
        logging.error('Dimension of test data does not match dimension of prediction')
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
