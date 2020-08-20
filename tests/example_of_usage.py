import numpy as np
from sklearn.datasets import load_boston
import asgl

# Import test data #
boston = load_boston()
x = boston.data
y = boston.target
group_index = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5])

# Define parameters grid
lambda1 = (10.0 ** np.arange(-3, 1.01, 0.6)).tolist()
alpha = np.arange(0, 1, 0.2).tolist()
power_weight = [0, 0.2, 0.4, 0.6, 0.8, 1]

# Define model parameters
model = 'lm'
penalization = 'asgl'
tau = 0.5

# Define cv class
cross_validation_class = asgl.CV(model=model, penalization=penalization, lambda1=lambda1, alpha=alpha,
                               tau=0.5, parallel=True, weight_technique='pca_pct',
                               lasso_power_weight=power_weight, gl_power_weight=power_weight, variability_pct=0.85,
                               nfolds=5, error_type='QRE', random_state=99)

# Compute error using k-fold cross validation
error = cross_validation_class.cross_validation(x, y, group_index)

# Obtain the mean error across different folds
error = np.mean(error, axis=1)

# Select the minimum error
minimum_error_idx = np.argmin(error)

# Select the parameters index associated to mininum error values
optimal_parameters = cross_validation_class.retrieve_parameters_value(minimum_error_idx)

# Define asgl class using optimal values
asgl_model = asgl.ASGL(model=model, penalization=penalization, tau=tau,
                       intercept=cross_validation_class.intercept,
                       lambda1=optimal_parameters.get('lambda1'), 
                       alpha=optimal_parameters.get('alpha'),
                       lasso_weights=optimal_parameters.get('lasso_weights'),
                       gl_weights=optimal_parameters.get('gl_weights'))

# Split data into train / test
train_idx, test_idx = asgl.train_test_split(nrows=x.shape[0], train_pct=0.7, random_state=1)

# Solve the model
asgl_model.fit(x=x[train_idx, :], y=y[train_idx], group_index=group_index)

# Obtain betas
final_beta_solution = asgl_model.coef_

# Obtain predictions
final_prediction = asgl_model.predict(x_new=x[test_idx, :])

# Obtain final errors
final_error = asgl.error_calculator(y_true=y[test_idx], prediction_list=final_prediction,
                                    error_type=cross_validation_class.error_type, tau=tau)
