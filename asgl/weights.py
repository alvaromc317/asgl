import logging
import sys

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA

from .asgl import ASGL

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WEIGHTS:
    """
    Parameters
    ----------
    model: str, default = 'lm'
        Model to be fit. Currently, accepts:
            - 'lm': linear regression models.
            - 'qr': quantile regression models.
    penalization: str, default = 'asgl'
        Penalization to use. Currently, accepts:
            - None: unpenalized model.
            - 'lasso': lasso model.
            - 'gl': group lasso model.
            - 'sgl': sparse group lasso model.
            - 'alasso': adaptive lasso model.
            - 'agl': adaptive group lasso model.
            - 'asgl': adaptive sparse group lasso model.
    tau: float, defaul=0.5
        quantile level in quantile regression models. Valid values are between 0 and 1. It only has effect if
        ``model='qr'``
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
    """
    def __init__(self, model='lm', penalization='asgl', tau=0.5, weight_technique='pca_pct', weight_tol=1e-4,
                 lasso_power_weight=1, gl_power_weight=1, variability_pct=0.9, lambda1_weights=0.1, spca_alpha=1e-5,
                 spca_ridge_alpha=1e-2):
        self.valid_penalizations = ['alasso', 'agl', 'asgl']
        self.model = model
        self.penalization = penalization
        self.tau = tau
        self.weight_technique = weight_technique
        self.weight_tol = weight_tol
        self.lasso_power_weight = lasso_power_weight
        self.gl_power_weight = gl_power_weight
        self.variability_pct = variability_pct
        self.lambda1_weights = lambda1_weights
        self.spca_alpha = spca_alpha
        self.spca_ridge_alpha = spca_ridge_alpha

    # PREPROCESSING ###################################################################################################

    def _preprocessing(self, power_weight):
        if isinstance(power_weight, (int, float)):
            power_weight = [power_weight]
        return power_weight

    # WEIGHT TECHNIQUES ###############################################################################################

    def _pca_1(self, x, y):
        """
        Computes the adpative weights based on the first principal component
        """
        pca = PCA(n_components=1)
        pca.fit(x)
        tmp_weight = np.abs(pca.components_).flatten()
        return tmp_weight

    def _pca_pct(self, x, y):
        """
        Computes the adpative weights based on principal component analysis
        """
        # If var_pct is equal to one, the algorithm selects just 1 component, not 100% of the variability.
        if self.variability_pct == 1:
            var_pct2 = np.min((x.shape[0], x.shape[1]))
        else:
            var_pct2 = self.variability_pct
        pca = PCA(n_components=var_pct2)
        # t is the matrix of "scores" (the projection of x into the PC subspace)
        # p is the matrix of "loadings" (the PCs, the eigen vectors)
        t = pca.fit_transform(x)
        p = pca.components_.T
        # Solve an unpenalized qr model using the obtained PCs
        unpenalized_model = ASGL(model=self.model, penalization=None, intercept=True, tau=self.tau)
        unpenalized_model.fit(x=t, y=y)
        beta_qr = unpenalized_model.coef_[0][1:]  # Remove intercept
        # Recover an estimation of the beta parameters and use it as weight
        tmp_weight = np.abs(np.dot(p, beta_qr)).flatten()
        return tmp_weight

    def _pls_1(self, x, y):
        """
        Computes the adpative weights based on the first partial least squares component
        """
        # x_loadings_ is the pls equivalent to the PCs
        pls = PLSRegression(n_components=1, scale=False)
        pls.fit(x, y)
        tmp_weight = np.abs(pls.x_rotations_).flatten()
        return tmp_weight

    def _pls_pct(self, x, y):
        """
        Computes the adpative weights based on partial least squares
        """
        total_variance_in_x = np.sum(np.var(x, axis=0))
        pls = PLSRegression(n_components=np.min((x.shape[0], x.shape[1])), scale=False)
        pls.fit(x, y)
        variance_in_pls = np.var(pls.x_scores_, axis=0)
        fractions_of_explained_variance = np.cumsum(variance_in_pls / total_variance_in_x)
        # Update variability_pct
        self.variability_pct = np.min((self.variability_pct, np.max(fractions_of_explained_variance)))
        n_comp = np.argmax(fractions_of_explained_variance >= self.variability_pct) + 1
        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(x, y)
        tmp_weight = np.abs(np.asarray(pls.coef_).flatten())
        return tmp_weight

    def _unpenalized(self, x, y):
        """
        Only for low dimensional frameworks. Computes the adpative weights based on unpenalized regression model
        """
        unpenalized_model = ASGL(model=self.model, penalization=None, intercept=True, tau=self.tau)
        unpenalized_model.fit(x=x, y=y)
        tmp_weight = np.abs(unpenalized_model.coef_[0][1:])  # Remove intercept
        return tmp_weight

    def _sparse_pca(self, x, y):
        """
        Computes the adpative weights based on sparse principal component analysis.
        """
        # Compute sparse pca
        x_center = x - x.mean(axis=0)
        total_variance_in_x = np.sum(np.var(x, axis=0))
        spca = SparsePCA(n_components=np.min((x.shape[0], x.shape[1])), alpha=self.spca_alpha,
                         ridge_alpha=self.spca_ridge_alpha)
        t = spca.fit_transform(x_center)
        p = spca.components_.T
        # Obtain explained variance using spca as explained in the original paper (based on QR decomposition)
        t_spca_qr_decomp = np.linalg.qr(t)
        # QR decomposition of modified PCs
        r_spca = t_spca_qr_decomp[1]
        t_spca_variance = np.diag(r_spca) ** 2 / x.shape[0]
        # compute variance_ratio
        fractions_of_explained_variance = np.cumsum(t_spca_variance / total_variance_in_x)
        # Update variability_pct
        self.variability_pct = np.min((self.variability_pct, np.max(fractions_of_explained_variance)))
        n_comp = np.argmax(fractions_of_explained_variance >= self.variability_pct) + 1
        unpenalized_model = ASGL(model=self.model, penalization=None, intercept=True, tau=self.tau)
        unpenalized_model.fit(x=t[:, 0:n_comp], y=y)
        beta_qr = unpenalized_model.coef_[0][1:]
        # Recover an estimation of the beta parameters and use it as weight
        tmp_weight = np.abs(np.dot(p[:, 0:n_comp], beta_qr)).flatten()
        return tmp_weight

    def _lasso(self, x, y):
        lasso_model = ASGL(model=self.model, penalization='lasso', lambda1=self.lambda1_weights, intercept=True,
                           tau=self.tau)
        lasso_model.fit(x=x, y=y)
        tmp_weight = np.abs(lasso_model.coef_[0][1:])  # Remove intercept
        return tmp_weight

    def _weight_techniques_names(self):
        return '_' + self.weight_technique

    def _lasso_weights_calculation(self, tmp_weight):
        self.lasso_power_weight = self._preprocessing(self.lasso_power_weight)
        lasso_weights = [1 / (tmp_weight ** elt + self.weight_tol) for elt in self.lasso_power_weight]
        return lasso_weights

    def _gl_weights_calculation(self, tmp_weight, group_index):
        self.gl_power_weight = self._preprocessing(self.gl_power_weight)
        unique_index = np.unique(group_index)
        gl_weights = []
        for glpw in self.gl_power_weight:
            tmp_list = [1 / (np.linalg.norm(tmp_weight[np.where(group_index == unique_index[i])[0]], 2) ** glpw +
                             self.weight_tol) for i in range(len(unique_index))]
            tmp_list = np.asarray(tmp_list)
            gl_weights.append(tmp_list)
        return gl_weights

    def fit(self, x, y=None, group_index=None):
        """
        Main function of the module, given the input specified in the class definition, this function computes
        the specified weights.
        """
        tmp_weight = getattr(self, self._weight_techniques_names())(x=x, y=y)
        if self.penalization == 'alasso':
            lasso_weights = self._lasso_weights_calculation(tmp_weight)
            gl_weights = None
        elif self.penalization == 'agl':
            lasso_weights = None
            gl_weights = self._gl_weights_calculation(tmp_weight, group_index)
        elif self.penalization == 'asgl':
            lasso_weights = self._lasso_weights_calculation(tmp_weight)
            gl_weights = self._gl_weights_calculation(tmp_weight, group_index)
        else:
            lasso_weights = None
            gl_weights = None
            logging.error(f'Not a valid penalization for weight calculation. Valid penalizations '
                          f'are {self.valid_penalizations}')
        return lasso_weights, gl_weights
