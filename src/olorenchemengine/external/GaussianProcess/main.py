# Adapted from github.com/PatWalters/yamc under the MIT license
# which in turn cites:
# A minor refactoring of code from
# https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/examples/gp_regression_on_molecules.ipynb

import numpy as np

import olorenchemengine as oce
from olorenchemengine.base_class import log_arguments, BaseSKLearnModel

from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Kernel, Hyperparameter

class Tanimoto(Kernel):
    def __init__(self, sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5)):
        self.sigma_0 = sigma_0
        self.sigma_0_bounds = sigma_0_bounds

    @property
    def hyperparameter_sigma_0(self):
        return Hyperparameter("sigma_0", "numeric", self.sigma_0_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))

        :param X: N x D array
        :param Y: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if Y is None:
            Y = X

        X = np.sum(X**2, axis=-1)  # Squared L2-norm of X
        Y = np.sum(Y**2, axis=-1)  # Squared L2-norm of X2
        outer_product = np.tensordot(X, Y, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -outer_product + X + Y

        return self.variance * outer_product / denominator
    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y).
        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X).
        """
        return np.einsum("ij,ij->i", X, X) + self.sigma_0**2

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    def __repr__(self):
        return "{0}(sigma_0={1:.3g})".format(self.__class__.__name__, self.sigma_0)

class GaussianProcessModel(BaseSKLearnModel):
    """Gaussian process model
    """

    @log_arguments
    def __init__(
        self,
        representation,
        kernel = None,
        alpha = 1e-10,
        random_state=None,
        **kwargs
    ):
        if kernel == "tanimoto":
            kernel = Tanimoto()
        elif kernel == "RBF":
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
        else:
            kernel = None
       
        regressor = GaussianProcessRegressor(
            kernel = kernel,
            alpha = alpha,
            random_state=random_state
        )
        classifier = GaussianProcessClassifier(
            kernel = kernel,
            alpha = alpha,
            random_state=random_state,
        )
        super().__init__(representation, regressor, classifier, log=False, **kwargs)