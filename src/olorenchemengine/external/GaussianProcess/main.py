# Adapted from github.com/PatWalters/yamc under the MIT license
# which in turn states:
# A minor refactoring of code from
# https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/examples/gp_regression_on_molecules.ipynb

from unittest.mock import MagicMock
import numpy as np

import olorenchemengine as oce
from olorenchemengine.internal import *
from olorenchemengine.base_class import BaseEstimator, log_arguments, BaseSKLearnModel, BaseClass
from olorenchemengine.representations import BaseVecRepresentation

from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel, WhiteKernel, PairwiseKernel
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel

class GaussianProcessRegressor(BaseEstimator):

    """Wrapper for sklearn GaussianProcessRegressor"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.gaussian_process import GaussianProcessRegressor

        self.obj = GaussianProcessRegressor(*args, **kwargs)
        
class GaussianProcessClassifier(BaseEstimator):

    """Wrapper for sklearn GaussianProcessClassifier"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.gaussian_process import GaussianProcessClassifier

        self.obj = GaussianProcessClassifier(*args, **kwargs)
class Tanimoto(Kernel):
    r"""Tanimoto kernel.
    
    Adapted from DotProduct kernel in sklearn.gaussian_process.kernels and from
    Ryan-Rhys Griffiths' implementation of the Tanimoto kernel in 
    https://towardsdatascience.com/gaussian-process-regression-on-molecules-in-gpflow-ee6fedab2130.
    """

    def __init__(self, sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5)):
        self.sigma_0 = sigma_0
        self.sigma_0_bounds = sigma_0_bounds

    @property
    def hyperparameter_sigma_0(self):
        return Hyperparameter("sigma_0", "numeric", self.sigma_0_bounds)

    def bew(self, A, B):
        flatres = np.reshape(A, (-1, 1)) + np.reshape(B, (1, -1))
        return np.reshape(flatres, np.concatenate([A.shape, B.shape], 0))
    
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            product = np.tensordot(X, X, axes=[[-1], [-1]])
            K = self.sigma_0**2 * product / (2*np.sum(X**2, axis=-1) - product)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            product = np.tensordot(X, Y, axes=[[-1], [-1]])
            
            K = self.sigma_0**2 * product / (self.bew(np.sum(X**2, axis=-1), np.sum(Y**2, axis=-1)) - product)

        if eval_gradient:
            if not self.hyperparameter_sigma_0.fixed:
                return K, K[..., np.newaxis]*2*self.sigma_0**2
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K

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
        return np.diag(self(X))
    
    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    def __repr__(self):
        return "{0}(sigma_0={1:.3g})".format(self.__class__.__name__, self.sigma_0)
    
class GaussianProcessModel(BaseSKLearnModel):
    """Gaussian process model using SciPy's implementation of Gaussian process models.
    
    Parameters:
        representation (BaseVecRepresentation): Representation to use.
        kernel (str): Kernel to use. Default is None which uses SciPy's default.
        kernel_params (dict): Parameters for the kernel. Default is {}, where no
            kernel parameters are passed and SciPy defaults are used.
        alpha (float): Value of alpha to use. Default is 1e-10.
        random_state (int): Random state to use. Default is None.
    """

    @log_arguments
    def __init__(
        self,
        representation: BaseVecRepresentation,
        kernel: str = None,
        kernel_params: dict = {},
        alpha: float = 1e-10,
        random_state: int = None,
        **kwargs
    ):
        if kernel == "RBF":
            kernel = RBF(**kernel_params)
        elif kernel == "Matern":
            kernel = Matern(**kernel_params)
        elif kernel == "DotProduct":
            kernel = DotProduct(**kernel_params)
        elif kernel == "WhiteKernel":
            kernel = WhiteKernel(**kernel_params)
        elif kernel == "RationalQuadratic":
            kernel = RationalQuadratic(**kernel_params)
        elif kernel == "ExpSineSquared":
            kernel = ExpSineSquared(**kernel_params)
        elif kernel == "PairwiseKernel":
            kernel = PairwiseKernel(**kernel_params)
        elif kernel == "ConstantKernel":
            kernel = ConstantKernel(**kernel_params)
        elif kernel == "Tanimoto":
            kernel = Tanimoto(**kernel_params)
        else:
            kernel = None
       
        regressor = GaussianProcessRegressor(
            kernel = kernel,
            alpha= alpha,
            random_state=random_state
        )
        classifier = GaussianProcessClassifier(
            kernel = kernel,
            random_state=random_state,
        )
        super().__init__(representation, regressor, classifier, log=False, **kwargs)

if package_available("gpflow"):
    from abc import abstractproperty
    import gpflow
    import tensorflow as tf
    import numpy as np
    from gpflow.mean_functions import Constant
    from sklearn.preprocessing import StandardScaler
    from gpflow.utilities import positive
    from gpflow.utilities.ops import broadcasting_elementwise

    import os
    from os import listdir
    import zipfile
    import io

    # A minor refactoring of code from
    # https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/examples/gp_regression_on_molecules.ipynb

    class Tanimoto(gpflow.kernels.Kernel):
        def __init__(self):
            super().__init__()
            # We constrain the value of the kernel variance to be positive when it's being optimised
            self.variance = gpflow.Parameter(1.0, transform=positive())

        def K(self, X, X2=None):
            """
            Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))

            :param X: N x D array
            :param X2: M x D array. If None, compute the N x N kernel matrix for X.
            :return: The kernel matrix of dimension N x M
            """
            if X2 is None:
                X2 = X

            Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
            X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
            outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

            # Analogue of denominator in Tanimoto formula

            denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

            return self.variance * outer_product / denominator

        def K_diag(self, X):
            """
            Compute the diagonal of the N x N kernel matrix of X
            :param X: N x D array
            :return: N x 1 array
            """
            return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
    import tensorflow_probability.python.bijectors as bijectors

    class GPFlow_Parameter(BaseClass):
        """A wrapper for GPFlow parameters that makes it possible to save a load
        gpflow.Parameter classes."""

        @log_arguments
        def __init__(self, log=True):
            pass
        
        def set(self, parameter: gpflow.Parameter):
            self.d = {
                "value": parameter.numpy(),
                "name": parameter.name,
                "transform": parameter.transform.__class__.__name__,
            }
            return self

        def get(self):
            return gpflow.Parameter(self.d["value"], 
                                    name=self.d["name"],
                                    transform=getattr(bijectors, self.d["transform"])())

        def _save(self):
            return self.d

        def _load(self, d: dict):
            self.d = d

    class GPFlowEstimator(BaseEstimator):
        """Base class for GPFlow estimators. This class is not meant to be used directly.
        Use one of the derived classes instead. Derived classes implement the
        abstract method get_GPFlowModel, which returns a GPFlow model consstructed
        from a passed X_train and y_train.
        
        Requires installing GPFlow and TensorFlow.
        """

        @log_arguments
        def __init__(self, maxiter = 100, objective = "training-loss",
                     inducing_points = float('inf'), setting="regression", log=True):
            self.maxiter = maxiter
            self.m = None
            self.objective = objective
            self.setting = setting
            self.inducing_points = inducing_points
            self.inducing_variable = None

        @abstractmethod
        def get_GPFlowModel(self, X_train, y_train):
            pass
        
        def get_objective(self):
            if self.objective == "log_marginal_likelihood":
                if self.setting == "regression":
                    return lambda: -self.m.log_marginal_likelihood()
                else:
                    raise ValueError("log_marginal_likelihood is only implemented for regression")
            else:
                return gpflow.models.training_loss_closure(self.m, (self.X_train.astype(np.float64), np.reshape(self.y_train, (-1,1))))                        
        
        def fit(self, X_train, y_train):
            self.X_train = X_train
            self.y_train = y_train
            
            opt = gpflow.optimizers.Scipy()

            if X_train.shape[0] <= self.inducing_points:
                self.m = self.get_GPFlowModel(X_train, y_train)
                opt.minimize(self.get_objective(), self.m.trainable_variables, 
                            options=dict(maxiter=self.maxiter))
            else:
                rng = np.random.default_rng(1234)
                self.inducing_variable = rng.choice(X_train.astype(np.float64), size=self.inducing_points, replace=False)
                self.m = self.get_GPFlowModel(X_train, y_train, inducing_variable = self.inducing_variable.astype(np.float64))
                opt.minimize(self.get_objective(), self.m.trainable_variables, 
                            options=dict(maxiter=self.maxiter))
            tf.keras.backend.clear_session()

        def predict(self, X_test):
            y_pred, y_var = self.m.predict_f(X_test.astype(np.float64))
            return y_pred.numpy().flatten()

        def _save(self):
            parameter_dict_org = gpflow.utilities.parameter_dict(self.m)
            parameter_dict_new = {k: GPFlow_Parameter().set(v) for k, v in parameter_dict_org.items()}
            d = {"parameter_dict": {k: v._save() for k, v in parameter_dict_new.items()},
                 "x_train": self.X_train.tolist(),
                 "y_train": self.y_train.tolist(),
                 "inducing_variable": (self.inducing_variable.tolist() if self.inducing_variable is not None else None),
                 "inducing_points": self.inducing_points,}

            return d

        def _load(self, d):
            self.X_train = np.array(d["x_train"])
            self.y_train = np.array(d["y_train"])
            self.inducing_points = d["inducing_points"]
            self.inducing_variable = (np.array(d["inducing_variable"]) if d["inducing_variable"] is not None else None)
            self.m = self.get_GPFlowModel(self.X_train, self.y_train, inducing_variable = self.inducing_variable)
            parameter_dict_new = {k: GPFlow_Parameter() for k, v in d["parameter_dict"].items()}
            for k, v in d["parameter_dict"].items():
                parameter_dict_new[k]._load(v)
            parameter_dict_new = {k: v.get() for k, v in parameter_dict_new.items()}
            gpflow.utilities.multiple_assign(self.m, parameter_dict_new)

    class TanimotoGPRegressor(GPFlowEstimator):
        """
        Estimator for a GP regressor using the Tanimoto kernel.
        
        Adapted from Pat Walters github.com/PatWalters/yamc under the MIT license which in turn states:
        A minor refactoring of code from Ryan Rhys-Griffiths
        https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/examples/gp_regression_on_molecules.ipynb.
        """
        def get_GPFlowModel(self, X_train, y_train, inducing_variable = None):
            k = Tanimoto()
            if inducing_variable is None:
                m = gpflow.models.GPR(data=(X_train.astype(np.float64), np.reshape(y_train, (-1,1))),
                                    mean_function=Constant(np.mean(y_train)),
                                    kernel=k,
                                    noise_variance=1)
            else:
                m = gpflow.models.SGPR(data=(X_train.astype(np.float64), np.reshape(y_train, (-1,1))),
                                    mean_function=Constant(np.mean(y_train)),
                                    kernel = k,
                                    noise_variance = 1,
                                    inducing_variable = inducing_variable)
            return m

    class TanimotoGPClassifier(GPFlowEstimator):
        """
        Estimator for a GP classifier using the Tanimoto kernel.
        Requires installing GPFlow and TensorFlow.
        
        Adapted from Pat Walters github.com/PatWalters/yamc under the MIT license which in turn states:
        A minor refactoring of code from Ryan Rhys-Griffiths
        https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/examples/gp_regression_on_molecules.ipynb.
        """

        def get_GPFlowModel(self, X_train, y_train, inducing_variable = None):
            k = Tanimoto()
            if inducing_variable is None:
                m = gpflow.models.VGP(data=(X_train.astype(np.float64), np.reshape(y_train, (-1,1))),
                                   mean_function=Constant(np.mean(y_train)),
                                   kernel=k,
                                   likelihood=gpflow.likelihoods.Bernoulli())
            else:
                m = gpflow.models.SVGP(mean_function=Constant(np.mean(y_train)),
                                   kernel=k,
                                   likelihood=gpflow.likelihoods.Bernoulli(),
                                   inducing_variable=inducing_variable)
            return m

    class TanimotoGPModel(BaseSKLearnModel):
        """
        Gaussian Process model using the Tanimoto kernel implemented with GPFlow. 
        Requires installing GPFlow and TensorFlow.
        
        Adapted from Pat Walters github.com/PatWalters/yamc under the MIT license which in turn states:
        A minor refactoring of code from Ryan Rhys-Griffiths
        https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/examples/gp_regression_on_molecules.ipynb.
        
        Parameters:
            representation (BaseVecRepresentation): The representation to use."""

        @log_arguments
        def __init__(self, representation: BaseVecRepresentation, inducing_points = float('inf'),
                     objective = "training-loss",log=True, **kwargs):
            classifier = TanimotoGPClassifier(objective=objective, inducing_points = inducing_points, setting="classification")
            regressor = TanimotoGPRegressor(objective=objective, inducing_points = inducing_points, setting="regression")
            super().__init__(representation, regressor, classifier, log=False, **kwargs)
