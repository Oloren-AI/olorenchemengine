# Adapted from github.com/PatWalters/yamc under the MIT license
# which in turn cites:
# A minor refactoring of code from
# https://github.com/Ryan-Rhys/The-Photoswitch-Dataset/blob/master/examples/gp_regression_on_molecules.ipynb

import numpy as np

import olorenchemengine as oce
from olorenchemengine.base_class import BaseEstimator, log_arguments, BaseSKLearnModel

from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel, WhiteKernel, PairwiseKernel
    
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

class GaussianProcessModel(BaseSKLearnModel):
    """Gaussian process model
    """

    @log_arguments
    def __init__(
        self,
        representation,
        kernel = None,
        kernel_params = {},
        alpha = 1e-10,
        random_state=None,
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
        else:
            kernel = None
       
        regressor = GaussianProcessRegressor(
            kernel = kernel,
            random_state=random_state
        )
        classifier = GaussianProcessClassifier(
            kernel = kernel,
            random_state=random_state,
        )
        super().__init__(representation, regressor, classifier, log=False, **kwargs)