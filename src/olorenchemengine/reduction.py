import olorenchemengine as oce

from .base_class import *

class PCA(BaseSKLearnReduction):
    
    """Wrapper for sklearn PCA"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.decomposition import PCA

        self.obj = PCA(*args, **kwargs)
    
class FactorAnalysis(BaseSKLearnReduction):
    
    """Wrapper for sklearn FactorAnalysis"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.decomposition import FactorAnalysis

        self.obj = FactorAnalysis(*args, **kwargs)