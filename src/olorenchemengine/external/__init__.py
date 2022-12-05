from olorenchemengine.internal import package_available
from .ChemProp.main import ChemPropModel
from .GINNetwork.main import GINModel
from .HondaSTRep.main import HondaSTRep
from .HondaSTRep.operations import WordVocab
from .mol2vec.main import Mol2Vec
from .MolCLR.main import MolCLR, MolCLRVecRep
from .piCalculax import calc_pI
from .SPGNN.main import SPGNN, SPGNNVecRep
from .stoned import *

from .GaussianProcess.main import GaussianProcessModel

if package_available("gpflow"):
    from .GaussianProcess.main import TanimotoGPModel