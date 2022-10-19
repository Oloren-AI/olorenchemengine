""" A library of various molecular representations.
"""

from abc import abstractmethod, abstractproperty
from ctypes.wintypes import COLORREF

import numpy as np
from tqdm import tqdm
from pandas.api.types import is_numeric_dtype
from rdkit import Chem

from typing import List, Union, Any

import olorenchemengine as oce

from .base_class import *
from .dataset import *


def get_all_reps():
    return list(oce.all_subclasses(BaseRepresentation))


class BaseRepresentation(BaseClass):

    """ BaseClass for all molecular representations (PyTorch Geometric graphs, descriptors, fingerprints, etc.)

    Parameters:
        log (boolean): Whether to log the representation or not

    Methods:
        _convert(smiles: str, y: Union[int, float, np.number] = None) -> Any: converts a single structure (represented by a SMILES string) to a representation
        convert(Xs: Union[list, pd.DataFrame, dict, str], ys: Union[list, pd.Series, np.ndarray]=None) -> List[Any]: converts input data to a list of representations
    """

    @log_arguments
    def __init__(self, log=True):
        pass

    @abstractmethod
    def _convert(self, smiles: str, y: Union[int, float, np.number] = None) -> Any:
        """ Converts a single structure (represented by a SMILES string) to a representation

        Parameters:
            smiles (str): SMILES string of the structure
            y (Union[int, float, np.number]): target value of the structure

        Returns:
            Any: representation of the structure
        """
        pass

    def _convert_list(self, smiles_list: List[str], ys: List[Union[int, float, np.number]] = None) -> List[Any]:
        """ Converts a list of structures (represented by a SMILES string) to a list of representations

        Parameters:
            smiles_list (List[str]): list of SMILES strings of the structures
            ys (List[Union[int, float, np.number]]): list of target values of the structures

        Returns:
            List[Any]: list of representations of the structures"""
        if ys is None:
            return [self._convert(s) for s in smiles_list]
            # with mp.Pool(mp.cpu_count()) as p:
            #   X = p.map(self._convert, smiles_list)
            # return X
        else:
            return [self._convert(s, y=y) for s, y in tqdm(zip(smiles_list, ys))]

    def _convert_cache(self, smiles: str, y: Union[int, float, np.number] = None) -> Any:
        """ Converts a single structure (represented by a SMILES string) to a representation

        Parameters:
            smiles (str): SMILES string of the structure
            y (Union[int, float, np.number]): target value of the structure
        Returns:
            Any: representation of the structure
        """
        if smiles in self.cache.keys():
            return self.cache[smiles]
        else:
            return self._convert(smiles, y=y)

    def convert(
        self, Xs: Union[list, pd.DataFrame, dict, str], ys: Union[list, pd.Series, np.ndarray] = None, **kwargs
    ) -> List[Any]:
        """ Converts input data to a list of representations

        Parameters:
            Xs (Union[list, pd.DataFrame, dict, str]): input data
            ys (Union[list, pd.Series, np.ndarray]=None): target values of the input data

        Returns:
            List[Any]: list of representations of the input data
        """
        if isinstance(Xs, list) and (isinstance(Xs[0], list) or isinstance(Xs[0], tuple)):
            smiles = [X[0] for X in Xs]
        elif isinstance(Xs, pd.DataFrame) or isinstance(Xs, dict):
            if isinstance(Xs, pd.DataFrame):
                keys = Xs.columns.tolist()
            elif isinstance(Xs, dict):
                keys = Xs.keys()
            if "smiles" in keys:
                smiles = Xs["smiles"]
            elif "Smiles" in keys:
                smiles = Xs["Smiles"]
            elif "SMILES" in keys:
                smiles = Xs["SMILES"]
            else:
                smiles = Xs.iloc[:, 0].tolist()
        elif isinstance(Xs, str):
            smiles = [Xs]
        else:
            smiles = Xs

        return self._convert_list(smiles, ys=ys)

    def _save(self):
        return {}

    def _load(self, d):
        pass


class SMILESRepresentation(BaseRepresentation):
    """ Extracts the SMILES strings from inputted data

    Methods:
        convert(Xs: Union[list, pd.DataFrame, dict, str], ys: Union[list, pd.Series, np.ndarray]=None) -> List[Any]: converts input data to a list of SMILES strings
            Data types:
                pd.DataFrames will have columns "smiles" or "Smiles" or "SMILES" extracted
                lists and tuples of multiple elements will have their first element extracted
                strings will be converted to a list of one element
                everything else will be returned as inputted
    """

    def _convert(self, smiles, y=None):
        pass

    def convert(self, Xs, ys=None, **kwargs):
        if isinstance(Xs, list) and (isinstance(Xs[0], list) or isinstance(Xs[0], tuple)):
            smiles = [X[0] for X in Xs]
        elif isinstance(Xs, pd.DataFrame) or isinstance(Xs, dict):
            if isinstance(Xs, pd.DataFrame):
                keys = Xs.columns.tolist()
            elif isinstance(Xs, dict):
                keys = Xs.keys()
            if "smiles" in keys:
                smiles = Xs["smiles"]
            elif "Smiles" in keys:
                smiles = Xs["Smiles"]
            elif "SMILES" in keys:
                smiles = Xs["SMILES"]
            elif "smi" in keys:
                smiles = Xs["smi"]
            else:
                smiles = Xs.iloc[:, 0]
        elif isinstance(Xs, str):
            smiles = [Xs]
        elif isinstance(Xs, pd.Series):
            smiles = Xs.tolist()
        else:
            smiles = Xs

        if isinstance(smiles, pd.Series):
            smiles = smiles.tolist()
        return smiles


class AtomFeaturizer(BaseClass):
    """ Abstract class for atom featurizers, which create a vector representation for a single atom.

    Methods:
        length(self) -> int: returns the length of the atom vector representation, to be implemented by subclasses
        convert(self, atom: Chem.Atom) -> np.ndarray: converts a single Chem.Atom string to a vector representation, to be implemented by subclasses
    """

    @abstractproperty
    def length(self) -> int:
        pass

    @abstractmethod
    def convert(self, atom: Chem.Atom) -> np.ndarray:
        pass

    def _save(self) -> dict:
        return {}

    def _load(self, d: dict):
        pass


class BondFeaturizer(BaseClass):
    """ Abstract class for bond featurizers, which create a vector representation for a single bond.

    Methods:
        length(self) -> int: returns the length of the bond vector representation, to be implemented by subclasses
        convert(self, bond: Chem.Bond) -> np.ndarray: converts a single Chem.Bond string to a vector representation, to be implemented by subclasses
    """

    @abstractproperty
    def length(self) -> int:
        pass

    @abstractmethod
    def convert(self, bond: Chem.Bond) -> np.ndarray:
        pass

    def _save(self):
        return {}

    def _load(self, d: dict):
        pass


class ConcatenatedAtomFeaturizers(AtomFeaturizer):
    """ Concatenates multiple atom featurizers into a single vector representation.

    Methods:
        length(self) -> int: returns the length of the atom vector representation, to be implemented by subclasses
        convert(self, atom: Chem.Atom) -> np.ndarray: converts a single Chem.Atom string to a vector representation, to be implemented by subclasses
    """

    def __init__(self, atom_featurizers: List[AtomFeaturizer]):
        self.atom_featurizers = atom_featurizers

    @property
    def length(self) -> int:
        return sum([f.length for f in self.atom_featurizers])

    def convert(self, atom: Chem.Atom) -> np.ndarray:
        return np.concatenate([f.convert(atom) for f in self.atom_featurizers])

    def _save(self):
        return {"atom_featurizers": [f._save() for f in self.atom_featurizers]}

    def _load(self, d: dict):
        for d_, af in zip(d["atom_featurizers"], self.atom_featurizers):
            af._load(d_)


class ConcatenatedBondFeaturizers(BondFeaturizer):
    """ Concatenates multiple bond featurizers into a single vector representation.

    Methods:
        length(self) -> int: returns the length of the bond vector representation, to be implemented by subclasses
        convert(self, bond: Chem.Bond) -> np.ndarray: converts a single Chem.Bond string to a vector representation, to be implemented by subclasses
    """

    def __init__(self, bond_featurizers: List[BondFeaturizer]):
        self.bond_featurizers = bond_featurizers

    @property
    def length(self) -> int:
        return sum([f.length for f in self.bond_featurizers])

    def convert(self, bond: Chem.Bond) -> np.ndarray:
        return np.concatenate([f.convert(bond) for f in self.bond_featurizers])

    def _save(self):
        return {"bond_featurizers": [f._save() for f in self.bond_featurizers]}

    def _load(self, d: dict):
        for d_, bf in zip(d["bond_featurizers"], self.bond_featurizers):
            bf._load(d_)


class OGBAtomFeaturizer(AtomFeaturizer):
    """ Creates a vector representation for a single atom using the Open Graph Benchmark's atom_to_feature_vector function."""

    @log_arguments
    def __init__(self):
        pass

    @property
    def length(self):
        return 9

    def convert(self, atom: Chem.Atom):
        from ogb.utils.features import atom_to_feature_vector

        return atom_to_feature_vector(atom)


class OGBBondFeaturizer(BondFeaturizer):
    """ Creates a vector representation for a single bond using the Open Graph Benchmark's bond_to_feature_vector function."""

    @log_arguments
    def __init__(self):
        pass

    @property
    def length(self):
        return 3

    def convert(self, bond: Chem.Bond):
        from ogb.utils.features import bond_to_feature_vector

        return bond_to_feature_vector(bond)


class TorchGeometricGraph(BaseRepresentation):
    """ Representation which returns torch_geometric.data.Data objects.

    Parameters:
        atom_featurizer (AtomFeaturizer): featurizer for atoms
        bond_featurizer (BondFeaturizer): featurizer for bonds

    Attributes:
        dimensions (Tuple[int, int]): number of dimensions for the atom and bond representations

    Methods:
        _convert(self, smiles: str, y: Any=None) -> Data: converts a single SMILES string to a torch_geometric.data.Data object
    """

    @log_arguments
    def __init__(
        self,
        atom_featurizer: AtomFeaturizer = OGBAtomFeaturizer(),
        bond_featurizer: BondFeaturizer = OGBBondFeaturizer(),
        **kwargs,
    ):

        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer

    @property
    def dimensions(self):
        return (self.atom_featurizer.length, self.bond_featurizer.length)

    def _convert(self, smiles, y=None, addHs=False, **kwargs):
        from torch_geometric.data import Data
        from torch import from_numpy, Tensor

        data = Data()

        mol = Chem.MolFromSmiles(smiles)
        if addHs:
            mol = Chem.AddHs(mol)

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(self.atom_featurizer.convert(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = self.bond_featurizer.convert(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)
        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, self.bond_featurizer.length), dtype=np.int64)

        graph = dict()
        graph["edge_index"] = edge_index
        graph["edge_feat"] = edge_attr
        graph["node_feat"] = x
        graph["num_nodes"] = len(x)

        data.__num_nodes__ = int(graph["num_nodes"])
        data.edge_index = from_numpy(graph["edge_index"])
        data.edge_attr = from_numpy(graph["edge_feat"])
        data.x = from_numpy(graph["node_feat"])
        if not y is None:
            data.y = Tensor([y])

        del graph["num_nodes"]
        del graph["edge_index"]
        del graph["edge_feat"]
        del graph["node_feat"]

        return data

    def convert(
        self, Xs: Union[list, pd.DataFrame, dict, str], ys: Union[list, pd.Series, np.ndarray] = None, **kwargs
    ) -> List[Any]:
        Xs = SMILESRepresentation().convert(Xs)
        return super().convert(Xs, ys = ys, **kwargs)

    def _save(self):
        return {"atom_featurizer": self.atom_featurizer._save(), "bond_featurizer": self.bond_featurizer._save()}

    def _load(self, d):
        self.atom_featurizer._load(d["atom_featurizer"])
        self.bond_featurizer._load(d["bond_featurizer"])


class BaseVecRepresentation(BaseRepresentation):
    """ Representation where given input data, returns a vector representation for each compound."""

    @log_arguments
    def __init__(self, *args, collinear_thresh=1.01, scale=StandardScaler(), names = None, log=True, **kwargs):
        self.collinear_thresh = collinear_thresh
        self.to_drop = None
        if not scale is None:
            scale = scale.copy()
        self.scale = scale
        self.scale_fitted = False
        self._names = names

        from os import path
        if not path.exists(path.join(path.expanduser("~"), f".oce/cache/")):
            os.mkdir(path.join(path.expanduser("~"), f".oce/cache/"))
        if not path.exists(path.join(path.expanduser("~"), f".oce/cache/vecrep/")):
            os.mkdir(path.join(path.expanduser("~"), f".oce/cache/vecrep/"))
        if not path.exists(path.join(path.expanduser("~"), f".oce/cache/vecrep/{self.__class__.__name__}")):
            os.mkdir(path.join(path.expanduser("~"), f".oce/cache/vecrep/{self.__class__.__name__}"))

        super().__init__(*args, log=False, **kwargs)

    @property
    def names(self):
        if not self._names is None:
            return self._names
        else:
            raise ValueError(f"Names not set for representation {self.__class__.__name__}")

    def convert(
        self,
        Xs: Union[list, pd.DataFrame, dict, str],
        ys: Union[list, pd.Series, np.ndarray] = None,
        fit=False,
        **kwargs,
    ) -> List[np.ndarray]:
        """ BaseVecRepresentation's convert returns a list of numpy arrays.

        Args:
            Xs (Union[list, pd.DataFrame, dict, str]): input data
            ys (Union[list, pd.Series, np.ndarray], optional): included for compatibility, unused argument. Defaults to None.

        Returns:
            List[np.ndarray]: list of molecular vector representations
        """
        import joblib

        input_hash = (joblib.hash(Xs) +
            joblib.hash(ys) +
            joblib.hash(self._save()) +
            joblib.hash(oce.parameterize(self)))

        from os import path
        if path.exists(path.join(path.expanduser("~"), f".oce/cache/vecrep/{self.__class__.__name__}/{input_hash}.npy")):
            return np.load(path.join(path.expanduser("~"), f".oce/cache/vecrep/{self.__class__.__name__}/{input_hash}.npy"), allow_pickle = True)

        feats = super().convert(Xs, ys)
        import pandas as pd

        feats = pd.DataFrame.from_records(feats, columns=[f"col{i}" for i in range(len(feats[0]))])
        if fit and len(Xs) > 2:
            # collinear
            corr_matrix = feats.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            self.to_drop = [column for column in upper.columns if any(upper[column] > self.collinear_thresh)]
            feats = feats.drop(columns=self.to_drop)
            # scale
            if not self.scale is None:
                x = self.scale.fit_transform(feats.values)
                feats.values[:] = np.nan_to_num(x.reshape(feats.values.shape))
                self.scale_fitted = True
        else:
            if not self.to_drop is None:
                feats = feats.drop(columns=self.to_drop)
            # scale
            if self.scale_fitted and self.scale is not None:
                x = self.scale.transform(feats.values)
                feats.values[:] = np.nan_to_num(x.reshape(feats.values.shape))

        output = np.nan_to_num(np.array(feats.to_records(index=False).tolist()))
        np.save(path.join(path.expanduser("~"), f".oce/cache/vecrep/{self.__class__.__name__}/{input_hash}.npy"), output, allow_pickle=True)
        return output

    def calculate_distance(self, x1: Union[str, List[str]], x2: Union[str, List[str]],
        metric: str = "cosine", **kwargs) -> np.ndarray:
        """ Calculates the distance between two molecules or list of molecules.
        
        Returns a 2D array of distances between each pair of molecules of shape 
        len(x1) by len(x2).
        
        This uses pairwise_distances from sklearn.metrics to calculate distances 
        between the vector representations of the molecules. Options for distances
        are Valid values for metric are:

            From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, 
                ‘manhattan’]. These metrics support sparse matrix inputs. 
                [‘nan_euclidean’] but it does not yet support sparse matrices.
            From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’,
                ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, 
                ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, 
                ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, 
                ‘yule’].
            
            See the documentation for scipy.spatial.distance for details on these metrics.
        """
        
        from sklearn.metrics import pairwise_distances
        
        if isinstance(x1, str):
            x1 = [x1]
        if isinstance(x2, str):
            x2 = [x2]
        x1 = self.convert(x1)
        x2 = self.convert(x2)
        return pairwise_distances(x1, x2, metric=metric, **kwargs)

    def __add__(self, other):
        """ Adds two representations together

        Parameters:
            other (BaseVecRepresentation): representation to add to the current representation

        Returns:
            BaseVecRepresentation: new representation"""
        return ConcatenatedVecRepresentation(self, other)

    def _save(self):
        d = {}
        d.update({"collinear_thresh": self.collinear_thresh})
        d.update({"to_drop": self.to_drop})
        if not self.scale is None:
            d.update({"scale": self.scale._save()})
            d.update({"scale_fitted": self.scale_fitted})
        return d

    def _load(self, d):
        self.collinear_thresh = d["collinear_thresh"]
        self.to_drop = d["to_drop"]
        if not self.to_drop is None:
            self.to_drop = set(self.to_drop)
        if not self.scale is None and "scale" in d:
            self.scale._load(d["scale"])
            self.scale_fitted = d["scale_fitted"]

class ConcatenatedVecRepresentation(BaseVecRepresentation):
    """ Creates a structure vector representation by concatenating multiple representations.

    Parameters:
        rep1 (BaseVecRepresentation): first representation to concatenate
        rep2 (BaseVecRepresentation): second representation to concatenate
        log (bool): whether to log the representations or not

    Can be created by adding two representations together using + operator.

    Example
    ------------------------------
    import olorenautoml as oam
    combo_rep = oam.MorganVecRepresentation(radius=2, nbits=2048) + oam.Mol2Vec()
    model = oam.RandomForestModel(representation = combo_rep, n_estimators = 1000)

    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(self, rep1: BaseVecRepresentation, rep2: BaseVecRepresentation, log=True, **kwargs):
        self.rep1 = rep1
        self.rep2 = rep2
        try:
            self._names = rep1.names + rep2.names
        except Exception as e:
            self._names = None
        super().__init__(names=self._names, log=False, **kwargs)

    def _convert(self, smiles, y=None, fit = False):
        converted_1 = self.rep1._convert(smiles, y=y, fit = fit)
        converted_2 = self.rep2._convert(smiles, y=y, fit = fit)
        return np.concatenate((converted_1, converted_2))

    def _convert_list(self, smiles_list, ys=None, fit = False):
        converted_1 = self.rep1._convert_list(smiles_list, ys=ys, fit = fit)
        converted_2 = self.rep2._convert_list(smiles_list, ys=ys, fit = fit)
        return np.concatenate((converted_1, converted_2), axis=1)

    def convert(self, smiles_list, ys = None, fit = False, **kwargs):
        converted_1 = self.rep1.convert(smiles_list, ys=ys, fit = fit)
        converted_2 = self.rep2.convert(smiles_list, ys=ys, fit = fit)
        return np.concatenate((converted_1, converted_2), axis=1)

class NoisyVec(BaseVecRepresentation):
    """ Adds noise to a given BaseVecRepresentation

    Parameters:
        rep (BaseVecRepresentation): BaseVecRepresentation to add noise to
        a_std (float): standard deviation of the additive noise. Defaults to 0.1.
        m_std (float): standard deviation of the multiplicative noise. Defaults to 0.1.
        names (List[str]): list of the names of the features in the vector representation, optional.

    Example
    ------------------------------
    import olorenautoml as oam
    model = oam.RandomForestModel(representation = oam.'''BaseCompoundVecRepresentation(Params)''', n_estimators=1000)

    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(self, rep: BaseVecRepresentation, *args, a_std=0.1, m_std=0.1, **kwargs):
        self.a_std = a_std
        self.m_std = m_std
        self.rep = rep
        super().__init__(*args, **kwargs)

    def _convert(self, smiles: str, y=None) -> np.ndarray:
        """ Returns specified vector representation of inputted compound with added noise.

        Args:
            smiles (str): SMILES representation of compound
            y (optional): unused argument for compatibility. Defaults to None.

        Returns:
            np.ndarrary: vector representation of inputted compound with added noise
        """
        x = self.rep._convert(smiles)
        x = (x + np.random.normal(0, self.a_std, x.shape)) * np.random.normal(1, self.m_std, x.shape)
        return x

class DatasetFeatures(BaseVecRepresentation):
    """ Selects features from the input dataset as the vector representation """

    def _convert(self, smiles, y=None):
        pass

    def convert(self, X, **kwargs):
        assert isinstance(X, pd.DataFrame)
        self.numeric_cols = [c for c in X.columns.tolist() if is_numeric_dtype(X[c])]
        assert not len(self.numeric_cols) > 0, "No numeric feature columns found in dataset"
        X = X[self.numeric_cols].to_numpy()
        return X


class BaseCompoundVecRepresentation(BaseVecRepresentation):
    """ Computes a vector representation from each structure.

    Parameters:
        normalize (bool): whether to normalize the vector representation or not
        names (List[str]): list of the names of the features in the vector representation, optional."""

    @log_arguments
    def __init__(self, normalize=False, **kwargs):
        self.normalize = normalize
        super().__init__(**kwargs)

    def __len__(self):
        return len(self._convert("C"))

    @abstractmethod
    def _convert(self, smiles, y=None) -> np.ndarray:
        pass

    def convert(
        self,
        Xs: Union[list, pd.Series, pd.DataFrame, dict, str],
        ys: Union[list, pd.Series, np.ndarray] = None,
        fit=False,
        **kwargs,
    ) -> np.ndarray:
        """ Computes a vector representation from each structure in Xs."""
        feats = np.array(super().convert(Xs, ys, fit=fit))
        out = np.nan_to_num(feats.astype(np.float32))
        return out

    def inverse(self, Xs):
        """ Inverts the vector representation to the original feature values

        Parameters:
            Xs (np.ndarray): vector representation of the structures

        Returns:
            list: list of the original feature values"""

        pass

class ConcatenatedStructVecRepresentation(BaseCompoundVecRepresentation):
    """ Creates a structure vector representation by concatenating multiple representations.

    DEPRECEATED, use ConcatenatedVecRepresentation instead.

    Parameters:
        rep1 (BaseVecRepresentation): first representation to concatenate
        rep2 (BaseVecRepresentation): second representation to concatenate
        log (bool): whether to log the representations or not
    """

    @log_arguments
    def __init__(self, rep1: BaseCompoundVecRepresentation, rep2: BaseCompoundVecRepresentation, log=True, **kwargs):
        self.rep1 = rep1
        self.rep2 = rep2
        try:
            self._names = rep1.names + rep2.names
        except Exception as e:
            self._names = None
        super().__init__(names=self._names, log=False, **kwargs)

    def _convert(self, smiles, y=None):
        converted_1 = self.rep1._convert(smiles, y=y)
        converted_2 = self.rep2._convert(smiles, y=y)
        return np.concatenate((converted_1, converted_2))

    def _convert_list(self, smiles_list, ys=None):
        converted_1 = self.rep1._convert_list(smiles_list, ys=ys)
        converted_2 = self.rep2._convert_list(smiles_list, ys=ys)
        return np.concatenate((converted_1, converted_2), axis=1)


class DescriptastorusDescriptor(BaseCompoundVecRepresentation):
    """ Wrapper for DescriptaStorus descriptors (https://github.com/bp-kelley/descriptastorus)

    Parameters:
        name (str): name of the descriptor. Either "atompaircounts", "morgan3counts",
            "morganchiral3counts","morganfeature3counts","rdkit2d","rdkit2dnormalized",
            "rdkitfpbits"
        log (bool): whether to log the representations or not"""

    available_descriptors = [
        "atompaircounts",
        "morgan3counts",
        "morganchiral3counts",
        "morganfeature3counts",
        "rdkit2d",
        "rdkit2dnormalized",
        "rdkitfpbits",
    ]

    @log_arguments
    def __init__(self, name, *args, log=True, scale=None, **kwargs):
        self.name = name

        from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

        self.generator = MakeGenerator((name,))
        super().__init__(log=False, scale=scale, **kwargs)

    def _convert(self, smiles, y=None):
        results = self.generator.process(smiles)
        processed, features = results[0], np.nan_to_num(results[1:], nan=0)

        if processed is None:
            print("ERROR: %s" % smiles)
        return features

    @classmethod
    def AllInstances(cls):
        return [cls(name) for name in cls.available_descriptors]


from rdkit.Chem import Lipinski


class LipinskiDescriptor(BaseCompoundVecRepresentation):

    """ Wrapper for Lipinski descriptors (https://www.rdkit.org/docs/RDKit_Book.html#Lipinski_Descriptors)

    Parameters:
        log (bool): whether to log the representations or not"""

    @log_arguments
    def __init__(self, log=True, **kwargs):
        super().__init__(
            names=[
                "FractionCSP3",
                "HeavyAtomCount",
                "NHOHCount",
                "NOCount",
                "NumAliphaticCarbocycles",
                "NumAliphaticHeterocycles",
                "NumAliphaticRings",
                "NumAromaticCarbocycles",
                "NumAromaticHeterocycles",
                "NumAromaticRings",
                "NumHAcceptors",
                "NumHDonors",
                "NumHeteroatoms",
                "NumRotatableBonds",
                "NumSaturatedCarbocycles",
                "NumSaturatedHeterocycles",
                "NumSaturatedRings",
                "RingCount",
            ],
            log=False,
            **kwargs,
        )

    def _convert(self, smiles, y=None):
        m = Chem.MolFromSmiles(smiles)
        return [
            Lipinski.FractionCSP3(m),
            Lipinski.HeavyAtomCount(m),
            Lipinski.NHOHCount(m),
            Lipinski.NOCount(m),
            Lipinski.NumAliphaticCarbocycles(m),
            Lipinski.NumAliphaticHeterocycles(m),
            Lipinski.NumAliphaticRings(m),
            Lipinski.NumAromaticCarbocycles(m),
            Lipinski.NumAromaticHeterocycles(m),
            Lipinski.NumAromaticRings(m),
            Lipinski.NumHAcceptors(m),
            Lipinski.NumHDonors(m),
            Lipinski.NumHeteroatoms(m),
            Lipinski.NumRotatableBonds(m),
            Lipinski.NumSaturatedCarbocycles(m),
            Lipinski.NumSaturatedHeterocycles(m),
            Lipinski.NumSaturatedRings(m),
            Lipinski.RingCount(m),
        ]


class FragmentIndicator(BaseCompoundVecRepresentation):
    """ Indicator variables for all fragments in rdkit.Chem.Fragments

    http://rdkit.org/docs/source/rdkit.Chem.Fragments.html
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):
        super().__init__(
            names=[
                "Number of aliphatic carboxylic acids ",
                "Number of aliphatic hydroxyl groups  ",
                "Number of aliphatic hydroxyl groups excluding tert-OH",
                "Number of N functional groups attached to aromatics",
                "Number of Aromatic carboxylic acide  ",
                "Number of aromatic nitrogens         ",
                "Number of aromatic amines            ",
                "Number of aromatic hydroxyl groups   ",
                "Number of carboxylic acids           ",
                "Number of carboxylic acids           ",
                "Number of carbonyl O                 ",
                "Number of carbonyl O, excluding COOH ",
                "Number of thiocarbonyl               ",
                "Number of C(OH)CCN-Ctert-alkyl or C(OH)CCNcyclic",
                "Number of Imines                     ",
                "Number of Tertiary amines            ",
                "Number of Secondary amines           ",
                "Number of Primary amines             ",
                "Number of hydroxylamine groups       ",
                "Number of XCCNR groups               ",
                "Number of tert-alicyclic amines (no heteroatoms, not quinine-like bridged N)",
                "Number of H-pyrrole nitrogens        ",
                "Number of thiol groups               ",
                "Number of aldehydes                  ",
                "Number of alkyl carbamates (subject to hydrolysis)",
                "Number of alkyl halides              ",
                "Number of allylic oxidation sites excluding steroid dienone",
                "Number of amides                     ",
                "Number of amidine groups             ",
                "Number of anilines                   ",
                "Number of aryl methyl sites for hydroxylation",
                "Number of azide groups               ",
                "Number of azo groups                 ",
                "Number of barbiturate groups         ",
                "Number of benzene rings              ",
                "Number of benzodiazepines with no additional fused rings",
                "Bicyclic                             ",
                "Number of diazo groups               ",
                "Number of dihydropyridines           ",
                "Number of epoxide rings              ",
                "Number of esters                     ",
                "Number of ether oxygens (including phenoxy)",
                "Number of furan rings                ",
                "Number of guanidine groups           ",
                "Number of halogens                   ",
                "Number of hydrazine groups           ",
                "Number of hydrazone groups           ",
                "Number of imidazole rings            ",
                "Number of imide groups               ",
                "Number of isocyanates                ",
                "Number of isothiocyanates            ",
                "Number of ketones                    ",
                "Number of ketones excluding diaryl, a,b-unsat. dienones, heteroatom on Calpha",
                "Number of beta lactams               ",
                "Number of cyclic esters (lactones)   ",
                "Number of methoxy groups -OCH3       ",
                "Number of morpholine rings           ",
                "Number of nitriles                   ",
                "Number of nitro groups               ",
                "Number of nitro benzene ring substituents",
                "Number of non-ortho nitro benzene ring substituents",
                "Number of nitroso groups, excluding NO2",
                "Number of oxazole rings              ",
                "Number of oxime groups               ",
                "Number of para-hydroxylation sites   ",
                "Number of phenols                    ",
                "Number of phenolic OH excluding ortho intramolecular Hbond substituents",
                "Number of phosphoric acid groups     ",
                "Number of phosphoric ester groups    ",
                "Number of piperdine rings            ",
                "Number of piperzine rings            ",
                "Number of primary amides             ",
                "Number of primary sulfonamides       ",
                "Number of pyridine rings             ",
                "Number of quarternary nitrogens      ",
                "Number of thioether                  ",
                "Number of sulfonamides               ",
                "Number of sulfone groups             ",
                "Number of terminal acetylenes        ",
                "Number of tetrazole rings            ",
                "Number of thiazole rings             ",
                "Number of thiocyanates               ",
                "Number of thiophene rings            ",
                "Number of unbranched alkanes of at least 4 members (excludes halogenated alkanes)",
                "Number of urea groups                ",
            ],
            log=False,
            **kwargs,
        )

    def _convert(self, smiles, y=None):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)

        from rdkit.Chem import Fragments

        return [
            Fragments.fr_Al_COO(mol),
            Fragments.fr_Al_OH(mol),
            Fragments.fr_Al_OH_noTert(mol),
            Fragments.fr_ArN(mol),
            Fragments.fr_Ar_COO(mol),
            Fragments.fr_Ar_N(mol),
            Fragments.fr_Ar_NH(mol),
            Fragments.fr_Ar_OH(mol),
            Fragments.fr_COO(mol),
            Fragments.fr_COO2(mol),
            Fragments.fr_C_O(mol),
            Fragments.fr_C_O_noCOO(mol),
            Fragments.fr_C_S(mol),
            Fragments.fr_HOCCN(mol),
            Fragments.fr_Imine(mol),
            Fragments.fr_NH0(mol),
            Fragments.fr_NH1(mol),
            Fragments.fr_NH2(mol),
            Fragments.fr_N_O(mol),
            Fragments.fr_Ndealkylation1(mol),
            Fragments.fr_Ndealkylation2(mol),
            Fragments.fr_Nhpyrrole(mol),
            Fragments.fr_SH(mol),
            Fragments.fr_aldehyde(mol),
            Fragments.fr_alkyl_carbamate(mol),
            Fragments.fr_alkyl_halide(mol),
            Fragments.fr_allylic_oxid(mol),
            Fragments.fr_amide(mol),
            Fragments.fr_amidine(mol),
            Fragments.fr_aniline(mol),
            Fragments.fr_aryl_methyl(mol),
            Fragments.fr_azide(mol),
            Fragments.fr_azo(mol),
            Fragments.fr_barbitur(mol),
            Fragments.fr_benzene(mol),
            Fragments.fr_benzodiazepine(mol),
            Fragments.fr_bicyclic(mol),
            Fragments.fr_diazo(mol),
            Fragments.fr_dihydropyridine(mol),
            Fragments.fr_epoxide(mol),
            Fragments.fr_ester(mol),
            Fragments.fr_ether(mol),
            Fragments.fr_furan(mol),
            Fragments.fr_guanido(mol),
            Fragments.fr_halogen(mol),
            Fragments.fr_hdrzine(mol),
            Fragments.fr_hdrzone(mol),
            Fragments.fr_imidazole(mol),
            Fragments.fr_imide(mol),
            Fragments.fr_isocyan(mol),
            Fragments.fr_isothiocyan(mol),
            Fragments.fr_ketone(mol),
            Fragments.fr_ketone_Topliss(mol),
            Fragments.fr_lactam(mol),
            Fragments.fr_lactone(mol),
            Fragments.fr_methoxy(mol),
            Fragments.fr_morpholine(mol),
            Fragments.fr_nitrile(mol),
            Fragments.fr_nitro(mol),
            Fragments.fr_nitro_arom(mol),
            Fragments.fr_nitro_arom_nonortho(mol),
            Fragments.fr_nitroso(mol),
            Fragments.fr_oxazole(mol),
            Fragments.fr_oxime(mol),
            Fragments.fr_para_hydroxylation(mol),
            Fragments.fr_phenol(mol),
            Fragments.fr_phenol_noOrthoHbond(mol),
            Fragments.fr_phos_acid(mol),
            Fragments.fr_phos_ester(mol),
            Fragments.fr_piperdine(mol),
            Fragments.fr_piperzine(mol),
            Fragments.fr_priamide(mol),
            Fragments.fr_prisulfonamd(mol),
            Fragments.fr_pyridine(mol),
            Fragments.fr_quatN(mol),
            Fragments.fr_sulfide(mol),
            Fragments.fr_sulfonamd(mol),
            Fragments.fr_sulfone(mol),
            Fragments.fr_term_acetylene(mol),
            Fragments.fr_tetrazole(mol),
            Fragments.fr_thiazole(mol),
            Fragments.fr_thiocyan(mol),
            Fragments.fr_thiophene(mol),
            Fragments.fr_unbrch_alkane(mol),
            Fragments.fr_urea(mol),
        ]


from rdkit.Chem import Descriptors, rdMolDescriptors
from .external import calc_pI


class PeptideDescriptors1(BaseCompoundVecRepresentation):
    @log_arguments
    def __init__(self, log=True, **kwargs):
        self.calc_pI = calc_pI(ph=7.4)
        super().__init__(names=["NumAtoms", "MW", *self.calc_pI.names, "NumAmideBonds"], log=False, **kwargs)

    def _convert(self, smiles, y=None):
        m = Chem.MolFromSmiles(smiles)
        calc_pI_props = self.calc_pI._convert(smiles).tolist()
        x = np.nan_to_num(
            np.array(
                [m.GetNumAtoms(), Descriptors.MolWt(m), *calc_pI_props, rdMolDescriptors.CalcNumAmideBonds(m),]
            ).astype(float)
        )
        return x


from rdkit.Chem import AllChem


class MorganVecRepresentation(BaseCompoundVecRepresentation):
    @log_arguments
    def __init__(self, radius=2, nbits=1024, log=True, **kwargs):
        self.radius = radius
        self.nbits = nbits
        super().__init__(log=False, **kwargs)

    def _convert(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, self.radius, nBits=self.nbits)

        fp_np = np.array(fp)
        return fp_np

    def info(self, smiles):
        info = {}
        m = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, self.radius, nBits=self.nbits, bitInfo=info)
        return info

class MACCSKeys(BaseCompoundVecRepresentation):
    """Calculate MACCS (Molecular ACCess System) Keys fingerprint.

    Durant, Joseph L., et al. "Reoptimization of MDL keys for use in drug discovery."
    Journal of chemical information and computer sciences 42.6 (2002): 1273-1280."""

    @log_arguments
    def __init__(self):
        super().__init__(log=False)

    def _convert(self, s: str) -> np.ndarray:
        return AllChem.GetMACCSKeysFingerprint(Chem.MolFromSmiles(s))

class PubChemFingerprint(BaseCompoundVecRepresentation):
    """PubChem Fingerprint.
    The PubChem fingerprint is a 881 bit structural key,
    which is used by PubChem for similarity searching.
    References
    ----------
    .. [1] ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.pdf
    .. [2] https://github.com/deepchem/deepchem/blob/master/deepchem/feat/molecule_featurizers/pubchem_fingerprint.py
    Note
    -----
    PubChemPy uses REST API to get the fingerprint, so you need internet access.

    PubChem fingerprint calculations fail for certain compounds (i.e. 'CCCCCCC[NH+](CC)CCCC(O)c1ccc(NS(C)(=O)=O)cc1').
    Certain SMILES strings encountered in datasets are not recognized by PubChem.
    In these cases, the fingerprint is returned with all bits equal to zero.
    """
    @log_arguments
    def __init__(self):
        super().__init__(log=False)

    def _convert(self, s: str) -> np.ndarray:
        oce.import_or_install("pubchempy")

        import pubchempy as pcp
        #Check if retrieval of compound and subsequent descriptor calculation succeed without error
        try:
            pubchem_compound = pcp.get_compounds(s, 'smiles')[0]
            fp = [int(bit) for bit in pubchem_compound.cactvs_fingerprint]
        except:
            fp = [0] * 881
        return np.array(fp)

class MordredDescriptor(BaseCompoundVecRepresentation):

    """ Wrapper for Mordred descriptors (https://github.com/mordred-descriptor/mordred)

    Parameters:
        log (bool): whether to log the representations or not
        descriptor_set (str): name of the descriptor set to use
        normalize (bool): whether to normalize the descriptors or not
    """

    @log_arguments
    def __init__(self, descriptor_set: Union[str, list] = "all", log: bool = True, normalize: bool = False, **kwargs):
        oce.import_or_install("mordred")

        from mordred import Calculator, descriptors

        if descriptor_set == "all":
            self.calc = Calculator(descriptors, ignore_3D=False)

        self.normalize = normalize

        super().__init__(log=False, normalize=self.normalize, **kwargs)

    def _convert(self, smiles, y=None):
        pass

    def convert(self, Xs, ys=None, **kwargs):
        """ Convert list of SMILES to descriptors in the form of a numpy array.

        Parameters:
            Xs (list): List of SMILES strings.
            ys (list): List of labels.
            normalize (bool): Whether to normalize the descriptors.

        Returns:
            np.ndarray: Array of descriptors.
            Shape: (len(Xs), len(self.names))"""

        smiles = SMILESRepresentation().convert(Xs, ys=ys)
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        feats = pd.DataFrame([self.calc(mol).asdict() for mol in mols])
        for col in feats.columns:
            feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(0)
        feats = np.nan_to_num(feats.to_numpy())
        return feats


from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D


class GobbiPharma2D(BaseCompoundVecRepresentation):

    """ 2D Gobbi pharmacophore descriptor (implemented in RDKit, from https://doi.org/10.1002/(SICI)1097-0290(199824)61:1<47::AID-BIT9>3.0.CO;2-Z) """

    def _convert(self, smiles, y=None):
        m = Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(m)  # gen 3d

        factory = Gobbi_Pharm2D.factory
        # calc 2d p4 fp
        fp = Generate.Gen2DFingerprint(m, factory)
        # dMat option is Key!
        return fp


class GobbiPharma3D(BaseCompoundVecRepresentation):

    """ 3D Gobbi pharmacophore descriptor (implemented in RDKit, from https://doi.org/10.1002/(SICI)1097-0290(199824)61:1<47::AID-BIT9>3.0.CO;2-Z) """

    def _convert(self, smiles, y=None):
        m = Chem.MolFromSmiles(smiles)
        s = AllChem.EmbedMolecule(m, useRandomCoords=True)

        factory = Gobbi_Pharm2D.factory
        if s == -1:
            fp = Generate.Gen2DFingerprint(m, factory)
        else:
            fp = Generate.Gen2DFingerprint(m, factory, dMat=Chem.Get3DDistanceMatrix(m))

        return fp


from collections import OrderedDict

try:
    import torch
except ImportError:
    oce.mock_imports(globals(), "torch")

try:
    import torch_geometric.data
    from torch_geometric.data import DataLoader as PyGDataLoader
except:
    oce.mock_imports(globals(), "torch_geometric", "PyGDataLoader")

from rdkit import Chem


class OlorenCheckpoint(BaseCompoundVecRepresentation):
    """ Use OlorenVec from checkpoint as a molecular representation

    Parameters:
        model_path (str): path to checkpoint file for OlorenVec. Use "default" if unsure
        num_tasks (int): number of coordinates in the vector representation
        log (bool, optional): Log arguments or not. Should only be true if it is not nested. Defaults to True.
    """

    @log_arguments
    def __init__(self, model_path: str, num_tasks: int = 2048, log: bool = True, **kwargs):
        if model_path == "default":
            import os
            from os.path import expanduser

            path = os.path.join(expanduser("~"), ".oce/olorenvec.ckpt")

            if not os.path.exists(path):
                print("Downloading OlorenVec checkpoint...")
                import gcsfs

                fs = gcsfs.GCSFileSystem()
                with fs.open("gs://oloren-public-data/saves/olorenvec.ckpt", "rb") as f:
                    with open(path, "wb") as out:
                        out.write(f.read())
        else:
            path = model_path

        state_dict = OrderedDict(
            [
                (k.replace("model.", ""), v)
                for k, v in torch.load(path, map_location=oce.CONFIG["DEVICE"])["state_dict"].items()
            ]
        )

        from olorenchemengine.pyg.gcn import GNN

        self.model = GNN(
            gnn_type="gcn", num_tasks=num_tasks, num_layer=5, emb_dim=300, drop_ratio=0.5, virtual_node=False
        )

        self.model.to(oce.CONFIG["DEVICE"])

        self.model.load_state_dict(state_dict)
        self.model.eval()

        super().__init__(log=False, **kwargs)

    def _convert(self, smiles, y=None):
        try:
            from torch_geometric.data import Batch

            graph = self.smiles2pyg(smiles, y)
            batch = Batch.from_data_list([graph])
            batch.to(oce.CONFIG["DEVICE"])
            x = self.model(batch).detach().cpu().numpy()
            x = np.squeeze(x)
            return x
        except Exception as e:
            print(e)
            print("ERROR ON %s" % smiles)
            return self._convert("C1CCCCC1", y=y)

    def _convert_list(self, smiles_list, ys=None):
        xs = [self.smiles2pyg(s, None) for s in smiles_list]

        kwargs = {"num_workers": oce.CONFIG["NUM_WORKERS"], "pin_memory": True} if oce.CONFIG["USE_CUDA"] else {}

        dataloader = PyGDataLoader(xs, batch_size=64, **kwargs)

        predictions = list()
        for batch in dataloader:
            batch.to(oce.CONFIG["DEVICE"])
            predictions.append(self.model(batch))

        predictions = torch.cat(predictions, dim=0)
        predictions = predictions.detach().cpu().numpy()
        return predictions

    @classmethod
    def AllInstances(cls):
        return [cls("default", num_tasks=2048)]

    def molecule2graph(self, mol, include_mol=False):
        """ Convert a molecule to a PyG graph with features and labels

        Parameters:
            mol (rdkit.Chem.rdchem.Mol): molecule to convert
            include_mol (bool, optional): Whether or not include the molecule in the graph. Defaults to False.

        Returns:
           graph: PyG graph"""

        # Convert to RDKit molecule
        if not isinstance(mol, Chem.Mol):
            mol = Chem.MolFromSmiles(mol)

        from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

        # Generate nodes of the graph
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        # Generate edges of the graph
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        graph = dict()
        graph["edge_index"] = edge_index
        graph["edge_feat"] = edge_attr
        graph["node_feat"] = x
        graph["num_nodes"] = len(x)

        if include_mol:
            graph["mol"] = mol

        return graph

    def smiles2pyg(self, smiles_str, y, morgan_params={"radius": 2, "nBits": 1024}):
        """ Convert a SMILES string to a PyG graph with features and labels

        Parameters:
            smiles_str (str): SMILES string to convert
            y (int): label of the molecule
            morgan_params (dict, optional): parameters for morgan fingerprint. Defaults to {"radius": 2, "nBits": 1024}.

        Returns:
            graph: PyG graph
        """
        graph = self.molecule2graph(smiles_str)  # construct ogb graph
        g = torch_geometric.data.Data()
        g.__num_nodes__ = graph["num_nodes"]
        g.edge_index = torch.from_numpy(graph["edge_index"])

        del graph["num_nodes"]
        del graph["edge_index"]

        if graph["edge_feat"] is not None:
            g.edge_attr = torch.from_numpy(graph["edge_feat"])
            del graph["edge_feat"]

        if graph["node_feat"] is not None:
            g.x = torch.from_numpy(graph["node_feat"])
            del graph["node_feat"]

        if y is None:
            return g

        if type(y) == bool:
            g.y = torch.LongTensor([1 if y else 0])
        else:
            g.y = torch.FloatTensor([y])

        return g


class MCSClusterRep(BaseCompoundVecRepresentation):
    """ Clusters a train set of compounds and then finds the maximum common
    substructure (MCS) within each set. The presence of each cluster's MCS is
    used as a feature
    """

    @log_arguments
    def __init__(
        self,
        dataset: BaseDataset,
        *args,
        eval_set="train",
        timeout: int = 5,
        threshold: float = 0.9,
        cached=False,
        log=True,
        **kwargs,
    ):
        if not self.kwargs["cached"]:
            from rdkit.Chem import AllChem
            from rdkit import DataStructs

            if eval_set == "train":
                eval_set = dataset.data[dataset.data["split"] == "train"]
            elif eval_set == "valid":
                eval_set = dataset.data[dataset.data["split"] == "valid"]
            elif eval_set == "test":
                eval_set = dataset.data[dataset.data["split"] == "test"]
            else:
                eval_set = dataset.data

            fingerprinter = lambda s: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(s), 4)
            eval_set["fps"] = eval_set[dataset.structure_col].apply(fingerprinter)
            corr = np.array(
                [DataStructs.BulkTanimotoSimilarity(fp, eval_set["fps"].tolist()) for fp in eval_set["fps"]]
            )

            from sklearn.cluster import AgglomerativeClustering
            from rdkit.Chem.rdFMCS import FindMCS

            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=0.4, affinity="precomputed", linkage="single"
            ).fit(1 - corr)
            eval_set["cluster_id"] = clustering.labels_

            self.smarts = list()
            for cluster_id in tqdm(eval_set["cluster_id"].unique()):
                data_ = eval_set[eval_set["cluster_id"] == cluster_id]
                if len(data_) < 2:
                    continue
                data_["mols"] = data_[dataset.structure_col].apply(Chem.MolFromSmiles)
                mcs = FindMCS(data_["mols"].tolist(), threshold=threshold, timeout=timeout)
                self.smarts.append(mcs.smartsString)
            self.args = [""]
            self.kwargs["cached"] = True
        super().__init__(*args, **kwargs, log=False)

    def _convert(self, smiles):
        return np.array([Chem.MolFromSmiles(smiles).HasSubstructMatch(Chem.MolFromSmarts(s)) for s in self.smarts])

    def _save(self):
        d = super()._save()
        d["smarts"] = self.smarts
        return d

    def _load(self, d):
        self.smarts = d["smarts"]
        super()._load(d)


class ModelAsRep(BaseCompoundVecRepresentation):
    """ Uses a trained model itself as a representation.

    If we are trying to predict property A, and there is a highly related property B
    that has a lot of data we could train a model on property B and use that model
    with ModelAsRep as a representation for property A.

    Parameters:
        model (BaseModel, str): A trained model to be used as the representation,
            either a BaseModel object or a path to a saved model
        download_public_file (bool, optional): If True, will download the specified
            model from OCE's public warehouse of models. Defaults to False.
        name (str): Name of the property the passed model predicts, which
            is usefully for clear save files/interpretability visualizations.
             Optional.
    """

    @log_arguments
    def __init__(self, model: Union[BaseModel, str], name="ModelAsRep", 
            download_public_file = False, log=True, **kwargs):
        if isinstance(model, str):
            if download_public_file:
                self.model = oce.load(oce.download_public_file(model))
            else:
                self.model = oce.load(model)
        else:
            self.model = model
        super().__init__(log=False, names=[name], **kwargs)

    def _convert(self, smiles, y=None):
        x = self.model.predict([smiles])
        return x

    def _convert_list(self, smiles_list, ys=None):
        x = np.expand_dims(self.model.predict(smiles_list), axis=1)
        return x

    def _save(self):
        return super()._save()

    def _load(self, d):
        super()._load(d)
