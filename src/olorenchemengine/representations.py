""" A library of various molecular representations.
"""

from abc import abstractmethod, abstractproperty
from typing import Any, List, Union

import numpy as np
from pandas.api.types import is_numeric_dtype
from rdkit import Chem
from tqdm import tqdm

import olorenchemengine as oce

from .base_class import *
from .dataset import *

class AtomFeaturizer(BaseClass):
    """Abstract class for atom featurizers, which create a vector representation for a single atom.

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
    """Abstract class for bond featurizers, which create a vector representation for a single bond.

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
    """Concatenates multiple atom featurizers into a single vector representation.

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
    """Concatenates multiple bond featurizers into a single vector representation.

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
    """Creates a vector representation for a single atom using the Open Graph Benchmark's atom_to_feature_vector function."""

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
    """Creates a vector representation for a single bond using the Open Graph Benchmark's bond_to_feature_vector function."""

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
    """Representation which returns torch_geometric.data.Data objects.

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
        from torch import Tensor, from_numpy
        from torch_geometric.data import Data

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
        self,
        Xs: Union[list, pd.DataFrame, dict, str],
        ys: Union[list, pd.Series, np.ndarray] = None,
        **kwargs,
    ) -> List[Any]:
        Xs = SMILESRepresentation().convert(Xs)
        return super().convert(Xs, ys=ys, **kwargs)

    def _save(self):
        return {
            "atom_featurizer": self.atom_featurizer._save(),
            "bond_featurizer": self.bond_featurizer._save(),
        }

    def _load(self, d):
        self.atom_featurizer._load(d["atom_featurizer"])
        self.bond_featurizer._load(d["bond_featurizer"])

class NoisyVec(BaseVecRepresentation):
    """Adds noise to a given BaseVecRepresentation

    Parameters:
        rep (BaseVecRepresentation): BaseVecRepresentation to add noise to
        a_std (float): standard deviation of the additive noise. Defaults to 0.1.
        m_std (float): standard deviation of the multiplicative noise. Defaults to 0.1.
        names (List[str]): list of the names of the features in the vector representation, optional.

    Example
    ------------------------------
    import olorenchemengine as oce
    model = oce.RandomForestModel(representation = oce.'''BaseCompoundVecRepresentation(Params)''', n_estimators=1000)

    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(
        self, rep: BaseVecRepresentation, *args, a_std=0.1, m_std=0.1, **kwargs
    ):
        self.a_std = a_std
        self.m_std = m_std
        self.rep = rep
        super().__init__(*args, **kwargs)

    def _convert(self, smiles: str, y=None) -> np.ndarray:
        """Returns specified vector representation of inputted compound with added noise.

        Args:
            smiles (str): SMILES representation of compound
            y (optional): unused argument for compatibility. Defaults to None.

        Returns:
            np.ndarrary: vector representation of inputted compound with added noise
        """
        x = self.rep._convert(smiles)
        x = (x + np.random.normal(0, self.a_std, x.shape)) * np.random.normal(
            1, self.m_std, x.shape
        )
        return x


class DatasetFeatures(BaseVecRepresentation):
    """Selects features from the input dataset as the vector representation"""

    def _convert(self, smiles, y=None):
        pass

    def convert(self, X, **kwargs):
        assert isinstance(X, pd.DataFrame)
        self.numeric_cols = [c for c in X.columns.tolist() if is_numeric_dtype(X[c])]
        assert (
            not len(self.numeric_cols) > 0
        ), "No numeric feature columns found in dataset"
        X = X[self.numeric_cols].to_numpy()
        return X


class BaseCompoundVecRepresentation(BaseVecRepresentation):
    """Computes a vector representation from each structure.

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
        lambda_convert: Callable = None,
        fit=False,
        **kwargs,
    ) -> np.ndarray:
        """Computes a vector representation from each structure in Xs."""
        feats = np.array(super().convert(Xs, ys, fit=fit, lambda_convert = lambda_convert))
        out = np.nan_to_num(feats.astype(np.float32))
        return out

    def inverse(self, Xs):
        """Inverts the vector representation to the original feature values

        Parameters:
            Xs (np.ndarray): vector representation of the structures

        Returns:
            list: list of the original feature values"""

        pass


class ConcatenatedStructVecRepresentation(BaseCompoundVecRepresentation):
    """Creates a structure vector representation by concatenating multiple representations.

    DEPRECEATED, use ConcatenatedVecRepresentation instead.

    Parameters:
        rep1 (BaseVecRepresentation): first representation to concatenate
        rep2 (BaseVecRepresentation): second representation to concatenate
        log (bool): whether to log the representations or not
    """

    @log_arguments
    def __init__(
        self,
        rep1: BaseCompoundVecRepresentation,
        rep2: BaseCompoundVecRepresentation,
        log=True,
        **kwargs,
    ):
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
    """Wrapper for DescriptaStorus descriptors (https://github.com/bp-kelley/descriptastorus)

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

        try: 
            processed, features = results[0], np.nan_to_num(results[1:], nan=0)
        except Exception as e:
            raise ValueError("%s can not be converted" % smiles)

        if processed is None:
            print("ERROR: %s" % smiles)
        return features

    @classmethod
    def AllInstances(cls):
        return [cls(name) for name in cls.available_descriptors]


from rdkit.Chem import Lipinski


class LipinskiDescriptor(BaseCompoundVecRepresentation):

    """Wrapper for Lipinski descriptors (https://www.rdkit.org/docs/RDKit_Book.html#Lipinski_Descriptors)

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
    """Indicator variables for all fragments in rdkit.Chem.Fragments

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
        super().__init__(
            names=["NumAtoms", "MW", *self.calc_pI.names, "NumAmideBonds"],
            log=False,
            **kwargs,
        )

    def _convert(self, smiles, y=None):
        m = Chem.MolFromSmiles(smiles)
        calc_pI_props = self.calc_pI._convert(smiles).tolist()
        x = np.nan_to_num(
            np.array(
                [
                    m.GetNumAtoms(),
                    Descriptors.MolWt(m),
                    *calc_pI_props,
                    rdMolDescriptors.CalcNumAmideBonds(m),
                ]
            ).astype(float)
        )
        return x


from rdkit.Chem import AllChem


class MorganVecRepresentation(BaseCompoundVecRepresentation):
    @log_arguments
    def __init__(self, radius=2, nbits=1024, scale =None, log=True, **kwargs):
        self.radius = radius
        self.nbits = nbits
        super().__init__(scale=None, log=False, **kwargs)

    def _convert(self, smiles):
        m = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, self.radius, nBits=self.nbits)

        fp_np = np.array(fp)
        return fp_np

    def info(self, smiles):
        info = {}
        m = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            m, self.radius, nBits=self.nbits, bitInfo=info
        )
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
    """PubChem Fingerprint
    
    Implemented as a fingerprint, which runs locally vs by calling the PubChem
    Fingerprint (PCFP) webservice, using RDKIT to calculate the fingerprint.
    
    Specs described in ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt.
    Search patterns from https://bitbucket.org/caodac/pcfp/src/master/src/tripod/fingerprint/PCFP.java.
    -----
    """

    @log_arguments
    def __init__(self):
        super().__init__(log=False)

    def _convert(self, s: str) -> np.ndarray:
        oce.import_or_install("pubchempy")

        import pubchempy as pcp

        # Check if retrieval of compound and subsequent descriptor calculation succeed without error
        try:
            pubchem_compound = pcp.get_compounds(s, "smiles")[0]
            fp = [int(bit) for bit in pubchem_compound.cactvs_fingerprint]
        except:
            fp = [0] * 881
        return np.array(fp)


class MordredDescriptor(BaseCompoundVecRepresentation):

    """Wrapper for Mordred descriptors (https://github.com/mordred-descriptor/mordred)

    Parameters:
        log (bool): whether to log the representations or not
        descriptor_set (str): name of the descriptor set to use
        normalize (bool): whether to normalize the descriptors or not
    """

    @log_arguments
    def __init__(
        self,
        descriptor_set: Union[str, list] = "2d",
        log: bool = True,
        normalize: bool = False,
        **kwargs,
    ):
        oce.import_or_install("mordred")

        from mordred import Calculator, descriptors

        if descriptor_set == "all":
            self.calc = Calculator(descriptors, ignore_3D=False)
        elif descriptor_set == "2d":
            self.calc = Calculator(descriptors, ignore_3D=True)

        self.normalize = normalize

        super().__init__(log=False, normalize=self.normalize, **kwargs)

    def _convert(self, smiles, y=None):
        pass

    def convert_full(self, Xs, ys=None, **kwargs):
        """Convert list of SMILES to descriptors in the form of a numpy array.

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
    
    def convert(self, Xs, ys=None, **kwargs):
        return super().convert(Xs, ys=ys, lambda_convert = self.convert_full,**kwargs)


from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D


class GobbiPharma2D(BaseCompoundVecRepresentation):

    """2D Gobbi pharmacophore descriptor (implemented in RDKit, from https://doi.org/10.1002/(SICI)1097-0290(199824)61:1<47::AID-BIT9>3.0.CO;2-Z)"""

    def _convert(self, smiles, y=None):
        m = Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(m)  # gen 3d

        factory = Gobbi_Pharm2D.factory
        # calc 2d p4 fp
        fp = Generate.Gen2DFingerprint(m, factory)
        # dMat option is Key!
        return fp


class GobbiPharma3D(BaseCompoundVecRepresentation):

    """3D Gobbi pharmacophore descriptor (implemented in RDKit, from https://doi.org/10.1002/(SICI)1097-0290(199824)61:1<47::AID-BIT9>3.0.CO;2-Z)"""

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

import torch
import torch_geometric.data
from rdkit import Chem
from torch_geometric.data import DataLoader as PyGDataLoader


class OlorenCheckpoint(BaseCompoundVecRepresentation):
    """Use OlorenVec from checkpoint as a molecular representation

    Parameters:
        model_path (str): path to checkpoint file for OlorenVec. Use "default" if unsure
        num_tasks (int): number of coordinates in the vector representation
        log (bool, optional): Log arguments or not. Should only be true if it is not nested. Defaults to True.
    """

    @log_arguments
    def __init__(
        self, model_path: str, num_tasks: int = 2048, log: bool = True, **kwargs
    ):
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
                for k, v in torch.load(path, map_location=oce.CONFIG["DEVICE"])[
                    "state_dict"
                ].items()
            ]
        )

        from olorenchemengine.pyg.gcn import GNN

        self.model = GNN(
            gnn_type="gcn",
            num_tasks=num_tasks,
            num_layer=5,
            emb_dim=300,
            drop_ratio=0.5,
            virtual_node=False,
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

        kwargs = (
            {"num_workers": oce.CONFIG["NUM_WORKERS"], "pin_memory": True}
            if oce.CONFIG["USE_CUDA"]
            else {}
        )

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
        """Convert a molecule to a PyG graph with features and labels

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
        """Convert a SMILES string to a PyG graph with features and labels

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
    """Clusters a train set of compounds and then finds the maximum common
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
            from rdkit import DataStructs
            from rdkit.Chem import AllChem

            if eval_set == "train":
                eval_set = dataset.data[dataset.data["split"] == "train"]
            elif eval_set == "valid":
                eval_set = dataset.data[dataset.data["split"] == "valid"]
            elif eval_set == "test":
                eval_set = dataset.data[dataset.data["split"] == "test"]
            else:
                eval_set = dataset.data

            fingerprinter = lambda s: AllChem.GetMorganFingerprint(
                Chem.MolFromSmiles(s), 4
            )
            eval_set["fps"] = eval_set[dataset.structure_col].apply(fingerprinter)
            corr = np.array(
                [
                    DataStructs.BulkTanimotoSimilarity(fp, eval_set["fps"].tolist())
                    for fp in eval_set["fps"]
                ]
            )

            from rdkit.Chem.rdFMCS import FindMCS
            from sklearn.cluster import AgglomerativeClustering

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.4,
                affinity="precomputed",
                linkage="single",
            ).fit(1 - corr)
            eval_set["cluster_id"] = clustering.labels_

            self.smarts = list()
            for cluster_id in tqdm(eval_set["cluster_id"].unique()):
                data_ = eval_set[eval_set["cluster_id"] == cluster_id]
                if len(data_) < 2:
                    continue
                data_["mols"] = data_[dataset.structure_col].apply(Chem.MolFromSmiles)
                mcs = FindMCS(
                    data_["mols"].tolist(), threshold=threshold, timeout=timeout
                )
                self.smarts.append(mcs.smartsString)
            self.args = [""]
            self.kwargs["cached"] = True
        super().__init__(*args, **kwargs, log=False)

    def _convert(self, smiles):
        return np.array(
            [
                Chem.MolFromSmiles(smiles).HasSubstructMatch(Chem.MolFromSmarts(s))
                for s in self.smarts
            ]
        )

    def _save(self):
        d = super()._save()
        d["smarts"] = self.smarts
        return d

    def _load(self, d):
        self.smarts = d["smarts"]
        super()._load(d)


class ModelAsRep(BaseCompoundVecRepresentation):
    """Uses a trained model itself as a representation.

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
    def __init__(
        self,
        model: Union[BaseModel, str],
        name="ModelAsRep",
        download_public_file=False,
        log=True,
        **kwargs,
    ):
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

from rdkit.Chem import rdmolops

def isCarbonOnlyRing(mol, atoms):
    for i in range(len(atoms)):
        if mol.GetAtomWithIdx(atoms[i]).GetAtomicNum() != 6:
            return False
    return True

def isRingSaturated(mol, atoms):
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            b = mol.GetBondBetweenAtoms(atoms[i], atoms[j])
            if b is not None and b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                return False
    return True

def isRingUnsaturated(mol, atoms, all_rings):
    db = 0
    hetero = 0
    nitro = 0
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            b = mol.GetBondBetweenAtoms(atoms[i], atoms[j])
            if b is not None and b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                ring_count_i = [atoms[i] in ring for ring in all_rings]
                ring_count_j = [atoms[j] in ring for ring in all_rings]
                if len(ring_count_i) + len(ring_count_j) < 4:
                    db += 1

        atno = mol.GetAtomWithIdx(atoms[i]).GetAtomicNum()
        if atno != 6 and atno != 1:
            if atno == 7:
                nitro += 1
            hetero += 1

    if ((db == 1 and len(atoms) == 3)
        or (db == 2 and (len(atoms) == 4 
                        or len(atoms) == 5 
                        or (len(atoms) == 6 and hetero == 1)))
        or (db == 3 and len(atoms) == 7)
        or (db > 0 and hetero == 0)):
        return True
    return False

def isAromaticRing(mol, atoms):
    bondsInRing = []
    for i in range(len(atoms)):
        a = mol.GetAtomWithIdx(atoms[i])
        for b in a.GetBonds():
            if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms:
                bondsInRing.append(b)

    for b in bondsInRing:
        if b.GetBondType() not in [Chem.rdchem.BondType.AROMATIC]:
            return False
    return True

def countAnyRing(mol, rings, size):
    count = 0
    for ring in rings:
        if len(ring) == size:
            count += 1
    return count
            
def countAromaticRing(mol, rings):
    count = 0
    for ring in rings:
        if isAromaticRing(mol, ring):
            count += 1
    return count

def countHeteroAromaticRing(mol, rings):
    count = 0
    for ring in rings:
        if isAromaticRing(mol, ring) and not isCarbonOnlyRing(mol, ring):
            count += 1
    return count

def countSaturatedOrAromaticCarbonOnlyRing(mol, rings, size):
    count = 0
    for ring in rings:
        if len(ring) == size and (isAromaticRing(mol, ring) or isRingSaturated(mol, ring)) and isCarbonOnlyRing(mol, ring):
            count += 1
    return count

def countSaturatedOrAromaticNitrogenContainingRing(mol, rings, size):
    count = 0
    for ring in rings:
        if len(ring) == size and (isAromaticRing(mol, ring) or isRingSaturated(mol, ring)):
            for i in range(len(ring)):
                if mol.GetAtomWithIdx(ring[i]).GetAtomicNum() == 7:
                    count += 1
                    break
    return count

def countSaturatedOrAromaticHeteroContainingRing(mol, rings, size):
    count = 0
    for ring in rings:
        if len(ring) == size and (isAromaticRing(mol, ring) or isRingSaturated(mol, ring)):
            for i in range(len(ring)):
                if mol.GetAtomWithIdx(ring[i]).GetAtomicNum() != 6:
                    count += 1
                    break
    return count

def countUnsaturatedCarbonOnlyRing(mol, rings, size):
    count = 0
    for ring in rings:
        if len(ring) == size and isRingUnsaturated(mol, ring, rings):
            count += 1
    return count

def countUnsaturatedNitrogenContainingRing(mol, rings, size):
    count = 0
    for ring in rings:
        if len(ring) == size and isRingUnsaturated(mol, ring, rings):
            for i in range(len(ring)):
                if mol.GetAtomWithIdx(ring[i]).GetAtomicNum() == 7:
                    count += 1
                    break
    return count

def countUnsaturatedHeteroContainingRing(mol, rings, size):
    count = 0
    for ring in rings:
        if len(ring) == size and isRingUnsaturated(mol, ring, rings):
            for i in range(len(ring)):
                if mol.GetAtomWithIdx(ring[i]).GetAtomicNum() != 6:
                    count += 1
                    break
    return count

def countNitrogenInRing(mol, rings, size):
    count = 0
    for ring in rings:
        if len(ring) == size:
            for i in range(len(ring)):
                if mol.GetAtomWithIdx(ring[i]).GetAtomicNum() == 7:
                    count += 1
                    break
    return count

def countHeteroInRing(mol, rings, size):
    count = 0
    for ring in rings:
        if len(ring) == size:
            for i in range(len(ring)):
                if mol.GetAtomWithIdx(ring[i]).GetAtomicNum() != 6:
                    count += 1
                    break
    return count

from itertools import combinations

def get_valid_combinations(sets):
    valid_combinations = []
    # get all possible combinations of merging the sets
    for merge in combinations(sets, 2):
        # check if there are at least two overlapping elements
        overlap = set.intersection(*merge)
        if len(overlap) >= 2:
            # add the combination to the list of valid combinations
            valid_combinations.append(set.union(*merge))
      
    for merge in combinations(sets, 3):
        # check if there are at least two overlapping elements
        overlap = set.intersection(*merge)
        if len(overlap) >= 2:
          # add the combination to the list of valid combinations
          valid_combinations.append(set.union(*merge))
        
    return valid_combinations + sets

class PubChemFingerprint_local(BaseCompoundVecRepresentation):
    """PubChem Fingerprint
    
    Implemented as a fingerprint, which runs locally vs by calling the PubChem
    Fingerprint (PCFP) webservice, using RDKIT to calculate the fingerprint.
    
    On a validation set of 400 compounds from the FDA Orange Book, the PubCheFP_local
    matches the PubChem server-based version on 331/400 compounds, and is within 1 bit
    on 360/400 compounds. There are however 28/400 compounds where it is between 50
    and 100 bits off.
    
    Specs described in ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt.
    Search patterns from https://bitbucket.org/caodac/pcfp/src/master/src/tripod/fingerprint/PCFP.java.
    -----
    """

    @log_arguments
    def __init__(self):
        filepath = oce.download_public_file("pcfp/pcfp-patterns.json")
        with open(filepath, "r") as f:
            self.patterns = json.load(f)["SMARTS"]
        
        self.smarts = [Chem.MolFromSmarts(x, mergeHs=True) for x in self.patterns]
        super().__init__(log=False)

    def _convert(self, s: str) -> np.ndarray:
        m = Chem.MolFromSmiles(s)
        m = Chem.AddHs(m)
        
        rings = [set(x) for x in list(rdmolops.GetSSSR(m))]
        rings = [list(x) for x in get_valid_combinations(rings)]
        
        output = np.zeros(881)
        
        # Section 1
        
        # get element counts up to uranium 
        counts = {i: 0 for i in range(1, 93)}
        for atom in m.GetAtoms():
            counts[atom.GetAtomicNum()] = counts.get(atom.GetAtomicNum(), 0) + 1

        if counts[1] >= 4: output[0] = 1
        if counts[1] >= 8: output[1] = 1
        if counts[1] >= 16: output[2] = 1
        if counts[1] >= 32: output[3] = 1
        if counts[3] >= 1: output[4] = 1
        if counts[3] >= 2: output[5] = 1
        if counts[5] >= 1: output[6] = 1
        if counts[5] >= 2: output[7] = 1
        if counts[5] >= 4: output[8] = 1
        if counts[6] >= 2: output[9] = 1
        if counts[6] >= 4: output[10] = 1
        if counts[6] >= 8: output[11] = 1
        if counts[6] >= 16: output[12] = 1
        if counts[6] >= 32: output[13] = 1
        if counts[7] >= 1: output[14] = 1
        if counts[7] >= 2: output[15] = 1
        if counts[7] >= 4: output[16] = 1
        if counts[7] >= 8: output[17] = 1
        if counts[8] >= 1: output[18] = 1
        if counts[8] >= 2: output[19] = 1
        if counts[8] >= 4: output[20] = 1
        if counts[8] >= 8: output[21] = 1
        if counts[8] >= 16: output[22] = 1
        if counts[9] >= 1: output[23] = 1
        if counts[9] >= 2: output[24] = 1
        if counts[9] >= 4: output[25] = 1
        if counts[11] >= 1: output[26] = 1
        if counts[11] >= 2: output[27] = 1
        if counts[14] >= 1: output[28] = 1
        if counts[14] >= 2: output[29] = 1
        if counts[15] >= 1: output[30] = 1
        if counts[15] >= 2: output[31] = 1
        if counts[15] >= 4: output[32] = 1
        if counts[16] >= 1: output[33] = 1
        if counts[16] >= 2: output[34] = 1
        if counts[16] >= 4: output[35] = 1
        if counts[16] >= 8: output[36] = 1
        if counts[17] >= 1: output[37] = 1
        if counts[17] >= 2: output[38] = 1
        if counts[17] >= 4: output[39] = 1
        if counts[17] >= 8: output[40] = 1
        if counts[19] >= 1: output[41] = 1
        if counts[19] >= 2: output[42] = 1
        if counts[35] >= 1: output[43] = 1
        if counts[35] >= 2: output[44] = 1
        if counts[35] >= 4: output[45] = 1
        if counts[53] >= 1: output[46] = 1
        if counts[53] >= 2: output[47] = 1
        if counts[53] >= 4: output[48] = 1
        if counts[4] >= 1: output[49] = 1
        if counts[12] >= 1: output[50] = 1
        if counts[13] >= 1: output[51] = 1
        if counts[20] >= 1: output[52] = 1
        if counts[21] >= 1: output[53] = 1
        if counts[22] >= 1: output[54] = 1
        if counts[23] >= 1: output[55] = 1
        if counts[24] >= 1: output[56] = 1
        if counts[25] >= 1: output[57] = 1
        if counts[26] >= 1: output[58] = 1
        if counts[27] >= 1: output[59] = 1
        if counts[28] >= 1: output[60] = 1
        if counts[29] >= 1: output[61] = 1
        if counts[30] >= 1: output[62] = 1
        if counts[31] >= 1: output[63] = 1
        if counts[32] >= 1: output[64] = 1
        if counts[33] >= 1: output[65] = 1
        if counts[34] >= 1: output[66] = 1
        if counts[36] >= 1: output[67] = 1
        if counts[37] >= 1: output[68] = 1
        if counts[38] >= 1: output[69] = 1
        if counts[39] >= 1: output[70] = 1
        if counts[40] >= 1: output[71] = 1
        if counts[41] >= 1: output[72] = 1
        if counts[42] >= 1: output[73] = 1
        if counts[44] >= 1: output[74] = 1
        if counts[45] >= 1: output[75] = 1
        if counts[46] >= 1: output[76] = 1
        if counts[47] >= 1: output[77] = 1
        if counts[48] >= 1: output[78] = 1
        if counts[49] >= 1: output[79] = 1
        if counts[50] >= 1: output[80] = 1
        if counts[51] >= 1: output[81] = 1
        if counts[52] >= 1: output[82] = 1
        if counts[54] >= 1: output[83] = 1
        if counts[55] >= 1: output[84] = 1
        if counts[56] >= 1: output[85] = 1
        if counts[71] >= 1: output[86] = 1
        if counts[72] >= 1: output[87] = 1
        if counts[73] >= 1: output[88] = 1
        if counts[74] >= 1: output[89] = 1
        if counts[75] >= 1: output[90] = 1
        if counts[76] >= 1: output[91] = 1
        if counts[77] >= 1: output[92] = 1
        if counts[78] >= 1: output[93] = 1
        if counts[79] >= 1: output[94] = 1
        if counts[80] >= 1: output[95] = 1
        if counts[81] >= 1: output[96] = 1
        if counts[82] >= 1: output[97] = 1
        if counts[83] >= 1: output[98] = 1
        if counts[57] >= 1: output[99] = 1
        if counts[58] >= 1: output[100] = 1
        if counts[59] >= 1: output[101] = 1
        if counts[60] >= 1: output[102] = 1
        if counts[61] >= 1: output[103] = 1
        if counts[62] >= 1: output[104] = 1
        if counts[63] >= 1: output[105] = 1
        if counts[64] >= 1: output[106] = 1
        if counts[65] >= 1: output[107] = 1
        if counts[66] >= 1: output[108] = 1
        if counts[67] >= 1: output[109] = 1
        if counts[68] >= 1: output[110] = 1
        if counts[69] >= 1: output[111] = 1
        if counts[70] >= 1: output[112] = 1
        if counts[43] >= 1: output[113] = 1
        if counts[92] >= 1: output[114] = 1
        
        # Section 2
        
        def block(size, count, i):
            if countAnyRing(m, rings, size) >= count: output[i] = 1
            if countSaturatedOrAromaticCarbonOnlyRing(m, rings, size) >= count: output[i+1] = 1
            if countSaturatedOrAromaticNitrogenContainingRing(m, rings, size) >= count: output[i+2] = 1
            if countSaturatedOrAromaticHeteroContainingRing(m, rings, size) >= count: output[i+3] = 1
            if countUnsaturatedCarbonOnlyRing(m, rings, size) >= count: output[i+4] = 1
            if countUnsaturatedNitrogenContainingRing(m, rings, size) >= count: output[i+5] = 1
            if countUnsaturatedHeteroContainingRing(m, rings, size) >= count: output[i+6] = 1
        
        block(3,1,115)
        block(3,2,122)
        
        block(4,1,129)
        block(4,2,136)
        
        block(5,1,143)
        block(5,2,150)
        block(5,3,157)
        block(5,4,164)
        block(5,5,171)
        
        block(6,1,178)
        block(6,2,185)
        block(6,3,192)
        block(6,4,199)
        block(6,5,206)
        
        block(7,1,213)
        block(7,2,220)
        
        block(8,1,227)
        block(8,2,234)
        
        block(9,1,241)
        
        block(10,1,248)
        
        if countAromaticRing(m, rings) >= 1: output[255] = 1
        if countHeteroAromaticRing(m, rings) >= 1: output[256] = 1
        if countAromaticRing(m, rings) >= 2: output[257] = 1
        if countHeteroAromaticRing(m, rings) >= 2: output[258] = 1
        if countAromaticRing(m, rings) >= 3: output[259] = 1
        if countHeteroAromaticRing(m, rings) >= 3: output[260] = 1
        if countAromaticRing(m, rings) >= 4: output[261] = 1
        if countHeteroAromaticRing(m, rings) >= 4: output[262] = 1
        
        i = 263
        for smarts in self.smarts:
            if m.HasSubstructMatch(smarts, recursionPossible=False, useChirality=True, ): output[i] = 1
            i+=1
        
        return output.astype(int)