import copy
import io

from abc import abstractmethod
from tqdm import tqdm

import numpy as np

from .base_class import *
from .dataset import *
from .representations import *
from .external.stoned import *

import PIL

import selfies as sf

from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity, TanimotoSimilarity
from rdkit.Chem import AllChem

class PerturbationEngine(BaseClass):

    """PerturbationEngine is the base class for techniques which mutate or perturb a compound
    into a similar one with a small difference.

    Methods:
        get_compound_at_idx:
        get_compound: returns a compound with a randomly chosen modification
        get_compound_list: returns a list of compounds with modifications, the list
            is meant to be comprehensive of the result of the application of an
            entire class of modifications"""
    @abstractmethod
    def get_compound_at_idx(self, smiles, idx):
        pass

    @abstractmethod
    def get_compound(self, smiles, n=1, **kwargs) -> str:
        pass

    @abstractmethod
    def get_compound_list(self, smiles, **kwargs) -> list:
        pass

class SwapMutations(PerturbationEngine):
    """SwapMutations replaces substructures with radius r with another substructure
    with radius < r. The substructure is chosen such that it has the same outgoing
    bonds, and this set of substructures is identified through a comprehensive
    ennumeration of a large set of lead-like compounds.

    Methods:
        get_compound: returns a compound with a randomly chosen modification
        get_compound_list: returns a list of compounds with modifications, the list
            is meant to be comprehensive of the result of the application of an
            entire class of modifications"""

    @log_arguments
    def __init__(self, radius = 0, log = True):
        self.radius = radius

        if not radius in (0, 1, 2):
            raise ValueError("radius must be 0, 1, or 2")

        transformation_path = download_public_file(f"swap-mutations/trans_{radius}.json")

        with open(transformation_path, "r") as f:
            self.trans = json.load(f)

    def get_substitution(self, m, idx, r = 1):

        # finding atoms within radius r of atom idx
        substructure_atoms = set({idx})
        for i in range(r):
            bonds = [b for a_ in substructure_atoms for b in m.GetAtomWithIdx(a_).GetBonds()]
            substructure_atoms_ = {b.GetBeginAtomIdx() for b in bonds}.union({b.GetEndAtomIdx() for b in bonds})
            substructure_atoms = substructure_atoms.union(substructure_atoms_)

        # finding atoms connecting to substructure
        surface_atoms = {b.GetEndAtomIdx() for a_ in substructure_atoms for b in m.GetAtomWithIdx(a_).GetBonds() if not b.GetEndAtomIdx() in substructure_atoms}
        surface_atoms.update({b.GetBeginAtomIdx() for a_ in substructure_atoms for b in m.GetAtomWithIdx(a_).GetBonds() if not b.GetBeginAtomIdx() in substructure_atoms})
        surface_atoms = list(surface_atoms)
        surface_l = list()
        for surface_atom in surface_atoms:
            bonds = [b.GetIdx() for b in m.GetAtomWithIdx(surface_atom).GetBonds() if b.GetEndAtomIdx() in substructure_atoms or b.GetBeginAtomIdx() in substructure_atoms]
            surface_l.append((m.GetAtomWithIdx(surface_atom).GetAtomicNum(),
                              [m.GetBondWithIdx(b).GetBondTypeAsDouble() for b in bonds]))

        tmp = [(x1, x2, x3) for x1, x2, x3 in zip([sum(surface_b) for surface_a, surface_b in surface_l], surface_atoms, surface_l)]
        tmp.sort(key = lambda x: x[0])
        surface_l = [x for _, _, x in tmp]
        surface_atoms = [x for _, x, _ in tmp]

        return surface_l, surface_atoms, substructure_atoms

    def get_entry(self, m, idx, r = 1):
        m = copy.deepcopy(m)

        # finding atoms within radius r of atom idx
        substructure_atoms = set({idx})
        for i in range(r):
            bonds = [b for a_ in substructure_atoms for b in m.GetAtomWithIdx(a_).GetBonds()]
            substructure_atoms_ = {b.GetBeginAtomIdx() for b in bonds}.union({b.GetEndAtomIdx() for b in bonds})
            substructure_atoms = substructure_atoms.union(substructure_atoms_)

        # finding atoms connecting to substructure
        surface_atoms = {b.GetEndAtomIdx() for a_ in substructure_atoms for b in m.GetAtomWithIdx(a_).GetBonds() if not b.GetEndAtomIdx() in substructure_atoms}
        surface_atoms.update({b.GetBeginAtomIdx() for a_ in substructure_atoms for b in m.GetAtomWithIdx(a_).GetBonds() if not b.GetBeginAtomIdx() in substructure_atoms})
        surface_atoms = list(surface_atoms)
        surface_l = list()
        for surface_atom in surface_atoms:
            bonds = [b.GetIdx() for b in m.GetAtomWithIdx(surface_atom).GetBonds() if b.GetEndAtomIdx() in substructure_atoms or b.GetBeginAtomIdx() in substructure_atoms]
            surface_l.append((m.GetAtomWithIdx(surface_atom).GetAtomicNum(),
                              [m.GetBondWithIdx(b).GetBondTypeAsDouble() for b in bonds]))

        tmp = [(x1, x2, x3) for x1, x2, x3 in zip([sum(surface_b) for surface_a, surface_b in surface_l], surface_atoms, surface_l)]
        tmp.sort(key = lambda x: x[0])
        surface_l = [x for _, _, x in tmp]
        surface_atoms = [x for _, x, _ in tmp]

        for i, j in enumerate(surface_atoms):
            m.GetAtomWithIdx(j).SetAtomMapNum(i+1)

        substruct = Chem.RWMol(m)

        removal = list()
        for i in range(substruct.GetNumAtoms()):
            if i not in substructure_atoms and i not in surface_atoms:
                removal.append(substruct.GetAtomWithIdx(i))

        for atom in removal:
            substruct.RemoveAtom(atom.GetIdx())
        return surface_l, substruct.GetMol()

    def stitch(self, m):
        m = Chem.RWMol(m)
        mappings = {}
        for a in m.GetAtoms():
            if a.GetAtomMapNum() != 0:
                if not a.GetAtomMapNum() in mappings:
                    mappings[a.GetAtomMapNum()] = list()
                mappings[a.GetAtomMapNum()].append(a)
        removal = []
        for k, v in mappings.items():
            i = m.AddAtom(Chem.Atom(v[0].GetAtomicNum()))
            a = m.GetAtomWithIdx(i)

            added = []
            for v_ in v:
                for b in v_.GetBonds():
                    if v_.GetIdx() == b.GetEndAtomIdx():
                        i = b.GetBeginAtomIdx()
                    else:
                        i = b.GetEndAtomIdx()
                    if not i in added:
                        m.AddBond(a.GetIdx(), i, b.GetBondType())
                        added.append(i)

                removal.append(v_)


        for remove in removal:
            m.RemoveAtom(remove.GetIdx())
        return m

    def get_compound_at_idx(self, smiles, idx, **kwargs):
        ref_m = Chem.MolFromSmiles(smiles)

        l, a, sub_a = self.get_substitution(ref_m, idx, r = self.radius)

        # if the current substructure to replace has not replacements in the database
        # retry getting compound on failure
        if str(l) not in self.trans:
            return self.get_compound(ref_m, **kwargs)

        for x in sub_a:
            ref_m.GetAtomWithIdx(x).SetAtomMapNum(10000)

        for k, j in enumerate(a):
            ref_m.GetAtomWithIdx(j).SetAtomMapNum(k+1)

        surface_atoms = [ref_m.GetAtomWithIdx(j) for j in a]

        sub = np.random.choice(self.trans[str(l)])
        m = Chem.CombineMols(ref_m, Chem.MolFromSmiles(sub, sanitize=False))

        m = Chem.RWMol(m)

        removal = []
        for i in range(len(m.GetAtoms())):
            if m.GetAtomWithIdx(i).GetAtomMapNum() == 10000:
                removal.append(m.GetAtomWithIdx(i))

        for remove in removal:
            m.RemoveAtom(remove.GetIdx())
        if m is None:
            errors+=1
        try:
            s1 = Chem.MolToSmiles(m)
            m = self.stitch(m)
            if not m is None:
                return Chem.MolToSmiles(m)
        except Exception as e:
            pass

        # retry getting compound on failure
        return self.get_compound(smiles, **kwargs)

    def get_compound(self, smiles, **kwargs):
        if isinstance(smiles, Chem.Mol):
            mol = smiles
            smiles = Chem.MolToSmiles(mol)
        else:
            mol = Chem.MolFromSmiles(smiles)
        idx = np.random.choice(mol.GetNumAtoms(), replace = False)
        return self.get_compound_at_idx(smiles, idx, **kwargs)

    def get_compound_list(self, smiles, **kwargs) -> list:
        outs = []
        ref_m = Chem.MolFromSmiles(smiles)
        for i in tqdm(range(len(ref_m.GetAtoms()))):
            ref_m = Chem.MolFromSmiles(smiles)
            l, a, sub_a = self.get_substitution(ref_m, i, r= self.radius)

            for x in sub_a:
                ref_m.GetAtomWithIdx(x).SetAtomMapNum(10000)

            for k, j in enumerate(a):
                ref_m.GetAtomWithIdx(j).SetAtomMapNum(k+1)

            surface_atoms = [ref_m.GetAtomWithIdx(j) for j in a]
            if str(l) not in self.trans.keys():
                print("Continuing")
                continue
            for sub in self.trans[str(l)]:
                m = Chem.CombineMols(ref_m, Chem.MolFromSmiles(sub, sanitize=False))

                m = Chem.RWMol(m)

                removal = []
                for i in range(len(m.GetAtoms())):
                    if m.GetAtomWithIdx(i).GetAtomMapNum() == 10000:
                        removal.append(m.GetAtomWithIdx(i))

                for remove in removal:
                    m.RemoveAtom(remove.GetIdx())
                try:
                    s1 = Chem.MolToSmiles(m)
                    m = self.stitch(m)
                    if not m is None:
                        outs.append(Chem.MolToSmiles(m))
                except Exception as e:
                    pass
        return outs

    def _save(self) -> dict:
        return super()._save()

    def _load(self, d: dict):
        return super()._load(d)

class STONEDMutations(PerturbationEngine):
    """Implements STONED-SELFIES algorithm for generating modified compounds.

    `STONED-SELFIES GitHub repository <https://github.com/aspuru-guzik-group/stoned-selfies>`_
    `Beyond Generative Models: Superfast Traversal, Optimization, Novelty, Exploration and Discovery
    (STONED) Algorithm for Molecules using SELFIES <https://doi.org/10.26434/chemrxiv.13383266.v2>`_

    Methods:
        get_compound_at_idx: returns a compound with a randomly chosen modification at `idx`
        get_compound: returns a compound with a randomly chosen modification
        get_compound_list: returns a list of `num_samples` compounds with randomly chosen modifications
    """

    @log_arguments
    def __init__(self, mutations: int = 1, log = True):
        self.mutations = mutations

    def get_compound_at_idx(self, smiles: str, idx: int, **kwargs) -> str:
        selfie = sf.encoder(smiles)
        selfie_chars = get_selfie_chars(selfie)
        max_molecules_len = len(selfie_chars) + self.mutations
        selfie_mutated, _ = mutate_selfie(selfie, max_molecules_len, index = idx)
        smiles_back = sf.decoder(selfie_mutated)
        return smiles_back

    def get_compound(self, smiles: str, **kwargs) -> str:
        ref_m = Chem.MolFromSmiles(smiles)
        randomized_smiles_ordering = randomize_smiles(ref_m)
        selfie = sf.encoder(randomized_smiles_ordering)
        selfie_mutated = get_mutated_SELFIES([selfie], num_mutations = self.mutations)[0]
        smiles_back = sf.decoder(selfie_mutated)
        return smiles_back

    def get_compound_list(self, smiles: str, num_samples: int = 1000, **kwargs) -> list:
        ref_m = Chem.MolFromSmiles(smiles)
        randomized_smile_orderings = [randomize_smiles(ref_m) for _ in range(num_samples)]
        selfies_ls = [sf.encoder(x) for x in randomized_smile_orderings]
        selfies_mut = get_mutated_SELFIES(selfies_ls.copy(), num_mutations = self.mutations)
        smiles_back = [sf.decoder(x) for x in selfies_mut]
        return smiles_back

    def _save(self) -> dict:
        return super()._save()

    def _load(self, d: dict):
        return super()._load(d)

class CounterfactualEngine(BaseClass):
    """Generates counterfactual compounds based on:

    `exmol GitHub repository <https://github.com/ur-whitelab/exmol>`_
    `Model agnostic generation of counterfactual explanations for molecules <http://dx.doi.org/10.1039/D1SC05259D>`_
    """

    @log_arguments
    def __init__(self, model: BaseModel, perturbation_engine: PerturbationEngine = "default"):
        self.model = model
        if perturbation_engine == "default":
            self.perturbation_engine = SwapMutations(radius=1)
        else:
            self.perturbation_engine = perturbation_engine
        self.samples = []
        self.cfs = []

    def generate_samples(self, smiles: str) -> None:
        """Generates candidate counterfactuals and stores them in `self.samples` as a list of dictionaries.

        Args:
            smiles: SMILES string of the target prediction
        """
        perturbed_smiles = [smiles] + self.perturbation_engine.get_compound_list(smiles)
        filtered_smiles = []
        seen = set()
        for smi in tqdm(perturbed_smiles):
            mol, smi_canon, conversion_successful = sanitize_smiles(smi)
            if conversion_successful and smi_canon not in seen:
                filtered_smiles.append(smi_canon)
                seen.add(smi_canon)

        ref_m = Chem.MolFromSmiles(smiles)
        filtered_mols = [Chem.MolFromSmiles(s) for s in filtered_smiles]

        ref_fp = AllChem.GetMorganFingerprint(ref_m, 2)
        fps = [AllChem.GetMorganFingerprint(m, 2) for m in filtered_mols]
        scores = BulkTanimotoSimilarity(ref_fp, fps)

        values = self.model.predict(filtered_smiles)

        dmat = 1 - np.array([BulkTanimotoSimilarity(fp, fps) for fp in tqdm(fps)])

        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
        coords = PCA(n_components = 2).fit_transform(dmat)
        clustering = DBSCAN(eps=0.15, min_samples=5).fit(coords)

        samples = {
            'SMILES': filtered_smiles,
            'Similarity': scores,
            'Value': values,
            'Coordinate': list(coords),
            'Cluster': clustering.labels_,
            'Original': [True] + [False for _ in filtered_smiles[1:]]
        }
        self.samples = pd.DataFrame(samples).to_dict('records')

    def get_samples(self) -> pd.DataFrame:
        """Returns candidate counterfactuals as a pandas dataframe.

        Returns:
            pandas dataframe of candidate counterfactuals
        """
        return pd.DataFrame(self.samples)

    def generate_cfs(self, delta: Union[int, float, Tuple] = (-1, 1), n: int = 4) -> None:
        """Generates counterfactuals and stores them in `self.cfs` as a list of dictionaries.

        Args:
            delta: margin defining counterfactuals for regression models
            n: number of counterfactuals
        """
        base = self.samples[0]['Value']

        if self.model.setting == "classification":
            cfs = self._select_cfs(lambda s: s['Value'] != base, n)
            self.cfs = self.samples[:1] + cfs
        else:
            if type(delta) in (float, int):
                delta = (-delta, delta)
            lcfs = self._select_cfs(lambda s: s['Value'] - delta[0] < base, n // 2)
            hcfs = self._select_cfs(lambda s: s['Value'] - delta[1] > base, n // 2)
            self.cfs = self.samples[:1] + lcfs + hcfs

    def get_cfs(self) -> pd.DataFrame:
        """Returns counterfactuals as a pandas dataframe.

        Returns:
            pandas dataframe of counterfactuals
        """
        return pd.DataFrame(self.cfs)

    def _select_cfs(self, fn: Callable, n: int) -> list:
        """Selects counterfactuals from `self.samples` using a validity function.

        Args:
            fn: boolean function for identifying counterfactuals
            n: maximum number of counterfactuals
        Returns:
            cfs: list of dictionary of counterfactuals
        """
        cfs = []

        def cluster_score(s, i):
            """Score for clustering for a given model.

            Parameters:
                s: sample
                i: index of sample

            Returns:
                score: score for clustering
            """
            return (s['Cluster'] == i) * fn(s) * s['Similarity']

        clusters = {s['Cluster'] for s in self.samples[1:]}
        for i in clusters:
            candidate = max(self.samples[1:], key=lambda s: cluster_score(s, i))
            if cluster_score(candidate, i):
                cfs.append(candidate)

        cfs = sorted(cfs, key=lambda s: s['Similarity'], reverse=True)[:n]
        return cfs

    def _save(self) -> dict:
        return super()._save()

    def _load(self, d: dict):
        return super()._load(d)

def model_molecule_sensitivity(model: BaseModel, smiles: str, perturbation_engine: PerturbationEngine = "default", n: int = 30) -> Chem.Mol:

    """
    Calculates the sensitivity of a model to perturbations in on each of a molecule's atoms,
    outputting a rdkit molecule, with sensitivity as an atom property.

    Parameters:
        model: model to be used for sensitivity calculation
        smiles: SMILES string of the target prediction
        perturbation_engine: perturbation engine to be used for sensitivity calculation
        n: number of perturbations to be used for sensitivity calculation

    Returns:
        rdkit molecule with sensitivity as an atom property
    """

    if perturbation_engine == "default":
        perturbation_engine = SwapMutations(radius=1)

    mol = Chem.MolFromSmiles(smiles)

    idxs = []

    for a in tqdm(mol.GetAtoms()):
        idx = a.GetIdx()
        smiles = [Chem.MolToSmiles(mol)]

        count = 1
        while count < n:
            perturbed_smiles = perturbation_engine.get_compound_at_idx(smiles[0], idx)
            try:
                mol2 = Chem.MolFromSmiles(perturbed_smiles)
                if not mol2 is None:
                    smiles.append(perturbed_smiles)
                    idxs.append(idx)
                    count += 1
            except:
                pass
        predictions = model.predict(smiles)
        v = np.std(predictions)
        a.SetProp("sensitivity", str(v))
        a.SetProp("atomNote", str(np.format_float_scientific(v, precision=2)))
    return mol

from PIL import Image

class BaseVis(BaseClass):
    """Base class for visualizations.
    """
    @log_arguments
    def __init__(self, log=True):
        pass

    def _run_image(self, *args, **kwargs) -> PIL.Image:

        raise NotImplementedError

    def _run_html(self) -> io.StringIO:

        raise NotImplementedError

    def run(self, display="ipynb", **kwargs):
        """run is the core method of the `BaseVis` class. It will run and then
        output the visualization in the specified display format. On the backend
        it will call one of `_run_image` or `_run_html` method, in order by
        priority.

        Args:
            display (str, optional): The format in which to display the output
            to. Defaults to "ipynb", other options are "html", "image",
            "save_html", "save_image", where "save_*" requires a filename
            parameter specifying where to save the output to

        Returns:
            visualization: Type depends on specified display format.
        """
        if display == "ipynb":
            try:
                return self._run_image(**kwargs)
            except NotImplementedError:
                pass
            try:
                from IPython.display import display, HTML
                return display(HTML(self._run_html(**kwargs).getvalue()))
            except NotImplementedError:
                pass
        elif display == "image":
            try:
                return self._run_image(**kwargs)
            except NotImplementedError:
                pass
            try:
                from IPython.display import display, HTML
                return display(HTML(self._run_html(**kwargs).getvalue()))
            except NotImplementedError:
                pass


    def _save(self):
        pass

    def _load(self, d: dict):
        pass

def fig_to_PIL(fig):
    """Converts a plotnine figure to a PIL image.

    Parameters:
        fig: plotnine figure

    Returns:
        PIL image generated from plotnine figure
    """
    return PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

class RegressionFeaturesContribution(BaseVis):
    """Visualization of a regression model and its features.
    """

    @log_arguments
    def __init__(self, rep: BaseCompoundVecRepresentation, log=True, **kwargs):
        self.feat_names = rep.names
        self.rep = rep

    def _run_image(self, dataset: BaseDataset = None, alpha=0.4, wspace = 0.1, hspace=0.1, regression_fit = False):
        """_run_image returns a `PIL.Image` of the visualization for given
        inputs

        Parameters:
            dataset: dataset to be used for visualization
            alpha: transparency of the regression line
            wspace: width space between the regression line and the feature
            hspace: height space between the regression line and the feature
            regression_fit: whether to show the regression line

        Returns:
            PIL image generated from the visualization
        """
        assert not dataset is None, "No dataset provided"
        N = len(self.rep)

        all_feats = self.rep.convert(dataset.entire_dataset[0])
        col_feats = [[row[i] for row in all_feats] for i in range(len(self.rep))]
        ys = dataset.entire_dataset[1]

        if regression_fit:
            GR = oce.GuessingRegression()
            GR.fit(pd.DataFrame({name: feat for name, feat in zip(self.feat_names, col_feats)}), dataset.entire_dataset[1])

        df = pd.DataFrame(columns=["Feature Name", "Feature Value", "Property"])
        for i, name in enumerate(self.feat_names):
            feats = [x[i] for x in all_feats]
            df2 = pd.DataFrame({"Feature Name": name,
                "Feature Value": feats,
                "Property": ys})
            df = df.append(df2)

        df["Feature Value"] = df["Feature Value"].astype(float)
        df["Property"] = df["Property"].astype(float)
        print(df)
        from plotnine import ggplot, aes, geom_point, facet_wrap, theme, ylab
        fig = (ggplot(df, aes(x="Feature Value", y="Property")) +
            geom_point(alpha=alpha) +
            facet_wrap("Feature Name", scales="free") +
            theme(subplots_adjust={'wspace':wspace, 'hspace':hspace}) +
            ylab("% Compounds with Value")).draw()

        fig.canvas.draw()
        return PIL.Image.frombytes('RGB',
            fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

class ClassificationFeaturesContribution(BaseVis):
    """Visualization of a classification model and its features.
    """

    @log_arguments
    def __init__(self, rep: BaseCompoundVecRepresentation, log=True, **kwargs):
        self.feat_names = rep.names
        self.rep = rep

    def _run_image(self, dataset: BaseDataset = None, alpha=0.4, wspace = 0.1, hspace=0.1):
        """_run_image returns a `PIL.Image` of the visualization for given
        inputs

        Parameters:
            dataset: dataset to be used for visualization
            alpha: transparency of the regression line
            wspace: width space between the regression line and the feature
            hspace: height space between the regression line and the feature

        Returns:
            PIL image generated for the visualization
        """
        assert not dataset is None, "No dataset provided"
        N = len(self.rep)

        all_feats = self.rep.convert(dataset.entire_dataset[0])
        col_feats = [[row[i] for row in all_feats] for i in range(len(self.rep))]
        ys = dataset.entire_dataset[1]

        df = pd.DataFrame(columns=["Feature Name", "Feature Value", "Property"])
        for i, name in enumerate(self.feat_names):
            feats = [x[i] for x in all_feats]
            df2 = pd.DataFrame({"Feature Name": name,
                "Feature Value": feats,
                "Property": ys})
            df = df.append(df2)

        df["Feature Value"] = df["Feature Value"].astype(float)
        df["Property"] = df["Property"]#.astype(float)

        from plotnine import ggplot, aes, geom_histogram, facet_wrap, theme, ylab
        fig = (ggplot(df, aes(x="Feature Value", color="Property")) +
            geom_histogram(aes(y='..width..*..density..'), fill="white", alpha=0.5, position="dodge2", bins=10) +
            facet_wrap("Feature Name", scales="free", ncol=1) +
            theme(subplots_adjust={'wspace':wspace, 'hspace':hspace}) +
            theme(figure_size=(7.5, 3.5*N)) +
            ylab("% Compounds with Value")).draw()

        fig.canvas.draw()
        return PIL.Image.frombytes('RGB',
            fig.canvas.get_width_height(),fig.canvas.tostring_rgb())