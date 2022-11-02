from typing import *

from olorenchemengine.base_class import *
from olorenchemengine.dataset import *
from olorenchemengine.visualizations.visualization import *
from olorenchemengine.representations import *
from olorenchemengine.basics import *

class CompoundDistMatchedPairsViewer(BaseVisualization):
    """
    This view allows the user to browse all matched pairs in a dataset based on 
    compound distances matched pairs
    
    Parameters:
        dataset (BaseDataset): The dataset to conduct the compound distance-based
            matched pairs analysis on
        id (str): The id column in the dataset to define the pairs by. 
            Must be in dataset.feature_cols. Default is None meaning the index 
            which identifies pairs by (row_num_1, row_num_2).
        rep (BaseRepresentation): The representation to use for the analysis.
            Defaults to MorganVecRepresentation.
        invert_colors (bool): invert the colors of the MCS such that the MCS is
            red and the differences are green. Default False.
    """
    
    @log_arguments
    def __init__(
        self,
        dataset: BaseDataset,
        *args,
        id: str = None,
        rep: BaseCompoundVecRepresentation = MorganVecRepresentation(),
        invert_colors: bool = False,
        timeout: float = 1,
        show_display_smiles: bool = False,
        show_structure_col: bool = False,
        match_threshold: int = 10,
        log = True,
        **kwargs
    ):
        # calculate distances and similarities and filter
        smiles = dataset.data[dataset.structure_col]
        dists = rep.calculate_distance(smiles, 
                                       smiles, 
                                       metric="rogerstanimoto")
        sim = 1 - dists
        pairs = np.argwhere(sim > 0.95)
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        print(f"{len(pairs)} pairs with similarity > 0.95")
        
        # construct the list of pairs
        pair_df = pd.DataFrame(pairs, columns = ["index1", "index2"])
        pair_df["sim"] = sim[pairs[:, 0], pairs[:, 1]]
        
        # map structure columns
        pair_df["smiles1"] = smiles.iloc[pairs[:, 0]].tolist()
        pair_df["smiles2"] = smiles.iloc[pairs[:, 1]].tolist()
        
        # map feature columns
        cols = [dataset.structure_col] + [dataset.property_col] + dataset.feature_cols
        pair_df = pair_df.merge(dataset.data[cols], left_on = "smiles1", 
                right_on = dataset.structure_col, suffixes=("", " 1"))
        pair_df = pair_df.rename(columns = {col: col + " 1" for col in cols})
        pair_df = pair_df.merge(dataset.data[cols], left_on = "smiles2", 
                right_on = dataset.structure_col, suffixes=("", " 2"))
        pair_df = pair_df.rename(columns = {col: col + " 2" for col in cols})
        
        # helper method to get fragments with MCS differences
        def get_diff(pair):
            mols = [Chem.MolFromSmiles(pair[0]), Chem.MolFromSmiles(pair[1])]

            mols_ = []
            for s, m in zip(smiles, mols):
                if m is None:
                    print(f"Could not parse {s}, skipping")
                mols_.append(m)
            mols = mols_

            continue_mcs = True
            patts = []
            while continue_mcs:
                try:
                    from rdkit.Chem.rdFMCS import FindMCS
                    result = FindMCS(mols, completeRingsOnly=True, timeout = timeout)
                    smarts = result.smartsString
                    patt = Chem.MolFromSmarts(smarts)
                    if len(patt.GetAtoms()) > match_threshold:
                        mol_frags = []
                        mols_ = []
                        for mol in mols:
                            mol = Chem.Mol(mol)
                            matches = mol.GetSubstructMatches(patt)

                            mol = Chem.RWMol(mol)
                            mol.BeginBatchEdit()
                            for a_idx in matches[0]:
                                mol.RemoveAtom(a_idx)
                            mol.CommitBatchEdit()
                            mol.UpdatePropertyCache()
                            Chem.FastFindRings(mol)
                            mols_.append(mol)
                        mols = mols_
                        patts.append(patt)
                    else:
                        continue_mcs = False
                except Exception as e:
                    print(e)
                    continue_mcs = False
                    
            mol_frags = []
            for mol in mols:
                mol = Chem.RWMol(mol)
                frags = Chem.GetMolFrags(mol, asMols=False)
                frags_ = []
                for frag in frags:
                    mol_ = copy.deepcopy(mol)
                    mol_.BeginBatchEdit()
                    for i in range(mol_.GetNumAtoms()):
                        if i not in frag:
                            mol_.RemoveAtom(i)
                    mol_.CommitBatchEdit()
                    frags_.append(Chem.MolToSmiles(mol_))
                mol_frags.append(frags_)
            return mol_frags, patts

        # get the fragments
        pair_df["pair"] = [pair for pair in pair_df[["smiles1", "smiles2"]].values]
        from tqdm import tqdm
        tqdm.pandas()
        res =  list(zip(*pair_df["pair"].progress_apply(get_diff).tolist()))
        pair_df["diff"] = res[0]
        pair_df["patt"] = res[1]
        
        pair_df["diff1"] = [diff[0] for diff in pair_df["diff"]]
        pair_df["diff2"] = [diff[1] for diff in pair_df["diff"]]
        
        # get the feature annotations for both sides of the pair
        self.annotations = []
        
        from pandas.api.types import is_numeric_dtype
        for col in dataset.feature_cols + [dataset.property_col]:
            if is_numeric_dtype(dataset.data[col]):
                self.annotations.append(col + " diff (2-1)")
                pair_df[col + " diff (2-1)"] = (pair_df[col + " 2"] - 
                    pair_df[col + " 1"])
            else:
                self.annotations.append(col + " 1")
                self.annotations.append(col + " 2")
        
        if show_display_smiles:
            self.annotations.append("smiles1")
            self.annotations.append("smiles2")
        
        if show_structure_col:
            self.annotations.append(dataset.structure_col + " 1")
            self.annotations.append(dataset.structure_col + " 2")
                
        # Find the common MCS between pairs and highlight it
        from rdkit.Chem.rdFMCS import FindMCS
        display_smiles_1 = []
        display_smiles_2 = []
        for pair, patts in zip(pair_df["pair"], pair_df["patt"]):
            mols = [Chem.MolFromSmiles(pair[0]), Chem.MolFromSmiles(pair[1])]
            for mol in mols:
                for i in range(mol.GetNumAtoms()):
                    mol.GetAtomWithIdx(i).SetAtomMapNum(1)
                for patt in patts:
                    try:
                        patt = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(patt)))
                        matches = mol.GetSubstructMatches(patt)
                        for a_idx in matches[0]:
                            mol.GetAtomWithIdx(a_idx).SetAtomMapNum(2)
                    except:
                        pass
            display_smiles_1.append(Chem.MolToSmiles(mols[0]))
            display_smiles_2.append(Chem.MolToSmiles(mols[1]))
        
        pair_df["display_smiles_1"] = display_smiles_1
        pair_df["display_smiles_2"] = display_smiles_2
        
        pair_df = pair_df.sort_values(dataset.property_col + " diff (2-1)", ascending = False)
        
        if id is None:
            self.ids = [
                str((row["index1"], row["index2"]))\
                for i, row in pair_df.iterrows()
            ]
        else:
            self.ids = [
                str((row[id + " 1"], row[id + " 2"]))\
                for i, row in pair_df.iterrows()
            ]
        self.ids = [id + " " + str(val) for id, val in zip(self.ids, pair_df[dataset.property_col + " diff (2-1)"].tolist())]
        
        pair_df = pair_df.drop(columns = ["pair", "diff", "patt"])
        self.pair_df = pair_df
        self.invert_colors = invert_colors
        
        super().__init__(*args, log = False, **kwargs)
        self.packages += ["olorenrenderer"]
        
    def get_diff_table(self):
        return self.pair_df
        
    def get_data(self):
        if self.invert_colors:
            highlights = [[2, "#fb8fff"], [1, "#8fff9c"]]
        else:
            highlights = [[1, "#fb8fff"], [2, "#8fff9c"]]
            
        return {
            "annotations": self.annotations,
            "ids": self.ids,
            "table": self.pair_df.to_dict("r"),
            "highlights": highlights,
        }

class MatchedPairsFeaturesTable(BaseVisualization):
    """
    This visualization is intended to show matched pairs of molecules in a dataset
    which differ based on a set of feature columns defined in the dataset object.
    """
    @log_arguments
    def __init__(
        self,
        dataset: BaseDataset,
        mode: str = "features",
        annotations: Union[str, List[str]] = [],
    ):

        self.packages = ["d3"]

        self.dataset = dataset

        if not isinstance(annotations, list):
            annotations = [annotations]
        self.annotations = annotations

        self.mps = self.matched_pairs(dataset.data, dataset.feature_cols)
        if mode == "features":
            self.mp_diffs = pd.DataFrame()

            for mp in self.mps:
                r1 = self.dataset.data.iloc[mp[0]]
                r2 = self.dataset.data.iloc[mp[1]]
                diff = mp[2][0]
                self.mp_diffs = self.mp_diffs.append(
                    {
                        "col": diff[0],
                        "initial": diff[1],
                        "final": diff[2],
                        **{
                            f"Delta (Final - Initial) {property_col}": r2[
                                property_col + " mean"
                            ]
                            - r1[property_col + " mean"]
                            for property_col in dataset.property_col
                        },
                        **{
                            f"Fold (Final/Initial) {property_col}": r2[
                                property_col + " mean"
                            ]
                            / r1[property_col + " mean"]
                            for property_col in dataset.property_col
                        },
                        **{
                            f"Initial {annotation}": r1[annotation]
                            for annotation in self.annotations
                        },
                        **{
                            f"Final {annotation}": r2[annotation]
                            for annotation in self.annotations
                        },
                    },
                    ignore_index=True,
                )
            self.mp_diffs = (
                self.mp_diffs.sort_values(by="col", ascending=True)
                .reset_index()
                .drop(columns=["index"])
            )
        elif mode == "property":
            self.mp_diffs = pd.DataFrame()
            for mp in self.mps:
                r1 = self.dataset.data.iloc[mp[0]]
                r2 = self.dataset.data.iloc[mp[1]]
                diff = mp[2][0]

                for property in self.dataset.property_col:
                    if not r1[property] == r2[property]:
                        self.mp_diffs = self.mp_diffs.append(
                            {
                                "col": diff[0],
                                "initial": diff[1],
                                "final": diff[2],
                                f"Initial {property}": r1[property],
                                f"Final {property}": r2[property],
                                **{
                                    f"Initial {annotation}": r1[annotation]
                                    for annotation in annotations
                                },
                                **{
                                    f"Final {annotation}": r2[annotation]
                                    for annotation in annotations
                                },
                            },
                            ignore_index=True,
                        )
            self.mp_diffs = (
                self.mp_diffs.sort_values(
                    by=["col", "initial", "final"], ascending=True
                )
                .reset_index()
                .drop(columns=["index"])
            )

    def matched_pairs(self, df, cols, dist=1):
        out = []
        for i_ in tqdm(range(len(df.index))):
            i = df.index[i_]
            for j_ in range(len(df.index)):
                j = df.index[j_]
                if not i_ == j_:
                    count = []
                    close = True
                    for col in cols:
                        if str(df.iloc[i][col]) != str(df.iloc[j][col]):
                            count.append((col, df.iloc[i][col], df.iloc[j][col]))
                        if len(count) > dist:
                            close = False
                            break
                    if close:
                        out.append((i, j, count))
        return out

    def get_data(self):
        data = []
        for i, col in enumerate(self.mp_diffs["col"].unique()):
            data.append(
                {
                    "col": col,
                    "i": i,
                    "col_data": self.mp_diffs.loc[self.mp_diffs["col"] == col].to_dict(
                        "r"
                    ),
                }
            )
        return data

    @property
    def JS_NAME(self):
        return "MatchedPairsTable"


class MatchedPairsFeaturesHeatmap(MatchedPairsFeaturesTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.packages = ["d3", "plotly"]

    def get_data(self):
        data = []
        for property_col in self.dataset.property_col:
            property_col = f"Delta (Final - Initial) {property_col}"
            for i, col in enumerate(self.mp_diffs["col"].unique()):
                col_data = self.mp_diffs[self.mp_diffs["col"] == col]
                mapping = {
                    str(x): i for i, x in enumerate(col_data["initial"].unique())
                }
                col_heatmap = np.zeros((len(mapping), len(mapping)))
                for i, col_row in col_data.iterrows():
                    try:
                        x = float(col_row[property_col])
                        if not np.isnan(x):
                            col_heatmap[
                                mapping[str(col_row["final"])],
                                mapping[str(col_row["initial"])],
                            ] += col_row[property_col]
                    except:
                        pass

                data.append(
                    {
                        "z": col_heatmap.tolist(),
                        "x": list(mapping.keys()),
                        "y": list(mapping.keys()),
                        "xlabel": "Initial",
                        "ylabel": "Final",
                        "title": f"Residue number {col}, colored by sum of {property_col}",
                    }
                )
        return data

    @property
    def JS_NAME(self):
        return "MatchedPairsHeatmap"
