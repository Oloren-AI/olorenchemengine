"""
for basic analysis of compounds/groups of compounds
"""

from olorenchemengine.base_class import *
from olorenchemengine.dataset import *
from olorenchemengine.internal import *
from olorenchemengine.interpret import *
from olorenchemengine.manager import *
from olorenchemengine.representations import *
from olorenchemengine.uncertainty import *
from olorenchemengine.visualizations import *

from typing import *

class VisualizeMCS(VisualizeCompounds):
    """
    Visualize the maximum common substructure between a list of compounds.
    
    Parameters:
        smiles (List[str]): the smiles of the compounds to visualize
        timeout (int): the timeout for the MCS calculation in seconds
        invert_colors (bool): invert the colors of the MCS such that the MCS is 
            red and the differences are green. Default False.
        completeRingsOnly (bool): only consider complete rings in the MCS, parameter
            for rdkit.Chem.rdFMCS.FindMCS. Default True.
    """
    
    @log_arguments
    def __init__(self, smiles: List[str], *args, timeout = 30, kekulize = False, 
            completeRingsOnly=True, invert_colors=False, log=True, **kwargs):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        mols_ = []
        for s, m in zip(smiles, mols):
            if m is None:
                print(f"Could not parse {s}, skipping")
            mols_.append(m)
        mols = mols_
        
        from rdkit.Chem.rdFMCS import FindMCS
        result = FindMCS(mols, completeRingsOnly=completeRingsOnly, timeout = timeout)
        smarts = result.smartsString
        for mol in mols:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            for a_idx in matches[0]:
                mol.GetAtomWithIdx(a_idx).SetAtomMapNum(2)
            for a_idx in range(mol.GetNumAtoms()):
                if a_idx not in matches[0]:
                    mol.GetAtomWithIdx(a_idx).SetAtomMapNum(1)
        
        smiles_highlighted = [Chem.MolToSmiles(mol) for mol in mols]
        
        if invert_colors:
            highlights = [[2, "#fb8fff"], [1, "#8fff9c"]]
        else:
            highlights = [[1, "#fb8fff"], [2, "#8fff9c"]]
        super().__init__(smiles_highlighted, *args, highlights = highlights, kekulize=kekulize, log=False, **kwargs)

    @property
    def JS_NAME(self):
        return "VisualizeCompounds"