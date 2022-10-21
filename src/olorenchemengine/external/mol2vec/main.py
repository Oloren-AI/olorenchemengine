import os
from typing import Any, List, Union

import numpy as np
from rdkit import Chem

import olorenchemengine as oce
from olorenchemengine.base_class import log_arguments
from olorenchemengine.internal import download_public_file
from olorenchemengine.representations import BaseVecRepresentation

from .operations import DfVec, MolSentence, mol2alt_sentence, sentences2vec


class Mol2Vec(BaseVecRepresentation):
    @log_arguments
    def __init__(self):
        print(
            "Requires gensim installation https://github.com/RaRe-Technologies/gensim"
        )

        model_path = download_public_file("mol2vec/model_300dim.pkl")

        oce.import_or_install("gensim")

        from gensim.models import word2vec

        self.model = word2vec.Word2Vec.load(model_path)
        super().__init__(log=False)

    def _convert(self, s: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(s)
        sentence = MolSentence(mol2alt_sentence(mol, 1))
        X = DfVec(sentences2vec([sentence], self.model, unseen="UNK")[0]).vec
        return np.array(X)

    def _convert_list(
        self, smiles_list: List[str], ys: List[Union[int, float, np.number]] = None
    ) -> List[np.ndarray]:
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        sentences = [MolSentence(mol2alt_sentence(mol, 1)) for mol in mols]
        X = [
            DfVec(sentence).vec
            for sentence in sentences2vec(sentences, self.model, unseen="UNK")
        ]
        return np.array(X)
