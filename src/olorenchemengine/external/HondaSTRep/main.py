import numpy as np
import pandas as pd

from typing import List, Union, Any

import olorenchemengine as oce
from olorenchemengine.base_class import log_arguments, BaseModel
from olorenchemengine.representations import BaseVecRepresentation, SMILESRepresentation
from olorenchemengine.internal import download_public_file
from .operations import WordVocab, TrfmSeq2seq, Seq2seqDataset

try:
    from torch.utils.data import DataLoader
    import torch
except ImportError:
    oce.mock_imports(globals(), "DataLoader", "torch")

class HondaSTRep(BaseVecRepresentation):
    """ HondaSTRep is an implementation of the molecular representation provided
    by Honda et al. in `SMILES Transformer: Pre-trained Molecular Fingerprint for Low Data Drug Discovery
    <https://arxiv.org/abs/1911.04738>`."""

    @log_arguments
    def __init__(self):
        model_path = download_public_file("HondaSTRep/trfm_12_23000.pkl")
        vocab_path = download_public_file("HondaSTRep/vocab2.pkl")
        self.vocab = oce.WordVocab.load_vocab(vocab_path)
        self.trfm = TrfmSeq2seq(len(self.vocab), 256, len(self.vocab), 4)
        self.trfm.load_state_dict(torch.load(model_path))
        self.trfm.eval()
        super().__init__(log=False)

    def _convert(self, s: str) -> np.ndarray:
        seq2seqdataset = Seq2seqDataset([s], self.vocab)
        loader = DataLoader(seq2seqdataset, batch_size=32, shuffle=False, num_workers=oce.CONFIG["NUM_WORKERS"])
        output = self.trfm.encode(next(iter(loader)))
        return output

    def _convert_list(self, smiles_list: List[str], ys: List[Union[int, float, np.number]] = None) -> List[np.ndarray]:
        if isinstance(smiles_list, pd.Series):
            smiles_list = smiles_list.tolist()
        seq2seqdataset = Seq2seqDataset(smiles_list, self.vocab)
        loader = DataLoader(seq2seqdataset, batch_size=32, shuffle=False, num_workers=oce.CONFIG["NUM_WORKERS"])
        output = None
        for b, sm in enumerate(loader):
            sm = torch.t(sm)
            sm = sm.to(oce.CONFIG["DEVICE"])
            encoding = self.trfm.encode(sm)
            if output is None:
                output = encoding
            else:
                output = np.concatenate([output, encoding])
        return output