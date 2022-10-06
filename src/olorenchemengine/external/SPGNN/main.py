""" Wraps the model presented in `Strategies for pre-training graph neural networks <https://arxiv.org/abs/1905.12265>`_

`GitHub repository <https://github.com/snap-stanford/pretrain-gnns>`_
"""

from olorenchemengine.internal import mock_imports

try:
    from torch_geometric.data import DataLoader
except ImportError:
    mock_imports(globals(), "DataLoader")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    mock_imports(globals(), "torch", "nn", "optim")


from tqdm import tqdm
import numpy as np

from .model import GNN_graphpred, GNN, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from sklearn.metrics import roc_auc_score

import olorenchemengine as oce
from olorenchemengine.representations import AtomFeaturizer, BaseRepresentation, BondFeaturizer, TorchGeometricGraph, BaseVecRepresentation

import os
import io
from typing import Union, List, Tuple, Dict, Optional

from olorenchemengine.base_class import BaseModel, log_arguments, QuantileTransformer
from olorenchemengine.internal import download_public_file

from rdkit import Chem

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

class SPGNN_AF(AtomFeaturizer):
    @property
    def length(self):
        return 2

    def convert(self, atom: Chem.Atom):
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        x = np.array(atom_feature)
        return x

class SPGNN_BF(BondFeaturizer):
    @property
    def length(self):
        return 2

    def convert(self, bond: Chem.Bond):
        edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
        x = np.array(edge_feature)
        return x

class SPGNN_PYG(TorchGeometricGraph):
    @log_arguments
    def __init__(self):
        super().__init__(SPGNN_AF(), SPGNN_BF(), log = False)

class SPGNNVecRep(BaseVecRepresentation):
    """SPGNN_REP gives the output of the model presented in `Strategies for pre-training graph neural networks <https://arxiv.org/abs/1905.12265>`_
    `GitHub repository <https://github.com/snap-stanford/pretrain-gnns>` as a molecular representation.

    Attributes:
        available_pretrained_models (List[str]): List of available pretrained models; passed in the model_type parameter

    Parameters:
        model_type (str): Type of model to use; default: "contextpred"
    """

    available_pretrained_models = ["contextpred",
        "edgepred",
        "infomax",
        "masking",
        "supervised_contextpred",
        "supervised_edgepred",
        "supervised_infomax",
        "supervised_masking",
        "supervised",
        "gat_supervised_contextpred",
        "gat_supervised",
        "gat_contextpred"]

    @log_arguments
    def __init__(self, model_type = "contextpred",
        batch_size = 32,
        epochs = 100,
        lr = 0.001,
        lr_scale = 1,
        decay = 0,
        num_layer = 5,
        emb_dim = 300,
        dropout_ratio = 0.5,
        graph_pooling = "mean",
        JK = "last",
        gnn_type = "gin", **kwargs):

        self.model_type = model_type

        if "gat" in model_type:
            gnn_type = "gat"
            emb_dim= 300

        self.representation = SPGNN_PYG()

        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = oce.CONFIG["NUM_WORKERS"]
        self.map_location = oce.CONFIG["MAP_LOCATION"]
        self.device = oce.CONFIG["DEVICE"]

        self.model = GNN(num_layer, emb_dim, JK, dropout_ratio, gnn_type = gnn_type)

        input_model_file = download_public_file(f"SPGNN_saves/{self.model_type}.pth")

        self.model.load_state_dict(torch.load(input_model_file, map_location=self.map_location))
        self.model.eval()
        self.model.to(self.device)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

    def _convert(self, smiles: str, y: Union[int, float, np.number] = None) -> np.ndarray:
        assert Exception, "Please directly use convert method"

    def convert(self, smiles, **kwargs):
        X = self.representation.convert(smiles)
        loader = DataLoader(X, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)

        self.model.eval()
        y_pred = []

        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(self.device)

            with torch.no_grad():
                pred = self.model(batch.x, batch.edge_index, batch.edge_attr)
                pred = self.pool(pred, batch.batch).detach().cpu().numpy()

            y_pred.append(pred)
        return np.concatenate(y_pred)

def train(model, device, loader, optimizer, setting):
    if setting == "classification":
        criterion = nn.BCEWithLogitsLoss(reduction = "none")
    else:
        criterion = nn.MSELoss()
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Loss matrix
        loss_mat = criterion(pred.double(), y)

        optimizer.zero_grad()

        loss = torch.mean(loss_mat)
        loss.backward()

        optimizer.step()

def predict(model, device, loader, setting = "classification"):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_pred.append(pred)
    if setting == "classification":
        y_pred = nn.Sigmoid()(torch.cat(y_pred, dim = 0)).cpu().numpy()
    else:
        y_pred = torch.cat(y_pred, dim = 0).cpu().numpy()
    return y_pred.flatten()

class SPGNN(BaseModel):
    """SPGNN is the model presented in `Strategies for pre-training graph neural networks <https://arxiv.org/abs/1905.12265>`_
    `GitHub repository <https://github.com/snap-stanford/pretrain-gnns>`_

    Attributes:
        available_pretrained_models (List[str]): List of available pretrained models; passed in the model_type parameter

    Parameters:
        model_type (str): Type of model to use; default: "contextpred"
    """

    available_pretrained_models = ["contextpred",
        "edgepred",
        "infomax",
        "masking",
        "supervised_contextpred",
        "supervised_edgepred",
        "supervised_infomax",
        "supervised_masking",
        "supervised",
        "gat_supervised_contextpred",
        "gat_supervised",
        "gat_contextpred"]

    @log_arguments
    def __init__(self, model_type = "contextpred",
        batch_size = 32,
        epochs = 100,
        lr = 0.001,
        lr_scale = 1,
        decay = 0,
        num_layer = 5,
        emb_dim = 300,
        dropout_ratio = 0.5,
        graph_pooling = "mean",
        JK = "last",
        gnn_type = "gin", **kwargs):

        self.model_type = model_type

        if "gat" in model_type:
            gnn_type = "gat"
            emb_dim= 300
        self.representation = SPGNN_PYG()

        self.batch_size = batch_size
        self.epochs = epochs

        self.num_workers = oce.CONFIG["NUM_WORKERS"]
        self.map_location = oce.CONFIG["MAP_LOCATION"]
        self.device = oce.CONFIG["DEVICE"]

        self.model = GNN_graphpred(num_layer,
            emb_dim,
            1,
            JK = JK,
            drop_ratio = dropout_ratio,
            graph_pooling = graph_pooling,
            gnn_type = gnn_type)

        input_model_file = download_public_file(f"SPGNN_saves/{self.model_type}.pth")

        self.model.from_pretrained(input_model_file, map_location=self.map_location)

        super().__init__(**kwargs)

        self.model.to(self.device)

        model_param_group = []
        model_param_group.append({"params": self.model.gnn.parameters()})
        if graph_pooling == "attention":
            model_param_group.append({"params": self.model.pool.parameters(), "lr":lr*lr_scale})
        model_param_group.append({"params": self.model.graph_pred_linear.parameters(), "lr":lr* lr_scale})

        self.optimizer = optim.Adam(model_param_group, lr=lr, weight_decay=decay)
        self.sigmoid = nn.Sigmoid()

    def preprocess(self, X, y, **kwargs):
        if y is None:
            y = [None]*len(X)
        return self.representation.convert(X, ys=y)

    def _fit(self, X, y, **kwargs):
        dataloader = DataLoader(X, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        for epoch in range(self.epochs):
            train(self.model, self.device, dataloader, self.optimizer, self.setting)

    def _predict(self, X, **kwargs):
        dataloader = DataLoader(X, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return predict(self.model, self.device, dataloader, setting = self.setting)

    def _save(self) -> str:
        d = super()._save()
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        d.update({"save": buffer.getvalue()})
        return d

    def _load(self, d):
        super()._load(d)
        self.model = torch.load(io.BytesIO(d["save"]))

    @classmethod
    def AllInstances(cls):
        return [cls(model_type=mt) for mt in cls.available_pretrained_models]