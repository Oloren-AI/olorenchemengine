"""
Wraps the model presented in `Analyzing Learned Molecular Representations for Property Prediction <hhttps://doi.org/10.1021/acs.jcim.9b00237>`.
Here, we adapt its PyTorch Geometric implementation as in the `Github repository <https://github.com/itakigawa/pyg_chemprop>`

"""

from olorenchemengine.representations import AtomFeaturizer, BondFeaturizer, TorchGeometricGraph
from olorenchemengine.base_class import BaseModel, log_arguments, QuantileTransformer

from olorenchemengine.internal import mock_imports

try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data, Dataset
    from torch_geometric.data.data import size_repr
    from torch_geometric.nn import global_mean_pool
    from torch_scatter import scatter_sum
except ImportError:
    mock_imports(globals(), "DataLoader", "Data", "Dataset", "size_repr", "global_mean_pool", "scatter_sum")


try:
    import torch
    import torch.nn as nn
    import torch.utils.data
except ImportError:
    mock_imports(globals(), "torch", "nn")

import numpy as np
from rdkit import Chem
import io
from tqdm import tqdm


class RevIndexedData(Data):
    def __init__(self, orig):
        super(RevIndexedData, self).__init__()
        if orig:
            for key in orig.keys:
                self[key] = orig[key]
            edge_index = self["edge_index"]
            revedge_index = torch.zeros(edge_index.shape[1]).long()
            for k, (i, j) in enumerate(zip(*edge_index)):
                edge_to_i = edge_index[1] == i
                edge_from_j = edge_index[0] == j
                revedge_index[k] = torch.where(edge_to_i & edge_from_j)[0].item()
            self["revedge_index"] = revedge_index

    def __inc__(self, key, value, *args, **kwargs):
        if key == "revedge_index":
            if len(self.revedge_index) > 0:
                return self.revedge_index.max().item() + 1
            else:
                return 0
        else:
            return super().__inc__(key, value)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))

class RevIndexedDataset(Dataset):
    def __init__(self, orig):
        super(RevIndexedDataset, self).__init__()
        self.dataset = [RevIndexedData(data) for data in tqdm(orig)]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

class ChemProp_AF (AtomFeaturizer):
    @property
    def length(self):
        return 151

    def convert(self, atom: Chem.Atom) -> np.ndarray:

        ATOM_FEATURES = {
            "atomic_num": list(range(1,119)),
            "degree": [0, 1, 2, 3, 4, 5],
            "formal_charge": [-1, -2, 1, 2, 0],
            "chiral_tag": [0, 1, 2, 3],
            "num_Hs": [0, 1, 2, 3, 4],
            "hybridization": [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        }
        features = (
            onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_num"])
            + onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
            + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
            + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
            + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
            + onek_encoding_unk(
                int(atom.GetHybridization()), ATOM_FEATURES["hybridization"]
            )
            + [1 if atom.GetIsAromatic() else 0]
            + [atom.GetMass() * 0.01]
        )  # scaled to about the same range as other features
        return np.array(features)

class ChemProp_BF (BondFeaturizer):
    @property
    def length (self):
        return 14

    def convert(self, bond: Chem.Bond) -> np.ndarray:
        if bond is None:
            fbond = [1] + [0] * (13)
        else:
            bt = bond.GetBondType()
            fbond = [
                0,  # bond is not None
                bt == Chem.rdchem.BondType.SINGLE,
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE,
                bt == Chem.rdchem.BondType.AROMATIC,
                (bond.GetIsConjugated() if bt is not None else 0),
                (bond.IsInRing() if bt is not None else 0),
            ]
            fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        return np.array(fbond)

class ChemPropDataLoading (TorchGeometricGraph):
    @log_arguments
    def __init__(self):
        super().__init__(ChemProp_AF(), ChemProp_BF(), log = False)

    def _convert(self, smiles, y=None, **kwargs):
        data = super()._convert(smiles, y=y, **kwargs)
        return RevIndexedData(data)


def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.
    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

def directed_mp(message, edge_index, revedge_index):
    m = scatter_sum(message, edge_index[1], dim=0)
    m_all = m[edge_index[0]]
    m_rev = message[revedge_index]
    return m_all - m_rev

def aggregate_at_nodes(num_nodes, message, edge_index):
    m = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
    return m[torch.arange(num_nodes)]

#### Key Message-Passing algorithm in the model ####
class DMPNNEncoder(nn.Module):
    def __init__(self, hidden_size, node_fdim, edge_fdim, depth=3):
        super(DMPNNEncoder, self).__init__()
        self.act_func = nn.ReLU()
        self.W1 = nn.Linear(node_fdim + edge_fdim, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W3 = nn.Linear(node_fdim + hidden_size, hidden_size, bias=True)
        self.depth = depth

    def forward(self, data):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act_func(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = directed_mp(h, edge_index, revedge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        # readout: pyg global pooling
        return global_mean_pool(node_attr, batch)

#### General ChemProp model training method, to be used with _fit in the ChemPropModel() class implementation. ####
def train(config, loader, setting, device=torch.device("cpu")):
    criterion = config["loss"]
    model = config["model"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]

    if setting == "classification":
        criterion = nn.BCEWithLogitsLoss(reduction = "none")
    else:
        criterion = nn.MSELoss()

    model = model.to(device)
    model.train()
    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(torch.squeeze(out,1), batch.y.float())
        loss.sum().backward()

        optimizer.step()
        scheduler.step()

#### General ChemProp model prediction method, to be used with _predict in the ChemPropModel() class implementation. ####
def predict(config, loader, setting = "classification", device=torch.device("cpu")):
    model = config["model"]

    model = model.to(device)
    model.eval()
    y_pred = []
    for batch in tqdm(loader, total=len(loader)):
        batch = batch.to(device)
        with torch.no_grad():
            if setting == "classification":
                batch_preds = torch.sigmoid(model(batch))
            else:
                batch_preds = model(batch)
        y_pred.extend(batch_preds)

    predictions = np.concatenate([y.cpu().numpy() for y in y_pred])
    return predictions.flatten()

class ChemPropModel(BaseModel):
    """ChemProp is the model presented in `Analyzing Learned Molecular Representations for Property Prediction <hhttps://doi.org/10.1021/acs.jcim.9b00237>` _.
    Here, we adapt its PyTorch Geometric implementation as in the `Github repository <https://github.com/itakigawa/pyg_chemprop>`_

    "ChemProp" is a simple but effective Graph Neural Network (GNN) for Molecular Property Prediction. The PyG implementation makes it compatible with Oloren AI software.

    Attributes:
        representation (TorchGeometricGraph): Returns smiles-inputted data as PyTorch graph objects for use in the ChemProp model
        setting (str): whether the model is a "classification" model or a "regression" model
        optimizer: function to modify model parameters such as weights and learning rate; we use Adam.
        criterion: loss function; we use BCEWithLogitsLoss for classification and MSELoss for regression.
        scheduler: reduces LR as number of training epochs increases; we use PyTorch's OneCycleLR.

    Parameters:
        dropout_rate (float): fraction of layer outputs that are randomly ignored; default = 0.
        epochs (int): number of complete passes of the training dataset through the model; default = 3.
        batch_size (int): number of training examples utilized in one model iteration; default = 50.
        lr (float): amount that the weights are updated during training; default = 1e-3.
        hidden_size (int): number of hidden neurons between input and output; default = 300.
        depth (int): number of layers between input and output; default = 3.
    """

    @log_arguments
    def __init__(self,
        dropout_rate: float = 0.0,
        epochs: int = 3,
        batch_size: int = 50,
        lr: float = 1e-3,
        hidden_size: int = 300,
        depth: int =3,
        map_location = "cuda:0",
         **kwargs):

        self.representation = ChemPropDataLoading()

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.map_location = map_location
        self.device = torch.device(self.map_location)

        AF = ChemProp_AF()
        BF = ChemProp_BF()

        self.model = nn.Sequential(
            DMPNNEncoder(self.hidden_size, AF.length, BF.length, depth),
            nn.Sequential(
                nn.Dropout(p=self.dropout_rate, inplace=False),
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate, inplace=False),
                nn.Linear(hidden_size, 1, bias=True),
            ),
        )
        initialize_weights(self.model)

        super().__init__(preprocessor=QuantileTransformer(), **kwargs)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()

    def preprocess(self, X, y, **kwargs):
        if y is None:
            y = [None]*len(X)
        return self.representation.convert(X, ys=y)

    def _fit(self, X, y, **kwargs):

        loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr, steps_per_epoch=len(X), epochs=self.epochs
        )
        self.config = {
            "loss": self.criterion,
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

        if (len(np.unique(y, return_counts=True)) > 2 and (((np.unique(y)).size / (y.size)) > 0.1)):
            self.setting = "regression"

        for epoch in range(self.epochs):
            train(self.config, loader, self.setting)

    def _predict(self, X, **kwargs):
        loader = DataLoader(X, batch_size=self.batch_size, shuffle=False)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr, steps_per_epoch=len(X), epochs=self.epochs
        )
        self.config = {
            "loss": self.criterion,
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }
        return predict(self.config, loader, self.setting)

    def _save(self) -> str:
        d = super()._save()
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        d.update({"save": buffer.getvalue()})
        return d

    def _load(self, d):
        super()._load(d)
        self.model = torch.load(io.BytesIO(d["save"]))