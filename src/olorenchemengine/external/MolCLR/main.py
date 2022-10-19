import olorenchemengine as oce

from olorenchemengine.base_class import log_arguments, BaseModel
from olorenchemengine.representations import AtomFeaturizer, BondFeaturizer, TorchGeometricGraph, SMILESRepresentation, BaseVecRepresentation
from olorenchemengine.internal import download_public_file

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

try:
    from torch_geometric.data import DataLoader
except ImportError:
    oce.mock_imports(globals(), "DataLoader")

import numpy as np
from tqdm import tqdm

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

class MolCLR_AF(AtomFeaturizer):
    @property
    def length(self):
        return 2

    def convert(self, atom: Chem.Atom):
        atom_feature = [ATOM_LIST.index(atom.GetAtomicNum()), CHIRALITY_LIST.index(atom.GetChiralTag())]
        x = np.array(atom_feature)
        return x

class MolCLR_BF(BondFeaturizer):
    @property
    def length(self):
        return 2

    def convert(self, bond: Chem.Bond):
        edge_feature = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
        x = np.array(edge_feature)
        return x

class MolCLR_PYG(TorchGeometricGraph):
    @log_arguments
    def __init__(self):
        super().__init__(MolCLR_AF(), MolCLR_BF(), log = False)

class MolCLR(BaseModel):

    model_config = {"num_layer": 5,      # number of graph conv layers
            "emb_dim": 300,                  # embedding dimension in graph conv layers
            "feat_dim": 512,                 # output feature dimention
            "drop_ratio": 0.3,               # dropout ratio
            "pool": "mean"}

    @log_arguments
    def __init__(self, model_type = "ginet", epochs = 100, batch_size = 32,
                 init_lr = 0.0005, init_base_lr = 0.0001, weight_decay = 1e-6,
                 **kwargs):
        self.model_type = model_type

        self.config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "init_lr": init_lr,
            "init_base_lr": init_base_lr,
            "weight_decay": weight_decay
        }

        self.representation = MolCLR_PYG()

        super().__init__(**kwargs)

    def preprocess(self, X, y, **kwargs):
        if y is None:
            y = [None]*len(X)
        return self.representation.convert(X, ys=y)

    def _fit(self, X, y = None, **kwargs):

        import torch
        from torch import nn
        import torch.nn.functional as F
        from torch.utils.tensorboard import SummaryWriter
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

        if self.setting == 'regression':
            self.criterion = nn.MSELoss()
        elif self.setting == 'classification':
            self.criterion = nn.BCEWithLogitsLoss(reduction = "none")

        if self.model_type == "ginet":
            from .model import GINet
            self.model = GINet(self.setting, **self.model_config)
            save_path = download_public_file("MolCLR/pretrained_gin.pth")
        elif self.model_type == "gcn":
            from .model import GCN
            self.model = GCN(self.setting, **self.model_config)
            save_path = download_public_file("MolCLR/pretrained_gcn.pth")

        self.model.load_my_state_dict(torch.load(save_path, map_location = oce.CONFIG["MAP_LOCATION"]))
        self.model.to(oce.CONFIG["DEVICE"])

        layer_list = []
        for name, param in self.model.named_parameters():
            if 'pred_lin' in name:
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, self.model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, self.model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=self.config['weight_decay']
        )

        train_loader = DataLoader(X, batch_size=self.config["batch_size"], shuffle=True, num_workers=oce.CONFIG["NUM_WORKERS"])
        for epoch_counter in range(self.config['epochs']):
            print(f"Epoch {epoch_counter+1}/{self.config['epochs']}")
            for bn, batch in enumerate(train_loader):

                batch = batch.to(oce.CONFIG["DEVICE"])
                _, pred = self.model(batch)
                y = batch.y.view(pred.shape).to(torch.float64)

                #Loss matrix
                loss_mat = self.criterion(pred.double(), y)

                optimizer.zero_grad()

                loss = torch.mean(loss_mat)
                loss.backward()

                optimizer.step()

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data)  # [N,C]
        loss = self.criterion(pred.flatten(), data.y)

        return loss

    def _predict(self, X, **kwargs):
        import torch

        loader = DataLoader(X, batch_size=self.config["batch_size"], shuffle=False, num_workers=oce.CONFIG["NUM_WORKERS"])
        self.model.eval()
        y_pred = []

        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(oce.CONFIG["DEVICE"])

            with torch.no_grad():
                _, pred = self.model(batch)
            y_pred.append(pred.cpu().detach().numpy())

        return np.vstack(y_pred).flatten()

class MolCLRVecRep(BaseVecRepresentation):

    model_config = {"num_layer": 5,      # number of graph conv layers
            "emb_dim": 300,                  # embedding dimension in graph conv layers
            "feat_dim": 512,                 # output feature dimention
            "drop_ratio": 0.3,               # dropout ratio
            "pool": "mean"}

    @log_arguments
    def __init__(self, model_type = "ginet", epochs = 100, batch_size = 32,
                 init_lr = 0.0005, init_base_lr = 0.0001, weight_decay = 1e-6,
                 **kwargs):
        self.model_type = model_type

        self.config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "init_lr": init_lr,
            "init_base_lr": init_base_lr,
            "weight_decay": weight_decay
        }

        import torch
        from torch import nn
        import torch.nn.functional as F
        from torch.utils.tensorboard import SummaryWriter
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

        if self.model_type == "ginet":
            from .model import GINet
            self.model = GINet("regression", **self.model_config)
            save_path = download_public_file("MolCLR/pretrained_gin.pth")
        elif self.model_type == "gcn":
            from .model import GCN
            self.model = GCN("regression", **self.model_config)
            save_path = download_public_file("MolCLR/pretrained_gcn.pth")

        self.model.load_my_state_dict(torch.load(save_path, map_location = oce.CONFIG["MAP_LOCATION"]))
        self.model.to(oce.CONFIG["DEVICE"])

    def _convert(self, X, **kwargs):
        return self.convert(X)

    def convert(self, X, **kwargs):
        X = MolCLR_PYG().convert(X, ys=None)
        import torch

        loader = DataLoader(X, batch_size=self.config["batch_size"], shuffle=True, num_workers=oce.CONFIG["NUM_WORKERS"])
        self.model.eval()
        y_pred = []

        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(oce.CONFIG["DEVICE"])

            with torch.no_grad():
                rep, _ = self.model(batch)
            y_pred.append(rep.cpu().detach().numpy())

        return np.concatenate(y_pred)