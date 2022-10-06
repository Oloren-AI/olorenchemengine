""" Building blocks for graph neural networks.
"""

from .base_class import *
from .representations import BaseRepresentation, TorchGeometricGraph
from .internal import mock_imports

try:
    import torch
    import torch.nn as nn
except:
    mock_imports(globals(), "torch", "nn")

try:

    from pytorch_lightning import LightningModule
except:
    LightningModule = object


class BaseLightningModule(BaseClass, LightningModule):

    """ BaseLightningModule allows for the use of a Pytorch Lightning module as a BaseClass to be incorporated into the framework.

    Parameters:
        optim (str, optional): parameter describing what kind of optimizer to use. Defaults to "adam".
        input_dimensions (Tuple, optional): Tulpe describing the dimensions of the input data. Defaults to None.
    """

    haspreprocess = False
    hascollate_fn = False

    def __init__(self, optim: str = "adam", input_dimensions: Tuple = None):
        super().__init__()
        self.optim = optim

    def set_task_type(self, task_type, pos_weight=torch.tensor([1])):
        """ Sets the task type for the model.

        Parameters:
            task_type (str): the task type to set the model to.
            pos_weight (torch.tensor, optional): the weight to use for the positive class. Defaults to torch.tensor([1]).

        """
        if task_type == "classification":
            self.loss_fun = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        else:
            self.loss_fun = nn.MSELoss()

    def loss(self, y_pred, y_true):
        """ Calculate the loss for the model.

        Parameters:
            y_pred (torch.tensor): the predictions for the model.
            y_true (torch.tensor): the true labels for the model.

        Returns:
            torch.tensor: the loss for the model."""
        if len(y_true.shape) == 1:
            y_true = torch.unsqueeze(y_true, 1)
        y_available = ~torch.isnan(y_true)
        loss = self.loss_fun(y_pred[y_available], y_true[y_available])
        loss = loss.mean()
        return loss

    def _save(self):
        return ""

    def _load(self, d):
        pass

    def forward(self, batch):
        return self.network(batch)

    ##########################
    # train, val, test, steps
    ##########################
    def training_step(self, batch, batch_idx):
        y_true = batch.y.float()
        y_pred = self.forward(batch)
        loss = self.loss(y_pred, y_true)
        return loss

    def validation_step(self, batch, batch_idx):
        y_true = batch.y.float()
        y_pred = self.forward(batch)
        loss = self.loss(y_pred, y_true)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    ##################
    # Optimizers
    ##################
    def configure_optimizers(self):
        if self.optim == "adamw":
            if hasattr(self, "lr"):
                return torch.optim.AdamW(self.parameters(), lr=self.lr)
            else:
                return torch.optim.AdamW(self.parameters())
        else:
            if hasattr(self, "lr"):
                return torch.optim.Adam(self.parameters(), lr=self.lr)
            else:
                return torch.optim.Adam(self.parameters())


class MultiSequential(nn.Sequential):

    """ Helper class to allow for the use of nn.Sequential with multi-input torch.nn modules"""

    def forward(self, *input):
        """ Forward pass of the module.

        Parameters:
            *input (torch.tensor): the input data.

        Returns:
            torch.tensor: the output data."""
        x = None
        for module in self._modules.values():
            if x is None:
                x = module(*input)
            else:
                x = module(x)
        del input
        return x


class AttentiveFP(BaseLightningModule):

    """ AttentiveFP is a wrapper for the PyTorch Geometric interpretation of https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959.

    Parameters:
        hidden_channels (int, optional): the number of hidden channels to use in the model. Defaults to 4.
        out_channels (int, optional): the number of output channels to use in the model. Defaults to 1.
        num_layers (int, optional): the number of layers to use in the model. Defaults to 1.
        num_timesteps (int, optional): the number of timesteps to use in the model. Defaults to 1.
        dropout (float, optional): the dropout rate to use in the model. Defaults to 0.
        skip_lin (bool, optional): the number of skip connections to use in the model. Defaults to True.
        layer_dims (List, optional): the dimensions to use for each layer in the model. Defaults to [512, 128].
        activation (str, optional): the activation function to use in the model. Defaults to "leakyrelu".
        optim (str, optional): the optimizer to use in the model. Defaults to "adamw"."""

    @log_arguments
    def __init__(
        self,
        hidden_channels=4,
        out_channels=1,
        num_layers=1,
        num_timesteps=1,
        dropout=0,
        skip_lin=True,
        layer_dims=[512, 128],
        activation="leakyrelu",
        optim="adamw",
        **kwargs
    ):
        super().__init__(optim=optim, **kwargs)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.layer_dims = [out_channels] + layer_dims

        self.skip_lin = skip_lin
        if skip_lin:
            self.out_channels = 1
            return

        if activation == "leakyrelu":
            self.activation = nn.LeakyReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        elif activation == "elu":
            self.activation = nn.ELU

    def create(self, dimensions):
        """ Create the model.

        Parameters:
            dimensions (list): the dimensions of the input data.

        """

        from torch_geometric.nn.models import AttentiveFP as AttentiveFP_

        if self.skip_lin:
            self.network = AttentiveFP_(
                dimensions[0],
                self.hidden_channels,
                self.out_channels,
                dimensions[1],
                self.num_layers,
                self.num_timesteps,
                dropout=self.dropout,
            )
        else:
            self.network = MultiSequential(
                AttentiveFP_(
                    dimensions[0],
                    self.hidden_channels,
                    self.out_channels,
                    dimensions[1],
                    self.num_layers,
                    self.num_timesteps,
                    dropout=self.dropout,
                ),
                *[
                    nn.Sequential(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]), self.activation())
                    for i in range(len(self.layer_dims) - 1)
                ],
                nn.Linear(self.layer_dims[-1], 1)
            )

    def forward(self, data):
        if data.x.is_cuda:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        x = data.x.type(dtype)
        edge_index = data.edge_index
        edge_attr = data.edge_attr.type(dtype)
        batch = data.batch

        del dtype
        del data
        return self.network(x, edge_index, edge_attr, batch)


class BaseTorchGeometricModel(BaseModel):

    """ BaseTorchGeometricModel is a base class for models in the PyTorch Geometric framework.

    Parameters:
        network (nn.Module): The network to be used for the model.
        representation (BaseRepresentation, optional): The representation to be used for the model.
            Note that the representation must be compatible with the network,
            so the default, TorchGeometricGraph() is highly reccomended
        epochs (int, optional): The number of epochs to train the model for.
        batch_size (int, optional): The batch size to use for training.
        lr (float, optional): The learning rate to use for training.
        auto_lr_find (bool, optional): Whether to automatically adjust the learning rate.
        pos_weight (str, optional): Strategy for weighting positives in classification
        preinitialized (bool, optional): Whether to the network is pre-initialized.
        log (bool, optional): Log arguments or not. Should only be true if it is not nested. Defaults to True.
    """

    @log_arguments
    def __init__(
        self,
        network: BaseLightningModule,
        representation: BaseRepresentation = TorchGeometricGraph(),
        epochs: int = 1,
        batch_size: int = 16,
        lr: float = 1e-4,
        auto_lr_find: bool = True,
        pos_weight: str = "balanced",
        preinitialized: bool = False,
        log: bool = True,
        **kwargs
    ):
        self.representation = representation
        self.preinitialized = preinitialized
        self.network = network
        self.epochs = epochs
        self.batch_size = batch_size
        self.pos_weight = pos_weight

        from pytorch_lightning import Trainer

        self.trainer = Trainer(
            accelerator="auto", max_epochs=self.epochs, auto_lr_find=auto_lr_find, num_sanity_val_steps=0
        )

        super().__init__(log=False, **kwargs)

    def preprocess(self, X, y, fit=False):
        if self.network.haspreprocess:
            X_ = self.network.preprocess(X, y)
            return X_
        if y is None:
            y = [None] * len(X)

        return self.representation.convert(X, ys=y)

    def _fit(self, X_train, y_train):
        if hasattr(self.network, "create"):
            self.network.create((self.representation.dimensions))
        self.network.train()

        values, counts = np.unique(y_train, return_counts=True)
        if len(values) == 2:
            if self.pos_weight == "balanced":
                self.network.set_task_type(
                    "classification", pos_weight=torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)])
                )
            else:
                self.network.set_task_type("classification")
        else:
            self.network.set_task_type("regression")

        if self.network.hascollate_fn:
            from torch.utils.data import DataLoader as TorchDataLoader

            dataloader = TorchDataLoader(
                X_train, batch_size=self.batch_size, shuffle=True, num_workers=8, collate_fn=self.network.collate_fn
            )
        else:
            from torch_geometric.data import DataLoader as PyGDataLoader

            dataloader = PyGDataLoader(X_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

        self.trainer.fit(self.network, dataloader)

    def _predict(self, X):
        self.network.eval()
        if self.network.hascollate_fn:
            from torch.utils.data import DataLoader as TorchDataLoader

            dataloader = TorchDataLoader(
                X, batch_size=self.batch_size, num_workers=8, collate_fn=self.network.collate_fn
            )
        else:
            from torch_geometric.data import DataLoader as PyGDataLoader

            dataloader = PyGDataLoader(X, batch_size=self.batch_size, num_workers=8)

        predictions = [x for x in self.trainer.predict(self.network, dataloader)]

        predictions = np.concatenate([x.cpu().numpy() for x in predictions])

        return predictions.flatten()

    def _save(self) -> str:
        d = super()._save()
        buffer = io.BytesIO()
        torch.save(self.network, buffer)
        d.update({"save": buffer.getvalue()})
        return d

    def _load(self, d):
        super()._load(d)
        import sys

        self.network = torch.load(io.BytesIO(d["save"]))


from collections import OrderedDict


class TLFromCheckpoint(BaseLightningModule):

    """ TLFromCheckpoint is a base class for transfer-learning from an OlorenVec PyTorch-lightning checkpoint.

    Parameters:
        model_path (str, option): The path to the PyTorch-lightning checkpoint. Ise
            "default" to use a pretrained OlorenVec model.
        map_location (str, optional): The location to map the model to. Default is "cuda:0".
        num_tasks (int, optional): The number of tasks in the OlorenVec model
        dropout (float, optional): The dropout rate to use for the model. Default is 0.1.
        lr (float, optional): The learning rate to use for training. Default is 1e-4.
        optim (str, optional): The optimizer to use for training. Default is "adam".
    """

    @log_arguments
    def __init__(
        self,
        model_path,
        map_location: str = "cuda:0",
        num_tasks: int = 2048,
        dropout: float = 0.1,
        lr: float = 1e-4,
        optim: str = "adam",
        reset: bool = False,
    ):
        self.lr = lr
        super().__init__(optim=optim)

        if model_path == "default":
            path = download_public_file("saves/olorenvec.ckpt")
        else:
            path = model_path

        if not torch.cuda.is_available():
            map_location = torch.device("cpu")
            logging.warn("Overriding map_location to cpu as no GPUs are available.")

        state_dict = OrderedDict(
            [(k.replace("model.", ""), v) for k, v in torch.load(path, map_location=map_location)["state_dict"].items()]
        )

        from olorenchemengine.pyg.gcn import GNN

        self.A = GNN(gnn_type="gcn", num_tasks=num_tasks, num_layer=5, emb_dim=300, drop_ratio=0.5, virtual_node=False)
        if not reset:
            self.A.load_state_dict(state_dict)
        self.B = nn.Sequential(
            OrderedDict(
                [
                    ("dense1", nn.Linear(num_tasks, 64)),
                    ("relu1", nn.LeakyReLU()),
                    ("dropout1", nn.Dropout(dropout)),
                    ("dense2", nn.Linear(64, 1)),
                ]
            )
        )
        self.network = nn.Sequential(OrderedDict([("A", self.A), ("B", self.B)]))

