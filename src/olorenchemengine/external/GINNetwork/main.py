"""
Wrapping the GIN network family based on:

`GINEPLUS GitHub repository <https://github.com/RBrossard/GINEPLUS>`_
`Graph convolutions that can finally model local structure <https://arxiv.org/abs/2011.15069>`_
"""
from olorenchemengine.base_class import log_arguments
from olorenchemengine.gnn import BaseLightningModule
from .operations import GINNetwork

class GINModel(BaseLightningModule):
    
    """
    GINModel class from `GINEPLUS GitHub repository <https://github.com/RBrossard/GINEPLUS>`.
    
    Parameters:
        task_type (str): Type of task to perform. Options are 'regression' or 'classification'. This
            task_type will automatically be set by BaseTorchGeometricModel and usually
            does not need to be set manually.
        hidden (int): Number of nodes per hidden layer.
        lr (float): Learning rate.
        layers (int): Number of hidden layers.
        dropout (float): Dropout rate.
        virtual_node (bool): Whether to use virtual node.
        conv_type (str): Type of convolution to use. Options are 'gcn', 'gin', 'gin+', 'naivegin+'.
        conv_radius (int): Radius of convolution.
        optim (str): Optimizer to use. Options are 'adam'.
    """
    @log_arguments
    def __init__(self, task_type = "classification", hidden: str = 100, lr: float = 0.001,
                 layers=3, dropout=0.5, virtual_node=False,
                 conv_radius=3, conv_type='gin+',optim="adam",**kwargs):
        assert conv_type in ['gcn', 'gin', 'gin+', 'naivegin+']
        super().__init__(optim=optim)

        # Network
        out_dim = 1
        self.network = GINNetwork(hidden=hidden,
                                         out_dim=out_dim,
                                         layers=layers,
                                         dropout=dropout,
                                         virtual_node=virtual_node,
                                         k=conv_radius,
                                         conv_type=conv_type)