"""
Wrapping the GIN network family based on:

`GINEPLUS GitHub repository <https://github.com/RBrossard/GINEPLUS>`_
`Graph convolutions that can finally model local structure <https://arxiv.org/abs/2011.15069>`_
"""
from olorenchemengine.base_class import log_arguments
from olorenchemengine.gnn import BaseLightningModule
from .operations import GINNetwork

class GINModel(BaseLightningModule):
    @log_arguments
    def __init__(self, task_type = "classification", dataset="molpcba", batch_size=100, hidden=100, lr=0.001,
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