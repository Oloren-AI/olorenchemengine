from olorenchemengine.base_class import log_arguments, BaseModel
from olorenchemengine.representations import BaseVecRepresentation

class MolCLR(BaseModel):

    model_config = {"num_layer": 5,      # number of graph conv layers
            "emb_dim": 300,                  # embedding dimension in graph conv layers
            "feat_dim": 512,                 # output feature dimention
            "drop_ratio": 0.3,               # dropout ratio
            "pool": "mean"}

    @log_arguments
    def __init__(self, model_type = "ginet", epochs = 100, batch_size = 32,
                 init_lr = 0.0005, init_base_lr = 0.0001, weight_decay = 1e-6,):
        self.model_type = model_type
        
    
    def _fit(self, X, y = None):

        import torch
        from torch import nn
        import torch.nn.functional as F
        from torch.utils.tensorboard import SummaryWriter
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

        if self.setting == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif self.setting == 'regression':
            self.criterion = nn.MSELoss()
        
        if model_type == "ginet":
            from .model import GINet
            self.model = GINet(self.setting, **model_config)
            save_path = download_public_file("MolCLR/pretrained_gin.pth")
        elif model_type == "gcn":
            from .model import GCN
            self.model = GCN(self.setting, **model_config)
            save_path = download_public_file("MolCLR/pretrained_gcn.pth")

        self.model.load_my_state_dict(torch.load(save_path, map_location = CONFIG["MAP_LOCATION"]))
        print(type(self.model))

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_lin' in name:
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(self.model, data, n_iter)
                loss.backward()

                optimizer.step()
                n_iter += 1
                
    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            loss = self.criterion(pred, data.y)

        return loss