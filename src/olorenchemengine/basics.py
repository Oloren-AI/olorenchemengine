""" Machine learning algorithms for use with molecular vector representations and features from experimental data.
"""

from tempfile import NamedTemporaryFile
import joblib
import io
from sklearn.model_selection import RandomizedSearchCV

import olorenchemengine as oce
from olorenchemengine.representations import BaseCompoundVecRepresentation, BaseVecRepresentation
from .base_class import *
from rdkit import Chem

import pandas as pd
import numpy as np
import scipy
import json

from pandas.api.types import is_numeric_dtype

class RandomizedSearchCVModel(RandomizedSearchCV, BaseEstimator):

    """Wrapper class for RandomizedSearchCV"""
    def __init__(self, *args, **kwargs):
        self.obj = None
        super().__init__(*args, **kwargs)

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self.obj = self.best_estimator_

    def predict(self, *args, **kwargs):
        if self.obj is None:
            raise ValueError("Model not fitted")
        return self.obj.predict(*args, **kwargs)

class RandomForestRegressor(BaseEstimator):

    """Wrapper for sklearn RandomForestRegressor"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.ensemble import RandomForestRegressor

        self.obj = RandomForestRegressor(*args, **kwargs)


class RandomForestClassifier(BaseEstimator):

    """Wrapper for sklearn RandomForestClassifier"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.ensemble import RandomForestClassifier

        self.obj = RandomForestClassifier(*args, **kwargs)

    def predict(self, X):
        return [x[1] for x in self.obj.predict_proba(X)]


class SVC(BaseEstimator):

    """Wrapper for sklearn SVC"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.obj = make_pipeline(StandardScaler(), SVC(*args, **kwargs))


class SVR(BaseEstimator):

    """Wrapper for sklearn SVR"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.obj = make_pipeline(StandardScaler(), SVR(*args, **kwargs))

class KBestLinearRegression(BaseEstimator):

    """Selects the K-best features and then does linear regression"""

    @log_arguments
    def __init__(self, k=1, *args, **lwargs):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        from sklearn.feature_selection import SelectKBest, mutual_info_regression

        self.obj = Pipeline(
            [
                ("KBest", SelectKBest(mutual_info_regression, k=k)),
                ("LR", LinearRegression()),
            ]
        )

class LogisticRegression(BaseEstimator):

    """Wrapper for sklearn LogisticRegression"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.linear_model import LogisticRegression

        self.obj = LogisticRegression(*args, **kwargs)

class RandomForestModel(BaseSKLearnModel):
    """ Random forest model

        Parameters:
            n_estimators (int): The number of trees in the forest.
            max_depth (int): The maximum depth of the tree.
            max_features (int): The number of features to consider when looking for the best split.
            bootstrap (bool): Whether bootstrap samples are used when building trees.
            criterion (str): The function to measure the quality of a split.
            class_weight (str): Dict or 'balanced', defaults to None.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(
        self,
        representation,
        max_features="log2",
        max_depth=None,
        criterion="entropy",
        class_weight=None,
        bootstrap=True,
        n_estimators=100,
        **kwargs
    ):
        if not class_weight is None:
            try:
                class_weight = {int(k):float(v) for k,v in json.loads(class_weight).items()}
            except:
                 class_weight = None
        regressor = RandomForestRegressor(n_estimators = n_estimators,
            max_features=max_features,
            min_samples_split=4,
            min_samples_leaf=4,
            max_depth = max_depth,
            n_jobs=-1
        )
        classifier = RandomForestClassifier(
            max_features=max_features,
            min_samples_split=4,
            min_samples_leaf=4,
            n_estimators=n_estimators,
            n_jobs=-1,
            criterion=criterion,
            bootstrap=bootstrap,
            max_depth=max_depth,
            class_weight=class_weight,
        )
        super().__init__(representation, regressor, classifier, log=False, **kwargs)

class AutoRandomForestModel(BaseSKLearnModel, BaseObject):
    """ RandomForestModel where parameters have automatically tuned hyperparameters

        Parameters:
            representation (str): The representation to use for the model.
            n_iter (int): The number of iterations to run the hyperparameter tuning.
            scoring (str): The scoring metric to use for the hyperparameter tuning.
            verbose (int): The verbosity level of the hyperparameter tuning.
            cv (int): The number of folds to use for the hyperparameter tuning.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.AutoRandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048))
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(
        self, representation, n_iter=100, scoring=None, verbose=2, cv=5, **kwargs
    ):
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        regressor = self.autofit(
            RandomForestRegressor(), n_iter, cv, scoring, verbose
        )
        classifier = self.autofit(
            RandomForestClassifier(), n_iter, cv, scoring, verbose
        )
        super().__init__(representation, regressor, classifier, log=False, **kwargs)

    def autofit(self, model, n_iter, cv, scoring, verbose):
        """Takes an XGBoost model and replaces its fit function with one that
        automatically tunes the model hyperparameters

        Parameters:
            model (sklearn model): The model to be tuned
            n_iter (int): Number of iterations to run the hyperparameter tuning
            cv (int): Number of folds to use for cross-validation
            scoring (str): Scoring metric to use for cross-validation
            verbose (int): Verbosity level

        Returns:
            model (sklearn model): The tuned model
    """

        params = {
            "n_estimators": [50, 100, 200, 500, 1000],
            "max_depth": [3, 4, 5, 6, 7],
            "min_samples_split": [2, 3, 4, 5, 6],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_features": ["sqrt", "log2", "auto"],
            "min_impurity_decrease": [0.0, 0.1, 0.2, 0.3, 0.4],
            "max_samples": [0.5, 0.75, 1.0],
        }

        return RandomizedSearchCVModel(
            estimator=model,
            param_distributions=params,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
        )
class BaseMLPClassifier(BaseEstimator):

    """Wrapper for sklearn MLP"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.neural_network import MLPClassifier

        self.obj = MLPClassifier(*args, **kwargs)

class BaseMLPRegressor(BaseEstimator):

    """Wrapper for sklearn MLP"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.neural_network import MLPRegressor

        self.obj = MLPRegressor(*args, **kwargs)

class SklearnMLP(BaseSKLearnModel):
    """ MLP Model based on sklearn implementation

        Parameters:
            representation (BaseVecRepresentation): The representation to use for the model.
            hidden_layer_sizes (list): The number of neurons in each hidden layer.
            activation (str): The activation function to use.
            solver (str): The solver to use.
            alpha (float): Learning rate.
            batch_size (int): The size of the minibatches for stochastic optimizers.
            learning_rate (str): The learning rate schedule.
            learning_rate_init (float): The initial learning rate for the solver.
            power_t (float): The exponent for inverse scaling learning rate.
            max_iter (int): Maximum number of iterations.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.SklearnMLP(representation = oce.MorganVecRepresentation(radius=2, nbits=2048))
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """
    @log_arguments
    def __init__(self, representation,
        hidden_layer_sizes=[100], activation="relu", solver="adam",
        alpha=0.0001, batch_size=32, learning_rate="constant",
        learning_rate_init=0.001, power_t=0.5, max_iter=200, log=True, **kwargs):

        regressor = BaseMLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, activation=activation, learning_rate=learning_rate,
            solver=solver, alpha=alpha, batch_size=batch_size, learning_rate_init=learning_rate_init,
            power_t=power_t, max_iter=max_iter)
        classifier = BaseMLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, activation=activation, learning_rate=learning_rate,
            solver=solver, alpha=alpha, batch_size=batch_size, learning_rate_init=learning_rate_init,
            power_t=power_t, max_iter=max_iter)
        super().__init__(representation, regressor, classifier, log=False, **kwargs)

try:
    import pytorch_lightning as pl
except ImportError:
    oce.mock_imports(globals(), "pl")

try:
    import torch.nn as nn
    import torch
except ImportError:
    oce.mock_imports(globals(), "nn", "torch")

class plMLP(pl.LightningModule):

    def __init__(self, hidden_layer_sizes, norm_layer, activation_layer, setting):
        super().__init__()
        hidden_layer_sizes = hidden_layer_sizes + [1]
        layers = []
        for i in range(len(hidden_layer_sizes)-1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            if not activation_layer is None and i < len(hidden_layer_sizes)-2:
                layers.append(activation_layer())
            if not norm_layer is None and i < len(hidden_layer_sizes)-2:
                layers.append(norm_layer())
        self.layers = nn.Sequential(
          *layers
        )
        if setting == "classification":
            self.layers.append(nn.Sigmoid())
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

class TorchMLP(BaseModel):
    """ MLP Model based on torch implementation

        Parameters:
            representation (BaseVecRepresentation): The representation to use for the model.
            hidden_layer_sizes (list): The number of neurons in each hidden layer.
            norm_layer (str): The normalization to use for a final normalization layer. Default None.
            activation_layer (str): The activation function to use for a final activation layer. Default None.
            dropout (float): The dropout rate to use for the model.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.TorchMLP(representation = oce.MorganVecRepresentation(radius=2, nbits=2048))
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(self, representation, hidden_layer_sizes=[100],
            norm_layer: str = None, activation_layer: str = None, dropout: float = 0.0,
            log=True, **kwargs):
        self.representation = representation
        self.hidden_layer_sizes = hidden_layer_sizes
        import torch.nn as nn
        if activation_layer == "leakyrelu":
            self.activation_layer = nn.LeakyReLU
        elif activation_layer == "relu":
            self.activation_layer = nn.ReLU
        elif activation_layer == "tanh":
            self.activation_layer = nn.Tanh
        elif activation_layer == "sigmoid":
            self.activation_layer = nn.Sigmoid
        else:
            self.activation_layer = None

        if norm_layer == "batchnorm":
            self.norm_layer = nn.BatchNorm1d
        elif norm_layer == "layernorm":
            self.norm_layer = nn.LayerNorm
        else:
            self.norm_layer = None

        self.dropout = dropout
        super().__init__(representation, log=False)

    def preprocess(self, X, y, fit=False):
        if self.representation is None:
            return X
        else:
            return self.representation.convert(X, fit=fit)

    def _fit(self, X, y):
        import pytorch_lightning as pl
        import torch
        import torch.nn as nn

        self.network = plMLP(hidden_layer_sizes=[len(X[0])] + self.hidden_layer_sizes,
            norm_layer=self.norm_layer,
            activation_layer=self.activation_layer,
            setting = self.setting)

        from torch.utils.data import TensorDataset, DataLoader
        y = np.array(y.tolist())
        device = torch.device(oce.CONFIG["MAP_LOCATION"])
        dataset = TensorDataset(torch.from_numpy(X).float().to(device), torch.from_numpy(y).float().to(device))
        loader = DataLoader(dataset, batch_size=32)

        self.trainer = pl.Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            max_epochs=10,
        )

        self.trainer.fit(self.network, loader)

    def _predict(self, X):
        import torch
        # from torch.utils.data import TensorDataset, DataLoader
        device = torch.device(oce.CONFIG["MAP_LOCATION"])
        # dataset = TensorDataset(torch.from_numpy(X).float().to(device))
        # loader = DataLoader(dataset, batch_size=1)
        # return self.trainer.predict(self.network, loader)
        self.network.to(device)
        return self.network(torch.from_numpy(X).float().to(device)).detach().cpu().numpy()

    def _save(self) -> str:
        d = super()._save()
        buffer = io.BytesIO()
        import torch
        torch.save(self.network, buffer)
        d.update({"save": buffer.getvalue()})
        return d

    def _load(self, d):
        super()._load(d)
        import torch
        self.network = torch.load(io.BytesIO(d["save"]))

class XGBoostModel(BaseSKLearnModel, BaseObject):
    """ XGBoost model

        Parameters:
            representation (str): The representation to use for the model
            n_iter (int): Number of iterations to run the hyperparameter tuning
            cv (int): Number of folds to use for cross-validation
            scoring (str): Scoring metric to use for cross-validation
            verbose (int): Verbosity level

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.XGBoostModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048))
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(
        self, representation, n_estimators=2000, max_depth=7, subsample=0.5,max_leaves=5,
            learning_rate=0.05, colsample_bytree=0.8, **kwargs
    ):
        from xgboost import XGBRegressor, XGBClassifier

        regressor = XGBRegressor(tree_method="gpu_hist", n_estimators=n_estimators,
            max_depth=max_depth, max_leaves=max_leaves, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree)

        classifier = XGBClassifier(tree_method="gpu_hist", n_estimators=n_estimators,
            max_depth=max_depth, max_leaves=max_leaves, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree)

        super().__init__(representation, regressor, classifier, log=False, **kwargs)

class ZWK_XGBoostModel(BaseSKLearnModel, BaseObject):
    """ XGBoost model from https://github.com/smu-tao-group/ADMET_XGBoost

        Parameters:
            representation (str): The representation to use for the model.
            n_iter (int): The number of iterations to run the hyperparameter tuning.
            scoring (str): The scoring metric to use for the hyperparameter tuning.
            verbose (int): The verbosity level of the hyperparameter tuning.
            cv (int): The number of folds to use for the hyperparameter tuning.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.ZWK_XGBoostModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048))
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(
        self, representation, n_iter=100, scoring=None, verbose=2, cv=5, **kwargs
    ):
        from xgboost import XGBRegressor, XGBClassifier

        if scoring == 'spearman':
            from scipy.stats import spearmanr
            from sklearn.metrics import make_scorer
            def spearman_loss_func(y_true, y_pred):
                """spearman metric
                """
                return spearmanr(y_true, y_pred)[0]
            spearman = make_scorer(spearman_loss_func)
            scoring = spearman

        regressor = self.autofit(
            XGBRegressor(tree_method="gpu_hist"), n_iter, cv, scoring, verbose
        )
        classifier = self.autofit(
            XGBClassifier(tree_method="gpu_hist"), n_iter, cv, scoring, verbose
        )

        super().__init__(representation, regressor, classifier, log=False, **kwargs)

    def autofit(self, model, n_iter, cv, scoring, verbose):
        """Takes an XGBoost model and replaces its fit function with one that
        automatically tunes the model hyperparameters

        Parameters:
            model (sklearn model): The model to be tuned
            n_iter (int): Number of iterations to run the hyperparameter tuning
            cv (int): Number of folds to use for cross-validation
            scoring (str): Scoring metric to use for cross-validation
            verbose (int): Verbosity level

        Returns:
            model (sklearn model): The tuned model
        """

        params = {
            "n_estimators": [50, 100, 200, 500, 1000],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.001, 0.005, 0.01, 0.1],
            "subsample": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 1, 5, 10, 20, 25, 30],
            "reg_lambda": [0, 0.1, 1, 5, 10, 20, 25, 30],
            "min_child_weight": [1, 2, 3],
        }

        return RandomizedSearchCVModel(
            estimator=model,
            param_distributions=params,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
        )

class SupportVectorMachine(BaseSKLearnModel):

    """ Support vector machine

        Parameters:
            representations (str): The representation to use for the model.
            kernel (str): The kernel to use for the model.
            C (float): The C parameter for the model.
            gamma (float): The gamma parameter for the model.
            coef0 (float): The coef0 parameter for the model.
            cache_size (int): The cache size parameter for the model.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.SupportVectorMachine(representation = oce.MorganVecRepresentation(radius=2, nbits=2048))
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(
        self,
        representation,
        C=0.8,
        kernel="rbf",
        gamma="scale",
        coef0=0,
        cache_size=500,
        **kwargs
    ):
        regressor = SVR(
            C=C,
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            cache_size=cache_size,
        )
        classifier = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            cache_size=cache_size,
        )
        super().__init__(representation, regressor, classifier, log=False, **kwargs)


class KNeighborsRegressor(BaseEstimator):

    """Wrapper for sklearn KNeighborsRegressor"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.neighbors import KNeighborsRegressor

        self.obj = KNeighborsRegressor(*args, **kwargs)


class KNeighborsClassifier(BaseEstimator):

    """Wrapper for sklearn KNeighborsClassifier"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.neighbors import KNeighborsClassifier

        self.obj = KNeighborsClassifier(*args, **kwargs)

    def predict(self, X):
        return [x[1] for x in self.obj.predict_proba(X)]


class KNN(BaseSKLearnModel):

    """ KNN model

        Parameters:
            representations (str): The representation to use for the model.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.KNN(representation = oce.MorganVecRepresentation(radius=2, nbits=2048))
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(self, representation, **kwargs):
        regressor = KNeighborsRegressor(**kwargs)
        classifier = KNeighborsClassifier(**kwargs)
        super().__init__(representation, regressor, classifier, log=False, **kwargs)


class MLP(BaseSKLearnModel):

    """ MLP model

        Parameters:
            representation (BaseVecRepresentation): The representation to use for the model.
            hidden_layer_sizes (List[int]): The hidden layer sizes to use for the model.
            activation (str): The activation function to use for the model.
            epochs (int): The number of epochs to use for the model.
            batch_size (int): The batch size to use for the model.
            lr (float): The learning rate to use for the model.
            dropout (float): The dropout rate to use for the model.
            kernel_regularizer (float): The kernel regularizer to use for the model.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.MLP(representation = oce.MorganVecRepresentation(radius=2, nbits=2048))
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
     """

    def preprocess(self, X, y, fit = False):
        if self.representation is None:
            return np.array(X)
        else:
            return np.array(self.representation.convert(X, fit = fit))
    # specify parameters
    @log_arguments
    def __init__(
        self,
        representation: BaseVecRepresentation,
        layer_dims=[2048, 512, 128],
        activation="tanh",
        epochs=100,
        batch_size=16,
        lr=0.0005,
        dropout=0,
        kernel_regularizer=1e-4,
        **kwargs
    ):

        super().__init__(
            representation,
            TorchMLP(
                None,
                layer_dims=layer_dims,
                activation=activation,
                epochs=epochs,
                batch_size=batch_size,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                lr=lr,
                task="regression",
            ),
            TorchMLP(
                None,
                layer_dims=layer_dims,
                activation=activation,
                epochs=epochs,
                batch_size=batch_size,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                lr=lr,
                task="classification",
            ),
            log=False,
            **kwargs
        )


class GuessingRegression(BaseModel):

    """Guessing model for regression, used to infer non-linear relationships.

    This model tries different non-linear relationships between each feature and
    property, selecting the best such relationship for each feature. Then the
    features are transformed then either aggregated using linear regression or
    averages to obtain the final prediction for the property. This is best used
    for using experimental features with direct relationships to the properties.

    Attributes:
        transformations (List[Callable]): List of transformations to apply to
            the data i.e. possible relationships between feature and property.
        state (Dict[str, Tuple[int, float, float]]): State of the model,
            best transformation for each feature.
        reg (str): Method to use for combining features,
            either "lr" linear regression or "avg" average."""

    @log_arguments
    def __init__(self, config="full", reg="lr", **kwargs):

        safe_log = lambda x: np.log10(np.abs(x + 1e-3))
        safe_divide = lambda x1, x2: x1 / (np.abs(x2) + 1e-3)

        self.transformations = [
            (
                lambda x: safe_log(x),
                lambda y: safe_log(y),
                lambda A, B: lambda x: np.exp(B + A * np.log(x)),
                "x: log x, y: log y",
            ),
            (lambda x: x, lambda y: y, lambda A, B: lambda x: A * x + B, "x: x, y: y"),
            (
                lambda x: x,
                lambda y: safe_log(y),
                lambda A, B: lambda x: np.exp(B + A * x),
                "x: x, y: log y",
            ),
            (
                lambda x: safe_log(x),
                lambda y: y,
                lambda A, B: lambda x: B * safe_log(x) + A,
                "x: log x, y: y",
            ),
            (
                lambda x: safe_divide(1, x),
                lambda y: y,
                lambda A, B: lambda x: safe_divide((A + B * x), x),
                "x: 1/x, y: y",
            ),
            (
                lambda x: x,
                lambda y: safe_divide(1, y),
                lambda A, B: lambda x: safe_divide(1, (B + A * x)),
                "x: x, y: 1/y",
            ),
            (
                lambda x: safe_divide(1, x),
                lambda y: safe_divide(1, y),
                lambda A, B: lambda x: safe_divide(1, (B + A * x)),
                "x: 1/x, y: 1/y",
            ),
        ]
        self.state = {}
        if reg == "lr":
            self.reg = LinearRegression()
        if reg == "avg":
            self.reg = None
        super().__init__(log=False, **kwargs)

    def preprocess(self, X, y, fit = False):
        """This method is used to preprocess the data before training.

        Parameters:
            X (np.ndarray): List of lists of features.
            y (np.array): List of properties.

        Returns:
            np.ndarray: List of lists of features.
            Shape: (n_samples, n_features)"""
        return X.copy()

    def linearize(self, X):
        """Linearize the data--apply the best transformation for each feature.

        Parameters:
            X (np.ndarray): List of lists of features.

        Returns:
            np.ndarray: List of lists of features.
            Shape: (n_samples, n_features)"""
        out = pd.DataFrame()
        for c in self.state.keys():
            trans = self.transformations[self.state[c][0]][2]
            out[c] = trans(self.state[c][1], self.state[c][2])(X[c])
        return out

    def _fit(self, X_train, y_train):
        assert isinstance(X_train, pd.DataFrame)
        best = ""
        for c in X_train.columns.tolist():
            if is_numeric_dtype(X_train[c]):
                df2 = pd.DataFrame({"x": X_train[c], "y": y_train}).dropna()
                r2max = -1
                for i, trans in enumerate(self.transformations):
                    t0 = trans[0](df2["x"])
                    t1 = trans[1](df2["y"])
                    A, B, r, p, se = scipy.stats.linregress(t0, t1)
                    if r ** 2 > r2max:
                        r2max = r ** 2
                        best = trans[3]
                        self.state[c] = (i, A, B, r2max)
                print(c)
                print(best)
                print(r2max)

        X_train = self.linearize(X_train)
        X_train["y"] = y_train
        X_train = X_train.dropna()
        if not self.reg is None:
            self.reg.fit(X_train.loc[:, X_train.columns != "y"], X_train["y"])

    def _predict(self, X):
        X = self.linearize(X)
        if self.reg == "lr":
            return self.reg.predict(X)
        else:
            return X.mean(axis=1)

    def _save(self):
        d = super()._save()
        if not self.reg is None:
            d.update({"reg": self.reg._save(), "state": self.state})
        else:
            d.update({"state": self.state})
        return d

    def _load(self, d):
        super()._load(d)
        if not self.reg is None:
            self.reg._load(d["reg"])
        self.state = d["state"]


class FeaturesClassification(BaseModel):

    """ FeaturesClassification uses machine learning models to classify features
        based on their experimental data

        Attributes:
            obj (sklearn.base.BaseEstimator): Machine learning model to use.

        Parameters:
            config (str): Configuration to use for the model."""

    @log_arguments
    def __init__(self, config="lineardiscriminant"):
        """FeaturesClassfication constructor

        Args:
            config (str, optional): Type of machine learning model to use to
                classify features. Options are "lineardiscriminant",
                "gaussiannb", "quadraticdiscriminant", "knearestneighbors".
                Defaults to "lineardiscriminant".
        """
        self.numeric_cols = []
        if config == "lineardiscriminant":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            self.obj = LinearDiscriminantAnalysis()
        elif config == "gaussiannb":
            from sklearn.naive_bayes import GaussianNB

            self.obj = GaussianNB()
        elif config == "quadraticdiscriminant":
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

            self.obj = QuadraticDiscriminantAnalysis()
        elif config == "knearestneighbors":
            from sklearn.neighbors import KNeighborsClassifier

            self.obj = KNeighborsClassifier()

    def _fit(self, X_train, y_train):
        self.numeric_cols = [
            c for c in X_train.columns.tolist() if is_numeric_dtype(X_train[c])
        ]
        X_train = X_train[self.numeric_cols]
        self.obj.fit(X_train, y_train)

    def _predict(self, X):
        X = X[self.numeric_cols]
        return self.obj.predict(X).astype(float)

    def _save(self):
        b = io.BytesIO()
        joblib.dump(self.obj, b)
        return {"obj": b.getvalue(), "numeric_cols": self.numeric_cols}

    def _load(self, d):
        super()._load(d)
        self.obj = joblib.load(io.BytesIO(d["obj"]))
        self.numeric_cols = d["numeric_cols"]
