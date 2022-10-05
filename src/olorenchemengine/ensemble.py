""" Ensembling methods to combine `BaseModel`s to create better, combined models.
"""
from .base_class import *
from .representations import *
from .basics import *

def get_oof(self, model, X, y, kf):
    # https://www.kaggle.com/code/arthurtok/introduction-to-ensembling-stacking-in-python/notebook
    oof_train = np.zeros((len(X),))

    for train_index, test_index in kf.split(X):
        x_tr = X.iloc[train_index]
        y_tr = y.iloc[train_index]
        x_te = X.iloc[test_index]

        model.fit(x_tr, y_tr)

        oof_train[test_index] = model.predict(x_te)

    return oof_train

class Averager(BaseModel):
    """ Averager averages the predictions of multiple models for an ensembled prediction.

    Parameters:
        models (list): list of BaseModel objects to be averaged.
        n (int, optional): Number of times to repeat the given models. Defaults to 1.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.Averager(models = [
        oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000),
        oce.SupportVectorMachine(representation = oce.Mol2Vec())
    ])
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """
    @log_arguments
    def __init__(self, models: List[BaseModel], n: int =1, log: bool = True, **kwargs):
        """ BaseStacking constructor

        Args:
            models (List[BaseModel]): List of models to use for the learners to be stacked together.
            stacker_model (BaseModel):a model to use for stacking the models.
            n (int, optional): Number of times to repeat the given models. Defaults to 1.
            log (bool, optional): Whether or not to log the arguments of this constructor,
                should be True only for unnested classes. Defaults to True.
        """

        if issubclass(type(models), BaseModel):
            models = [models]
        self.models = list()
        for i in range(n):
            for model in models:
                self.models.append(model)
        super().__init__(log=False, **kwargs)

    def preprocess(self, X, y, fit = False):
        """ Preprocesses the data for the model.

        Parameters:
            X (pd.DataFrame): Dataframe of features.
            y (pd.DataFrame): Dataframe of labels.

        Returns:
            X (pd.DataFrame): Dataframe of features."""
        return X

    def _fit(self, X, y, valid = None):
        for model in self.models:
            if issubclass(type(model), BaseModel):
                model.fit(X, y)

    def _predict(self, X):
        results = np.zeros(X.shape[0])
        for model in self.models:
            results += model.predict(X)/len(self.models)
        return results

    def _save(self):
        d = super()._save()
        model_save_list = list()
        for i, model in enumerate(self.models):
            model_save_list.append(model._save())
        d.update({"model_save_dict": model_save_list})
        return d

    def _load(self, d):
        super()._load(d)
        for i, model in enumerate(self.models):
            model._load(d["model_save_dict"][i])

class BaseStacking(BaseModel):

    """ BaseStacking stacks the predictions of models for an ensembled prediction.

    Parameters:
        models (List[BaseModel]): list of models to use for the learners to be stacked together.
        stacker_model (BaseModel): a model to use for stacking the models.

    Called only by child classes. Not to be called directly by user.
    """

    @log_arguments
    def __init__(self, models: List[BaseModel], stacker_model: BaseModel, n: int =1, log: bool = True,
        oof = False, nfolds = 5, **kwargs):
        """ BaseStacking constructor

        Args:
            models (List[BaseModel]): List of models to use for the learners to be stacked together.
            stacker_model (BaseModel):a model to use for stacking the models.
            n (int, optional): Number of times to repeat the given models. Defaults to 1.
            log (bool, optional): Whether or not to log the arguments of this constructor,
                should be True only for unnested classes. Defaults to True.
        """

        if issubclass(type(models), BaseModel):
            models = [models]
        self.models = list()
        for i in range(n):
            for model in models:
                self.models.append(model)
        self.stacker_model = stacker_model
        self.oof = oof
        self.nfolds = nfolds
        super().__init__(log=False, **kwargs)

    def preprocess(self, X, y, fit = False):
        return X

    def featurize(self, X):
        """
        Featurizes the data for the model.

        Parameters:
            X (pd.DataFrame): Dataframe of features.
            y (pd.DataFrame): Dataframe of labels.

        Returns:
            data: featurized dataset."""
        data = []
        for i, model in enumerate(self.models):
            if issubclass(type(model), BaseModel):
                data.append(np.squeeze(np.array(model.predict(X))))
            elif issubclass(type(model), BaseRepresentation):
                data_ = model.convert(X)
                for j in range(len(data_[0])):
                    row = np.squeeze(np.array([d[j] for d in data_]))
                    data.append(row)
        data = np.array(data)
        data = np.transpose(np.squeeze(data))
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data

    def _fit(self, X, y, valid = None):
        if not self.oof:
            for model in self.models:
                if issubclass(type(model), BaseModel):
                    model.fit(X, y)
            if valid is None:
                data = self.featurize(X)
                self.stacker_model.fit(data, y)
            else:
                X_valid, y_valid = valid
                data = self.featurize(X_valid)
                self.stacker_model.fit(data, y_valid)
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits = self.nfolds)

            data = []
            for model in self.models:
                if issubclass(type(model), BaseModel):
                    feats = get_oof(model, X, y, kf)
                    data.append(np.squeeze(np.array(feats)))
                elif issubclass(type(model), BaseRepresentation):
                    data_ = model.convert(X)
                    for j in range(len(data_[0])):
                        row = np.squeeze(np.array([d[j] for d in data_]))
                        data.append(row)
            data = np.array(data)
            data = np.transpose(np.squeeze(data))
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            self.stacker_model.fit(data, y)

            for model in self.models:
                if issubclass(type(model), BaseModel):
                    model.fit(X, y)

    def _predict(self, X):
        data = self.featurize(X)
        if hasattr(self.stacker_model, "predict_proba"):
            return self.stacker_model.predict_proba(data)
        else:
            return self.stacker_model.predict(data)

    def _save(self):
        d = super()._save()
        model_save_list = list()
        for i, model in enumerate(self.models):
            model_save_list.append(model._save())
        d.update({"model_save_dict": model_save_list})
        d.update({"stacker_model": self.stacker_model._save()})
        return d

    def _load(self, d):
        super()._load(d)
        for i, model in enumerate(self.models):
            model._load(d["model_save_dict"][i])
        self.stacker_model._load(d["stacker_model"])

class BestStacker(BaseStacking):

    """ BestStacker is a stacking method that uses the best model from a collection of models to make an ensembled prediction.

    Parameters:
        models (List[BaseModel]): list of models to use for the learners to be stacked together.
        n (int, optional): Number of times to repeat the given models. Defaults to 1.
        log (bool, optional): Whether or not to log the arguments of this constructor. Defaults to True.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.BestStacker(models = [
        oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000),
        oce.SupportVectorMachine(representation = oce.Mol2Vec())
    ])
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(self, models: List[BaseModel], n: int = 1, k: int = 1, log: bool = True):
        """ BestStacker constructor

        Args:
            models (List[BaseModel]): List of models to use for the learners to be stacked together.
            n (int, optional): Number of times to repeat the given models. Defaults to 1.
            k (int, optional): Number of "best" models to choose for the stacking. Defaults to 1."""

        super().__init__(models, KBestLinearRegression(k=k), n=n, log=False)

class LinearRegressionStacker(BaseStacking):

    """ LinearRegressionStacker is a stacking method that uses linear regression on the predictions
        from a collection of models to make an ensembled prediction.

        Parameters:
            models (List[BaseModel]): list of models to use for the learners to be stacked together.
            n (int, optional): Number of times to repeat the given models. Defaults to 1.
            log (bool, optional): Whether or not to log the arguments of this constructor. Defaults to True.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.LinearRegressionStacker(models = [
        oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000),
        oce.SupportVectorMachine(representation = oce.Mol2Vec())
    ])
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(self, models: List[BaseModel],  n: int = 1, log: bool = True):
        """ LinearRegressionStacker constructor

        Args:
            models (List[BaseModel]): List of models to use for the learners to be stacked together.
            n (int, optional): Number of times to repeat the given models. Defaults to 1.
            log (bool, optional): Whether or not to log the arguments of this constructor,
        """
        super().__init__(models, LinearRegression(),
        n=n, log=False)


class SKLearnStacker(BaseSKLearnModel):

    """ SKLearnStacker is a stacking method that uses a sklearn-like models to make an ensembled prediction.

    Attributes:
        reg_stack: a sklearn-like model to use for stacking the models for regression tasks.
        class_stack: a sklearn-like model to use for stacking the models for classification tasks.

    Called only by child classes. Not to be called directly by user.
    """

    @log_arguments
    def __init__(self, models: List[BaseModel], regression_stacker_model: BaseEstimator,
        classification_stacker_model: BaseEstimator, n: int = 1, log: bool = True):
        """ SKLearnStacker constructor

        Args:
            models (List[BaseModel]): List of models to use for the learners to be stacked together.
            regression_stacker_model (BaseEstimator): a sklearn-like model to use for stacking the models for regression tasks.
            classification_stacker_model (BaseEstimator): a sklearn-like model to use for stacking the models for classification tasks.
            n (int, optional): Number of times to repeat the given models. Defaults to 1.
            log (bool, optional): Whether or not to log the arguments of this constructor,
                should be True only for unnested classes. Defaults to True."""
        self.reg_stack = BaseStacking(models, regression_stacker_model, n=n)
        self.class_stack = BaseStacking(models, classification_stacker_model, n=n)
        super().__init__(None,
            self.reg_stack,
            self.class_stack,
            log = False)

class RFStacker(SKLearnStacker):

    """ RFStacker is a subclass of SKLearnStacker that uses a random forest models to make an ensembled prediction.

        Parameters:
            models (List[BaseModel]): list of models to use for the learners to be stacked together.
            n_estimators (int, optional): Number of trees in the forest. Defaults to 100.
            n (int, optional): Number of times to repeat the given models. Defaults to 1.
            log (bool, optional): Whether or not to log the arguments of this constructor. Defaults to True.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RFStacker(
        models = [
            oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000),
            oce.SupportVectorMachine(representation = oce.Mol2Vec())],
        n_estimators = 100
    )
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(self, models: List[BaseModel], n_estimators: int = 100,
        n: int = 1, log: bool = True):
        """ RFStacker constructor

        Args:
            models (List[BaseModel]): List of models to use for the learners to be stacked together.
            n_estimators (int, optional): Number of trees in the forest. Defaults to 100.
            n (int, optional): Number of times to repeat the given models. Defaults to 1.
            log (bool, optional): Whether or not to log the arguments of this constructor,
        """
        rf_regressor = RandomForestRegressor(n_estimators = n_estimators,
            max_features="log2",
            min_samples_split=4,
            min_samples_leaf=4)
        rf_classifier = RandomForestClassifier(
            max_features="log2",
            min_samples_split=4,
            min_samples_leaf=4,
            n_estimators = n_estimators,
            n_jobs=-1,
            criterion='entropy',
            class_weight="balanced_subsample"
            )

        super().__init__(models,
            rf_regressor,
            rf_classifier,
            n=n,
            log = False)

class MLPStacker(SKLearnStacker):

    """ MLPStacker is a subclass of SKLearnStacker that uses a multi-layer perceptron model to make an ensembled prediction.

        Parameters:
            models (List[BaseModel]): list of models to use for the learners to be stacked together.
            layer_dims (List[int]): list of layer dimensions for the MLP.
            activation (str, optional): activation function to use for the MLP. Defaults to 'tanh'.
            epochs (int, optional): number of epochs to train the MLP. Defaults to 100.
            batch_size (int, optional): batch size for the MLP. Defaults to 16.
            verbose (int, optional): verbosity level for the MLP. Defaults to 0.
            n (int, optional): Number of times to repeat the given models. Defaults to 1.
            log (bool, optional): Whether or not to log the arguments of this constructor. Defaults to True.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.MLPStacker(
        models = [
            oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000),
            oce.SupportVectorMachine(representation = oce.Mol2Vec())],
        layer_dims = [32, 32],
        activation = 'tanh',
        epochs = 15,
        batch_size = 16,
        verbose = 0,
    )
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(self, models, layer_dims = [2048,512,128], activation = "tanh", epochs = 100, batch_size = 16, verbose = 0, n=1,log = True):

        super().__init__(models,
            TorchMLP(None, layer_dims= layer_dims, activation = activation, epochs=epochs, batch_size=batch_size, verbose = 0),
            TorchMLP(None, layer_dims= layer_dims, activation = activation, epochs=epochs, batch_size=batch_size, verbose = 0),
            n=n,
            log = False)

class BaseBoosting(BaseModel):
    """ BaseBoosting uses models in a gradient boosting fashion to create an ensembled model.

    Parameters:
        models (List[BaseModel]): list of models to use for the learners to be stacked together.
        n (int, optional): Number of times to repeat the given models. Defaults to 1.
        oof (bool, optional): Whether or not to use out-of-fold predictions for the ensembled model. Defaults to False.
        log (bool, optional): Whether or not to log the arguments of this constructor. Defaults to True.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.BaseBoosting(
        models = [
            oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000),
            oce.SupportVectorMachine(representation = oce.Mol2Vec())]
    )
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """


    @log_arguments
    def __init__(self, models: List[BaseModel], n: int =1, oof=False, nfolds = 5, log: bool =True, **kwargs):
        """

        Args:
            models (List[BaseModel]): List of models to use in gradient boosting.
            n (int, optional): Number of times to replicate the inputted models. Defaults to 1.
            oof (bool, optional): Whether to use out of fold predictions for boosting. Defaults to False.
            log (bool, optional): Log arguments or not. Should only be true if it is not nested. Defaults to True.
        """

        if issubclass(type(models), BaseModel):
            models = [models]

        self.models = []

        for i in range(n):
            for model in models:
                self.models.append(model.copy())

        self.oof = oof
        self.nfolds = nfolds

        super().__init__(log = False, **kwargs)

    def _fit(self, X_train, y_train):
        if self.oof:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits = self.nfolds)

        for i, model in enumerate(self.models):
            if self.oof:
                y_pred = get_oof(model, X_train, y_train, kf)
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
                y_pred = np.array(model.predict(X_train)).flatten()
            y_train = y_train - y_pred
    
    def _error_calc(self, residuals):
        initial_residual, other_residuals = residuals[0], residuals[1:]
        y = np.array(([initial_residual]))
        for index in range(len(other_residuals)):
            curr_residual = other_residuals[index]
            total_residual = y[-1] - curr_residual
            y = np.append(y, [total_residual], axis = 0)
        
        stdevs = list(np.std(y, axis = 1))
        return stdevs

    def _waterfall(self, y_data = None, normalize = False):
        self.stdevs = self._error_calc(self.residuals)
        if y_data is not None: 
            naive_mean = np.average(y_data)
            naive_residuals = [(y - naive_mean)/np.std(y_data) for y in y_data] #normalized so they're in the same format as predict's input
            if normalize:    
                naive_residuals = naive_residuals 
            else: 
                naive_residuals = self._unnormalize(np.array(naive_residuals))
            self.residuals.insert(0, naive_residuals)
            self.stdevs.insert(0, np.std(naive_residuals))  
        return self.stdevs

    def _predict(self, X, waterfall = False, normalize = False):
        if waterfall: 
            self.residuals = [] 
        y = np.zeros((len(X)))
        for model in self.models:
            prediction = np.array(model.predict(X)).flatten()
            y = y + prediction
            if waterfall:
                if normalize: 
                    self.residuals.append(prediction)
                else:
                    self.residuals.append(self._unnormalize(prediction))
        if self.setting == "classification":
            y = np.clip(y, 0, 1)
            if waterfall: 
                self.residuals = [np.clip(residual, 0, 1) for residual in self.residuals]
        return y.flatten()

    def _save(self):
        d = super()._save()
        model_save_list = list()
        for i, model in enumerate(self.models):
            model_save_list.append(model._save())
        d.update({"model_save_dict": model_save_list})
        return d

    def _load(self, d):
        super()._load(d)
        for i, model in enumerate(self.models):
            model._load(d["model_save_dict"][i])

class Resample1(BaseModel):
    """ Sample from imbalanced dataset. Take all compounds from smaller class and
    then sample an equal number from the larger class.

    Parameters:
        model (BaseModel): Model to use for classification.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.Resample1(
        oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    )
    model.fit(train['Drug'], train['Y'])
    preds = model.predict(test['Drug'])
    ------------------------------

    Note: may only be used on binary classification data.
    """

    @log_arguments
    def __init__(self, model: BaseModel, log = True):
        self.model = model
        super().__init__(log = False)

    def _fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def fit(self, X_train, y_train):
        y_train = np.array(y_train)
        assert len(np.unique(y_train)) == 2, "Resample1 can only be used on binary classification tasks"
        v1 = np.unique(y_train)[0]
        v2 = np.unique(y_train)[1]
        ix1 = np.where(y_train == v1)[0]
        ix2 = np.where(y_train == v2)[0]
        if len(ix1) > len(ix2):
            ix = np.concatenate((ix2, np.random.choice(ix1, size = len(ix2), replace = False)))
        else:
            ix = np.concatenate((ix1, np.random.choice(ix2, size = len(ix1), replace = False)))
        if isinstance(X_train, list):
            X_train = [X_train[i] for i in ix]
        elif isinstance(X_train, pd.Series) or isinstance(X_train, pd.DataFrame):
            X_train = X_train.iloc[ix]
        else:
            X_train = X_train[ix]
        print(len(X_train))
        super().fit(X_train, y_train[ix])

    def _predict(self, X):
        return self.model.predict(X)

    def _save(self):
        d = {"model_save": self.model._save()}
        return d

    def _load(self, d):
        self.model._load(d["model_save"])

class Resample2(BaseModel):
    """ Sample from imbalanced dataset. Take all compounds from smaller class and
    then sample an equal number from the larger class."""

    @log_arguments
    def __init__(self, model: BaseModel, log = True):
        self.model = model
        super().__init__(log = False)

    def _fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def fit(self, X_train, y_train):
        y_train = np.array(y_train)
        assert len(np.unique(y_train)) == 2, "Resample2 can only be used on binary classification tasks"
        v1 = np.unique(y_train)[0]
        v2 = np.unique(y_train)[1]
        ix1 = np.where(y_train == v1)[0]
        ix2 = np.where(y_train == v2)[0]
        print(len(ix1))
        print(len(ix2))
        if len(ix1) > len(ix2):
            ix = np.concatenate((ix1, ix2, np.random.choice(ix2, size = len(ix1) - len(ix2), replace = True)))
        else:
            ix = np.concatenate((ix1, ix2, np.random.choice(ix1, size = len(ix2) - len(ix1), replace = True)))
        if isinstance(X_train, list):
            X_train = [X_train[i] for i in ix]
        elif isinstance(X_train, pd.Series) or isinstance(X_train, pd.DataFrame):
            X_train = X_train.iloc[ix]
        else:
            X_train = X_train[ix]
        print(len(ix))
        print(len(X_train))
        super().fit(X_train, y_train[ix])

    def _predict(self, X):
        return self.model.predict(X)

    def _save(self):
        d = {"model_save": self.model._save()}
        return d

    def _load(self, d):
        self.model._load(d["model_save"])


class ResampleAdaboost(BaseBoosting):
    """ ResampleAdaBoost performs the Adaboost with sampling weighting being done
    via resampling of the dataset to create an ensembled model.

    Parameters:
        models (List[BaseModel]): list of models to use for the learners to be stacked together.
        n (int, optional): Number of times to repeat the given models. Defaults to 1.
        size (int, optional): Size of the resampled dataset. Defaults to None.
        factor (int, optional): Factor by which to resample the dataset. Defaults to 8.
        equation (str, optional): Equation to use for resampling. Defaults to "abs".
        log (bool, optional): Whether or not to log the arguments of this constructor. Defaults to True.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.ResampleAdaboost(
        models = [
            oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000),
            oce.SupportVectorMachine(representation = oce.Mol2Vec())],
        factor = 8,
        equation = 'abs'
    )
    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(self, models: List[BaseModel], n: int =1, factor: int =8, size: int = None, equation: str ="abs",log: bool=True, **kwargs):
        """_summary_

        Args:
            models (List[BaseModel]): List of models to use in gradient boosting.
            n (int, optional): The number of times to replicate the inputted models. Defaults to 1.
            factor (int, optional): Factor to increase the number of samples resampled by. Mutually exclusive with size. Defaults to 8.
            size (int, optional): _description_. Amount of samples to use in resampling. Mutually exclusive with size. Defaults to None.
            equation (str, optional): What equation to use as loss for resampling weighting, can be "abs", "sqr", or "exp. Defaults to "abs".
            log (bool, optional): Log arguments or not. Should only be true if it is not nested. Defaults to True.
        """
        super().__init__(models, n=n, log=False, **kwargs)
        self.size = size
        self.equation = equation
        self.factor = factor

    def _fit(self, X_train, y_train):

        if self.size is None:
            size = int(len(X_train) * self.factor)
        else:
            size = self.size
        X_train = pd.DataFrame(X_train)
        y_train_df = pd.DataFrame(y_train)
        y_train = np.array(y_train)

        w = np.array([1/len(X_train)] * len(X_train))
        self.betas = list()

        i = 0
        for model in self.models:
            i+=1

            p = w/np.sum(w)
            idxs = np.random.choice(len(X_train), size=size,replace=True, p=p)
            X_train_temp = X_train.iloc[idxs, :]
            y_train_temp = y_train_df.iloc[idxs, :].to_numpy().flatten()
            model.fit(X_train_temp, y_train_temp)
            y_pred = np.squeeze(np.array(model.predict(X_train)).flatten())
            res = np.abs(y_train - y_pred)

            D = np.max(res)
            if self.equation == "abs":
                loss = res/D
            elif self.equation == "sqr":
                loss = res**2/D**2
            else:
                loss = 1 - np.exp(-res/D)

            loss = loss/np.max(loss)
            avg_loss = np.sum(loss*p)
            beta = avg_loss/(1-avg_loss)

            w = w * np.power(beta, 1-loss)
            self.betas.append(beta)

        self.betas = np.array(self.betas)

    def _predict(self, X):
        ys = list()
        for model in self.models:
            ys.append(model.predict(X))
        ys = np.array(ys)

        predictions = list()

        predictions = np.dot(ys.transpose(), np.log10(1/self.betas))/np.sum(np.log10(1/self.betas))

        return np.array(predictions)

    def _save(self):
        d = super()._save()
        d.update({"betas": self.betas})
        return d

    def _load(self, d):
        super()._load(d)
        self.betas = d["betas"]