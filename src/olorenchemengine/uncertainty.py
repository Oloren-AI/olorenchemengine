from cmath import exp
from tkinter import N
import olorenchemengine as oce

from .base_class import *
from .representations import *
from .dataset import *
from .basics import *

import pandas as pd
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from rdkit.Chem import AllChem


class BaseFingerprintModel(BaseErrorModel):
    """ BaseFingerprintModel is the base class for error models that require the
        computation of Morgan Fingerprints.
    """
    def build(
        self,
        model: BaseModel,
        X: Union[pd.DataFrame, np.ndarray, list, pd.Series],
        y: Union[np.ndarray, list, pd.Series],
    ):
        super().build(model, X, y)
        if isinstance(self.X_train, pd.DataFrame):
            self.X_train.columns = ["SMILES"]
        smiles = SMILESRepresentation().convert(self.X_train)
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        self.train_fps = [AllChem.GetMorganFingerprint(mol, 2) for mol in mols]

    def _save(self) -> dict:
        d = super()._save()
        if hasattr(self, "train_fps"):
            d.update({"train_fps": self.train_fps})
        return d

    def _load(self, d) -> None:
        super()._load(d)
        if "train_fps" in d.keys():
            self.train_fps = d["train_fps"]

class SDC(BaseFingerprintModel):
    """ SDC is an error model that predicts error bars based on the Sum of
        Distance-weighted Contributions: `Molecular Similarity-Based Domain
        Applicability Metric Efficiently Identifies Out-of-Domain Compounds
        <http://dx.doi.org/10.1021/acs.jcim.8b00597>`_

        Parameters:
            a (int or float, optional): Value of a in the SDC formula. Default 3.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.SDC())
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    @log_arguments
    def __init__(self, a=3):
        self.a = a

    def calculate(self, X, y_pred):
        def sdc(smi):
            mol = Chem.MolFromSmiles(smi)
            ref_fp = AllChem.GetMorganFingerprint(mol, 2)
            TD = 1 - np.array(BulkTanimotoSimilarity(ref_fp, self.train_fps))
            return np.sum(np.exp(-self.a * TD / (1 - TD)))

        return np.array([sdc(smi) for _, smi in tqdm(np.ndenumerate(X))])


class SDCwRMSD1(SDC):
    """ SDCwRMSD1 is an error model that calculates the root-mean-square
        difference between the predicted activity of the target molecule and
        the observed activities of all training molecules, weighted by SDC.

        Parameters:
            a (int or float, optional): Value of a in the SDC formula. Default 3.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.SDCwRMSD1())
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """
    def calculate(self, X, y_pred):
        def wrmsd(smi, pred):
            mol = Chem.MolFromSmiles(smi)
            ref_fp = AllChem.GetMorganFingerprint(mol, 2)
            TD = 1 - np.array(BulkTanimotoSimilarity(ref_fp, self.train_fps))
            sdc = np.exp(-self.a * TD / (1 - TD))
            return np.sqrt(np.dot(sdc, np.abs(self.y_train - pred)) / np.sum(sdc))

        X = np.array(X).flatten()
        return np.array([wrmsd(smi, y_pred[i]) for i, smi in tqdm(np.ndenumerate(X))])

class SDCwRMSD2(SDC):
    """ SDCwRMSD2 is an error model that calculates the root-mean-square
        difference between the predicted and observed activities of all
        training molecules, weighted by SDC.

        Parameters:
            a (int or float, optional): Value of a in the SDC formula. Default 3.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.SDCwRMSD2())
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """
    def calculate(self, X, y_pred):
        residuals = np.abs(self.y_train - self.y_pred_train)
        def wrmsd(smi):
            mol = Chem.MolFromSmiles(smi)
            ref_fp = AllChem.GetMorganFingerprint(mol, 2)
            TD = 1 - np.array(BulkTanimotoSimilarity(ref_fp, self.train_fps))
            sdc = np.exp(-self.a * TD / (1 - TD))
            return np.sqrt(np.dot(sdc, residuals) / np.sum(sdc) )

        return np.array([wrmsd(smi) for _, smi in tqdm(np.ndenumerate(X))])

class kNNSimilarity(BaseFingerprintModel):
    """ kNNSimilarity is an error model that calculates mean Tanimoto similarity
        between the target molecule and the k most similar training molecules
        using a Morgan Fingerprint with a radius of 2 bonds.

        Parameters:
            k (int, optional): Number of nearest neighbors to consider. Default 5.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.kNNSimilarity())
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """
    @log_arguments
    def __init__(self, k=5):
        self.k = k

    def calculate(self, X, y_pred):
        def mean_sim(smi):
            mol = Chem.MolFromSmiles(smi)
            ref_fp = AllChem.GetMorganFingerprint(mol, 2)
            similarity = np.array(BulkTanimotoSimilarity(ref_fp, self.train_fps))
            return np.mean(sorted(similarity)[-self.k:])

        return np.array([mean_sim(smi) for _, smi in tqdm(np.ndenumerate(X))])

class kNNwRMSD1(kNNSimilarity):
    """ kNNwRMSD1 is an error model that calculates the root-mean-square
        difference between the predicted activity of the target molecule and
        the observed activities of the k most similar training molecules,
        weighted by their similarity.

        Parameters:
            k (int, optional): Number of nearest neighbors to consider. Default 5.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.kNNwRMSD1())
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """
    def calculate(self, X, y_pred):
        def wrmsd(smi, pred):
            mol = Chem.MolFromSmiles(smi)
            ref_fp = AllChem.GetMorganFingerprint(mol, 2)
            similarity = np.array(BulkTanimotoSimilarity(ref_fp, self.train_fps))
            diff = np.abs(self.y_train - pred)
            ind = np.argsort(similarity)[-self.k:]
            return np.sqrt(np.dot(similarity[ind], diff[ind]) / np.sum(similarity[ind]))

        X = np.array(X).flatten()
        return np.array([wrmsd(smi, y_pred[i]) for i, smi in tqdm(np.ndenumerate(X))])

class kNNwRMSD2(kNNSimilarity):
    """ kNNwRMSD2 is an error model that calculates the root-mean-square
        difference between the predicted and observed activities of the k most
        similar training molecules to the target molecule, weighted by their
        similarity.

        Parameters:
            k (int, optional): Number of nearest neighbors to consider. Default 5.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.kNNwRMSD2())
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """
    def calculate(self, X, y_pred):
        residuals = np.abs(self.y_train - self.y_pred_train)
        def wrmsd(smi):
            mol = Chem.MolFromSmiles(smi)
            ref_fp = AllChem.GetMorganFingerprint(mol, 2)
            similarity = np.array(BulkTanimotoSimilarity(ref_fp, self.train_fps))
            ind = np.argsort(similarity)[-self.k:]
            return np.sqrt(np.dot(similarity[ind], residuals[ind]) / np.sum(similarity[ind]))

        return np.array([wrmsd(smi) for _, smi in tqdm(np.ndenumerate(X))])

class PREDICTED(BaseErrorModel):
    """ PREDICTED is an error model that predicts error bars based on only the
        predicted value of a molecule. It is best used as part of an aggregate
        error model rather than by itself.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.RandomForestErrorModel([oce.SDC(), oce.PREDICTED()])
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    @log_arguments
    def __init__(self):
        pass

    def calculate(self, X, y_pred):
        return y_pred

class BaseADAN(BaseErrorModel):
    """ BaseADAN is an error model that predicts error bars based on one or
        multiple ADAN categories: `Applicability Domain Analysis (ADAN): A
        Robust Method for Assessing the Reliability of Drug Property Predictions
        <https://doi.org/10.1021/ci500172z>`_

        Parameters:
            criteria (Union[str, list, set]): the ADAN criteria to be considered

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.BaseADAN("E_raw"))
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    @log_arguments
    def __init__(self, criteria: Union[str, list, set]):
        if criteria == "Category":
            criteria = {"A","B","C","D","E","F"}
        elif isinstance(criteria, list):
            criteria = set(criteria)
        elif isinstance(criteria, str):
            criteria = {criteria}
        self.criteria = criteria

    def build(
        self,
        model: BaseModel,
        X: Union[pd.DataFrame, np.ndarray, list, pd.Series],
        y: Union[np.ndarray, list, pd.Series],
        rep: BaseCompoundVecRepresentation = None,
        dim_reduction: str = "pls",
        explvar: float = 0.8,
        threshold: float = 0.95
    ):
        super().build(model, X, y)
        self.rep = rep
        self.dim_reduction = dim_reduction

        self.X_train = self.preprocess(self.X_train)
        thresh = int(self.X_train.shape[0] * threshold)
        max_components = min(100, min(self.X_train.shape))

        if self.dim_reduction == "pls":
            self.reduction = PLSRegression(n_components = max_components)
            self.reduction.fit(self.X_train, self.y_train)
            x_var = np.var(self.reduction.x_scores_, axis=0)
            x_var /= np.sum(x_var)
        elif self.dim_reduction == "pca":
            self.reduction = PCA(n_components = max_components)
            self.reduction.fit(self.X_train, self.y_train)
            x_var = self.reduction.explained_variance_ratio_

        if np.sum(x_var) > explvar:
            self.n_components = np.where(np.cumsum(x_var) >= explvar)[0][0] + 1
        else:
            self.n_components = max_components

        self.Xp_train = self.reduction.transform(self.X_train)[:, :self.n_components]
        self.Xp_mean = np.mean(self.Xp_train, axis=0)
        self.y_mean = np.mean(self.y_train)

        nbrs = NearestNeighbors(n_neighbors = 2).fit(self.Xp_train)
        distances, indices = nbrs.kneighbors(self.Xp_train)

        centroid_dist = np.linalg.norm(self.Xp_train - self.Xp_mean, axis=1)
        neighbor_dist = distances[:, 1]
        model_dist = self.DModX(self.X_train, self.Xp_train)
        y_mean_dist = np.abs(self.y_train - self.y_mean)
        y_nei_dist = np.abs(self.y_train - self.y_train[indices[:, 1]])
        SDEP_dist = self.SDEP(self.Xp_train, n_drop = 1)

        self.training = {
            "A_raw": centroid_dist,
            "B_raw": neighbor_dist,
            "C_raw": model_dist,
            "D_raw": y_mean_dist,
            "E_raw": y_nei_dist,
            "F_raw": SDEP_dist,
        }

        self.thresholds = {
            "A": sorted(centroid_dist)[thresh],
            "B": sorted(neighbor_dist)[thresh],
            "C": sorted(model_dist)[thresh],
            "D": sorted(y_mean_dist)[thresh],
            "E": sorted(y_nei_dist)[thresh],
            "F": sorted(SDEP_dist)[thresh],
        }

    def calculate(self, X, y_pred):
        """Calcualtes confidence scores.
        """
        X = self.preprocess(X)
        Xp = self.reduction.transform(X)[:, :self.n_components]

        nbrs = NearestNeighbors(n_neighbors = 1).fit(self.Xp_train)
        distances, indices = nbrs.kneighbors(Xp)

        def calculate_criterion(criterion: str, standardize: bool = True):
            if criterion in ("A", "A_raw"):
                dist = np.linalg.norm(Xp - self.Xp_mean, axis=1)
            elif criterion in ("B", "B_raw"):
                dist = distances.flatten()
            elif criterion in ("C", "C_raw"):
                dist = self.DModX(X, Xp)
            elif criterion in ("D", "D_raw"):
                dist = self.SDEP(Xp, 0)
            elif criterion in ("E", "E_raw"):
                dist = np.abs(y_pred - self.y_mean)
            elif criterion in ("F", "F_raw"):
                dist = np.abs(y_pred - self.y_train[indices.flatten()])
            else:
                raise NameError("criterion {} is not recognized".format(criterion))

            if "raw" in criterion:
                if standardize:
                    return (dist - np.mean(self.training[criterion])) / np.std(self.training[criterion])
                else:
                    return dist
            else:
                return np.where(dist > self.thresholds[criterion], 1, 0)

        results = np.array([calculate_criterion(c) for c in self.criteria])
        return np.sum(results, axis=0)

    def preprocess(self, X, y = None):
        """Preprocesses data into the appropriate representation.
        """
        if self.rep is None:
            X = self.model.preprocess(X, y)
        else:
            X = np.array(self.rep.convert(X))

        assert isinstance(X, np.ndarray), "The preprocess for the model must return a np.ndarray, e.g. the model must use a BaseCompoundVecRepresentation or a representation must be passed"

        return X

    def DModX(self, X: np.ndarray, Xp: np.ndarray) -> np.ndarray:
        """Computes the distance to the model (DmodX).

        Computes the distance between a datapoint and the PLS model plane. See
        <https://www.jmp.com/support/help/en/15.2/index.shtml#page/jmp/dmodx-calculation.shtml>
        for more details about the statistic.

        Parameters:
            X (np.ndarray): queries
            Xp (np.ndarray): queries transformed into latent space
        """
        if self.dim_reduction == "pls":
            X_reconstructed = Xp @ self.reduction.x_loadings_[:, :self.n_components].T * self.reduction._x_std + self.reduction._x_mean
        elif self.dim_reduction == "pca":
            X_reconstructed = np.dot(Xp, self.reduction.components_[:self.n_components, :]) + self.reduction.mean_
        return np.linalg.norm(X - X_reconstructed, axis=1) / np.sqrt(self.X_train.shape[1] - self.n_components)

    def SDEP(self, Xp: np.ndarray, n_drop: int = 0, neighbor_thresh: float = 0.05) -> np.ndarray:
        """Computes the standard deviation error of predictions (SDEP).

        Computes the standard deviation training error of the `neighbor_thresh`
        fraction of closest training queries to each query in `Xp` in latent space.

        Parameters:
            Xp (np.ndarray): queries transformed into latent space
            n_drop (int): 1 if fitting, 0 if scoring
            neighbor_thresh (float): fraction of closest training queries to consider
        """
        n_neighbors = int(self.X_train.shape[0] * neighbor_thresh) + n_drop
        nbrs = NearestNeighbors(n_neighbors = n_neighbors).fit(self.Xp_train)
        distances, indices = nbrs.kneighbors(Xp)

        y_sqerr = np.array((self.y_train - self.y_pred_train) ** 2)
        y_mse = np.mean(y_sqerr[indices[:, n_drop:]], axis=1).astype(float)

        return np.sqrt(y_mse).flatten()

class RandomForestErrorModel(BaseAggregateErrorModel):
    """ RandomForestErrorModel is an aggregate error model that predicts error
        bars based on an aggregate score determined by a random forest model
        trained on several BaseErrorModels.

        Parameters:
            error_models (Union[BaseErrorModel, List[BaseErrorModel]]): a list
                of error models whose scores will be features for the random
                forest model

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.RandomForestErrorModel([oce.SDC(), oce.kNNSimilarity()])
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    def _train(self, scores, residuals):
        self.aggregate_model = RandomForestRegressor(**self.kwargs)
        self.aggregate_model.fit(scores, residuals)

    def _predict(self, scores, **kwargs):
        return self.aggregate_model.predict(scores, **kwargs)



class BaseDomainApplicability(BaseClass):
    """Base class for analyzing domain applicability of models and estimating
    prediction uncertainty.

    Parameters:
        model (BaseModel): trained model
        rep (BaseCompoundVecRepresentation): preprocessing representation

    Methods:
        _fit: fits a method to a trained model and preprocessed training dataset, to be implemented internally by child classes
        fit: fits a method to a trained model and training dataset
        _test evaluates prediction uncertainty on a preprocessed testing dataset, to be implemented internally by child classes
        test: evaluates prediction uncertainty on a testing dataset
    """

    @log_arguments
    def __init__(self, model: BaseModel, rep: BaseCompoundVecRepresentation = None):
        self.model = model
        self.rep = rep
        self.results = None

    @abstractmethod
    def _fit(self, X, y, y_pred, **kwargs):
        """To be implemented by the child class; fits the method on the internally stored training dataset.

        Note: _fit shouldn"t be directly called by the user, rather it should be called indirectly via the fit method.

        Parameters:
            X (np.ndarray): features, SMILES
            y (np.ndarray): values
            y_pred (np.ndarray): predicted values
        """
        pass

    def fit(self, dataset: BaseDataset, use_entire_dataset: bool = False, **kwargs):
        """Fits model uncertainty to a training dataset. Calls the _fit method on the provided dataset.

        Parameters:
            dataset (BaseDataset): dataset used for fitting the method
            use_entire_dataset (bool): whether to use the entire dataset or only the training data
        """
        if use_entire_dataset:
            X = dataset.entire_dataset[0]
            y = np.array(dataset.entire_dataset[1])
        else:
            X = dataset.train_dataset[0]
            y = np.array(dataset.train_dataset[1])

        y_pred = np.array(self.model.predict(X))

        self._fit(X, y, y_pred, **kwargs)

    @abstractmethod
    def _test(self, X, y_pred, **kwargs):
        """To be implemented by the child class; tests the method on the provided preprocessed dataset.

        Note: _test shouldn"t be directly called by the user, rather it should be called indirectly via the test method.

        Parameters:
            X (np.ndarray): features, SMILES
            y_pred (np.ndarray): predicted values
        """
        pass

    def test(self, dataset: Union[BaseDataset, list, str], use_entire_dataset: bool = False, **kwargs):
        """Evaluates model uncertainty on a test dataset. Calls the _test method on the provided dataset.

        Parameters:
            dataset (BaseDataset, list, str): dataset being tested, sequence of SMILES
            use_entire_dataset (bool): whether to use the entire dataset or only the testing data
        """
        if isinstance(dataset, BaseDataset):
            if use_entire_dataset:
                X = dataset.entire_dataset[0]
            else:
                X = dataset.test_dataset[0]
        elif isinstance(dataset, list):
            X = dataset
        else:
            X = [dataset]

        y_pred = np.array(self.model.predict(X))

        return self._test(X, y_pred, **kwargs)

    def preprocess(self, X, y = None):
        """Preprocesses data into the appropriate representation.

        The preprocess for the model must return a np.ndarray, e.g. the model must use a
        BaseStructVecRepresentation or a representation must be passed

        Parameters:
            X (np.ndarray): features, SMILES
            y (np.ndarray): values

        Returns:
            X (np.ndarray): features, representation
        """
        if self.rep is None:
            X = self.model.preprocess(X, y)
        else:
            X = np.array(self.rep.convert(X))

        assert isinstance(X, np.ndarray), "The preprocess for the model must return a np.ndarray, e.g. the model must use a BaseCompoundVecRepresentation or a representation must be passed"

        return X

    def _save(self) -> dict:
        return super()._save()

    def _load(self, d: dict):
        return super()._load(d)


class ADAN(BaseDomainApplicability):
    """ Implementation of Applicability Domain Analysis that predicts uncertainty
        scores for compounds: `Applicability Domain Analysis (ADAN): A Robust
        Method for Assessing the Reliability of Drug Property Predictions
        <https://doi.org/10.1021/ci500172z>`_
    """

    @log_arguments
    def __init__(self, model: BaseModel, rep: BaseCompoundVecRepresentation = None, dim_reduction: str = "pls"):
        self.dim_reduction = dim_reduction
        super().__init__(model = model, rep = rep)

    def _fit(self, X, y, y_pred, explvar: float = 0.8, threshold: float = 0.95):
        """Fits ADAN to a model and training dataset.

        Parameters:
            explvar (float): fraction of variance explained by latent variables
            threshold (float): percentile of training set that defines passing
        """
        self.X_train = self.preprocess(X, y)
        self.y_train = y
        self.y_pred_train = y_pred

        thresh = int(self.X_train.shape[0] * threshold)
        max_components = min(100, min(self.X_train.shape))

        if self.dim_reduction == "pls":
            self.reduction = PLSRegression(n_components = max_components)
            self.reduction.fit(self.X_train, self.y_train)
            x_var = np.var(self.reduction.x_scores_, axis=0)
            x_var /= np.sum(x_var)
        elif self.dim_reduction == "pca":
            self.reduction = PCA(n_components = max_components)
            self.reduction.fit(self.X_train, self.y_train)
            x_var = self.reduction.explained_variance_ratio_

        if np.sum(x_var) > explvar:
            self.n_components = np.where(np.cumsum(x_var) >= explvar)[0][0] + 1
        else:
            self.n_components = max_components
        self.Xp_train = self.reduction.transform(self.X_train)[:, :self.n_components]

        nbrs = NearestNeighbors(n_neighbors = 2).fit(self.Xp_train)
        distances, indices = nbrs.kneighbors(self.Xp_train)

        # criterion A
        centroid_dist = np.linalg.norm(self.Xp_train - np.mean(self.Xp_train), axis=1)
        self.centroid_thresh = sorted(centroid_dist)[thresh]

        # criterion B
        neighbor_dist = distances[:, 1]
        self.neighbor_thresh = sorted(neighbor_dist)[thresh]

        # criterion C
        model_dist = self._DModX(self.X_train, self.Xp_train)
        self.model_thresh = sorted(model_dist)[thresh]

        # criterion F
        SDEP_dist = self._SDEP(self.Xp_train, 1)
        self.SDEP_thresh = sorted(SDEP_dist)[thresh]

        if self.model.setting == "classification":
            # criterion D
            self.enc = OneHotEncoder()
            self.enc.fit(self.y_train.reshape(-1, 1))
            y_enc = self.enc.transform(self.y_train.reshape(-1, 1)).toarray()
            self.y_mean = np.mean(y_enc, axis=0)
            y_mean_dist = np.sum(np.abs(y_enc - self.y_mean), axis=1)
            self.y_mean_thresh = sorted(y_mean_dist)[thresh]

            # criterion E
            y_nei_dist = (self.y_train != self.y_train[indices[:, 1]]).astype(int)
            self.y_nei_thresh = sorted(y_nei_dist)[thresh]
        else:
            # criterion D
            self.y_mean = np.mean(self.y_train)
            y_mean_dist = np.abs(self.y_train - self.y_mean)
            self.y_mean_thresh = sorted(y_mean_dist)[thresh]

            # criterion E
            y_nei_dist = np.abs(self.y_train - self.y_train[indices[:, 1]])
            self.y_nei_thresh = sorted(y_nei_dist)[thresh]

        self.data_ = pd.DataFrame({
            "A": centroid_dist,
            "B": neighbor_dist,
            "C": model_dist,
            "D": y_mean_dist,
            "E": y_nei_dist,
            "F": SDEP_dist,
        })

        self.thresholds_ = {
            "A": self.centroid_thresh,
            "B": self.neighbor_thresh,
            "C": self.model_thresh,
            "D": self.y_mean_thresh,
            "E": self.y_nei_thresh,
            "F": self.SDEP_thresh,
        }

    def _test(self, X, y_pred):
        X = self.preprocess(X)
        Xp = self.reduction.transform(X)[:, :self.n_components]

        nbrs = NearestNeighbors(n_neighbors = 1).fit(self.Xp_train)
        distances, indices = nbrs.kneighbors(Xp)

        # criterion A
        centroid_dist = np.linalg.norm(Xp, axis=1)
        A_raw = centroid_dist
        A_res = np.where(centroid_dist > self.centroid_thresh, 1, 0)

        # criterion B
        neighbor_dist = distances.flatten()
        B_raw = neighbor_dist
        B_res = np.where(neighbor_dist > self.neighbor_thresh, 1, 0)

        # criterion C
        model_dist = self._DModX(X, Xp)
        C_raw = model_dist
        C_res = np.where(model_dist > self.model_thresh, 1, 0)

        # criterion F
        SDEP_dist = self._SDEP(Xp, 0)
        F_raw = SDEP_dist
        F_res = np.where(SDEP_dist > self.SDEP_thresh, 1, 0)

        if self.model.setting == "classification":
            # criterion D
            y_pred_enc = self.enc.transform(y_pred.reshape(-1, 1)).toarray()
            y_mean_dist = np.sum(np.abs(y_pred_enc - self.y_mean), axis=1)
            D_raw = y_mean_dist
            D_res = np.where(y_mean_dist > self.y_mean_thresh, 1, 0)

            # criterion E
            y_nei_dist = (y_pred != self.y_train[indices.flatten()]).astype(int)
            E_raw = y_nei_dist
            E_res = np.where(y_nei_dist > self.y_nei_thresh, 1, 0)
        else:
            # criterion D
            y_mean_dist = np.abs(y_pred - self.y_mean)
            D_raw = y_mean_dist
            D_res = np.where(y_mean_dist > self.y_mean_thresh, 1, 0)

            # criterion E
            y_nei_dist = np.abs(y_pred - self.y_train[indices.flatten()])
            E_raw = y_nei_dist
            E_res = np.where(y_nei_dist > self.y_nei_thresh, 1, 0)

        self.results = pd.DataFrame({
            "A": A_res,
            "B": B_res,
            "C": C_res,
            "D": D_res,
            "E": E_res,
            "F": F_res,
            "A_raw": A_raw,
            "B_raw": B_raw,
            "C_raw": C_raw,
            "D_raw": D_raw,
            "E_raw": E_raw,
            "F_raw": F_raw,
            "Category": A_res + B_res + C_res + D_res + E_res + F_res
        })

        self.vars_ = {
            "X": X,
            "Xp": Xp,
            "y_pred": y_pred,
            "X_train": self.X_train,
            "Xp_train": self.Xp_train,
            "y_train": self.y_train,
            "y_pred_train": self.y_pred_train,
        }

        return self.results

    def _DModX(self, X: np.ndarray, Xp: np.ndarray) -> np.ndarray:
        """Computes the distance to the model (DmodX).

        Computes the distance between a datapoint and the PLS model plane. See
        <https://www.jmp.com/support/help/en/15.2/index.shtml#page/jmp/dmodx-calculation.shtml>
        for more details about the statistic.

        Parameters:
            X (np.ndarray): queries
            Xp (np.ndarray): queries transformed into latent space
        """
        if self.dim_reduction == "pls":
            X_reconstructed = Xp @ self.reduction.x_loadings_[:, :self.n_components].T * self.reduction._x_std + self.reduction._x_mean
        elif self.dim_reduction == "pca":
            X_reconstructed = np.dot(Xp, self.reduction.components_[:self.n_components, :]) + self.reduction.mean_
        return np.linalg.norm(X - X_reconstructed, axis=1) / np.sqrt(self.X_train.shape[1] - self.n_components)

    def _SDEP(self, Xp: np.ndarray, n_drop: int, neighbor_thresh: float = 0.05) -> np.ndarray:
        """Computes the standard deviation error of predictions (SDEP).

        Computes the standard deviation training error of the `neighbor_thresh`
        fraction of closest training queries to each query in `Xp` in latent space.

        Parameters:
            Xp (np.ndarray): queries transformed into latent space
            n_drop (int): 1 if training, 0 if testing
            neighbor_thresh (float): fraction of closest training queries to consider
        """
        n_neighbors = int(self.X_train.shape[0] * neighbor_thresh) + n_drop
        nbrs = NearestNeighbors(n_neighbors = n_neighbors).fit(self.Xp_train)
        distances, indices = nbrs.kneighbors(Xp)

        if self.model.setting == "classification":
            y_sqerr = self.y_train != self.y_pred_train
        else:
            y_sqerr = (self.y_train - self.y_pred_train) ** 2
        y_sqerr_mean = np.mean(y_sqerr[indices[:, n_drop:]], axis=1).astype(float)

        return np.sqrt(y_sqerr_mean).flatten()

