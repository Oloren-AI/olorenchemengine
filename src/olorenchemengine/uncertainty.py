"""Techniques for quantifying uncertainty and estimating confidence intervals for all oce models.
"""

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity

import olorenchemengine as oce

from .base_class import *
from .basics import *
from .dataset import *
from .representations import *
from .reduction import *


class BaseFingerprintModel(BaseErrorModel):
    """Morgan fingerprint-based error models.
    
    BaseFingerprintModel is the base class for error models that require the
    computation of Morgan fingerprints.
    """

    @log_arguments
    def __init__(self, radius=2, log=True, **kwargs):
        self.radius = radius      
        super().__init__(log=False, **kwargs)

    def _build(self, **kwargs):
        if isinstance(self.X_train, pd.DataFrame):
            self.X_train.columns = ["SMILES"]
        self.train_fps = list(self.get_fps(self.X_train))

    def get_fps(self, smiles: List[str]) -> List:
        smiles = np.array(SMILESRepresentation().convert(smiles))
        get_fp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), self.radius)
        return list(np.vectorize(get_fp)(smiles))

    def _save(self) -> dict:
        d = super()._save()
        if hasattr(self, "train_fps"):
            d.update({"train_fps": self.train_fps})
        return d

    def _load(self, d) -> None:
        super()._load(d)
        if "train_fps" in d.keys():
            self.train_fps = d["train_fps"]

# List of kernels for kernel error models
KERNELS = {
    "sdc":          lambda X, h: np.exp(-h * X / (1 - X)),
    "uniform":      lambda X, h: h * X < 1,
    "linear":       lambda X, h: np.maximum(0, 1 - h * X),
    "parabolic":    lambda X, h: np.maximum(0, 1 - (h * X) ** 2),
    "power":        lambda X, h: (1 - X) ** h,
    "gaussian":     lambda X, h: np.exp(-(h * X) ** 2),
}

# List of predictors for kernel error models
PREDICTORS = {
    "property": lambda y, em: em.y_train - y,
    "error":    lambda y, em: em.y_train - em.y_pred_train,
}

class KernelError(BaseFingerprintModel):
    """Kernel error model. 
    
    KernelError uses a kernel-weighted average of prediction errors as the
    covariate for estimating confidence intervals. It is inspired by the 
    Nadaraya-Watson estimator, which generates a regression using a
    kernel-weighted average. The distance function used is 1 - Tanimoto
    Similarity.

    This is the recommended error model for general purposes and models.

    Parameters
    ------------------------------
    predictor (str, {"property", "error"}): Error predictor being estimated
    kernel (str, {"default"}): Kernel used as a weight-function
    h (int, float): Bandwidth

    Example
    ------------------------------
    import olorenchemengine as oce
    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.KernelError())
    model.predict(test["Drug"], return_ci = True)
    """

    @log_arguments
    def __init__(self, predictor="property", kernel="sdc", h=3, log=True, **kwargs):
        assert predictor in PREDICTORS, "Predictor `{}` is invalid. Please choose one of `{}`.".format(predictor, "`, `".join(KERNELS.keys()))
        assert kernel in KERNELS, "Kernel `{}` is invalid. Please choose one of `{}`.".format(kernel, "`, `".join(KERNELS.keys()))

        self.predictor = predictor
        self.kernel = kernel
        self.h = h
        super().__init__(log=False, **kwargs)

    def calculate(self, X, y_pred):
        K = KERNELS[self.kernel]
        P = PREDICTORS[self.predictor]

        def covariate(smiles, y):
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprint(mol, 2)
            TD = 1 - np.array(BulkTanimotoSimilarity(fp, self.train_fps))
            w = K(TD, self.h)
            p = P(y, self)

            norm = np.sum(w)
            if norm == 0:
                return np.mean(p)
            else:
                return np.dot(w, p) / norm

        X = np.array(X).flatten()
        return np.array([covariate(smiles, y) for smiles, y in tqdm(zip(X, y_pred))])


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
    def __init__(self, a=3, log=True, **kwargs):
        self.a = a
        super().__init__(log=False, **kwargs)

    def calculate(self, X, y_pred):
        ref_fps = self.get_fps(X)

        def sdc(fp):
            TS = np.array(BulkTanimotoSimilarity(fp, self.train_fps))
            return np.sum(np.exp(-self.a * (1 - TS) / TS))

        return np.array([sdc(fp) for fp in tqdm(ref_fps)])


class TargetDistDC(SDC):
    """ TargetDistDC is an error model that calculates the root-mean-square
        difference between the predicted activity of the target molecule and
        the observed activities of all training molecules, weighted by the DC.

        Parameters:
            a (int or float, optional): Value of a in the SDC formula. Default 3.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.TargetDistDC())
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):      
        super().__init__(log=False, **kwargs)

    def calculate(self, X, y_pred):        
        def dist(smi, pred):
            mol = Chem.MolFromSmiles(smi)
            ref_fp = AllChem.GetMorganFingerprint(mol, 2)
            TS = np.array(BulkTanimotoSimilarity(ref_fp, self.train_fps))
            DC = np.exp(-self.a * (1 - TS) / TS)
            error = np.abs(self.y_train - pred)
            if np.sum(DC) == 0:
                return np.sqrt(np.mean(error))
            return np.sqrt(np.dot(DC, error) / np.sum(DC))

        X = np.array(X).flatten()
        return np.array([dist(smi, pred) for smi, pred in tqdm(zip(X, y_pred))])


class TrainDistDC(SDC):
    """ TrainDistDC is an error model that calculates the root-mean-square
        difference between the predicted and observed activities of all
        training molecules, weighted by the DC.

        Parameters:
            a (int or float, optional): Value of a in the SDC formula. Default 3.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.TrainDistDC())
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):      
        super().__init__(log=False, **kwargs)

    def calculate(self, X, y_pred):
        residuals = np.abs(self.y_train - self.y_pred_train)
        ref_fps = self.get_fps(X)

        def dist(fp):
            TS = np.array(BulkTanimotoSimilarity(fp, self.train_fps))
            DC = np.exp(-self.a * (1 - TS) / TS)
            if np.sum(DC) == 0:
                return np.sqrt(np.mean(residuals))
            return np.sqrt(np.dot(DC, residuals) / np.sum(DC))

        return np.array([dist(fp) for fp in tqdm(ref_fps)])


class KNNSimilarity(BaseFingerprintModel):
    """ NNSimilarity is an error model that calculates mean Tanimoto similarity
        between the target molecule and the k most similar training molecules
        using a Morgan Fingerprint with a radius of 2 bonds.

        Parameters:
            k (int, optional): Number of nearest neighbors to consider. Default 5.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.KNNSimilarity())
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    @log_arguments
    def __init__(self, k=5, log=True, **kwargs):
        self.k = k
        super().__init__(log=False, **kwargs)

    def calculate(self, X, y_pred):
        ref_fps = self.get_fps(X)

        def mean_sim(fp):
            similarity = np.array(BulkTanimotoSimilarity(fp, self.train_fps))
            return np.mean(np.partition(similarity, self.k)[-self.k:])

        return np.array([mean_sim(fp) for fp in tqdm(ref_fps)])

class TargetDistKNN(KNNSimilarity):
    """ TargetDistKNN is an error model that calculates the root-mean-square
        difference between the predicted activity of the target molecule and
        the observed activities of the k most similar training molecules,
        weighted by their similarity.

        Parameters:
            k (int, optional): Number of nearest neighbors to consider. Default 5.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.TargetDistKNN())
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):      
        super().__init__(log=False, **kwargs)

    def calculate(self, X, y_pred):
        def dist(smi, pred):
            mol = Chem.MolFromSmiles(smi)
            ref_fp = AllChem.GetMorganFingerprint(mol, 2)
            TS = np.array(BulkTanimotoSimilarity(ref_fp, self.train_fps))
            error = np.abs(self.y_train - pred)
            idxs = np.argpartition(TS, self.k)[-self.k:]
            if np.sum(TS[idxs]) == 0:
                return np.sqrt(np.mean(error[idxs]))
            return np.sqrt(np.dot(TS[idxs], error[idxs]) / np.sum(TS[idxs]))

        X = np.array(X).flatten()
        return np.array([dist(smi, pred) for smi, pred in tqdm(zip(X, y_pred))])


class TrainDistKNN(KNNSimilarity):
    """ TrainDistKNN is an error model that calculates the root-mean-square
        difference between the predicted and observed activities of the k most
        similar training molecules to the target molecule, weighted by their
        similarity.

        Parameters:
            k (int, optional): Number of nearest neighbors to consider. Default 5.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.TrainDistKNN())
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):      
        super().__init__(log=False, **kwargs)

    def calculate(self, X, y_pred):
        residuals = np.abs(self.y_train - self.y_pred_train)
        ref_fps = self.get_fps(X)

        def dist(fp):
            similarity = np.array(BulkTanimotoSimilarity(fp, self.train_fps))
            idxs = np.argpartition(similarity, self.k)[-self.k:]
            if np.sum(similarity[idxs]) == 0:
                return np.sqrt(np.mean(residuals[idxs]))
            return np.sqrt(np.dot(similarity[idxs], residuals[idxs]) / np.sum(similarity[idxs]))
        
        return np.array([dist(fp) for fp in tqdm(ref_fps)])


class Predicted(BaseErrorModel):
    """ Predicted is an error model that predicts error bars based on only the
        predicted value of a molecule. It is best used as part of an aggregate
        error model rather than by itself.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.AggregateErrorModel([oce.SDC(), oce.Predicted()])
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):      
        super().__init__(log=False, **kwargs)

    def calculate(self, X, y_pred):
        return y_pred

class Naive(BaseErrorModel):
    """ Naive is an error model that predicts a uniform confidence interval
        based on the errors of the fitting dataset. Used exclusively for
        benchmarking error models.
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):
        super().__init__(log=False, **kwargs)

    def calculate(self, X, y_pred):
        return np.array([0 for _ in X])
    
    def _fit(
        self,
        residuals: np.ndarray,
        scores: np.ndarray,
    ):
        self.error = np.quantile(residuals, self.ci)
        self.reg = lambda X: np.array([self.error for _ in X])

    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray, list, pd.Series],
    ):
        return self.reg(X)

class BaseEnsembleModel(BaseErrorModel):
    """ BaseEnsembleModel is the base class for error models that estimate
        uncertainty based on the variance of an ensemble of models.
    """

    @log_arguments
    def __init__(self, ensemble_model = None, n_ensembles = 16, log=True, **kwargs):      
        self.ensemble_model = ensemble_model
        self.n_ensembles = n_ensembles
        super().__init__(log=False, **kwargs)

    def _build(self, **kwargs):
        if self.ensemble_model is None:
            self.ensemble_model = self.model.copy()
        self.ensembles = []
    
    def calculate(self, X, y_pred):
        predictions = np.stack([model.predict(X) for model in tqdm(self.ensembles)])
        return np.var(predictions, axis=0)

    def _save(self) -> dict:
        d = super()._save()
        if hasattr(self, "ensembles"):
            d.update({"ensembles": [saves(model) for model in self.ensembles]})
        return d

    def _load(self, d) -> None:
        super()._load(d)
        if "ensembles" in d.keys():
            self.ensembles = [loads(model) for model in d["ensembles"]]
    
class BootstrapEnsemble(BaseEnsembleModel):
    """ BootstrapEnsemble estimates uncertainty based on the variance of several
        models trained on bootstrapped samples of the training data.

        Parameters:
            ensemble_model (BaseModel): Model used for ensembling. Defaults to the same as the original model.
            n_ensembles (int): Number of ensembles
            bootstrap_size (float): Proportion of training data to train each ensemble model
    
    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.BootstrapEnsemble(n_ensembles = 10))
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """
    
    @log_arguments
    def __init__(self, ensemble_model = None, n_ensembles = 12, bootstrap_size = 0.25, log=True, **kwargs):
        self.bootstrap_size = bootstrap_size
        super().__init__(ensemble_model = ensemble_model, n_ensembles = n_ensembles, log=False, **kwargs)

    def _build(self, **kwargs):
        super()._build()

        from sklearn.model_selection import train_test_split
        
        for _ in tqdm(range(self.n_ensembles)):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_train, self.y_train, train_size=self.bootstrap_size
            )
            ensemble_model = self.ensemble_model.copy()
            ensemble_model.fit(X_train, y_train)
            self.ensembles.append(ensemble_model)

class RandomForestEnsemble(BaseEnsembleModel):
    """ RandomForestEnsemble estimates uncertainty based on the variance of several
        random forest models initialized to different random states.

        Parameters:
            ensemble_model (BaseModel): Model used for ensembling. Defaults to the same as the original model.
            n_ensembles (int): Number of ensembles
    
    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.RandomForestEnsemble(n_ensembles = 10))
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):      
        super().__init__(log=False, **kwargs)
    
    def _build(self, **kwargs):
        super()._build()

        for n in tqdm(range(self.n_ensembles)):
            ensemble_model = RandomForestModel(
                oce.MorganVecRepresentation(radius=2, nbits=2048), 
                random_state=n, 
                **kwargs
            )
            ensemble_model.fit(self.X_train, self.y_train)
            self.ensembles.append(ensemble_model)

class ADAN(BaseErrorModel):
    """ ADAN is an error model that predicts error bars based on one or
        multiple ADAN categories: `Applicability Domain Analysis (ADAN): A
        Robust Method for Assessing the Reliability of Drug Property Predictions
        <https://doi.org/10.1021/ci500172z>`_

        Parameters:
            criterion (str): the ADAN criteria to be considered.
            rep (BaseCompoundVecRepresentation): the representation to use. By default, 
                usees the representation of the BaseModel object.
            dim_reduction ({"pls", "pca"}): the dimensionality reduction to use. 
            explvar (float): the desired variance to be captured by the dimensionality
                reduction components as a proportion of total variance.
            threshold (float): the quantile for a criterion to be considered as out of
                its standard range.

    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit_cv(train["Drug"], train["Y"], error_model = oce.ADAN("E_raw"))
    model.predict(test["Drug"], return_ci = True)
    ------------------------------
    """

    @log_arguments
    def __init__(
        self, 
        criterion: str = "Category", 
        rep: BaseCompoundVecRepresentation = None,
        dim_reduction: str = "pls",
        explvar: float = 0.8,
        threshold: float = 0.95,
        log=True, 
        **kwargs
    ):
        self.criterion = criterion
        self.rep = rep
        self.dim_reduction = dim_reduction
        self.explvar = explvar
        self.threshold = threshold
        super().__init__(log=False, **kwargs)

    def _build(self, **kwargs):
        self.X_rep = self.preprocess(self.X_train)
        self.n_components = min(self.X_rep.shape)

        if self.dim_reduction == "pls":
            from sklearn.cross_decomposition import PLSRegression

            self.reduction = PLSRegression(n_components=self.n_components)
            self.reduction.fit(self.X_rep, self.y_train)
            x_var = np.var(self.reduction.x_scores_, axis=0)
            x_var /= np.sum(x_var)
        elif self.dim_reduction == "pca":
            from sklearn.decomposition import PCA

            self.reduction = PCA(n_components=self.n_components)
            self.reduction.fit(self.X_rep, self.y_train)
            x_var = self.reduction.explained_variance_ratio_
        else:
            raise NameError("dim_reduction {} is not recognized. Valid inputs are 'pls' and 'pca'.".format(self.dim_reduction))

        if np.sum(x_var) > self.explvar:
            self.n_components = np.where(np.cumsum(x_var) > self.explvar)[0][0] + 1

        self.Xp_train = self.reduction.transform(self.X_rep)[:, : self.n_components]
        self.Xp_mean = np.mean(self.Xp_train, axis=0)
        self.y_mean = np.mean(self.y_train)

        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=2).fit(self.Xp_train)
        distances, indices = nbrs.kneighbors(self.Xp_train)

        centroid_dist = np.linalg.norm(self.Xp_train - self.Xp_mean, axis=1)
        neighbor_dist = distances[:, 1]
        model_dist = self.DModX(self.X_rep, self.Xp_train)
        y_mean_dist = np.abs(self.y_train - self.y_mean)
        y_nei_dist = np.abs(self.y_train - self.y_train[indices[:, 1]])
        SDEP_dist = self.SDEP(self.Xp_train, n_drop=1)

        self.training_ = {
            "A_raw": centroid_dist,
            "B_raw": neighbor_dist,
            "C_raw": model_dist,
            "D_raw": y_mean_dist,
            "E_raw": y_nei_dist,
            "F_raw": SDEP_dist,
        }

        thresh = int(self.X_rep.shape[0] * self.threshold)
        self.thresholds_ = {
            "A": np.partition(centroid_dist, -thresh)[-thresh],
            "B": np.partition(neighbor_dist, -thresh)[-thresh],
            "C": np.partition(model_dist, -thresh)[-thresh],
            "D": np.partition(y_mean_dist, -thresh)[-thresh],
            "E": np.partition(y_nei_dist, -thresh)[-thresh],
            "F": np.partition(SDEP_dist, -thresh)[-thresh],
        }

    def calculate_full(self, X, standardize: bool = True):
        """Calculates complete confidence scores for visualization."""
        criteria = ["A", "B", "C", "D", "E", "F"]
        y_pred = np.array(self.model.predict(X)).flatten()
        X = self.preprocess(X)
        Xp = self.reduction.transform(X)[:, : self.n_components]

        self.results = {}
        for c in criteria:
            self.results[c] = self._calculate(X, Xp, y_pred, c)
        for c in criteria:
            c += "_raw"
            self.results[c] = self._calculate(X, Xp, y_pred, c, standardize=standardize)
        self.results["Category"] = np.sum([self.results[c] for c in criteria], axis=0)

        self.results = pd.DataFrame(self.results)

    def calculate(self, X, y_pred, standardize: bool = True):
        """Calcualtes confidence scores."""
        X = self.preprocess(X)
        Xp = self.reduction.transform(X)[:, : self.n_components]
        
        if self.criterion == "Category":
            criteria = ["A", "B", "C", "D", "E", "F"]
            return np.sum([self._calculate(X, Xp, y_pred, c) for c in criteria], axis=0)
        else:
            return self._calculate(X, Xp, y_pred, self.criterion, standardize=standardize)

    def _calculate(self, X, Xp, y_pred, criterion: str, standardize: bool = True):
        """Calcualtes confidence scores for a given criteron."""
        from sklearn.neighbors import NearestNeighbors
        
        if criterion in ("A", "A_raw"):
            dist = np.linalg.norm(Xp - self.Xp_mean, axis=1)
        elif criterion in ("B", "B_raw"):
            nbrs = NearestNeighbors(n_neighbors=1).fit(self.Xp_train)
            distances = nbrs.kneighbors(Xp)[0]
            dist = distances.flatten()
        elif criterion in ("C", "C_raw"):
            dist = self.DModX(X, Xp)
        elif criterion in ("D", "D_raw"):
            dist = self.SDEP(Xp, 0)
        elif criterion in ("E", "E_raw"):
            dist = np.abs(y_pred - self.y_mean)
        elif criterion in ("F", "F_raw"):
            nbrs = NearestNeighbors(n_neighbors=1).fit(self.Xp_train)
            indices = nbrs.kneighbors(Xp)[1]
            dist = np.abs(y_pred - self.y_train[indices.flatten()])
        else:
            raise NameError("Criterion {} is not recognized.".format(criterion))

        if len(criterion) > 1:
            if standardize:
                return (dist - np.mean(self.training_[criterion])) / np.std(
                    self.training_[criterion]
                )
            else:
                return dist
        else:
            return np.where(dist > self.thresholds_[criterion], 1, 0)

    def preprocess(self, X, y=None):
        """Preprocesses data into the appropriate representation."""
        if self.rep is None:
            X = self.model.preprocess(X, y)
        else:
            X = np.array(self.rep.convert(X))

        assert isinstance(
            X, np.ndarray
        ), "The preprocess for the model must return a np.ndarray, e.g. the model must use a BaseCompoundVecRepresentation or a representation must be passed"

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
            X_reconstructed = (
                Xp
                @ self.reduction.x_loadings_[:, : self.n_components].T
                * self.reduction._x_std
                + self.reduction._x_mean
            )
        elif self.dim_reduction == "pca":
            X_reconstructed = (
                np.dot(Xp, self.reduction.components_[: self.n_components, :])
                + self.reduction.mean_
            )
        return np.linalg.norm(X - X_reconstructed, axis=1) / np.sqrt(
            self.X_rep.shape[1] - self.n_components
        )

    def SDEP(
        self, Xp: np.ndarray, n_drop: int = 0, neighbor_thresh: float = 0.05
    ) -> np.ndarray:
        """Computes the standard deviation error of predictions (SDEP).

        Computes the standard deviation training error of the `neighbor_thresh`
        fraction of closest training queries to each query in `Xp` in latent space.

        Parameters:
            Xp (np.ndarray): queries transformed into latent space
            n_drop (int): 1 if fitting, 0 if scoring
            neighbor_thresh (float): fraction of closest training queries to consider
        """
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = int(self.X_rep.shape[0] * neighbor_thresh) + n_drop
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(self.Xp_train)
        distances, indices = nbrs.kneighbors(Xp)

        y_sqerr = np.array((self.y_train - self.y_pred_train) ** 2)
        y_mse = np.mean(y_sqerr[indices[:, n_drop:]], axis=1).astype(float)

        return np.sqrt(y_mse).flatten()

class AggregateErrorModel(BaseErrorModel):
    """ AggregateErrorModel estimates uncertainty by aggregating ucertainty scores from
        several different BaseErrorModels.

        Parameters:
            *error_models (BaseErrorModel): error models to be aggregated
            reduction (BaseReduction): reduction method used to aggregate uncertainty scores.
                Must output 1 component. Default FactorAnalysis().
    
    Example
    ------------------------------
    import olorenchemengine as oce

    model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
    model.fit(train["Drug"], train["Y"])
    error_model = oce.AggregateErrorModel(error_models = [oce.TargetDistDC(), oce.TrainDistDC()])
    error_model.build(model, train["Drug"], train["Y"])
    error_model.fit(valid["Drug"], valid["Y"])
    error_model.score(test["Drug"])
    ------------------------------
    """

    @log_arguments
    def __init__(
        self, 
        *error_models: BaseErrorModel, 
        reduction: BaseReduction = FactorAnalysis(n_components = 1),
        log=True,
        **kwargs
    ):
        self.error_models = list(error_models)
        self.reduction = reduction
        super().__init__(log=False, **kwargs)

    def _build(self, **kwargs):
        for error_model in self.error_models:
            error_model.build(self.model, self.X_train, self.y_train)

    def fit(self, X: Union[pd.DataFrame, np.ndarray, list, pd.Series], y: Union[np.ndarray, list, pd.Series], **kwargs):
        """Fits confidence scores to an external dataset

        Args:
            X (array-like): features, smiles
            y (array-like): true values
        
        Returns:
            plotly figure of fitted model against validation dataset
        """
        y_pred = np.array(self.model.predict(X)).flatten()
        scores = [error_model.calculate(X, y_pred) for error_model in self.error_models]
        scores = np.transpose(np.stack(scores))
        
        self.reduction.fit(scores)
        aggregate_scores = self.reduction.transform(scores).flatten()
        residuals = np.abs(np.array(y) - y_pred)
        self.fit_scores = scores

        return self._fit(residuals, aggregate_scores, **kwargs)

    def fit_cv(self, n_splits: int = 10, **kwargs):
        """Fits confidence scores to the training dataset via cross validation.

        Args:
            n_splits (int): Number of cross validation splits, default 5

        Returns:
            plotly figure of fitted model against validation dataset
        """
        from sklearn.model_selection import KFold

        self.X_train = np.array(self.X_train).flatten()

        residuals = None
        scores = None
        kf = KFold(n_splits=n_splits)

        split = 1
        for train_index, test_index in kf.split(self.X_train):
            print('evaluating split {} of {}'.format(split, n_splits))
            split += 1

            X_train, X_test = self.X_train[train_index], self.X_train[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]
            model = self.model.copy()
            model.fit(X_train, y_train)
            
            y_pred_test = np.array(model.predict(X_test)).flatten()
            pred_error = np.abs(y_test - y_pred_test)
            if residuals is None:
                residuals = pred_error
            else:
                residuals = np.concatenate((residuals, pred_error))

            for error_model in self.error_models:
                error_model.build(model, X_train, y_train)
            scores_fold = [error_model.calculate(X_test, y_pred_test) for error_model in self.error_models]
            scores_fold = np.transpose(np.stack(scores_fold))
            if scores is None:
                scores = scores_fold
            else:
                scores = np.concatenate((scores, scores_fold))
        
        self.reduction.fit(scores)
        aggregate_scores = self.reduction.transform(scores).flatten()
        self.fit_scores = scores

        return self._fit(residuals, aggregate_scores, **kwargs)

    def calculate(
        self, X: Union[pd.DataFrame, np.ndarray, list, pd.Series], y_pred: np.ndarray
    ) -> np.ndarray:
        """Computes aggregate error model score from inputs.

        Args:
            X: features, smiles
            y_pred: predicted values
        """
        scores = [error_model.calculate(X, y_pred) for error_model in self.error_models]
        scores = np.transpose(np.stack(scores))

        return self.reduction.transform(scores)
    
    def _save(self) -> dict:
        d = super()._save()
        if hasattr(self, "fit_scores"):
            d.update({"fit_scores": self.fit_scores})
        return d

    def _load(self, d) -> None:
        super()._load(d)
        if "fit_scores" in d.keys():
            self.fit_scores = d["fit_scores"]
            self.reduction.fit(self.fit_scores)