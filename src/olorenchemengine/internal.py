import base64
import contextlib
import hashlib
import inspect
import json
import os
import io
import joblib
import pickle
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from typing import Callable, Union, Any, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyrebase  # Default pyrebase is pyrebase3 which won't work. Need to install pyrebase4 (pip install pyrebase4)
from google.cloud.firestore import Client
from google.oauth2.credentials import Credentials
from tqdm import tqdm

import olorenchemengine
import olorenchemengine as oce

sys.modules[
    "olorenautoml"
] = olorenchemengine  # important for backwards compatibility of some models


def mock_imports(g, *args):
    for arg in args:
        g[arg] = MagicMock()


def all_subclasses(cls):
    """Helper function to return all subclasses of class"""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )

ignored_kwargs = ["map_location", "num_workers"]


def download_public_file(path, redownload=False):
    """Download a public file from Oloren's Public Storage, and returns the contents.

    @param path: The path to the file to read.
    @param redownload: Whether to redownload the file if it already exists.
    """

    local_path = os.path.join(olorenchemengine.CONFIG["CACHE_PATH"], path)

    if os.path.exists(local_path) and not redownload:
        return local_path

    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))

    print(f"Downloading {path}...")
    import urllib
    import urllib.request

    urllib.request.urlretrieve(
        urllib.parse.quote(f"https://storage.googleapis.com/oloren-public-data/{path}",safe='/:?=&'), local_path
    )
    return local_path


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def log_arguments(func: Callable[..., None]) -> Callable[..., None]:
    """
    log_arguments is a decorator which logs the arguments of a BaseClass constructor to instance variables for use in
        model parameterization.

    Args:
        func (function): a __init__(self, *args, **kwargs) function of a baseclass.

    Returns:
        wrapper (function): the same __init__ function with arguments saved to instance variables.
    """

    def wrapper(self, *args, **kwargs):
        if "log" not in kwargs.keys() or kwargs["log"]:
            kwds = get_default_args(func)
            for k, v in kwds.items():
                if k not in kwargs:
                    kwargs[k] = v
            self.args = args
            self.kwargs = {k: v for k, v in kwargs.items() if k not in ignored_kwargs}
        return func(self, *args, **kwargs)

    wrapper.__wrapped__ = func

    return wrapper


def _create_BC_if_necessary(obj):
    return obj if type(obj) is not dict else create_BC(obj)


def deparametrize_args_kwargs(params):
    args = params["args"]
    kwargs = params["kwargs"]
    return [_create_BC_if_necessary(arg) for arg in args], {
        key: create_BC(kwarg) for key, kwarg in kwargs.items()
    }

class BaseClass:
    """BaseClass is the base class for all models.

    All classes in Oloren ChemEngine should inherit from BaseClass to enable for universal saving and loading of both
    parameters and internal state. This requires the implementation of abstract methods _save and _load.

    Methods:
        Registry: returns a dictionary mapping the name of a class to the class itself for all subclasses of the class.
        _save: saves an instance of a BaseClass to a dictionary (abstract method to be implmented by subclasses)
        _load: loads an instance of a BaseClass from a dictionary (abstract method to be implmented by subclasses)
    """

    @log_arguments
    def __init__(self, log = True):
        pass

    @classmethod
    def Registry(cls):
        """Registry is a recursive method to create a dictionary of all subclasses of BaseClass, with the key being the name of the subclass and the value being the subclass itself."""
        d = {cls.__name__: cls}
        scs = cls.__subclasses__()
        if len(scs) > 0:
            for sc in scs:
                u = sc.Registry()
                d = {**d, **u}
        return d

    @classmethod
    def AllInstances(cls):
        """AllTypes returns a list of all standard instances of all subclasses of BaseClass.

        Standard instances means that all required parameters for instantiation of the
        subclasses are set with canonical values."""
        if hasattr(cls, "__abstractmethods__") and len(cls.__abstractmethods__) != 0:
            return [o for sc in cls.__subclasses__() for o in sc.AllInstances()]
        try:
            return [cls()] + [
                o for sc in cls.__subclasses__() for o in sc.AllInstances()
            ]
        except Exception:
            return [o for sc in cls.__subclasses__() for o in sc.AllInstances()]

    @classmethod
    def Opt(cls, *args, **kwargs):
        return {
            **{"BC_class_name": cls.__name__},
            **{"args": args},
            **{"kwargs": kwargs},
        }

    @abstractmethod
    def _save(self) -> dict:
        """_save is an abstract method that must be implemented by all subclasses. It should return a dictionary of variables which can passed to an instance of a model via _load to completely recreate a model.

        Returns:
            dict: a dictionary of variables which can be passed to _load to recreate the model.
        """
        return ""

    @abstractmethod
    def _load(self, d: dict):
        """_load is an abstract method that must be implemented by all subclasses. It should take an instance save by _save and recreate the model."""
        pass

    def copy(self):
        obj_copy = create_BC(parameterize(self))
        obj_copy._load(self._save())
        return obj_copy
    
class BaseDepreceated(BaseClass):
    """BaseDepreceated is a class which is used to deprecate a class.
    
    Depreceated classes will raise Exception and will not run.
    """
    
    @log_arguments
    def __init__(self, *args, **kwargs):
        raise Exception("This class has been depreceated and will not run, please reach out via email (contact@oloren.ai) or raise an issue on GitHub for more details if you have any questions")


def parameterize(object: Union[BaseClass, list, dict, int, float, str, None]) -> dict:
    """parameterize is a recursive method which creates a dictionary of all arguments necessary to instantiate a BaseClass object.

    Note that only objects which are instances of subclasses of BaseClass can be parameterized, other supported objects are to enable to recursive use of parameterize but cannot themselves be parameterized.

    Args:
        object (Union[BaseClass, list, dict, int, float, str, None]): parameterize is a recursive method which creates a dictionary of all arguments necessary to instantiate a BaseClass object.

    Raises:
        ValueError: Object is not of type that can be parameterized

    Returns:
        dict: dictionary of parameters necessary to instantiate the object.
    """
    if issubclass(type(object), BaseClass):
        return {
            **{"BC_class_name": type(object).__name__},
            **{"args": [parameterize(arg) for arg in object.args]},
            **{"kwargs": {k: parameterize(v) for k, v in object.kwargs.items()}},
        }
    elif (
        object is None
        or issubclass(type(object), int)
        or issubclass(type(object), float)
        or issubclass(type(object), str)
    ):
        return object
    elif issubclass(type(object), list):
        return [parameterize(x) for x in object]
    elif issubclass(type(object), dict):
        return {k: parameterize(v) for k, v in object.items()}
    else:
        raise ValueError(f"Invalid object {object}")


def model_name_from_params(param_dict: dict) -> str:
    """model_name_from_params creates a unique name for a model based on the parameters passed to it.

    Args:
        param_dict (dict): dictionary of parameters returned by parameterize neccessary to instantiate the model (note this is different from the instance save)

    Returns:
        str: the model name consisting of the the model class name with a hash of the parameters
    """
    return (
        param_dict["BC_class_name"]
        + " "
        + base64.urlsafe_b64encode(
            hashlib.md5(str(param_dict).encode("utf-8")).digest()
        )[:8].decode("utf-8")
    )


def model_name_from_model(model: BaseClass) -> str:
    """model_name_from_model creates a unique name for a model.

    Args:
        model (BaseClass): the model to be named

    Returns:
        str: the model name consisting of the the model class name with a hash of the parameters
    """
    return model_name_from_params(parameterize(model))


def saves(object: Union[BaseClass, dict, list, int, float, str, None]) -> dict:
    """saves is a method which saves BaseClass object, which can be recovered via loads.

    Args:
        object (Union[BaseClass, dict, list, int, float, str, None]): the object to be saved

    Returns:
        dict: a dictionary which can be passed to loads to recreate the object
    """
    if issubclass(type(object), BaseClass):
        return {
            **{"BC_class_name": type(object).__name__},
            **{"instance_save": object._save()},
            **{"args": [saves(arg) for arg in object.args]},
            **{"kwargs": {k: saves(v) for k, v in object.kwargs.items()}},
        }
    elif (
        issubclass(type(object), int)
        or issubclass(type(object), float)
        or issubclass(type(object), str)
        or issubclass(type(object), bool)
        or issubclass(type(object), bytes)
        or object is None
    ):
        return object
    elif issubclass(type(object), list):
        return [saves(x) for x in object]
    elif issubclass(type(object), dict):
        return {k: saves(v) for k, v in object.items()}
    else:
        print(object)
        raise ValueError


def create_BC(d: dict) -> BaseClass:
    """create_BC is a method which creates a BaseClass object from a dictionary of parameters.

    Note the instances variables of the object are not specified.

    Args:
        d (dict): a dictionary of parameters returned by parameterize

    Returns:
        BaseClass: the object created from the parameters
    """
    if isinstance(d, str):
        d = (
            d.replace("'", '"')
            .replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
        )
        d = json.loads(d)

    args = []
    if "args" in d.keys():
        for arg in d["args"]:
            if isinstance(arg, dict) and (
                "BC_class_name" in arg.keys()
            ):
                args.append(create_BC(arg))
            else:
                if isinstance(arg, list):
                    arg = [
                        create_BC(x)
                        if isinstance(x, dict)
                        and ("BC_class_name" in x.keys())
                        else x
                        for x in arg
                    ]
                elif isinstance(arg, dict):
                    arg = {
                        k: create_BC(v)
                        if isinstance(v, dict)
                        and ("BC_class_name" in v.keys())
                        else v
                        for k, v in arg.items()
                    }
                args.append(arg)

    kwargs = {}
    if "kwargs" in d.keys():
        for k, v in d["kwargs"].items():
            if isinstance(v, dict) and (
                "BC_class_name" in v.keys()
            ):
                kwargs[k] = create_BC(v)
            else:
                kwargs[k] = v

    return BaseClass.Registry()[d["BC_class_name"]](*args, **kwargs)


def loads(d: dict) -> BaseClass:
    """loads is a method which recreates a BaseClass object from a save.

    Args:
        d (dict): the dictionary returned by saves which saves the state of a BaseClass object

    Returns:
        BaseClass: the recreated object
    """

    import olorenchemengine as olorenautoml  # for backwards compatibility

    args = []
    for arg in d["args"]:
        if isinstance(arg, dict) and "BC_class_name" in arg.keys():
            args.append(loads(arg))
        else:
            if isinstance(arg, list):
                arg = [
                    loads(x)
                    if isinstance(x, dict) and "BC_class_name" in x.keys()
                    else x
                    for x in arg
                ]
            args.append(arg)

    kwargs = {
        k: loads(v) if isinstance(v, dict) and "BC_class_name" in v.keys() else v
        for k, v in d["kwargs"].items()
    }
    bc = BaseClass.Registry()[d["BC_class_name"]](*args, **kwargs)
    bc._load(d["instance_save"])
    return bc


def save(model: BaseClass, fname: str):
    """saves a BaseClass object to a file

    Args:
        model (BaseClass): the object to be saved
        fname (str): the file name to save the model to
    """
    save_dict = saves(model)
    with open(fname, "wb+") as f:
        pickle.dump(save_dict, f)


def load(fname: str) -> BaseClass:
    """loads a BaseClass from a file

    Args:
        fname (str): name of the file to load the object from

    Returns:
        BaseClass: the BaseClass object which as been recreated from the file
    """
    with open(fname, "rb") as f:
        d = pickle.load(f)
    return loads(d)


def pretty_params(base: Union[BaseClass, dict]) -> dict:
    """Returns a dictionary of the parameters of the passed BaseClass object, formatted such that they are in a
    human readable format, with the names of the arguments included."""
    if isinstance(base, dict):
        base = loads(base)

    if issubclass(type(base), BaseClass):
        args = list(base.args)
        kwargs = dict(base.kwargs.items())
        base_object_parameters = list(
            inspect.signature(base.__init__).parameters.keys()
        )

        for kwarg in kwargs:
            if kwarg in base_object_parameters:
                base_object_parameters.remove(kwarg)

        labeled_args = dict(zip(base_object_parameters, args))
        fully_labeled_args = {k: pretty_params(v) for k, v in labeled_args.items()}
        fully_labeled_kwargs = {k: pretty_params(v) for k, v in kwargs.items()}

        return {
            **{"BC_Class_name": type(base).__name__},
            **fully_labeled_args,
            **fully_labeled_kwargs,
        }
    elif (
        issubclass(type(base), int)
        or issubclass(type(base), float)
        or issubclass(type(base), str)
        or base is None
    ):
        return base
    elif issubclass(type(base), list):
        return [pretty_params(x) for x in base]
    elif issubclass(type(base), dict):
        return {k: pretty_params(v) for k, v in base.items()}
    else:
        print(type(base))
        raise ValueError


def pretty_params_str(base: Union[BaseClass, dict]) -> str:
    """Returns a string of the parameters of the passed BaseClass object, formatted such that they are in a human
    readable format"""
    return json.dumps(pretty_params(base), indent=4)


def json_params_str(base: Union[BaseClass, dict]) -> str:
    """Returns a json string of the parameters of the passed BaseClass object so that the model parameter dictionary can
    be reconstructed with json.load(params_str)"""
    return (
        json.dumps(pretty_params(base))
        .replace("'", '"')
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "null")
    )

def package_available(package_name: str) -> bool:
    """Checks if a package is available.

    Args:
        package_name (str): the name of the package to check for

    Returns:
        bool: True if the package is available, False otherwise
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_with_permission(package_name: str):
    inp = input(
        f"The required package {package_name} is not installed. Do you want to install it? [y/N]? "
    )
    if inp.lower() == "y":
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    else:
        print(
            f"Stopping program. You can install the package manually with: \n >> pip install {package_name}"
        )
        os._exit(1)

def import_or_install(package_name: str, statement: str = None, scope: dict = None):
    if scope is None:
        scope = globals()
    if statement is None:
        statement = f"import {package_name}"
    try:
        exec(statement, scope)
    except ImportError:
        install_with_permission(package_name)
    finally:
        exec(statement, scope)

def detect_setting(data):
    values, _ = np.unique(data, return_counts=True)
    if len(values) <= 2:
        return "classification"
    else:
        return "regression"

class BaseObject(BaseClass):
    """BaseObject is the parent class for all classes which directly wrap some object to be saved via joblib.

    Attributes:
        obj (object): the object which is wrapped by the BaseObject
    """

    @log_arguments
    def __init__(self, obj=None):
        self.obj = obj

    def _save(self):

        b = io.BytesIO()
        joblib.dump(self.obj, b)
        return {"obj": b.getvalue()}

    def _load(self, d):
        import joblib

        super()._load(d)
        self.obj = joblib.load(io.BytesIO(d["obj"]))


class BaseEstimator(BaseObject):

    """Utility class used to wrap any object with a fit and predict method"""

    def fit(self, X, y):
        """Fit the estimator to the data

        Parameters:
            X (np.array): The data to fit the estimator to
            y (np.array): The target data to fit the estimator to
        Returns:
            self (object): The estimator object fit to the data
        """

        return self.obj.fit(X, y)

    def predict(self, X):
        """Predict the output of the estimator

        Parameters:
            X (np.array): The data to predict the output of the estimator on
        Returns:
            y (np.array): The predicted output of the estimator
        """
        pred = self.obj.predict(X)
        pred = np.array(pred)
        if pred.ndim == 1:
            pred = pred
        else:
            pred = np.array([pred])
        return pred


class LinearRegression(BaseEstimator):

    """Wrapper for sklearn LinearRegression"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.linear_model import LinearRegression as LinearRegression_

        self.obj = LinearRegression_(*args, **kwargs)


class BasePreprocessor(BaseObject):
    """BasePreprocessor is the parent class for all preprocessors which transform the features or properties of a dataset.

    Methods:
        fit: fit the preprocessor to the dataset
        fit_transform: fit the preprocessor to the dataset return the transformed values
        transform: return the transformed values
        inverse_transform: return the original values from the transformed values
    """

    def fit(self, X):
        """Fits the preprocessor to the dataset.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The fit preprocessor instance
        """
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.obj.fit(X)

    def fit_transform(self, X):
        """Fits the preprocessor to the dataset and returns the transformed values.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The transformed values of the dataset as a numpy array
        """
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = self.obj.fit_transform(X)
        return X

    def transform(self, X):
        """Returns the transformed values of the dataset as a numpy array.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The transformed values of the dataset as a numpy array
        """
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = self.obj.transform(X)
        return X

    def inverse_transform(self, X):
        """Returns the original values from the transformed values.

        Parameters:
            X (np.ndarray): the transformed values

        Returns:
            The original values from the transformed values
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = np.array(X)
        return self.obj.inverse_transform(X)


class QuantileTransformer(BasePreprocessor):
    """QuantileTransformer is a BasePreprocessor which transforms a dataset by quantile transformation to specified distribution.

    Attributes:
        obj (sklearn.preprocessing.QuantileTransformer): the object which is wrapped by the BasePreprocessor
    """

    @log_arguments
    def __init__(
        self,
        n_quantiles=1000,
        output_distribution="normal",
        subsample=1e5,
        random_state=None,
    ):

        from sklearn.preprocessing import QuantileTransformer

        self.obj = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            subsample=subsample,
            random_state=random_state,
        )

class StandardScaler(BasePreprocessor):
    """StandardScaler is a BasePreprocessor which standardizes the data by removing the mean and scaling to unit variance.

    Attributes:
        obj (sklearn.preprocessing.StandardScaler): the object which is wrapped by the BasePreprocessor
    """

    @log_arguments
    def __init__(self, with_mean=True, with_std=True):

        from sklearn.preprocessing import StandardScaler

        self.obj = StandardScaler(with_mean=with_mean, with_std=with_std)


class LogScaler(BasePreprocessor):
    """LogScaler is a BasePreprocessor which standardizes the data by taking the log and then removing the mean and scaling to unit variance."""

    @log_arguments
    def __init__(self, min_value=0, with_mean=True, with_std=True):

        from sklearn.preprocessing import StandardScaler
        self.min=min_value
        self.obj = StandardScaler(with_mean=with_mean, with_std=with_std)

    def fit(self, X):
        """Fits the preprocessor to the dataset.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The fit preprocessor instance
        """
        X = np.array(X)
        self.min = np.min(X)
        X = np.log10(X - self.min + 1e-3)
        return self.obj.fit(X.reshape(-1, 1))

    def fit_transform(self, X):
        """Fits the preprocessor to the dataset and returns the transformed values.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The transformed values of the dataset as a numpy array
        """
        X = np.array(X)
        self.min = np.min(X)
        X = np.log10(np.maximum(X - self.min + 1e-3,0))
        result = self.obj.fit_transform(X.reshape(-1, 1)).reshape(-1)
        return result

    def transform(self, X):
        """Returns the transformed values of the dataset as a numpy array.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The transformed values of the dataset as a numpy array
        """
        X = np.log10(np.maximum(np.array(X) - self.min + 1e-3,0))
        return self.obj.transform(X.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, X):
        """Returns the original values from the transformed values.

        Parameters:
            X (np.ndarray): the transformed values

        Returns:
            The original values from the transformed values
        """
        X = np.array(X)
        return (
            10 ** (self.obj.inverse_transform(X.reshape(-1, 1)).reshape(-1))
            + self.min
            - 1e-3
        )

    def _save(self):
        d = super()._save()
        d["min"] = self.min
        return d

    def _load(self, d):
        super()._load(d)
        self.min = d["min"]
        return self

def get_all_reps():
    return list(all_subclasses(BaseRepresentation))

class BaseRepresentation(BaseClass):

    """BaseClass for all molecular representations (PyTorch Geometric graphs, descriptors, fingerprints, etc.)
    Parameters:
        log (boolean): Whether to log the representation or not
    Methods:
        _convert(smiles: str, y: Union[int, float, np.number] = None) -> Any: converts a single structure (represented by a SMILES string) to a representation
        convert(Xs: Union[list, pd.DataFrame, dict, str], ys: Union[list, pd.Series, np.ndarray]=None) -> List[Any]: converts input data to a list of representations
    """

    @log_arguments
    def __init__(self, log=True):
        pass

    @abstractmethod
    def _convert(self, smiles: str, y: Union[int, float, np.number] = None) -> Any:
        """Converts a single structure (represented by a SMILES string) to a representation
        Parameters:
            smiles (str): SMILES string of the structure
            y (Union[int, float, np.number]): target value of the structure
        Returns:
            Any: representation of the structure
        """
        pass

    def _convert_list(
        self, smiles_list: List[str], ys: List[Union[int, float, np.number]] = None
    ) -> List[Any]:
        """Converts a list of structures (represented by a SMILES string) to a list of representations
        Parameters:
            smiles_list (List[str]): list of SMILES strings of the structures
            ys (List[Union[int, float, np.number]]): list of target values of the structures
        Returns:
            List[Any]: list of representations of the structures"""
        if ys is None:
            return [self._convert(s) for s in smiles_list]
            # with mp.Pool(mp.cpu_count()) as p:
            #   X = p.map(self._convert, smiles_list)
            # return X
        else:
            return [self._convert(s, y=y) for s, y in tqdm(zip(smiles_list, ys))]

    def _convert_cache(
        self, smiles: str, y: Union[int, float, np.number] = None
    ) -> Any:
        """Converts a single structure (represented by a SMILES string) to a representation
        Parameters:
            smiles (str): SMILES string of the structure
            y (Union[int, float, np.number]): target value of the structure
        Returns:
            Any: representation of the structure
        """
        if smiles in self.cache.keys():
            return self.cache[smiles]
        else:
            return self._convert(smiles, y=y)

    def convert(
        self,
        Xs: Union[list, pd.DataFrame, dict, str],
        ys: Union[list, pd.Series, np.ndarray] = None,
        **kwargs,
    ) -> List[Any]:
        """Converts input data to a list of representations
        Parameters:
            Xs (Union[list, pd.DataFrame, dict, str]): input data
            ys (Union[list, pd.Series, np.ndarray]=None): target values of the input data
        Returns:
            List[Any]: list of representations of the input data
        """
        if isinstance(Xs, list) and (
            isinstance(Xs[0], list) or isinstance(Xs[0], tuple)
        ):
            smiles = [X[0] for X in Xs]
        elif isinstance(Xs, pd.DataFrame) or isinstance(Xs, dict):
            if isinstance(Xs, pd.DataFrame):
                keys = Xs.columns.tolist()
            elif isinstance(Xs, dict):
                keys = Xs.keys()
            if "smiles" in keys:
                smiles = Xs["smiles"]
            elif "Smiles" in keys:
                smiles = Xs["Smiles"]
            elif "SMILES" in keys:
                smiles = Xs["SMILES"]
            else:
                smiles = Xs.iloc[:, 0].tolist()
        elif isinstance(Xs, str):
            smiles = [Xs]
        else:
            smiles = Xs

        return self._convert_list(smiles, ys=ys)

    def _save(self):
        return {}

    def _load(self, d):
        pass

class BaseVecRepresentation(BaseRepresentation):
    """Representation where given input data, returns a vector representation for each compound."""

    @log_arguments
    def __init__(
        self,
        *args,
        collinear_thresh=1.01,
        scale=StandardScaler(),
        names=None,
        log=True,
        **kwargs,
    ):
        self.collinear_thresh = collinear_thresh
        self.to_drop = None
        if not scale is None:
            scale = scale.copy()
        self.scale = scale
        self.scale_fitted = False
        self._names = names

        from os import path

        if not path.exists(
            path.join(
                oce.CONFIG["CACHE_PATH"], f"vecrep/{self.__class__.__name__}/"
            )
        ):
            os.mkdir(
                path.join(
                oce.CONFIG["CACHE_PATH"], f"vecrep/{self.__class__.__name__}/"
                )
            )

        super().__init__(*args, log=False, **kwargs)

    @property
    def names(self):
        if not self._names is None:
            return self._names
        else:
            raise ValueError(
                f"Names not set for representation {self.__class__.__name__}"
            )

    def convert(
        self,
        Xs: Union[list, pd.DataFrame, dict, str],
        ys: Union[list, pd.Series, np.ndarray] = None,
        lambda_convert: Callable = None,
        fit=False,
        **kwargs,
    ) -> List[np.ndarray]:
        """BaseVecRepresentation's convert returns a list of numpy arrays.

        Args:
            Xs (Union[list, pd.DataFrame, dict, str]): input data
            ys (Union[list, pd.Series, np.ndarray], optional): included for compatibility, unused argument. Defaults to None.

        Returns:
            List[np.ndarray]: list of molecular vector representations
        """
        import joblib

        input_hash = (
            joblib.hash(Xs)
            + joblib.hash(ys)
            + joblib.hash(self._save())
            + joblib.hash(oce.parameterize(self))
        )

        from os import path

        if oce.CONFIG["CACHE"] and path.exists(
            path.join(
                oce.CONFIG["CACHE_PATH"],
                f"vecrep/{self.__class__.__name__}/{input_hash}.npy",
            )
        ):
            return np.load(
                path.join(
                    oce.CONFIG["CACHE_PATH"],
                    f"vecrep/{self.__class__.__name__}/{input_hash}.npy",
                ),
                allow_pickle=True,
            )
        if not lambda_convert is None:
            feats = lambda_convert(Xs, ys)
        else:
            feats = super().convert(Xs, ys)
        import pandas as pd

        feats = pd.DataFrame.from_records(
            feats, columns=[f"col{i}" for i in range(len(feats[0]))]
        )
        if fit and len(Xs) > 2:
            # collinear
            corr_matrix = feats.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
            )
            self.to_drop = [
                column
                for column in upper.columns
                if any(upper[column] > self.collinear_thresh)
            ]
            feats = feats.drop(columns=self.to_drop)
            # scale
            if not self.scale is None:
                x = self.scale.fit_transform(feats.values)
                feats.values[:] = np.nan_to_num(x.reshape(feats.values.shape))
                self.scale_fitted = True
        else:
            if not self.to_drop is None:
                feats = feats.drop(columns=self.to_drop)
            # scale
            if self.scale_fitted and self.scale is not None:
                x = self.scale.transform(feats.values)
                feats.values[:] = np.nan_to_num(x.reshape(feats.values.shape))

        output = np.nan_to_num(np.array(feats.to_records(index=False).tolist()))
        if oce.CONFIG["CACHE"]:
            np.save(
                path.join(
                    oce.CONFIG["CACHE_PATH"],
                    f"vecrep/{self.__class__.__name__}/{input_hash}.npy",
                ),
                output,
                allow_pickle=True,
            )
        return output

    def calculate_distance(
        self,
        x1: Union[str, List[str]],
        x2: Union[str, List[str]],
        metric: str = "cosine",
        **kwargs,
    ) -> np.ndarray:
        """Calculates the distance between two molecules or list of molecules.

        Returns a 2D array of distances between each pair of molecules of shape
        len(x1) by len(x2).

        This uses pairwise_distances from sklearn.metrics to calculate distances
        between the vector representations of the molecules. Options for distances
        are Valid values for metric are:

            From scikit-learn: [???cityblock???, ???cosine???, ???euclidean???, ???l1???, ???l2???,
                ???manhattan???]. These metrics support sparse matrix inputs.
                [???nan_euclidean???] but it does not yet support sparse matrices.
            From scipy.spatial.distance: [???braycurtis???, ???canberra???, ???chebyshev???,
                ???correlation???, ???dice???, ???hamming???, ???jaccard???, ???kulsinski???,
                ???mahalanobis???, ???minkowski???, ???rogerstanimoto???, ???russellrao???,
                ???seuclidean???, ???sokalmichener???, ???sokalsneath???, ???sqeuclidean???,
                ???yule???].

            See the documentation for scipy.spatial.distance for details on these metrics.
        """

        from sklearn.metrics import pairwise_distances

        if isinstance(x1, str):
            x1 = [x1]
        if isinstance(x2, str):
            x2 = [x2]
        x1 = self.convert(x1)
        x2 = self.convert(x2)
        return pairwise_distances(x1, x2, metric=metric, **kwargs)

    def __add__(self, other):
        """Adds two representations together

        Parameters:
            other (BaseVecRepresentation): representation to add to the current representation

        Returns:
            BaseVecRepresentation: new representation"""
        return ConcatenatedVecRepresentation(self, other)

    def _save(self):
        d = {}
        d.update({"collinear_thresh": self.collinear_thresh})
        d.update({"to_drop": self.to_drop})
        if not self.scale is None:
            d.update({"scale": self.scale._save()})
            d.update({"scale_fitted": self.scale_fitted})
        return d

    def _load(self, d):
        self.collinear_thresh = d["collinear_thresh"]
        self.to_drop = d["to_drop"]
        if not self.to_drop is None:
            self.to_drop = set(self.to_drop)
        if not self.scale is None and "scale" in d:
            self.scale._load(d["scale"])
            self.scale_fitted = d["scale_fitted"]


class ConcatenatedVecRepresentation(BaseVecRepresentation):
    """Creates a structure vector representation by concatenating multiple representations.

    Parameters:
        rep1 (BaseVecRepresentation): first representation to concatenate
        rep2 (BaseVecRepresentation): second representation to concatenate
        log (bool): whether to log the representations or not

    Can be created by adding two representations together using + operator.

    Example
    ------------------------------
    import olorenchemengine as oce
    combo_rep = oce.MorganVecRepresentation(radius=2, nbits=2048) + oce.Mol2Vec()
    model = oce.RandomForestModel(representation = combo_rep, n_estimators = 1000)

    model.fit(train['Drug'], train['Y'])
    model.predict(test['Drug'])
    ------------------------------
    """

    @log_arguments
    def __init__(
        self,
        rep1: BaseVecRepresentation,
        rep2: BaseVecRepresentation,
        log=True,
        **kwargs,
    ):
        self.rep1 = rep1
        self.rep2 = rep2
        try:
            self._names = rep1.names + rep2.names
        except Exception as e:
            self._names = None
        super().__init__(names=self._names, log=False, **kwargs)

    def _convert(self, smiles, y=None, fit=False):
        converted_1 = self.rep1._convert(smiles, y=y, fit=fit)
        converted_2 = self.rep2._convert(smiles, y=y, fit=fit)
        return np.concatenate((converted_1, converted_2))

    def _convert_list(self, smiles_list, ys=None, fit=False):
        converted_1 = self.rep1._convert_list(smiles_list, ys=ys, fit=fit)
        converted_2 = self.rep2._convert_list(smiles_list, ys=ys, fit=fit)
        return np.concatenate((converted_1, converted_2), axis=1)

    def convert(self, smiles_list, ys=None, fit=False, **kwargs):
        converted_1 = self.rep1.convert(smiles_list, ys=ys, fit=fit)
        converted_2 = self.rep2.convert(smiles_list, ys=ys, fit=fit)
        return np.concatenate((converted_1, converted_2), axis=1)

class SMILESRepresentation(BaseRepresentation):
    """Extracts the SMILES strings from inputted data

    Methods:
        convert(Xs: Union[list, pd.DataFrame, dict, str], ys: Union[list, pd.Series, np.ndarray]=None) -> List[Any]: converts input data to a list of SMILES strings
            Data types:
                pd.DataFrames will have columns "smiles" or "Smiles" or "SMILES" extracted
                lists and tuples of multiple elements will have their first element extracted
                strings will be converted to a list of one element
                everything else will be returned as inputted
    """

    def _convert(self, smiles, y=None):
        pass

    def convert(self, Xs, ys=None, **kwargs):
        if isinstance(Xs, list) and (
            isinstance(Xs[0], list) or isinstance(Xs[0], tuple)
        ):
            smiles = [X[0] for X in Xs]
        elif isinstance(Xs, pd.DataFrame) or isinstance(Xs, dict):
            if isinstance(Xs, pd.DataFrame):
                keys = Xs.columns.tolist()
            elif isinstance(Xs, dict):
                keys = Xs.keys()
            if "smiles" in keys:
                smiles = Xs["smiles"]
            elif "Smiles" in keys:
                smiles = Xs["Smiles"]
            elif "SMILES" in keys:
                smiles = Xs["SMILES"]
            elif "smi" in keys:
                smiles = Xs["smi"]
            else:
                smiles = Xs.iloc[:, 0]
        elif isinstance(Xs, str):
            smiles = [Xs]
        elif isinstance(Xs, pd.Series):
            smiles = Xs.tolist()
        else:
            smiles = Xs

        if isinstance(smiles, pd.Series):
            smiles = smiles.tolist()
        return smiles