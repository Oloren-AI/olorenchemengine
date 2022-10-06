import base64
import hashlib
import inspect
import json
import os
import pickle
from abc import ABC, abstractmethod
from typing import Callable, Union
import gcsfs
import subprocess
import pyrebase  # Default pyrebase is pyrebase3 which won't work. Need to install pyrebase4 (pip install pyrebase4)
from google.cloud.firestore import Client
from google.oauth2.credentials import Credentials
import sys
from unittest.mock import MagicMock

import olorenchemengine

sys.modules["olorenautoml"] = olorenchemengine # important for backwards compatibility of some models

def mock_imports(g, *args):
    for arg in args:
        g[arg] = MagicMock()

def all_subclasses(cls):
    """ Helper function to return all subclasses of class"""
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])

class OASConnector:
    """ Class which links oce to OAS and can move data between them using Firestore and an API"""

    def __init__(self) -> None:
        self.storage = None

    def authenticate(self):
        import json

        import requests
        from requests.exceptions import HTTPError

        FIREBASE_REST_API = "https://identitytoolkit.googleapis.com/v1/accounts"
        FIREBASE_API_KEY = "AIzaSyCSaFE5-qxy5-HAr-oqUd5del4bv5dqJ0Q"

        # Use the google verify password REST api to authenticate and generate user tokens
        def log_in_first_time(token):
            request_url = f"{FIREBASE_REST_API}:signInWithCustomToken?key={FIREBASE_API_KEY}"
            headers = {"content-type": "application/json; charset=UTF-8"}
            data = json.dumps({"token": token, "returnSecureToken": True})
            resp = requests.post(request_url, headers=headers, data=data)

            # Check for errors
            try:
                resp.raise_for_status()
            except HTTPError as e:
                raise HTTPError(e, resp.text)

            return resp.json()

        def get_id_token_from_refresh(refresh_token):
            request_url = f"https://securetoken.googleapis.com/v1/token?key={FIREBASE_API_KEY}"
            headers = {"content-type": "application/json; charset=UTF-8"}
            data = json.dumps({"refresh_token": refresh_token, "grant_type": "refresh_token"})
            resp = requests.post(request_url, headers=headers, data=data)
            try:
                resp.raise_for_status()
            except HTTPError as e:
                raise HTTPError(e, resp.text)

            return resp.json()

        from os.path import expanduser

        home = expanduser("~")
        PATH_TO_REFRESH_TOKEN = f"{home}/.oce/firebase_refresh_token.json"

        if not os.path.exists(PATH_TO_REFRESH_TOKEN):
            print("Navigate to https://oas.oloren.ai/auth to get auth token.")
            response = log_in_first_time(input("Paste token here: "))
            if not os.path.exists(f"{home}/.oce"):
                os.makedirs(f"{home}/.oce")
            with open(PATH_TO_REFRESH_TOKEN, "w") as f:
                f.write(json.dumps({"refreshToken": response["refreshToken"]}))

        with open(PATH_TO_REFRESH_TOKEN, "r") as f:
            refresh_token = json.loads(f.read())["refreshToken"]
            response = get_id_token_from_refresh(refresh_token)
            response["idToken"] = response["id_token"]
            response["refreshToken"] = response["refresh_token"]
            response["uid"] = response["user_id"]

        with open(PATH_TO_REFRESH_TOKEN, "w") as f:
            f.write(json.dumps({"refreshToken": response["refreshToken"]}))

        creds = Credentials(response["idToken"], response["refreshToken"])

        """Storage"""

        config = {
            "apiKey": "AIzaSyCSaFE5-qxy5-HAr-oqUd5del4bv5dqJ0Q",
            "authDomain": "oloren-ai.firebaseapp.com",
            "databaseURL": "https://oloren-ai-default-rtdb.firebaseio.com",
            "projectId": "oloren-ai",
            "storageBucket": "oloren-ai.appspot.com",
            "messagingSenderId": "602366687071",
            "appId": "1:602366687071:web:f35531f7142084b86a28ea",
            "measurementId": "G-MVNK4DPQ60",
        }
        firebase = pyrebase.initialize_app(config)

        auth = firebase.auth()
        user = auth.refresh(response["refreshToken"])

        self.storage = firebase.storage()
        self.logging_db = Client("oloren-ai", creds)
        self.uid = response["uid"]
        self.uid_token = response["idToken"]

    def upload_vis(self, visualization):
        self.authenticate()

        out = self.logging_db.collection("visualizations").add(
            {
                "uid": self.uid,
                "status": "Visualization rendered",
                "data_url": visualization.render_data_url(),
            }
        )

        print(f"Uploaded visualization to https://oas.oloren.ai/vis?vid={out[1].id}")

        return out


    def upload_model(self, model, model_name):
        self.authenticate()

        import tempfile

        with tempfile.NamedTemporaryFile() as tmp:
            pickle.dump(saves(model), tmp)
            tmp.flush()
            self.storage.child(self.uid + "/models/" + model_name + ".pkl").put(tmp.name, self.uid_token)

        out = self.logging_db.collection("models").add(
            {
                "uid": self.uid,
                "did": None,
                "source": "oce",
                "data": {
                    "file_path": "gs://oloren-ai.appspot.com/" + self.uid + "/models/" + model_name + ".pkl",
                    "name": model_name,
                    "description": pretty_params_str(model),
                    "parameters": json.dumps(parameterize(model)),
                    "metrics": None,
                    "status": "Runnable",
                },
            }
        )

        return out

oas_connector = OASConnector()

ignored_kwargs = ["map_location", "num_workers"]

def download_public_file(path, redownload=False):
    """Download a public file from Oloren's Public Storage, and returns the contents.

    @param path: The path to the file to read.
    @param redownload: Whether to redownload the file if it already exists.
    """

    local_path = os.path.join(os.path.expanduser("~"), ".oce", path)

    if os.path.exists(local_path) and not redownload: return local_path

    if not os.path.exists(os.path.dirname(local_path)):
        os.makedirs(os.path.dirname(local_path))

    print(f"Downloading {path}...")
    fs = gcsfs.GCSFileSystem()
    with fs.open(f'gs://oloren-public-data/{path}', 'rb') as f:
        with open(local_path, 'wb') as out:
            out.write(f.read())

    return local_path


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

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

        if "log" in kwargs.keys() and not kwargs["log"]:
            pass
        else:
            kwds = get_default_args(func)
            for k, v in kwds.items():
                if not k in kwargs.keys():
                    kwargs.update({k: v})
            self.args = args
            self.kwargs = {k: v for k, v in kwargs.items() if not k in ignored_kwargs}
        return func(self, *args, **kwargs)

    wrapper.__wrapped__ = func

    return wrapper

class BaseClass(ABC):
    """BaseClass is the base class for all models.

    All classes in Oloren ChemEngine should inherit from BaseClass to enable for universal saving and loading of both
    parameters and internal state. This requires the implementation of abstract methods _save and _load.

    Methods:
        Registry: returns a dictionary mapping the name of a class to the class itself for all subclasses of the class.
        _save: saves an instance of a BaseClass to a dictionary (abstract method to be implmented by subclasses)
        _load: loads an instance of a BaseClass from a dictionary (abstract method to be implmented by subclasses)
    """

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
        if not hasattr(cls, "__abstractmethods__") or len(cls.__abstractmethods__) == 0:
            try:
                return [cls()] + [o for sc in cls.__subclasses__() for o in sc.AllInstances()]
            except:
                return [o for sc in cls.__subclasses__() for o in sc.AllInstances()]
        else:
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


def parameterize(object: Union[BaseClass, list, int, float, str, None]) -> dict:
    """parameterize is a recursive method which creates a dictionary of all arguments necessary to instantiate a BaseClass object.

    Note that only objects which are instances of subclasses of BaseClass can be parameterized, other supported objects are to enable to recursive use of parameterize but cannot themselves be parameterized.

    Args:
        object (Union[BaseClass, list, int, float, str, None]): parameterize is a recursive method which creates a dictionary of all arguments necessary to instantiate a BaseClass object.

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
    else:
        print(object)
        raise ValueError


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
        + base64.urlsafe_b64encode(hashlib.md5(str(param_dict).encode("utf-8")).digest())[:8].decode("utf-8")
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
        d = d.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
        d = json.loads(d)
    args = []
    if "args" in d.keys():
        for arg in d["args"]:
            if isinstance(arg, dict) and "BC_class_name" in arg.keys():
                args.append(create_BC(arg))
            else:
                if isinstance(arg, list):
                    arg = [create_BC(x) if isinstance(x, dict) and "BC_class_name" in x.keys() else x for x in arg]
                args.append(arg)

    kwargs = {}
    if "kwargs" in d.keys():
        for k, v in d["kwargs"].items():
            if isinstance(v, dict) and "BC_class_name" in v.keys():
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

    import olorenchemengine as olorenautoml # for backwards compatibility

    args = []
    for arg in d["args"]:
        if isinstance(arg, dict) and "BC_class_name" in arg.keys():
            args.append(loads(arg))
        else:
            if isinstance(arg, list):
                arg = [loads(x) if isinstance(x, dict) and "BC_class_name" in x.keys() else x for x in arg]
            args.append(arg)

    kwargs = {k: loads(v) if isinstance(v, dict) and "BC_class_name" in v.keys() else v for k, v in d["kwargs"].items()}
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
    """ Returns a dictionary of the parameters of the passed BaseClass object, formatted such that they are in a
    human readable format, with the names of the arguments included."""
    if isinstance(base, dict):
        base = loads(base)

    if issubclass(type(base), BaseClass):
        args = [arg for arg in base.args]
        kwargs = {k: v for k, v in base.kwargs.items()}
        base_object_parameters = list(inspect.signature(base.__init__).parameters.keys())

        for kwarg in kwargs:
            if kwarg in base_object_parameters:
                base_object_parameters.remove(kwarg)

        labeled_args = {k: v for k, v in zip(base_object_parameters, args)}

        fully_labeled_args = {k: pretty_params(v) for k, v in labeled_args.items()}
        fully_labeled_kwargs = {k: pretty_params(v) for k, v in kwargs.items()}

        return {**{"BC_Class_name": type(base).__name__}, **fully_labeled_args, **fully_labeled_kwargs}
    elif issubclass(type(base), int) or issubclass(type(base), float) or issubclass(type(base), str) or base is None:
        return base
    elif issubclass(type(base), list):
        return [pretty_params(x) for x in base]
    elif issubclass(type(base), dict):
        return {k: pretty_params(v) for k, v in base.items()}
    else:
        print(type(base))
        raise ValueError


def pretty_params_str(base: Union[BaseClass, dict]) -> str:
    """ Returns a string of the parameters of the passed BaseClass object, formatted such that they are in a human
    readable format"""
    return json.dumps(pretty_params(base), indent=4)


def json_params_str(base: Union[BaseClass, dict]) -> str:
    """ Returns a json string of the parameters of the passed BaseClass object so that the model parameter dictionary can
    be reconstructed with json.load(params_str)"""
    return (
        json.dumps(pretty_params(base))
        .replace("'", '"')
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "null")
    )

def install_with_permission(package_name: str):
    inp = input(f"The required package {package_name} is not installed. Do you want to install it? [y/N]? ")
    if inp.lower() == "y":
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    else:
        print(f"Stopping program. You can install the package manually with: \n >> pip install {package_name}")
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