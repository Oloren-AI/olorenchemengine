"""
Contains the basic framework for hyperparameter optimization.

We use hyperopt as our framework for hyperparameter optimization, and the class Opt
functions as the bridge between olorenchemengine and hyperopt. Hyperparameters are defined
in Opt which is used as an argument in a BaseClass object's instantiation. These
hyperparameters are then collated and used for hyperparameter optimization.

The following is a brief introduction to hyperopt and is a useful starting point for
understanding our hyperparameter optimization engine:
https://github.com/hyperopt/hyperopt/wiki/FMin.
"""

from abc import abstractproperty
from typing import *

from .base_class import *
from .manager import *
from .benchmarks import *

from hyperopt import hp, fmin, tpe, hp
from hyperopt.pyll import scope

class Opt(BaseClass):

    @abstractproperty
    def get_hp(self):
        pass

    @log_arguments
    def __init__(self, label, *args, use_int = False, **kwargs):
        self.label = label
        if use_int:
            self.hp = scope.int(self.get_hp(label, *args, **kwargs))
        else:
            self.hp = self.get_hp(label, *args, **kwargs)

    def _load(self, d):
        pass

    def _save(self):
        return {}

class OptChoice(Opt):

    @property
    def get_hp(self):
        return hp.choice

class OptRandInt(Opt):

    @property
    def get_hp(self):
        return hp.randint

class OptUniform(Opt):

    @property
    def get_hp(self):
        return hp.uniform

class OptQUniform(Opt):

    @property
    def get_hp(self):
        return hp.quniform

class OptLogUniform(Opt):

    @property
    def get_hp(self):
        return hp.loguniform

class OptQLogUniform(Opt):

    @property
    def get_hp(self):
        return hp.qloguniform

class OptQNormal(Opt):

    @property
    def get_hp(self):
        return hp.qnormal

class OptLogNormal(Opt):

    @property
    def get_hp(self):
        return hp.lognormal

class OptQLogNormal(Opt):

    @property
    def get_hp(self):
        return hp.qnormal

def index_hyperparameters(object: BaseClass) -> dict:
    """
    Returns a dictionary of hyperparameters for the model.
    """
    if issubclass(type(object), Opt):
        d = {object.label: object.hp}
        for arg in object.args:
            d.update(index_hyperparameters(arg))
        return d
    if issubclass(type(object), BaseClass):
        d = {}
        for arg in object.args:
            count = len(d)
            d_ = index_hyperparameters(arg)
            d.update(d_)
            if not len(d) == count + len(d_):
                raise Exception(f"Hyperparameter indexing failed, overlapping keys in {d_}")
        for k, v in object.kwargs.items():
            count = len(d)
            d_ = index_hyperparameters(v)
            d.update(d_)
            if not len(d) == count + len(d_):
                raise Exception(f"Hyperparameter indexing failed, overlapping keys in {d_}")
        return d
    elif issubclass(type(object), dict):
        d = {}
        for arg in object["args"]:
            count = len(d)
            d_ = index_hyperparameters(arg)
            d.update(d_)
            if not len(d) == count + len(d_):
                raise Exception(f"Hyperparameter indexing failed, overlapping keys in {d_}")
        for k, v in object["kwargs"].items():
            count = len(d)
            d_ = index_hyperparameters(v)
            d.update(d_)
            if not len(d) == count + len(d_):
                raise Exception(f"Hyperparameter indexing failed, overlapping keys in {d_}")
        return d
    elif issubclass(type(object), list):
        d = {}
        for x in object:
            count = len(d)
            d_ = index_hyperparameters(x)
            d.update(d_)
            if not len(d) == count + len(d_):
                raise Exception(f"Hyperparameter indexing failed, overlapping keys in {d_}")
        return d
    elif (
        object is None
        or issubclass(type(object), int)
        or issubclass(type(object), float)
        or issubclass(type(object), str)
    ):
        return {}
    else:
        print(object)
        raise ValueError

def load_hyperparameters_(object: BaseClass, hyperparameter_dictionary: dict) -> dict:
    if issubclass(type(object), Opt):
        return load_hyperparameters_(hyperparameter_dictionary[object.label], hyperparameter_dictionary)

    if issubclass(type(object), BaseClass):
        return {
            **{"BC_class_name": type(object).__name__},
            **{"args": [load_hyperparameters_(arg, hyperparameter_dictionary) for arg in object.args]},
            **{"kwargs": {k: load_hyperparameters_(v, hyperparameter_dictionary) for k, v in object.kwargs.items()}},
        }
    elif issubclass(type(object), dict):
        return {
            **{"BC_class_name": object["BC_class_name"]},
            **{"args": [load_hyperparameters_(arg, hyperparameter_dictionary) for arg in object["args"]]},
            **{"kwargs": {k: load_hyperparameters_(v, hyperparameter_dictionary) for k, v in object["kwargs"].items()}},
        }
    elif (
        object is None
        or issubclass(type(object), int)
        or issubclass(type(object), float)
        or issubclass(type(object), str)
    ):
        return object
    elif issubclass(type(object), list):
        return [load_hyperparameters_(x, hyperparameter_dictionary) for x in object]
    else:
        print(object)
        raise ValueError

def load_hyperparameters(object: BaseClass, hyperparameter_dictionary: dict) -> dict:
    mp = load_hyperparameters_(object, hyperparameter_dictionary)
    return oce.create_BC(mp)

def optimize(model: Union[BaseModel, dict], manager: BaseModelManager, max_evals = 3):
    hyperparameter_index = index_hyperparameters(model)

    def objective(hyperparameter_dictionary, model = model, manager = manager):
        model = load_hyperparameters(model, hyperparameter_dictionary)
        metric = manager.run(model)
        return metric

    best = fmin(fn=objective,
        space=hyperparameter_index,
        algo=tpe.suggest,
        max_evals=max_evals,)
    return best