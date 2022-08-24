""" `automl` allows for the automatic selection and refinement of models for a given dataset.
"""
import json
import os
import random

from abc import ABC, abstractclassmethod
from typing import List, Tuple

from .base_class import *


class BaseAutoML(BaseClass):
    """ Base class for creating automl classes, which each have a get_model method with returns a suggested model to try.

    Methods:
        get_model: returns a suggested model parameters dictionary.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_model(cls, dataset = None, past_models = [], structure_column = None, value_columns=[], input_columns = []):
        """ Returns a suggested model parameters dictionary.

        Parameters:
            dataset: the dataset to use for the model.
            past_models: a list of past models and their performance.
            structure_column: the column to use for the structure.
            value_columns: the columns to use for the values.
            input_columns: the columns to use for the input.

        Returns:
            A suggested model parameters dictionary.
        """
        pass

    def _save(self):
        pass

    def _load(self):
        pass

import os

class RandomWalk(BaseAutoML):
    """ RandomWalk is a class that returns a selects model to try via a random walk with edges weighted by relative model performance statistics.

    Attributes:
        model_graph: a graph of model parameters and their relative performances.
    """
    def __init__(self, model_graph = None, force_reload = False):
        if not model_graph is None:
            self.model_graph = model_graph
        elif os.path.exists(os.path.join(os.path.dirname(__file__), "../saves/model_graph.json")) and not force_reload:
            self.model_graph = json.load(os.path.join(os.path.dirname(__file__), "../saves/model_graph.json"))
        else:
            import google.cloud.storage as storage
            client = storage.Client()
            storage_url = "oloren-ai.appspot.com"
            bucket = client.get_bucket(storage_url)
            blob = bucket.blob(f"internal/model_graph.json")
            self.model_graph = json.loads(blob.download_as_string(client=None))

    def get_model(self, dataset=None, past_models: List[Tuple[BaseModel, float]]=[], structure_column=None, value_columns=[], input_columns = []):
        from numpy import prod, average
        # performance should be normalized between 0 and 1 with 1 being the best
        weights = {k:min(1/np.prod([np.average(v2) if hasattr(v2, "__len__") and len(v2)> 0 else 1 for k2, v2 in v.items()]),8) for k, v in self.model_graph["graph"].items()}
        print(weights)

        past_model_names = list()
        if past_models is not None:
            for model, performance in past_models:
                model_name = model_name_from_model(model)
                past_model_names.append(model_name)
                if model_name in self.model_graph["graph"].keys():
                    for k, v in self.model_graph["graph"][model_name]:
                        weights[k] = weights[k]*2*performance*np.prod(v)

        model_names = list(weights.keys())
        weights = [weights[model_name] for model_name in model_names]
        i=0
        while True:
            model_choice = random.choices(model_names, weights= weights)[0]
            if not model_choice in past_models:
                break
            if i>100:
                break
        return self.model_graph["model_dict"][model_choice]

class QueuedAutoML(BaseAutoML):
    """ QueuedAutoML is a class that returns a selects model to try via a simple queue of models.

    Attributes:
        queue: a queue of model parameters dictionaries to be tried."""


    @log_arguments
    def __init__(self):
        self.i = 0
        template_models_path = os.path.join(os.path.dirname(__file__), "template_models.json")
        with open(template_models_path) as f:
            self.queue = json.load(f)

    def get_model(self, dataset=None, past_models=[], structure_column=None, value_columns=[], input_columns = []):
        self.i += 1
        return list(self.queue.items())[self.i - 1][1]

    def save_(self):
        return {"i": self.i}

    def load_(self, d):
        self.i = d["i"]

class NaiveSelection(BaseAutoML):
    """ NaiveSelection is a class that returns a selects model to try via random selection of models weighted by performance statistics.

    Attributes:
        modeldatabase_struct: for models using only structure, a pd.DataFrame of model parameters and their performance statistic.
        modeldatabase_structfeatures: for models using structure and user-inputted features, a pd.DataFrame of model parameters and their performance statistic."""

    def __init__(self):
        super().__init__()
        modeldatabase_struct_path = download_public_file("ModelDatabase/ModelDatabase_Struct.csv")
        modeldatabase_structfeatures_path = download_public_file("ModelDatabase/ModelDatabase_StructFeatures.csv")

        with open(modeldatabase_struct_path) as f:
            self.modeldatabase_struct = pd.read_csv(f)

        with open(modeldatabase_structfeatures_path) as f:
            self.modeldatabase_structfeatures = pd.read_csv(f)

    def get_model(self, dataset=None, past_models=[], structure_column=None, value_columns=[], input_columns = []):
        from numpy.random import choice
        i = 0
        while True:
            print(i)
            if input_columns is None or len(input_columns) == 0:
                mp = choice(self.modeldatabase_struct["Parameters"], p=self.modeldatabase_struct["Weight"]/sum(self.modeldatabase_struct["Weight"]))
            else:
                mp = choice(self.modeldatabase_structfeatures["Parameters"], p=self.modeldatabase_structfeatures["Weight"]/sum(self.modeldatabase_structfeatures["Weight"]))
            mp = json.loads(mp.replace("\'","\"").replace("False", "false").replace("True", "true"))
            mid = model_name_from_params(mp)
            print(mid)
            print(mp)
            i+=1
            if not mid in past_models or i>100:
                return mp

class QuickAutoML(BaseAutoML):
    """ QuickAutoML is a class that returns a selects model via a random selection of models which run speedily weighted by performance statistics. """
    modeldatabase_struct_path = download_public_file("ModelDatabase/ModelDatabase_quick.csv")

    with open(modeldatabase_struct_path) as f:
        modeldatabase_struct = pd.read_csv(f)

    def get_model(self, dataset=None, past_models=[], structure_column=None, value_columns=[], input_columns = []):
        i = 0
        while True:
            mp = np.random.choice(self.modeldatabase_struct["Parameters"], p=self.modeldatabase_struct["Prob Weight"]/sum(self.modeldatabase_struct["Prob Weight"]))

            mp = json.loads(mp.replace("\'","\"").replace("False", "false").replace("True", "true"))
            mid = model_name_from_params(mp)
            print(mid)
            print(mp)
            i+=1
            if not mid in past_models or i>100:
                return mp