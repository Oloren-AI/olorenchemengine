import contextlib
from locale import D_FMT
import sys
import contextlib

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "olorenchemengine"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import logging

logging.getLogger().setLevel(logging.ERROR)

import os
from os import path

if not path.exists(path.join(path.expanduser("~"), f".oce/")):
    os.mkdir(path.join(path.expanduser("~"), f".oce/"))

if not path.exists(path.join(path.expanduser("~"), f".oce/cache/")):
    os.mkdir(path.join(path.expanduser("~"), f".oce/cache/"))

if not path.exists(path.join(path.expanduser("~"), f".oce/cache/vecrep/")):
    os.mkdir(path.join(path.expanduser("~"), f".oce/cache/vecrep/"))

import pandas as pd

import json


CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".oce/CONFIG.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        CONFIG_ = json.load(f)
else:
    CONFIG_ = {"MAP_LOCATION": "cpu", "USE_CUDA": False, "CDD_TOKEN": None, "VAULT_ID": None, "NUM_WORKERS": 4}
    with open(CONFIG_PATH, "w+") as f:
        json.dump(CONFIG_, f)

CONFIG = CONFIG_.copy()

def update_config():
    """Update the configuration file.

    This function is called when a new parameter is added to the configuration file.
    """

    global CONFIG
    CONFIG = CONFIG_.copy()

    with contextlib.suppress(ImportError):
        import torch
        CONFIG["DEVICE"] = torch.device(CONFIG["MAP_LOCATION"])
        CONFIG["USE_CUDA"] = "cuda" in CONFIG["MAP_LOCATION"]

    with open(CONFIG_PATH, "w+") as f:
        json.dump(CONFIG_, f)


def set_config_param(param, value):
    """Set a configuration parameter.

    Parameters:
        param: the parameter to set.
        value: the value to set the parameter to.
    """
    CONFIG_[param] = value
    update_config()


def remove_config_param(param):
    """Remove a configuration parameter.

    Parameters:
        param: the parameter to remove.
    """
    CONFIG_.pop(param)
    update_config()


update_config()

def ExampleDataFrame():
    return pd.read_csv("gs://oloren-public-data/sample-csvs/sample_data3.csv")

from .base_class import *
from .basics import *
from .ensemble import *
from .gnn import *
from .representations import *
from .external import *
from .splitters import *
from .dataset import *
from .interpret import *
from .manager import *
from .visualizations import *
from .uncertainty import *
from .benchmarks import *
from .hyperparameters import *

def ExampleDataset():
    if os.path.exists(path.join(path.expanduser("~"), f".oce/exampledataset.oce")):
        return load(path.join(path.expanduser("~"), f".oce/exampledataset.oce"))
    else:
        dataset =  BaseDataset(data = ExampleDataFrame().to_csv(), structure_col = "Smiles",
            property_col = "pChEMBL Value") + RandomSplit()
        save(dataset, path.join(path.expanduser("~"), f".oce/exampledataset.oce"))
        return dataset

def BACEDataset():
    df = pd.read_csv(download_public_file("MoleculeNet/load_bace_regression.csv"))
    df["split"] = df["split"].replace({"Train": "train", "Valid": "valid", "Test": "test"})
    return oce.BaseDataset(data = df.to_csv(), structure_col = "smiles", property_col = "pIC50")

def test_oce():
    """Convenience function to test all functions of the oce package."""

    df = ExampleDataFrame()

    model = BaseBoosting(
        [
            RandomForestModel(DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1),
            BaseTorchGeometricModel(TLFromCheckpoint("default"), batch_size=8, epochs=1, preinitialized=True),
            RandomForestModel(OlorenCheckpoint("default"), n_estimators=1),
        ]
    )

    model.fit(df["Smiles"], df["pChEMBL Value"])
    _ = model.predict(["CC(=O)OC1=CC=CC=C1C(=O)O"])
    save(model, "model.oce")
    _ = load("model.oce")
    os.remove("model.oce")

