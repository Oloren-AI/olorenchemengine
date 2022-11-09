import contextlib
import sys

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


_OPTIONAL_IMPORTS_FOR_OCE_ONLINE = [
    "sklearn",
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "rdkit",
    "sklearn.metrics",
    "scipy",
    "scipy.stats",
    "joblib",
    "sklearn.model_selection",
    "tqdm",
    "sklearn.preprocessing",
    "rdkit.Chem",
    "torch.nn",
    "torch_geometric.nn",
    "torch_geometric.data",
    "torch.optim",
    "torch_geometric.utils",
    "torch.nn.functional",
    "torch_geometric.nn.inits",
    "rdkit.DataStructs",
    "rdkit.DataStructs.cDataStructs",
    "rdkit.Chem.AtomPairs",
    "rdkit.Chem.AtomPairs.Sheridan",
    "rdkit.Chem.Pharm2D",
    "torch_geometric.loader",
    "torch_geometric.data.data",
    "torch.utils",
    "torch.utils.data",
    "torch.autograd",
    "rdkit.Chem.rdchem",
    "pytorch_lightning",
    "rdkit.Chem.Scaffolds",
    "PIL",
    "selfies",
    "sklearn.cross_decomposition",
    "sklearn.decomposition",
    "sklearn.neighbors",
    "sklearn.manifold",
    "hyperopt",
    "hyperopt.pyll",
]

_mocked_imports = []

for imp in _OPTIONAL_IMPORTS_FOR_OCE_ONLINE:
    try:
        __import__(imp)
    except ImportError:
        import sys
        from unittest.mock import MagicMock

        sys.modules[imp] = MagicMock()
        _mocked_imports.append(imp)

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import logging

logging.getLogger().setLevel(logging.ERROR)

import os
from os import path

import json

import pandas as pd

if not path.exists(path.join(path.expanduser("~"), f".oce/")):
    os.mkdir(path.join(path.expanduser("~"), f".oce/"))
    
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".oce/CONFIG.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        CONFIG_ = json.load(f)
else:
    CONFIG_ = {}
    
defaults = {
    "MAP_LOCATION": "cpu",
    "USE_CUDA": False,
    "CDD_TOKEN": None,
    "VAULT_ID": None,
    "NUM_WORKERS": 0,
    "CACHE_PATH": os.path.join(os.path.expanduser("~"), ".oce/cache/"),
    "CACHE": True
}

for k, v in defaults.items():
    if k not in CONFIG_:
        CONFIG_[k] = v
        
with open(CONFIG_PATH, "w+") as f:
    json.dump(CONFIG_, f)

CONFIG = CONFIG_.copy()

if not path.exists(CONFIG["CACHE_PATH"]):
    os.mkdir(CONFIG["CACHE_PATH"])

if not path.exists(path.join(CONFIG["CACHE_PATH"], "vecrep/")):
    os.mkdir(path.join(CONFIG["CACHE_PATH"], "vecrep/")) # The passed CACHE_PATH must already exist

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
    return pd.read_csv(
        "https://storage.googleapis.com/oloren-public-data/sample-csvs/sample_data3.csv"
    )


from .base_class import *
from .basics import *
from .beta import *
from .benchmarks import *
from .dataset import *
from .ensemble import *
from .external import *
from .gnn import *
from .hyperparameters import *
from .internal import *
from .interpret import *
from .manager import *
from .reduction import *
from .representations import *
from .splitters import *
from .uncertainty import *
from .visualizations import *


def ExampleDataset():
    dataset = (
        BaseDataset(
            data=ExampleDataFrame().to_csv(),
            structure_col="Smiles",
            property_col="pChEMBL Value",
        )
        + RandomSplit()
    )
    return dataset


def BACEDataset():
    df = pd.read_csv(download_public_file("MoleculeNet/load_bace_regression.csv"))
    df["split"] = df["split"].replace(
        {"Train": "train", "Valid": "valid", "Test": "test"}
    )
    return oce.BaseDataset(
        data=df.to_csv(), structure_col="smiles", property_col="pIC50"
    )


def test_oce():
    """Convenience function to test all functions of the oce package."""

    df = ExampleDataFrame()

    model = BaseBoosting(
        [
            RandomForestModel(
                DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1
            ),
            BaseTorchGeometricModel(
                TLFromCheckpoint("default"), batch_size=8, epochs=1, preinitialized=True
            ),
            RandomForestModel(OlorenCheckpoint("default"), n_estimators=1),
        ]
    )

    model.fit(df["Smiles"], df["pChEMBL Value"])
    _ = model.predict(["CC(=O)OC1=CC=CC=C1C(=O)O"])
    save(model, "model.oce")
    _ = load("model.oce")
    os.remove("model.oce")

global online_session
online_session = None

def MISSING_DEPENDENCIES():
    if len(_mocked_imports) == 0:
        print("No missing dependencies.")
        return _mocked_imports
    else:
        print(textwrap.fill(f"Missing dependencies: {', '.join([x for x in _mocked_imports if '.' not in x])}.", 120))
        print("\nTo install OCE with all dependencies, run:")
        print("     bash <(curl -s https://raw.githubusercontent.com/Oloren-AI/olorenchemengine/master/install.sh)")

if len(_mocked_imports) > 0:
    import textwrap
    message = f"Some missing imports detected: {(','.join(_mocked_imports)[:30])}..." + \
    f"Automatically triggering online mode, where all OCE code is compiled locally and sent to AWS for execution." + \
    f"If you would like to use OCE online mode with private data or hosted on your own infrastructure, email." + \
    f" contact@oloren.ai for enterprise offerings." + \
    f"To exit online mode, run oce.online_session.__exit__()"

    message = f"""
Some missing imports detected: {(','.join(_mocked_imports)[:30])}...
Automatically triggering online mode, where all OCE code is compiled locally
and sent to AWS for execution. If you would like to use OCE online mode with private
data or hosted on your own infrastructure, email contact@oloren.ai for enterprise
offerings. To exit online mode, run olorenchemengine.online_session.__exit__()
""".replace("\n", " ")

    message = f"""
        To complete installation of Oloren ChemEngine, either:
            (1) Run oce.online() to use the demonstration package, Oloren ChemEngine Online,
            (2) Install the missing dependencies, instructions can be found oce.MISSING_DEPENDENCIES(), or
            (3) Email contact@oloren.ai with subject "Oloren ChemEngine Enterprise", for a secure privately hosted
                Server version.

        Oloren ChemEngine online is the public, dependency-free version of OCE, which compiles OCE code locally for fast,
        parallelized, remote execution on Oloren's cloud solution. Oloren ChemEngine online SHOULD NOT BE USED FOR
        CONFIDENTAIL DATA, and is only intended for demonstration purposes. The securely privately hosted Server version is
        called Oloren ChemEngine Enterprise.\n"""

    # print(textwrap.fill(textwrap.dedent(message).replace("\n", " ").replace("    (", "\n     ("), 80))
    print(textwrap.dedent(message))

def online(session_url="https://aws.chemengine.org"):
    global online_session
    online_session = Remote(session_url)
    online_session.__enter__()
