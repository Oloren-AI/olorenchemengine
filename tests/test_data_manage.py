import os

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from sklearn.model_selection import train_test_split

import olorenchemengine as oce
from olorenchemengine.internal import download_public_file
from olorenchemengine.interpret import *

__author__ = "Oloren AI"
__copyright__ = "Oloren AI"


def remote(func):
    def wrapper(*args, **kwargs):
        with oce.Remote("http://api.oloren.ai:5000") as remote:
            func(*args, **kwargs)

    return wrapper


def test_datasetfromcsv():
    dataset = oce.DatasetFromCSV(
        file_path=download_public_file("sample-csvs/sample_data1.csv"),
        structure_col="Smiles",
        property_col="pChEMBL Value",
    ) + oce.RandomSplit(split_proportions=[0.8, 0.0, 2])
    x_train, y_train = dataset.train_dataset
    x_test, y_test = dataset.test_dataset
    x_val, y_val = dataset.valid_dataset
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_val) == len(y_val)


def test_manage():
    dataset = oce.DatasetFromCSV(
        file_path=download_public_file("sample-csvs/sample_data1.csv"),
        structure_col="Smiles",
        property_col="pChEMBL Value",
    ) + oce.RandomSplit(split_proportions=[0.8, 0.1, 0.1])
    if os.path.exists("tmp.oce"):
        os.remove("tmp.oce")
    manager = oce.ModelManager(
        dataset, ["Root Mean Squared Error"], file_path="tmp.oce"
    )
    manager.run(
        oce.RandomForestModel(
            oce.DescriptastorusDescriptor("morgan3counts"), n_estimators=10
        )
    )
    manager.run(
        oce.MLP(oce.DescriptastorusDescriptor("morgan3counts"), n_estimators=10)
    )
    manager2 = oce.load("tmp.oce")

    error1 = manager.get_model_database()["Root Mean Squared Error"].to_numpy()
    error2 = manager2.get_model_database()["Root Mean Squared Error"].to_numpy()

    error1 = np.array(error1, dtype=float)
    error2 = np.array(error2, dtype=float)

    assert np.allclose(error1, error2)


import warnings


def test_sheets_manager():
    if (
        "GOOGLE_CREDENTIALS_FILENAME" not in oce.CONFIG
        and "GOOGLE_SERVICE_ACC_FILENAME" not in oce.CONFIG
    ):
        warnings.warn(
            UserWarning(
                "Skipping sheets manager test due to lack of supplied credentials."
            )
        )
        return

    dataset = oce.DatasetFromCSV(
        file_path=os.path.join(os.path.dirname(__file__), "sample_data1.csv"),
        structure_col="Smiles",
        property_col="pChEMBL Value",
    )
    if os.path.exists("tmp.oce"):
        os.remove("tmp.oce")
    manager = oce.SheetsModelManager(
        dataset, ["Root Mean Squared Error"], file_path="tmp.oce"
    )
    manager.run(
        oce.RandomForestModel(
            oce.DescriptastorusDescriptor("morgan3counts"), n_estimators=10
        )
    )
    manager.run(
        oce.MLP(oce.DescriptastorusDescriptor("morgan3counts"), n_estimators=10)
    )
    manager2 = oce.load("tmp.oce")

    error1 = manager.get_model_database()["Root Mean Squared Error"].to_numpy()
    error2 = manager2.get_model_database()["Root Mean Squared Error"].to_numpy()

    error1 = np.array(error1, dtype=float)
    error2 = np.array(error2, dtype=float)

    assert np.allclose(error1, error2)


def test_top_models_admet():
    models = oce.TOP_MODELS_ADMET()
    assert isinstance(models, List)
    for model in models:
        assert issubclass(type(model), oce.BaseModel)
