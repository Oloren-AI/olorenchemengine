from random import random

import pandas as pd
import pytest

import olorenchemengine as oce

__author__ = "Oloren AI"
__copyright__ = "Oloren AI"


def remote(func):
    def wrapper(*args, **kwargs):
        with oce.Remote("http://api.oloren.ai:5000") as remote:
            func(*args, **kwargs)

    return wrapper


@pytest.mark.timeout(300)
def test_0():
    import pandas as pd

    import olorenchemengine as oce

    df = pd.read_csv(
        "https://storage.googleapis.com/oloren-public-data/CHEMBL%20Datasets/997_2298%20-%20VEGFR1%20(CHEMBL1868).csv"
    )
    dataset = (
        oce.BaseDataset(
            data=df.to_csv(), structure_col="Smiles", property_col="pChEMBL Value"
        )
        + oce.CleanStructures()
        + oce.RandomSplit()
    )
    model = oce.RandomForestModel(
        oce.DescriptastorusDescriptor("morgan3counts"), n_estimators=1000
    )
    _ = model.fit_cv(*dataset.train_dataset, error_model=oce.kNNwRMSD1())
    oce.save(model, "vis-model.oce")
