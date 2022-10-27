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
    _ = model.fit_cv(*dataset.train_dataset, error_model=oce.TrainDistKNN())
    oce.save(model, "vis-model.oce")

@pytest.mark.timeout(300)
def test_1d():
    import olorenchemengine as oce

    # We'll use a dataset of BACE (beta secretase enzyme) pIC50 values for this
    # from a collection put together by MoleculeNet
    dataset = oce.BACEDataset() + oce.ScaffoldSplit()
    model = oce.BaseBoosting([oce.RandomForestModel(oce.DescriptastorusDescriptor("morgan3counts")),
                             oce.RandomForestModel(oce.OlorenCheckpoint("default"))])
    model.fit(*dataset.train_dataset)
    vis = oce.ChemicalSpacePlot(dataset, oce.DescriptastorusDescriptor("morgan3counts"), opacity = 0.4, dim_reduction = "tsne")
    vis.render_ipynb()
    vis = oce.VisualizeDatasetSplit(dataset, oce.DescriptastorusDescriptor("morgan3counts"), model = model, opacity = 0.4)
    vis.render_ipynb()
    vis = oce.VisualizeFeatures(dataset, features = [oce.LipinskiDescriptor()])
    vis.render_ipynb()
    ref_compound = dataset.data[dataset.structure_col].iloc[42]
    vis = oce.VisualizeCounterfactual(ref_compound, model, n=1000, delta=0.1)
    vis.render_ipynb()
    vis = oce.VisualizeCompounds(dataset.data[dataset.structure_col])
    vis.render_ipynb()
    query_smiles = "CCNc1cc(C(=O)N[C@@H](Cc2ccccc2)[C@H](O)CNCCC(F)(F)F)cc2c1CCCS(=O)(=O)N2C"
    vis = oce.VisualizePredictionSensitivity(model, query_smiles)
    vis.render_ipynb()