import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import olorenchemengine as oce
from olorenchemengine.base_class import *
from olorenchemengine.basics import *
from olorenchemengine.benchmarks import *
from olorenchemengine.ensemble import *
from olorenchemengine.internal import download_public_file
from olorenchemengine.representations import *

__author__ = "Oloren AI"
__copyright__ = "Oloren AI"


def remote(func):
    def wrapper(*args, **kwargs):
        with oce.Remote("http://api.oloren.ai:5000") as remote:
            func(*args, **kwargs)

    return wrapper


@pytest.fixture
def example_data():
    file_path = download_public_file("sample-csvs/sample_data1.csv")
    df = pd.read_csv(file_path)[:10]
    return train_test_split(
        df["Smiles"], df["pChEMBL Value"], test_size=0.33, random_state=42
    )


@pytest.fixture
def example_data2():
    file_path = download_public_file("sample-csvs/sample_data2.csv")
    df = pd.read_csv(file_path)[:10]
    return train_test_split(
        df["Smiles"], df["pChEMBL Value"], test_size=0.33, random_state=42
    )


@pytest.fixture
def example_data3():
    file_path = download_public_file("sample-csvs/sample_data3.csv")
    df = pd.read_csv(file_path)[:10]
    return train_test_split(
        df["Smiles"], df["pChEMBL Value"], test_size=0.33, random_state=42
    )


def bf_model(model):
    s1 = parameterize(model)
    model2 = create_BC(s1)
    s2 = parameterize(model2)
    assert s1 == s2
    assert type(model) == type(model2)


def train_predict_sl(model, example_data):
    X_train, X_test, y_train, y_test = example_data
    model.fit(X_train, y_train)
    p1 = model.predict(X_test)
    save = saves(model)
    model2 = loads(save)
    p2 = model2.predict(X_test)
    assert np.allclose(p1, p2)


def train_predict_slf(model, example_data):
    X_train, X_test, y_train, y_test = example_data
    model.fit(X_train, y_train)
    p1 = model.predict(X_test)
    save(model, "tmp.oce")
    model2 = load("tmp.oce")
    p2 = model2.predict(X_test)
    assert np.allclose(p1, p2)


def test_create_rf_model():
    model = RandomForestModel(
        DescriptastorusDescriptor("morgan3counts"), n_estimators=10
    )
    assert model is not None


def test_rf_back_and_forth():
    model = RandomForestModel(DescriptastorusDescriptor("morgan3counts"))
    bf_model(model)


def test_basic_train(example_data):
    model = RandomForestModel(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    X_train, X_test, y_train, y_test = example_data
    model.fit(X_train, y_train)
    p1 = model.predict(X_test)
    assert len(p1) == len(y_test)


def test_basic_test(example_data):
    model = RandomForestModel(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    X_train, X_test, y_train, y_test = example_data
    model.fit(X_train, y_train)
    model.test(X_test, y_test)


def test_basic_sl(example_data):
    model = RandomForestModel(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    train_predict_sl(model, example_data)


def test_basic_slf(example_data):
    model = RandomForestModel(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    train_predict_slf(model, example_data)


def test_basic_slf2(example_data):
    model = RandomForestModel(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
        normalization=QuantileTransformer(),
    )
    train_predict_slf(model, example_data)


def test_basic_slf3(example_data3):
    model = RandomForestModel(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    train_predict_slf(model, example_data3)


def test_basic_slf4(example_data3):
    model = RandomForestModel(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
        normalization=QuantileTransformer(),
    )
    train_predict_slf(model, example_data3)


def test_torchgeometric_save_load(example_data):
    from olorenchemengine.external import GINModel

    model = oce.BaseTorchGeometricModel(
        GINModel(hidden=1, layers=1, batch_size=10), epochs=1
    )
    train_predict_sl(model, example_data)


def test_mlp_saurus(example_data):
    model = oce.SklearnMLP(
        oce.DescriptastorusDescriptor("morgan3counts"),
        hidden_layer_sizes=[16, 4],
        activation="relu",
    )
    train_predict_slf(model, example_data)


def test_mlp_torch(example_data):
    model = oce.TorchMLP(
        oce.DescriptastorusDescriptor("morgan3counts"),
        hidden_layer_sizes=[16, 4],
        activation="relu",
    )
    train_predict_slf(model, example_data)


def test_boost(example_data):
    model = oce.BaseBoosting(
        [
            oce.RandomForestModel(
                oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1
            ),
            oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1),
        ],
        n=1,
    )
    train_predict_sl(model, example_data)


def test_basic_save_load_file(example_data):
    model = RandomForestModel(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    train_predict_slf(model, example_data)


def test_boost_file(example_data):
    model = oce.BaseBoosting(
        [
            oce.RandomForestModel(
                oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=10
            ),
            oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=10),
        ],
        n=1,
    )
    train_predict_slf(model, example_data)


def test_attentive_fp(example_data):
    model = oce.BaseTorchGeometricModel(oce.gnn.AttentiveFP(), epochs=1)
    train_predict_sl(model, example_data)


@pytest.mark.timeout(20)
def test_rfstacker(example_data):
    model = oce.RFStacker(
        [
            oce.RandomForestModel(
                oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=10
            ),
            oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=10),
        ],
        n=1,
    )
    train_predict_slf(model, example_data)


import warnings


def test_ns_spgnn_1(example_data3):
    from olorenchemengine.external import SPGNN

    model = SPGNN(model_type="contextpred", epochs=1)


def test_ns_spgnn_2(example_data3):
    from olorenchemengine.external import SPGNN

    model = SPGNN(model_type="contextpred", epochs=1)
    train_predict_sl(model, example_data3)


def test_ns_spgnn_3(example_data3):
    from olorenchemengine.external import SPGNN

    model = SPGNN(model_type="supervised_contextpred", epochs=1)


def test_chemprop_1(example_data3):
    from olorenchemengine.external import ChemPropModel

    model = ChemPropModel()


def test_chemprop_2(example_data3):
    from olorenchemengine.external import ChemPropModel

    model = ChemPropModel()
    train_predict_sl(model, example_data3)


def test_zwk_xgboost_model1(example_data):
    model = oce.basics.ZWK_XGBoostModel(
        oce.DescriptastorusDescriptor("morgan3counts"), n_iter=2, cv=2
    )
    train_predict_slf(model, example_data)


def test_zwk_xgboost_model2(example_data3):
    model = oce.basics.ZWK_XGBoostModel(
        oce.DescriptastorusDescriptor("morgan3counts"), n_iter=2, cv=2
    )
    train_predict_slf(model, example_data3)


def test_add_reps(example_data):
    model = oce.RandomForestModel(
        oce.OlorenCheckpoint("default") + oce.DescriptastorusDescriptor("morgan3counts")
    )
    train_predict_slf(model, example_data)


def test_pep_desc(example_data):
    model = oce.RandomForestModel(
        oce.PeptideDescriptors1(),
        n_estimators=10,
    )
    train_predict_slf(model, example_data)


def test_mol2vec(example_data):
    from olorenchemengine.external import Mol2Vec

    model = oce.RandomForestModel(
        Mol2Vec(),
        n_estimators=10,
    )
    train_predict_slf(model, example_data)


def test_smiles_transformer(example_data):
    from olorenchemengine.external import HondaSTRep

    model = RandomForestModel(
        HondaSTRep(),
        n_estimators=10,
    )
    train_predict_slf(model, example_data)


# @pytest.mark.parametrize("rep", [(rep) for rep in oce.BaseCompoundVecRepresentation.AllInstances()])
#
#
# def test_all_base_compound_vecs(example_data, rep):
#     if isinstance(rep, oce.MordredDescriptor):
#         return  # Mordred does not play well with pytest
#     model = oce.RandomForestModel(rep, n_estimators=10)
#     train_predict_slf(model, example_data)


def test_molnet_single_task():
    mn_datasets = [
        "bace_classification",
        "bbbp",
        "clintox",
        "hiv",
        "muv",
        "pcba",
        "sider",
        "tox21",
        "toxcast",
        "delaney" or "ESOL",
        "freesolv",
        "lipo",
        "bace_regression",
    ]

    MTC = oce.RandomForestModel(
        oce.DescriptastorusDescriptor("morgan3counts"),
        max_features="log2",
        n_estimators=1,
    )
    RDK = oce.RandomForestModel(
        oce.FragmentIndicator(), max_features="log2", n_estimators=1
    )
    models = [MTC, RDK]

    mn_bench = BenchmarkMolNet(datasets=[mn_datasets[0]])
    mn_bench.run(models)
