import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import olorenchemengine as oce
from olorenchemengine.base_class import *
from olorenchemengine.internal import download_public_file
from olorenchemengine.uncertainty import *

__author__ = "Oloren AI"
__copyright__ = "Oloren AI"


@pytest.fixture
def example_data():
    file_path = download_public_file("MoleculeNet/load_bace_regression.csv")
    df = pd.read_csv(file_path)[:100]
    return train_test_split(
        df["smiles"], df["pIC50"], test_size=0.5, random_state=42
    )

@pytest.fixture
def example_data2():
    file_path = download_public_file("MoleculeNet/load_bace_regression.csv")
    df = pd.read_csv(file_path)[100:200]
    return train_test_split(
        df["smiles"], df["pIC50"], test_size=0.5, random_state=42
    )

@pytest.fixture
def example_data3():
    file_path = download_public_file("MoleculeNet/load_bace_regression.csv")
    df = pd.read_csv(file_path)[200:300]
    return train_test_split(
        df["smiles"], df["pIC50"], test_size=0.5, random_state=42
    )

@pytest.fixture
def example_model(example_data):
    X_train, X_test, y_train, y_test = example_data
    model = SupportVectorMachine(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    model.fit(X_train, y_train)
    return model

@pytest.fixture
def example_model2(example_data2):
    X_train, X_test, y_train, y_test = example_data2
    model = SupportVectorMachine(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    model.fit(X_train, y_train)
    return model

@pytest.fixture
def example_model3(example_data3):
    X_train, X_test, y_train, y_test = example_data3
    model = SupportVectorMachine(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    model.fit(X_train, y_train)
    return model


def fit_score_sl(error_model, example_model, example_data):
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    error_model.fit(X_test, y_test)
    p1 = error_model.score(X_test)
    save = saves(error_model)
    error_model2 = loads(save)
    p2 = error_model2.score(X_test)
    assert np.allclose(p1, p2, equal_nan=True)


def fit_score_slf(error_model, example_model, example_data):
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    error_model.fit(X_test, y_test)
    p1 = error_model.score(X_test)
    save(error_model, "tmp.oce")
    error_model2 = load("tmp.oce")
    p2 = error_model2.score(X_test)
    assert np.allclose(p1, p2, equal_nan=True)

"""
Basic tests
"""

def test_basic_build(example_model, example_data):
    error_model = KernelRegressionError()
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    assert len(X_train) == len(error_model.X_train)
    assert len(y_train) == len(error_model.y_train)
    assert len(y_train) == len(error_model.y_pred_train)


def test_basic_fit_bin1(example_model, example_data):
    error_model = KernelRegressionError(method='bin')
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    error_model.fit(X_test, y_test)
    assert hasattr(error_model, 'reg')
    assert len(y_test) == len(error_model.residuals)
    assert len(y_test) == len(error_model.scores)


def test_basic_fit_bin2(example_model2, example_data2):
    error_model = KernelRegressionError(method='bin')
    X_train, X_test, y_train, y_test = example_data2
    error_model.build(example_model2, X_train, y_train)
    error_model.fit(X_test, y_test)
    assert hasattr(error_model, 'reg')
    assert len(y_test) == len(error_model.residuals)
    assert len(y_test) == len(error_model.scores)


def test_basic_fit_bin3(example_model3, example_data3):
    error_model = KernelRegressionError(method='bin')
    X_train, X_test, y_train, y_test = example_data3
    error_model.build(example_model3, X_train, y_train)
    error_model.fit(X_test, y_test)
    assert hasattr(error_model, 'reg')
    assert len(y_test) == len(error_model.residuals)
    assert len(y_test) == len(error_model.scores)


def test_basic_fit_qbin1(example_model, example_data):
    error_model = KernelRegressionError(method='qbin')
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    error_model.fit(X_test, y_test)
    assert hasattr(error_model, 'reg')
    assert len(y_test) == len(error_model.residuals)
    assert len(y_test) == len(error_model.scores)


def test_basic_fit_qbin2(example_model2, example_data2):
    error_model = KernelRegressionError(method='qbin')
    X_train, X_test, y_train, y_test = example_data2
    error_model.build(example_model2, X_train, y_train)
    error_model.fit(X_test, y_test)
    assert hasattr(error_model, 'reg')
    assert len(y_test) == len(error_model.residuals)
    assert len(y_test) == len(error_model.scores)


def test_basic_fit_qbin3(example_model3, example_data3):
    error_model = KernelRegressionError(method='qbin')
    X_train, X_test, y_train, y_test = example_data3
    error_model.build(example_model3, X_train, y_train)
    error_model.fit(X_test, y_test)
    assert hasattr(error_model, 'reg')
    assert len(y_test) == len(error_model.residuals)
    assert len(y_test) == len(error_model.scores)


def test_basic_fit_roll1(example_model, example_data):
    error_model = KernelRegressionError(method='roll', window=2)
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    error_model.fit(X_test, y_test)
    assert hasattr(error_model, 'reg')
    assert len(y_test) == len(error_model.residuals)
    assert len(y_test) == len(error_model.scores)


def test_basic_fit_roll2(example_model2, example_data2):
    error_model = KernelRegressionError(method='roll', window=2)
    X_train, X_test, y_train, y_test = example_data2
    error_model.build(example_model2, X_train, y_train)
    error_model.fit(X_test, y_test)
    assert hasattr(error_model, 'reg')
    assert len(y_test) == len(error_model.residuals)
    assert len(y_test) == len(error_model.scores)


def test_basic_fit_roll3(example_model3, example_data3):
    error_model = KernelRegressionError(method='roll', window=2)
    X_train, X_test, y_train, y_test = example_data3
    error_model.build(example_model3, X_train, y_train)
    error_model.fit(X_test, y_test)
    assert hasattr(error_model, 'reg')
    assert len(y_test) == len(error_model.residuals)
    assert len(y_test) == len(error_model.scores)


def test_basic_fit_cv(example_model, example_data):
    error_model = KernelRegressionError()
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    error_model.fit_cv(n_splits=2)
    assert hasattr(error_model, 'reg')
    assert len(y_train) == len(error_model.residuals)
    assert len(y_train) == len(error_model.scores)


def test_basic_score(example_model, example_data):
    error_model = KernelRegressionError()
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    error_model.fit(X_test, y_test)
    s = error_model.score(X_test)
    assert len(X_test) == len(s)


def test_basic_sl1(example_model, example_data):
    error_model = KernelRegressionError(ci=0.5)
    fit_score_sl(error_model, example_model, example_data)


def test_basic_sl2(example_model2, example_data2):
    error_model = KernelRegressionError(ci=0.5)
    fit_score_sl(error_model, example_model2, example_data2)


def test_basic_sl3(example_model3, example_data3):
    error_model = KernelRegressionError(ci=0.5)
    fit_score_sl(error_model, example_model3, example_data3)


def test_basic_slf1(example_model, example_data):
    error_model = KernelRegressionError(ci=0.5)
    fit_score_slf(error_model, example_model, example_data)


def test_basic_slf2(example_model2, example_data2):
    error_model = KernelRegressionError(ci=0.5)
    fit_score_slf(error_model, example_model2, example_data2)


def test_basic_slf3(example_model3, example_data3):
    error_model = KernelRegressionError(ci=0.5)
    fit_score_slf(error_model, example_model3, example_data3)


def test_basic_curvetype(example_model, example_data):
    error_model = KernelRegressionError(ci=0.5, curvetype = "linear")
    fit_score_sl(error_model, example_model, example_data)


def test_basic_curvetype(example_model2, example_data2):
    error_model = KernelRegressionError(ci=0.5, curvetype = "linear")
    fit_score_sl(error_model, example_model2, example_data2)


def test_basic_curvetype(example_model3, example_data3):
    error_model = KernelRegressionError(ci=0.5, curvetype = "linear")
    fit_score_sl(error_model, example_model3, example_data3)

"""
Fingerprint models
"""

def test_fingerprint_build(example_model, example_data):
    error_model = KernelRegressionError()
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    assert len(X_train) == len(error_model.train_fps)


def test_KernelRegression1(example_model, example_data):
    error_model = KernelRegressionError(predictor="residual", kernel="parabolic", h=1)
    fit_score_sl(error_model, example_model, example_data)


def test_KernelRegression2(example_model2, example_data2):
    error_model = KernelRegressionError(predictor="residual", kernel="parabolic", h=1)
    fit_score_sl(error_model, example_model2, example_data2)


def test_KernelRegression3(example_model3, example_data3):
    error_model = KernelRegressionError(predictor="residual", kernel="parabolic", h=1)
    fit_score_sl(error_model, example_model3, example_data3)


def test_KernelRegression_file1(example_model, example_data):
    error_model = KernelRegressionError(predictor="residual", kernel="parabolic", h=1)
    fit_score_slf(error_model, example_model, example_data)


def test_KernelRegression_file2(example_model2, example_data2):
    error_model = KernelRegressionError(predictor="residual", kernel="parabolic", h=1)
    fit_score_slf(error_model, example_model2, example_data2)


def test_KernelRegression_file3(example_model3, example_data3):
    error_model = KernelRegressionError(predictor="residual", kernel="parabolic", h=1)
    fit_score_slf(error_model, example_model3, example_data3)


def test_KernelDistance1(example_model, example_data):
    error_model = KernelDistanceError(weighted=False, kernel="parabolic", h=1)
    fit_score_sl(error_model, example_model, example_data)


def test_KernelDistance2(example_model2, example_data2):
    error_model = KernelDistanceError(weighted=False, kernel="parabolic", h=1)
    fit_score_sl(error_model, example_model2, example_data2)


def test_KernelDistance3(example_model3, example_data3):
    error_model = KernelDistanceError(weighted=False, kernel="parabolic", h=1)
    fit_score_sl(error_model, example_model3, example_data3)


def test_KernelDistance_file1(example_model, example_data):
    error_model = KernelDistanceError(weighted=False, kernel="parabolic", h=1)
    fit_score_slf(error_model, example_model, example_data)


def test_KernelDistance_file2(example_model2, example_data2):
    error_model = KernelDistanceError(weighted=False, kernel="parabolic", h=1)
    fit_score_slf(error_model, example_model2, example_data2)


def test_KernelDistance_file3(example_model3, example_data3):
    error_model = KernelDistanceError(weighted=False, kernel="parabolic", h=1)
    fit_score_slf(error_model, example_model3, example_data3)

"""
ADAN model
"""

def test_ADAN_preprocess(example_model, example_data):
    error_model = ADAN(threshold=0.5)
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    X_processed = error_model.preprocess(X_train)
    assert len(X_processed) == len(X_train)


def test_ADAN_build_pls(example_model, example_data):
    error_model = ADAN(threshold=0.5, dim_reduction = "pls")
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)

    assert len(X_train) == len(error_model.training_["A_raw"])
    assert len(X_train) == len(error_model.training_["B_raw"])
    assert len(X_train) == len(error_model.training_["C_raw"])
    assert len(X_train) == len(error_model.training_["D_raw"])
    assert len(X_train) == len(error_model.training_["E_raw"])
    assert len(X_train) == len(error_model.training_["F_raw"])


    thresh = int(error_model.X_train.shape[0] * error_model.threshold)

    assert sorted(error_model.training_["A_raw"])[thresh] == error_model.thresholds_["A"]
    assert sorted(error_model.training_["B_raw"])[thresh] == error_model.thresholds_["B"]
    assert sorted(error_model.training_["C_raw"])[thresh] == error_model.thresholds_["C"]
    assert sorted(error_model.training_["D_raw"])[thresh] == error_model.thresholds_["D"]
    assert sorted(error_model.training_["E_raw"])[thresh] == error_model.thresholds_["E"]
    assert sorted(error_model.training_["F_raw"])[thresh] == error_model.thresholds_["F"]


def test_ADAN_build_pca(example_model, example_data):
    error_model = ADAN(threshold=0.5, dim_reduction = "pca")
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)

    assert len(X_train) == len(error_model.training_["A_raw"])
    assert len(X_train) == len(error_model.training_["B_raw"])
    assert len(X_train) == len(error_model.training_["C_raw"])
    assert len(X_train) == len(error_model.training_["D_raw"])
    assert len(X_train) == len(error_model.training_["E_raw"])
    assert len(X_train) == len(error_model.training_["F_raw"])


    thresh = int(error_model.X_train.shape[0] * error_model.threshold)

    assert sorted(error_model.training_["A_raw"])[thresh] == error_model.thresholds_["A"]
    assert sorted(error_model.training_["B_raw"])[thresh] == error_model.thresholds_["B"]
    assert sorted(error_model.training_["C_raw"])[thresh] == error_model.thresholds_["C"]
    assert sorted(error_model.training_["D_raw"])[thresh] == error_model.thresholds_["D"]
    assert sorted(error_model.training_["E_raw"])[thresh] == error_model.thresholds_["E"]
    assert sorted(error_model.training_["F_raw"])[thresh] == error_model.thresholds_["F"] 


def test_ADAN_calculate_full(example_model, example_data):
    error_model = ADAN(threshold=0.5, )
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    error_model.calculate_full(X_test)

    assert all([val == 0 or val == 1 for val in error_model.results["A"]])
    assert all([val == 0 or val == 1 for val in error_model.results["B"]])
    assert all([val == 0 or val == 1 for val in error_model.results["C"]])
    assert all([val == 0 or val == 1 for val in error_model.results["D"]])
    assert all([val == 0 or val == 1 for val in error_model.results["E"]])
    assert all([val == 0 or val == 1 for val in error_model.results["F"]])

    assert all([val in [0, 1, 2, 3, 4, 5, 6] for val in error_model.results["Category"]])


def test_ADAN(example_model, example_data):
    error_model = ADAN(threshold=0.5, method='bin', bins=3)
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, method='bin', bins=3)
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_A(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="A", method='bin', bins=2)
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_A_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="A", method='bin', bins=2)
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_Araw(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="A_raw")
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_Araw_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="A_raw")
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_B(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="B", method='bin', bins=2)
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_B_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="B", method='bin', bins=2)
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_Braw(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="B_raw")
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_Braw_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="B_raw")
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_C(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="C", method='bin', bins=2)
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_C_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="C", method='bin', bins=2)
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_Craw(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="C_raw")
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_Craw_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="C_raw")
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_D(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="D", method='bin', bins=2)
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_D_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="D", method='bin', bins=2)
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_Draw(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="D_raw")
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_Draw_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="D_raw")
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_E(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="E", method='bin', bins=2)
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_E_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="E", method='bin', bins=2)
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_Eraw(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="E_raw")
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_Eraw_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="E_raw")
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_F(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="F", method='bin', bins=2)
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_F_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="F", method='bin', bins=2)
    fit_score_slf(error_model, example_model, example_data)


def test_ADAN_Fraw(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="F_raw")
    fit_score_sl(error_model, example_model, example_data)


def test_ADAN_Fraw_file(example_model, example_data):
    error_model = ADAN(threshold=0.5, criterion="F_raw")
    fit_score_slf(error_model, example_model, example_data)

"""
Aggregate model
"""

def test_aggregate_fit(example_model, example_data):
    error_model = AggregateErrorModel(
        KernelRegressionError(predictor="residual"), 
        KernelRegressionError(predictor="property")
    )
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    error_model.fit(X_test, y_test)
    assert hasattr(error_model, 'reg')
    assert len(y_test) == len(error_model.residuals)
    assert len(y_test) == len(error_model.scores)


def test_aggregate_fit_cv(example_model, example_data):
    error_model = AggregateErrorModel(
        KernelRegressionError(predictor="residual"), 
        KernelRegressionError(predictor="property")
    )
    X_train, X_test, y_train, y_test = example_data
    error_model.build(example_model, X_train, y_train)
    error_model.fit_cv(n_splits=2)
    assert hasattr(error_model, 'reg')
    assert len(y_train) == len(error_model.residuals)
    assert len(y_train) == len(error_model.scores)


def test_aggregate1(example_model, example_data):
    error_model = AggregateErrorModel(
        KernelRegressionError(predictor="residual"), 
        KernelRegressionError(predictor="property")
    )
    fit_score_sl(error_model, example_model, example_data)


def test_aggregate2(example_model2, example_data2):
    error_model = AggregateErrorModel(
        KernelRegressionError(predictor="residual"), 
        KernelRegressionError(predictor="property")
    )
    fit_score_sl(error_model, example_model2, example_data2)


def test_aggregate_file1(example_model, example_data):
    error_model = AggregateErrorModel(
        KernelRegressionError(predictor="residual"), 
        KernelRegressionError(predictor="property")
    )
    fit_score_slf(error_model, example_model, example_data)


def test_aggregate_file2(example_model2, example_data2):
    error_model = AggregateErrorModel(
        KernelRegressionError(predictor="residual"), 
        KernelRegressionError(predictor="property")
    )
    fit_score_slf(error_model, example_model2, example_data2)

"""
Ensemble model
"""

def test_ensemble1(example_model, example_data):
    error_model = BootstrapEnsemble(n_ensembles=2)
    fit_score_sl(error_model, example_model, example_data)


def test_ensemble2(example_model2, example_data2):
    error_model = BootstrapEnsemble(n_ensembles=2)
    fit_score_sl(error_model, example_model2, example_data2)


def test_ensemble_file1(example_model, example_data):
    error_model = BootstrapEnsemble(n_ensembles=2)
    fit_score_slf(error_model, example_model, example_data)


def test_ensemble_file2(example_model2, example_data2):
    error_model = BootstrapEnsemble(n_ensembles=2)
    fit_score_slf(error_model, example_model2, example_data2)

"""
Integration tests
"""

def test_create1(example_data):
    X_train, X_test, y_train, y_test = example_data
    model = SupportVectorMachine(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    model.fit(X_train, y_train)
    model.create_error_model(KernelRegressionError(ci=0.5), X_train, y_train, X_test, y_test)
    assert hasattr(model, "error_model")
    assert hasattr(model.error_model, "reg")


def test_create2(example_data2):
    X_train, X_test, y_train, y_test = example_data2
    model = SupportVectorMachine(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    model.fit(X_train, y_train)
    model.create_error_model(KernelRegressionError(ci=0.5), X_train, y_train, X_test, y_test)
    assert hasattr(model, "error_model")
    assert hasattr(model.error_model, "reg")


def test_concurrent1(example_data):
    X_train, X_test, y_train, y_test = example_data
    model = SupportVectorMachine(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    error_model = KernelRegressionError(ci=0.5)
    model.fit(X_train, y_train, error_model=error_model)

    assert hasattr(model, "error_model")
    
    model.test(X_test, y_test, fit_error_model=True)

    assert hasattr(model.error_model, "reg")

    model.predict(X_test, return_ci=True, return_vis=True)


def test_concurrent2(example_data2):
    X_train, X_test, y_train, y_test = example_data2
    model = SupportVectorMachine(
        OlorenCheckpoint(
            "default",
            num_tasks=2048,
        ),
        n_estimators=10,
    )
    error_model = KernelRegressionError(ci=0.5)
    model.fit(X_train, y_train, error_model=error_model)

    assert hasattr(model, "error_model")
    
    model.test(X_test, y_test, fit_error_model=True)

    assert hasattr(model.error_model, "reg")

    model.predict(X_test, return_ci=True, return_vis=True)


def test_model_sl1(example_model, example_data):
    X_train, X_test, y_train, y_test = example_data
    error_model = KernelRegressionError(ci=0.5)
    example_model.create_error_model(error_model, X_train, y_train, X_test, y_test)
    p1 = example_model.predict(X_test, return_ci=True)
    save = saves(example_model)
    model2 = loads(save)
    p2 = model2.predict(X_test, return_ci=True)
    assert np.allclose(p1['predicted'], p2['predicted'], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[0], list(zip(p2['ci']))[0], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[1], list(zip(p2['ci']))[1], equal_nan=True)


def test_model_sl2(example_model2, example_data2):
    X_train, X_test, y_train, y_test = example_data2
    error_model = KernelRegressionError(ci=0.5)
    example_model2.create_error_model(error_model, X_train, y_train, X_test, y_test)
    p1 = example_model2.predict(X_test, return_ci=True)
    save = saves(example_model2)
    model2 = loads(save)
    p2 = model2.predict(X_test, return_ci=True)
    assert np.allclose(p1['predicted'], p2['predicted'], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[0], list(zip(p2['ci']))[0], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[1], list(zip(p2['ci']))[1], equal_nan=True)


def test_model_sl3(example_model3, example_data3):
    X_train, X_test, y_train, y_test = example_data3
    error_model = KernelRegressionError(ci=0.5)
    example_model3.create_error_model(error_model, X_train, y_train, X_test, y_test)
    p1 = example_model3.predict(X_test, return_ci=True)
    save = saves(example_model3)
    model2 = loads(save)
    p2 = model2.predict(X_test, return_ci=True)
    assert np.allclose(p1['predicted'], p2['predicted'], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[0], list(zip(p2['ci']))[0], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[1], list(zip(p2['ci']))[1], equal_nan=True)


def test_model_slf1(example_model, example_data):
    X_train, X_test, y_train, y_test = example_data
    error_model = KernelRegressionError(ci=0.5)
    example_model.create_error_model(error_model, X_train, y_train, X_test, y_test)
    p1 = example_model.predict(X_test, return_ci=True)
    save(example_model, "tmp.oce")
    model2 = load("tmp.oce")
    p2 = model2.predict(X_test, return_ci=True)
    assert np.allclose(p1['predicted'], p2['predicted'], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[0], list(zip(p2['ci']))[0], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[1], list(zip(p2['ci']))[1], equal_nan=True)


def test_model_slf2(example_model2, example_data2):
    X_train, X_test, y_train, y_test = example_data2
    error_model = KernelRegressionError(ci=0.5)
    example_model2.create_error_model(error_model, X_train, y_train, X_test, y_test)
    p1 = example_model2.predict(X_test, return_ci=True)
    save(example_model2, "tmp.oce")
    model2 = load("tmp.oce")
    p2 = model2.predict(X_test, return_ci=True)
    assert np.allclose(p1['predicted'], p2['predicted'], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[0], list(zip(p2['ci']))[0], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[1], list(zip(p2['ci']))[1], equal_nan=True)


def test_model_slf3(example_model3, example_data3):
    X_train, X_test, y_train, y_test = example_data3
    error_model = KernelRegressionError(ci=0.5)
    example_model3.create_error_model(error_model, X_train, y_train, X_test, y_test)
    p1 = example_model3.predict(X_test, return_ci=True)
    save(example_model3, "tmp.oce")
    model2 = load("tmp.oce")
    p2 = model2.predict(X_test, return_ci=True)
    assert np.allclose(p1['predicted'], p2['predicted'], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[0], list(zip(p2['ci']))[0], equal_nan=True)
    assert np.allclose(list(zip(p1['ci']))[1], list(zip(p2['ci']))[1], equal_nan=True)