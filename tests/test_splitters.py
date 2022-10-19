from random import random

import pandas as pd
import pytest

import olorenchemengine as oce
from olorenchemengine.internal import download_public_file

__author__ = "Oloren AI"
__copyright__ = "Oloren AI"

"""There are currently no tests for DateSplitter as sample data does not have date values.
"""


def remote(func):
    def wrapper(*args, **kwargs):
        with oce.Remote("http://api.oloren.ai:5000") as remote:
            func(*args, **kwargs)

    return wrapper


@pytest.fixture
def example_data1():
    file_path = download_public_file("sample-csvs/sample_data1.csv")
    df = pd.read_csv(file_path)
    return df


@pytest.fixture
def example_data2():
    file_path = download_public_file("sample-csvs/sample_data2.csv")
    df = pd.read_csv(file_path)
    return df


@pytest.fixture
def example_data3():
    file_path = download_public_file("sample-csvs/sample_data3.csv")
    df = pd.read_csv(file_path)
    return df


def test_run_random(example_data3):
    splitter = oce.RandomSplit(split_proportions=[0.8, 0.1, 0.1])
    for i in splitter.split(example_data3):
        assert isinstance(i, pd.DataFrame)
        assert len(i) > 0


"""
Fail Info:
Stratify by categorical column of 0, 1.
Test passes with split_proportions = [0.8, 0.0, 0.2].
Test fails with split_proportions = [0.8, 0.1, 0.1]: "ValueError: The test_size = 1 should be greater or equal to the number of classes = 2"
Test fails with split_proportions = [0.8, 0.2, 0.0]: "ValueError: train_size=1.0 should be either positive and smaller than the number of samples 10 or a float in the (0, 1) range"
"""


def test_run_stratified(example_data3):
    splitter = oce.StratifiedSplitter(
        split_proportions=[0.8, 0.1, 0.1], value_col="pChEMBL Value"
    )
    for i in splitter.split(example_data3):
        assert isinstance(i, pd.DataFrame)
        assert len(i) > 0


def test_run_scaffold(example_data3):
    splitter = oce.ScaffoldSplit(
        scaffold_filter_threshold=1,
        split_type="murcko",
        split_proportions=[0.8, 0.1, 0.1],
    )
    for i in splitter.split(example_data3):
        assert isinstance(i, pd.DataFrame)
        assert len(i) > 0

    splitter = oce.ScaffoldSplit(
        scaffold_filter_threshold=1,
        split_type="kmeans_murcko",
        split_proportions=[0.8, 0.1, 0.1],
    )
    for i in splitter.split(example_data3):
        assert isinstance(i, pd.DataFrame)
        assert len(i) > 0


def test_run_property(example_data1):
    splitter = oce.PropertySplit(
        property_col="pChEMBL Value",
        threshold=None,
        noise=0.1,
        categorical=False,
        split_proportions=[0.8, 0.1, 0.1],
    )
    for i in splitter.split(example_data1):
        assert isinstance(i, pd.DataFrame)
        assert len(i) > 0

    splitter = oce.PropertySplit(
        property_col="pChEMBL Value", threshold=7, noise=0.1, categorical=False
    )
    split = splitter.split(example_data1)
    for i in range(len(split)):
        assert isinstance(split[i], pd.DataFrame)
        if i == 1:
            assert len(split[i]) == 0
        else:
            assert len(split[i]) > 0


"""Split Proportion Tests:
 - Check if true split proportions are within 0.05 or 0.1 of user-defined split proportions
 - Test case with empty validation split (TT/train test split)
"""


def test_split_props_random(example_data3):
    split_proportions = [0.8, 0.1, 0.1]

    splitter = oce.RandomSplit(split_proportions=split_proportions)
    print(example_data3)
    split = splitter.split(example_data3)
    total_samples = len(example_data3)

    for i in range(3):
        actual_split_prop = len(split[i]) / total_samples
        assert abs(actual_split_prop - split_proportions[i]) < 0.05

    split_proportions_TT = [0.8, 0.0, 0.2]

    splitter = oce.RandomSplit(split_proportions=split_proportions_TT)
    split = splitter.split(example_data3)

    for i in range(3):
        actual_split_prop = len(split[i]) / total_samples
        if i == 1:
            assert actual_split_prop == 0
        else:
            assert abs(actual_split_prop - split_proportions_TT[i]) < 0.05


"""For the [0.8, 0.0, 0.2] case, it is found that the validation split has samples in it despite 0 split setting.
Test won't show this as there is still the run error from the first test case [0.8, 0.1, 0.1], described in test_run_stratified.
"""


def test_split_props_stratified(example_data3):
    split_proportions = [0.8, 0.1, 0.1]

    splitter = oce.StratifiedSplitter(
        split_proportions=split_proportions, value_col="pChEMBL Value"
    )
    split = splitter.split(example_data3)
    total_samples = len(example_data3)

    for i in range(3):
        actual_split_prop = len(split[i]) / total_samples
        assert abs(actual_split_prop - split_proportions[i]) < 0.05

    split_proportions_TT = [0.8, 0.0, 0.2]

    splitter = oce.StratifiedSplitter(
        split_proportions=split_proportions_TT, value_col="pChEMBL Value"
    )
    split = splitter.split(example_data3)

    for i in range(3):
        actual_split_prop = len(split[i]) / total_samples
        if i == 1:
            assert actual_split_prop == 0
        else:
            assert abs(actual_split_prop - split_proportions_TT[i]) < 0.05


def test_split_props_scaffold_murcko(example_data3):
    split_proportions = [0.8, 0.1, 0.1]

    splitter = oce.ScaffoldSplit(
        scaffold_filter_threshold=1,
        split_type="murcko",
        split_proportions=split_proportions,
    )
    split = splitter.split(example_data3)
    total_samples = len(example_data3)

    for i in range(3):
        actual_split_prop = len(split[i]) / total_samples
        assert abs(actual_split_prop - split_proportions[i]) < 0.05

    split_proportions_TT = [0.8, 0.0, 0.2]

    splitter = oce.ScaffoldSplit(
        scaffold_filter_threshold=1,
        split_type="murcko",
        split_proportions=split_proportions_TT,
    )
    split = splitter.split(example_data3)

    for i in range(3):
        actual_split_prop = len(split[i]) / total_samples
        if i == 1:
            assert actual_split_prop == 0
        else:
            assert abs(actual_split_prop - split_proportions_TT[i]) < 0.05


def test_split_props_scaffold_kmeans_murcko(example_data3):
    split_proportions = [0.33, 0.33, 0.33]

    splitter = oce.ScaffoldSplit(
        scaffold_filter_threshold=1,
        split_type="kmeans_murcko",
        split_proportions=split_proportions,
    )
    split = splitter.split(example_data3)
    total_samples = len(example_data3)

    for i in range(3):
        actual_split_prop = len(split[i]) / total_samples
        assert abs(actual_split_prop - split_proportions[i]) < 0.1

    split_proportions_TT = [0.7, 0.0, 0.3]

    splitter = oce.ScaffoldSplit(
        scaffold_filter_threshold=1,
        split_type="kmeans_murcko",
        split_proportions=split_proportions_TT,
    )
    split = splitter.split(example_data3)

    for i in range(3):
        actual_split_prop = len(split[i]) / total_samples
        if i == 1:
            assert actual_split_prop == 0
        else:
            assert abs(actual_split_prop - split_proportions_TT[i]) < 0.1


def test_split_props_property(example_data1):
    split_proportions = [0.8, 0.1, 0.1]

    splitter = oce.PropertySplit(
        property_col="pChEMBL Value",
        threshold=None,
        noise=0.1,
        categorical=False,
        split_proportions=split_proportions,
    )
    split = splitter.split(example_data1)
    total_samples = len(example_data1)

    for i in range(3):
        actual_split_prop = len(split[i]) / total_samples
        assert abs(actual_split_prop - split_proportions[i]) < 0.1

    split_proportions_TT = [0.8, 0.0, 0.2]

    splitter = oce.PropertySplit(
        property_col="pChEMBL Value",
        threshold=None,
        noise=0.1,
        categorical=False,
        split_proportions=split_proportions_TT,
    )
    split = splitter.split(example_data1)

    for i in range(3):
        actual_split_prop = len(split[i]) / total_samples
        if i == 1:
            assert actual_split_prop == 0
        else:
            assert abs(actual_split_prop - split_proportions_TT[i]) < 0.1


def test_split_props_property_thresh(example_data1):
    splitter = oce.PropertySplit(
        property_col="pChEMBL Value", threshold=7, noise=0.1, categorical=False
    )
    split = splitter.split(example_data1)

    for i in range(3):
        if i == 1:
            assert len(split[i]) == 0
        else:
            assert len(split[i]) > 0


"""Check if saved and loaded splitter parameters equal the true parameters before save/load.
"""


def test_save_load(example_data1, example_data3, tmp_path):
    d = tmp_path / "sub"
    d.mkdir()

    random = oce.RandomSplit(split_proportions=[0.8, 0.1, 0.1])
    stratified = oce.StratifiedSplitter(
        split_proportions=[0.8, 0.0, 0.2], value_col="pChEMBL Value"
    )
    scaffold = oce.ScaffoldSplit(
        scaffold_filter_threshold=1,
        split_type="murcko",
        split_proportions=[0.8, 0.1, 0.1],
    )
    scaffold_kmeans = oce.ScaffoldSplit(
        scaffold_filter_threshold=1,
        split_type="kmeans_murcko",
        split_proportions=[0.8, 0.1, 0.1],
    )
    property = oce.PropertySplit(
        property_col="pChEMBL Value",
        threshold=None,
        noise=0.1,
        categorical=False,
        split_proportions=[0.8, 0.1, 0.1],
    )

    splitters = {
        "Random": random,
        "Stratified": stratified,
        "Scaffold": scaffold,
        "Scaffold_KMeans": scaffold_kmeans,
        "Property": property,
    }
    for name in splitters.keys():
        splitter = splitters[name]
        params = oce.base_class.parameterize(splitter)
        oce.base_class.save(splitter, d / f"{name}")

        loaded_splitter = oce.base_class.load(d / f"{name}")
        assert oce.base_class.parameterize(loaded_splitter) == params
