import os
from os.path import expanduser

import numpy as np

import olorenchemengine as oce


def remote(func):
    def wrapper(*args, **kwargs):
        with oce.Remote("http://api.oloren.ai:5000") as remote:
            func(*args, **kwargs)

    return wrapper


def get_model_for_testing(name):
    path = oce.download_public_file(f"test_models/{name}.oam")
    return oce.load(path)


test_molecules = ["O=C(C)Oc1ccccc1C(=O)O", "CC(CC(c1ccccc1)C2=C(c3c(OC2=O)cccc3)O)=O"]


def test_reg_0():
    model = get_model_for_testing("reg model 0")
    assert np.allclose(
        model.predict(test_molecules), [0.5017606, -0.8976576], atol=1e-3
    )


def test_reg_1():
    model = get_model_for_testing("reg model 1")
    assert np.allclose(
        model.predict(test_molecules), [0.25767391, -0.15310819], atol=1e-3
    )


def test_reg_2():
    model = get_model_for_testing("reg model 2")
    assert np.allclose(
        model.predict(test_molecules), [0.14640912, -0.19515301], atol=1e-3
    )


def test_reg_3():
    model = get_model_for_testing("reg model 3")
    print(model.predict(test_molecules))
    assert np.allclose(
        model.predict(test_molecules), [0.5101875, 0.27005324], atol=1e-3
    )


def test_reg_4():
    model = get_model_for_testing("reg model 4")
    assert np.allclose(
        model.predict(test_molecules), [0.69562391, 0.1398656], atol=1e-3
    )


def test_class_0():
    model = get_model_for_testing("class model 0")
    assert np.allclose(model.predict(test_molecules), [0.9907742, 0.9907742])


def test_class_1():
    model = get_model_for_testing("class model 1")
    assert np.allclose(model.predict(test_molecules), [1.0, 1.0])


def test_class_2():
    model = get_model_for_testing("class model 2")
    assert np.allclose(model.predict(test_molecules), [1.0, 1.0])


def test_class_3():
    model = get_model_for_testing("class model 3")
    assert np.allclose(model.predict(test_molecules), [0.580161, 0.56691015])


def test_class_4():
    model = get_model_for_testing("class model 4")
    assert np.allclose(model.predict(test_molecules), [1.0, 1.0])
