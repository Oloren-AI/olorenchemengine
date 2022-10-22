import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from sklearn.model_selection import train_test_split

import olorenchemengine as oce
from olorenchemengine.interpret import *

__author__ = "Oloren AI"
__copyright__ = "Oloren AI"


def remote(func):
    def wrapper(*args, **kwargs):
        with oce.Remote("http://api.oloren.ai:5000") as remote:
            func(*args, **kwargs)

    return wrapper


def test_swap_mutations():
    s = "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc3ccccc3)C(=O)O)C"
    sm = SwapMutations(radius=1)
    s_out = sm.get_compound(s)

    assert not s_out == s
    assert not s_out is None
    assert isinstance(s_out, str)
