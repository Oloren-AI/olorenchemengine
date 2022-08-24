import pytest
import olorenchemengine as oce
import numpy as np
from olorenchemengine.interpret import *

import pandas as pd
from sklearn.model_selection import train_test_split

from rdkit import Chem

__author__ = "Oloren AI"
__copyright__ = "Oloren AI"

def test_swap_mutations():
    s = "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc3ccccc3)C(=O)O)C"
    sm = SwapMutations(radius=1)
    s_out = sm.get_compound(s)

    assert not s_out == s
    assert not s_out is None
    assert isinstance(s_out, str)