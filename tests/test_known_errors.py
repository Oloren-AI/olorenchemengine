import pytest
import olorenchemengine as oce
import pandas as pd
import numpy as np
from olorenchemengine.internal import download_public_file

__author__ = "Oloren AI"
__copyright__ = "Oloren AI"

@pytest.fixture
def example_data1():
    file_path = download_public_file("sample-csvs/sample_data1.csv")
    df = pd.read_csv(file_path)
    return df

#RuntimeError: Not compiled with CUDA support
def test_TorchGeometric(example_data1):
    data = oce.RandomSplit(split_proportions=[0.8 ,0.0, 0.2]).split(example_data1)
    train, valid, test = data[0], data[1], data[2]
    s = "{'BC_class_name': 'BaseTorchGeometricModel', 'args': [{'BC_class_name': 'GINModel', 'args': [], 'kwargs': {'batch_size': 100, 'conv_radius': 3, 'conv_type': 'gin+', 'dataset': 'molpcba', 'dropout': 0.5, 'hidden': 100, 'layers': 3, 'lr': 0.001, 'optim': 'adam', 'task_type': 'classification', 'virtual_node': False}}], 'kwargs': {'auto_lr_find': True, 'batch_size': 16, 'epochs': 5, 'log': True, 'lr': 0.0001, 'pos_weight': 'balanced', 'preinitialized': False, 'representation': {'BC_class_name': 'TorchGeometricGraph', 'args': [], 'kwargs': {'atom_featurizer': {'BC_class_name': 'OGBAtomFeaturizer', 'args': [], 'kwargs': {}}, 'bond_featurizer': {'BC_class_name': 'OGBBondFeaturizer', 'args': [], 'kwargs': {}}}}}}"
    s = s.replace("'", "\"").replace("True", "true").replace("False", "false").replace("None", "null")
    model=(oce.create_BC(s))

    model.fit(train['Smiles'], train['pChEMBL Value'])
    preds=model.predict(test['Smiles'])

#TypeError: unsupported operand type(s) for +: 'Mol2Vec' and 'DescriptastorusDescriptor'
def test_ConcatMol2Vec():
    oce.DescriptastorusDescriptor('morgan3counts') + oce.Mol2Vec()
    oce.Mol2Vec() + oce.DescriptastorusDescriptor('morgan3counts')

#ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) and the array at index 1 has 1 dimension(s)
def test_ConcatMordred(example_data1):
    rep = oce.DescriptastorusDescriptor('morgan3counts') + oce.MordredDescriptor()
    rep.convert('C')

    model = oce.RandomForestModel(representation=rep, n_estimators=10)
    model.fit(example_data1['Smiles'], example_data1['pChEMBL Value'])

'''
Alternative code I found working taken from DeepChem
from mordred import Calculator, descriptors, is_missing

class MordredDescriptor(BaseCompoundVecRepresentation):

    """ Wrapper for Mordred descriptors (https://github.com/mordred-descriptor/mordred)

    Parameters:
        log (bool): whether to log the representations or not
        descriptor_set (str): name of the descriptor set to use, default all
        normalize (bool): whether to normalize the descriptors or not
    """

    @log_arguments
    def __init__(self, descriptor_set: Union[str, list] = "all", ignore_3D = False, log: bool = True, **kwargs):
        if descriptor_set == "all":
            self.calc = Calculator(descriptors, ignore_3D=ignore_3D)
        super().__init__(log=False, **kwargs)

    def _convert(self, s: str) -> np.ndarray:
        feature = self.calc(Chem.MolFromSmiles(s))
        feature = [
            0.0 if is_missing(val) or isinstance(val, str) else val
            for val in feature
        ]
        return np.array(feature)
'''