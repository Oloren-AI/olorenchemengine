Building your own molecular representation class
================================================

This notebook walks through the creation of a custom molecular
representation (``BaseCompoundVecRepresentation``) that can be used with
model training and predictions in OCE.

Parent Class Specs
~~~~~~~~~~~~~~~~~~

The custom representation class inherits from
BaseCompoundVecRepresentation. All representations have two parameters
defined from the parent class:       “scale”: sklearn scaler (or None)
which is used to scale the output representation data. Defaults to
StandardScaler.       “collinear_thresh”: Threshold for linear
collinearity. Representation features with correlation coefficient above
threshold with any other feature is removed.

Initialization
~~~~~~~~~~~~~~

You may define any custom parameters necessary for representation
calculation in the init function of your class. Ensure that
initialization allows for keyword arguments which will be passed to the
parent class (e.g. scale, collinear_thresh)

.. code:: ipython3

    from olorenchemengine import BaseCompoundVecRepresentation
    from olorenchemengine import log_arguments
    
    class CustomRepresentation(BaseCompoundVecRepresentation):
        @log_arguments
        def __init__(self, param1, log=True, **kwargs):
            self.param1 = param1
            super().__init__(log=False, **kwargs)

\_convert function
~~~~~~~~~~~~~~~~~~

The only function needed in your custom representation class is the
\_convert helper function, which takes as a parameter the SMILES string
representation of a single molecule and outputs a numpy array of its
representation vector. The numpy array should be of shape (n,), where n
is the number of bits/features for one molecule’s representation.

.. code:: ipython3

    class CustomRepresentation(BaseCompoundVecRepresentation):
        @log_arguments
        def __init__(self, radius = 2, nbits = 1024, log=True, **kwargs):
            self.radius = 2
            self.nbits = 1024
            super().__init__(log=False, **kwargs)
        
        def _convert(self, smiles):
            '''Run calculations on molecule SMILES string.
            Example calculation using Morgan Fingerprint from RDKit.
            '''
            from rdkit import Chem
            from rdkit.Chem import AllChem
            import numpy as np
    
            m = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=self.radius, nBits=self.nbits)
    
            return np.array(fp)

Convert!
~~~~~~~~

The parent class’s convert function automatically calls the child’s
\_convert function to convert lists and single molecule strings.

.. code:: ipython3

    representation = CustomRepresentation(radius = 2, nbits = 1024)

.. code:: ipython3

    single_rep = representation.convert('CN=C=O')

.. code:: ipython3

    print(single_rep)
    print(single_rep.shape)

.. code:: ipython3

    list_rep = representation.convert(['CN=C=O', '[Cu+2].[O-]S(=O)(=O)[O-]', 'O=Cc1ccc(O)c(OC)c1 COc1cc(C=O)ccc1O'])

.. code:: ipython3

    print(list_rep)
    print(list_rep.shape)

Using the representation in a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may now use your newly created representation class in any OCE model

.. code:: ipython3

    import olorenchemengine as oce
    import pandas as pd
    
    #load dataset
    df = pd.read_csv("https://storage.googleapis.com/oloren-public-data/CHEMBL%20Datasets/997_2298%20-%20VEGFR1%20(CHEMBL1868).csv")
    dataset = (oce.BaseDataset(data = df.to_csv(),
        structure_col = "Smiles", property_col = "pChEMBL Value") +
               oce.CleanStructures() + 
               oce.RandomSplit()
    )

.. code:: ipython3

    model = oce.BaseBoosting([
        oce.RandomForestModel(representation = representation, n_estimators=1000),
        oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000),
        oce.ChemPropModel(epochs=20, batch_size=64)
    ])

.. code:: ipython3

    model.fit(*dataset.train_dataset)
    model.test(*dataset.test_dataset)
