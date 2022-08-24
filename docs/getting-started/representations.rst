Representing Molecules
===========================

In order to build molecules on chemical databases, we need to be able to represent chemical structures to our models.
olorenchemengine is equipped with a wide array of the most performant representations available in the literature, in addition to our own proprietary representation, called OlorenVec.
The :py:mod:`olorenchemengine.representations` module contains a full list of the available representations.


Transforming Molecules with Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most basic thing we can do with the molecular representation is transform a molecule into a machine readable format using the representatiion.

::

    acetaminophen = "CC(=O)NC1=CC=C(C=C1)O"
    ibuprofen = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"

    olorenvec_repr = oce.OlorenCheckpoint("default")
    olorenvec_repr.convert(acetaminophen) # convert a single molecule
    olorenvec_repr.convert([acetaminophen, ibuprofen]) # converts multiple molecules

Defining Models with Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All of the models defined in :py:mod:`olorenchemengine.basics` require a representation to be passed in as an additional input.


For example, this code block generates a RandomForestModel using varying underlying molecular representations.
::

    olorenvec_model = oce.RandomForestModel(oce.OlorenCheckpoint("default"))
    rdkit2d_model = oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"))
    morgan_model = oce.RandomForestModel(oce.DescriptastorusDescriptor("morgan3counts"))

Due to the structure of olorenchemengine, running a hyperparameter sweep over different representations is as simple as writing a for loop:

::

    representations = [oce.OlorenCheckpoint("default"),
                       oce.DescriptastorusDescriptor("rdkit2dnormalized"),
                       oce.DescriptastorusDescriptor("morgan3counts")]

    for representation in representations:
        model = oce.RandomForestModel(representation)
        # Evaluate model here...

Concatenating Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to concatenate multiple representations together, you can just add them together!
::

    concatenated_representation = oce.OlorenCheckpoint("default") +
        oce.DescriptastorusDescriptor("rdkit2dnormalized") +
        oce.DescriptastorusDescriptor("morgan3counts")
    concatenated_model = oce.RandomForestModel(concatenated_representation)

Defining New Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New representations can be defined by extending the :py:class:`olorenchemengine.representations.BaseCompoundVecRepresentation` class:

::

    class MyCustomRepresentation(oce.BaseCompoundVecRepresentation):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def _convert(self, smiles, y=None):
            return vector_representation_of_smiles # should be a numpy array

    my_representation = MyCustomRepresentation()
    my_model = oce.RandomForestModel(my_representation)



"""


