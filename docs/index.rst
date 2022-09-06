=================
Oloren ChemEngine
=================

Made by `Oloren AI <oloren.ai>`_, the Python library **Oloren ChemEngine** enables the creation of state-of-the-art, complex molecular property predictors in just a few lines of code. Models defined and trained with olorenchemengine achieve super-leaderboard performance in less than 10 lines.

Installation
************

In a Python 3.8 environment, you can install the package with the following command:

.. code-block:: bash

    bash <(curl -s https://raw.githubusercontent.com/Oloren-AI/olorenchemengine/master/install.sh)

Feel free to inspect the install script to see what is going on under the hood. This will work fine in both a conda environment and a pip environment.

Quick Start
***********
::

    import olorenchemengine as oce

    ## Loading in a dataset

    # df is a Pandas Dataframe with the following columns:
    # "Smiles" (structure)
    # "pChEMBL Value" (property to be predicted)
    df = oce.ExampleDataFrame()

    ## Defining a model

    # The model is a gradient boosted model with the learners being:
    # 1. Random Forest model learning from a set of molecular descriptors
    # 2. GIN model pretrained using contrastive learning on PubChem
    # 3. a Random Forest model trained using a representation
    #    learned using contrastive learning on PubChem
   model = oce.BaseBoosting([
            oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1000),
            oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000)])

    ## Training the model
    model.fit(df["Smiles"], df["pChEMBL Value"])
   
    ## Saving the model
    oce.save(model, "model.oce")

    ## Loading the model
    model2 = oce.load("model.oce")

    ## Predicting property of new compounds
    y_pred = model2.predict(["CC(=O)OC1=CC=CC=C1C(=O)O"])


Philosophy
**********
With the rapidly changing and diverse landscape of molecular property predictors, **Oloren ChemEngine** provides a common framework for the development, testing, and usage of AI model. Previously an exercise in chaos, with **Oloren ChemEngine** many different types of models--including ensembles of different modelling strategies and features--can be defined, trained and saved, imposing *structure* on the development of molecular property predictors while maintaining *flexibility*.

The following are the guiding principles for the development of **Oloren ChemEngine** models:

* | **Simplicity**: models can be defined, trained, saved, and used with minimal effort.
* | **Flexibility**: differing molecular representations, experimental datapoints, model architectures, ensembling strategies, and other innovative methodologies can be implemented in a consistent framework
* | **Accuracy**: the capabilities of the library match or supercede top-of-the-leaderboard molecular property predictors, with a concerted focus on improving the utility of molecular property predictors in real-world settings, leveraging available experimental data.

Defined as subclasses of ``BaseModel``, models including graph neural networks, descriptor- and fingerprint-based machine learning models, and Oloren AI proprietary models are all supported. 

Contents
========

.. toctree::
   :maxdepth: 3

   Module Reference <api/modules>
   Getting Started <getting-started/*>

.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Getting Started

   getting-started/*

.. toctree::
   :glob:
   :caption: Examples
   :maxdepth: 3

   examples/*

.. toctree::
   :caption: Project Links
   :maxdepth: 3

   Contributing <contributing>
   Authors <authors>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: https://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain
.. _Sphinx: https://www.sphinx-doc.org/
.. _Python: https://docs.python.org/
.. _Numpy: https://numpy.org/doc/stable
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: https://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: https://scikit-learn.org/stable
.. _autodoc: https://www.sphinx-doc.org/en/master/ext/autodoc.html
.. _Google style: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: https://www.sphinx-doc.org/en/master/domains.html#info-field-lists
.. _conda-pack: https://conda.github.io/conda-pack/
