=============================
|oceLogo| Oloren ChemEngine
=============================

.. |oceLogo| image:: assets/oce_logo.png
  :height: 30

Oloren ChemEngine (oce) is a software package developed and maintained by Oloren AI containing a
unified API for the development and use of molecular property predictors enabling

* Direct development of high-performing predictors
* Integration of predictors into model interpretability, uncertainty quantification, and analysis frameworks

Here's an example of what we mean by this. In less than ten lines of code, we'll
train, save, load, and predict with a gradient-boosted model with two different
molecular vector representations.

.. code-block:: python

    import olorenchemengine as oce

    df = oce.ExampleDataFrame()

    model = oce.BaseBoosting([
                oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1000),
                oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000)])
    model.fit(df["Smiles"], df["pChEMBL Value"])

    oce.save(model, "model.oce")

    model2 = oce.load("model.oce")

    y_pred = model2.predict(["CC(=O)OC1=CC=CC=C1C(=O)O"])

It's that simple! And it's just as simple to train a graph neural network, generate
visualizations, and create error models. More information on features and
capabilities is available in our documentation at `docs.oloren.ai <https://docs.oloren.ai>`_.

-------------------------------
oce at a high level
-------------------------------

Everything in oce is built around Oloren's ``BaseClass`` system, which all classes stem from.
Any ``BaseClass`` derived objects has its parameters and complete state saved
via ``parmeterize`` and ``saves`` respectively. A blank object (no internal state)
can be recreated via ``create_BC`` and a complete object (with internal state) can
be recreated via ``loads``.

The system includes abstract subclasses of ``BaseClass`` are named ``Base{Class Type}``
and their interactions, most prominently

    * ``BaseModel``, a base class for all any molecular property predictor
    * ``BaseRepresentation``, a base class for all molecular representations
    * ``BaseVisualization``, a base class for all types of visualizations and analyses

This abstraction system is provided free of charge by Oloren AI in the internals.

-------------------------------
Getting Started with oce
-------------------------------
_______________________________
Installation
_______________________________

In a Python 3.8 environment, you can install the package with the following command:

.. code-block:: bash

    bash <(curl -s https://raw.githubusercontent.com/Oloren-AI/olorenchemengine/master/install.sh)

Feel free to check out install.sh to see what is happening under the hood. This will work fine in both a conda environment and a pip environment.

_______________________________
Docker
_______________________________

Alternatively, you can also run OCE from one of our docker images. After cloning the repo, just run:

.. code-block:: bash

    docker build -t oce:latest -f docker/Dockerfile.gpu . # build the docker image
    docker run -it -v ~/.oce:/root/.oce oce:latest python # run the docker image

Replace ".gpu" with ".cpu" in the docker path if you want to run the project in a dockerized environment.

_______________________________
Basic Usage
_______________________________
We have an examples folder, which we'd highly reccomend you checkout--1A and 1B
in particular--the rest of the examples can be purused when the topics come up.

_______________________________
Contributing
_______________________________
First, thank you for contributing to OCE! To install OCE in editable/development mode, simply clone the repository and run:

.. code-block:: bash

    bash install.sh --dev

This will install the repo in an editable way, so your changes will reflect immediately in your python environment. All tests for OCE are in the `tests` directory and can be run by running `pytest` in this directory. Please contact support@oloren.ai if you need any assistance in your development process!

PRs from external collaborators will require a Contributor License Agreement (CLA) to be signed before the code is merged into the repository.

_______________________________
Notice
_______________________________
NOTICE: IN CHEMENGINE, WE LOG MODEL PERFORMANCE AND MODEL HYPERPARAMETERS—NO THERAPEUTIC DATA—AND WE REQUIRE CONTRIBUTOR AGREEMENTS

-------------------------------
Our Thanks
-------------------------------
First, our thanks to the community of developers and scientists, who've built and maintained
a repotoire of software libraries and scripts which have been invaluable. We'd like
to particularly thank the folks creating RDKit, PyTorch Geometric, and SKLearn who've
developed software we strive to emulate and exceed.

Second, we'd like to thank the amazing developers at Oloren who've created Oloren
ChemEngine through enoromous effort and dedication. And, we'd like to thank our future
collaborators and contributors ahead, who we're excited meet and work with.

Third, huge gratitude goes to our investors, clients, and customers who've been
ever patient and ever gracious, who've provided us with the opportunity to bring
something we believe to be truly valuable into the world.
