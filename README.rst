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
                oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000),  # RF w/ our proprietary fingerprint
                oce.SPGNN(model_type="contextpred"),  # fine tune a trained GNN on your data
            ])
            
    model.fit(df["Smiles"], df["pChEMBL Value"])

    oce.save(model, "model.oce")

    model2 = oce.load("model.oce")

    y_pred = model2.predict(["CC(=O)OC1=CC=CC=C1C(=O)O"])

It's that simple! And it's just as simple to train a graph neural network, generate
visualizations, and create error models. More information on features and
capabilities is available in our documentation at `docs.oloren.ai <https://docs.oloren.ai>`_.

-------------------------------
Getting Started with oce
-------------------------------
_______________________________
Installation
_______________________________

In a fresh Python 3.8 environment, you can install the package with the following command, pasted into your terminal after activating your virtual environment:

.. code-block:: bash

    bash <(curl -s https://raw.githubusercontent.com/Oloren-AI/olorenchemengine/master/install.sh)

Feel free to check out install.sh to see what is happening under the hood. This will work fine in both a conda environment and a pip environment. The reason why a fresh environment is preferred is because PyTorch Geometric/ PyTorch/ CUDA are very particular about versioning, which often are more muddled in existing environment.

Here are some common error messages and solutions:
https://oloren-ai.notion.site/Oloren-ChemEngine-Installation-FAQ-f2edec771a7f4350af5fdc361d494604

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
Notice
_______________________________
Maintaining and developing Oloren ChemEngine requires a lot of resources. As such, we would like to log for each evaluated model the model hyperparameters, the model performance metrics and a unique, non-identifying hash of the dataset. These logs are used to improve our models. Below is a representative example of such a log:

.. code-block:: javascript
    
    {dataset_hash: "149eae5c763afcc14f6355007df298b05f4a51c6a334ea933fbe7fc496adb271",

    metric_direction: null,

    metrics: "{"Average Precision": 0.9479992350277128, "ROC-AUC": 0.7450549450549451}",

    name: "BaseBoosting 1zpI0dIb",

    params: "{"BC_class_name": "BaseBoosting", "args": [[{"BC_class_name": "RandomForestModel", "args": [{"BC_class_name": "DescriptastorusDescriptor", "args": ["morgan3counts"], "kwargs": {"log": true, "scale": null}}], "kwargs": {"bootstrap": true, "criterion": "entropy", "max_features": "log2", "n_estimators": 2000, "max_depth": null, "class_weight": null}}, {"BC_class_name": "RandomForestModel", "args": [{"BC_class_name": "DescriptastorusDescriptor", "args": ["morganchiral3counts"], "kwargs": {"log": true, "scale": null}}], "kwargs": {"bootstrap": true, "criterion": "entropy", "max_features": "log2", "n_estimators": 2000, "max_depth": null, "class_weight": null}}, {"BC_class_name": "RandomForestModel", "args": [{"BC_class_name": "DescriptastorusDescriptor", "args": ["morganfeature3counts"], "kwargs": {"log": true, "scale": null}}], "kwargs": {"bootstrap": true, "criterion": "entropy", "max_features": "log2", "n_estimators": 2000, "max_depth": null, "class_weight": null}}, {"BC_class_name": "RandomForestModel", "args": [{"BC_class_name": "DescriptastorusDescriptor", "args": ["rdkit2dnormalized"], "kwargs": {"log": true, "scale": null}}], "kwargs": {"bootstrap": true, "criterion": "entropy", "max_features": "log2", "n_estimators": 2000, "max_depth": null, "class_weight": null}}, {"BC_class_name": "RandomForestModel", "args": [{"BC_class_name": "OlorenCheckpoint", "args": ["default"], "kwargs": {"log": true, "num_tasks": 2048}}], "kwargs": {"bootstrap": true, "criterion": "entropy", "max_features": "log2", "n_estimators": 2000, "max_depth": null, "class_weight": null}}]], "kwargs": {"log": true, "n": 1, "oof": false, "nfolds": 5}}"}

The dataset hash is created with the following code:

.. code-block:: python

    import joblib

    dataset_hash = joblib.hash(X) + joblib.hash(y)

This means that **we log no therapeutics-related data whatsoever.** We just log hashes of model performance. 

If you would still prefer a logging-free version, please fill out the following form to obtain a version with all logging code excised: https://y09gl0qf49q.typeform.com/to/brGMidJ0. 

We also require contributor agreements for all versions of Oloren ChemEngine.

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

-------------------------------
Contributing
-------------------------------
First, thank you for contributing to OCE! To install OCE in editable/development mode, simply clone the repository and run:

.. code-block:: bash

    bash install.sh --dev

This will install the repo in an editable way, so your changes will reflect immediately in your python environment. All tests for OCE are in the `tests` directory and can be run by running `pytest` in this directory. Please contact support@oloren.ai if you need any assistance in your development process!

PRs from external collaborators will require a Contributor License Agreement (CLA) to be signed before the code is merged into the repository.

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
