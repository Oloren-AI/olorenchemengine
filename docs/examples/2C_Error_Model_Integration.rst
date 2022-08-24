Integrated Error Models
=======================

In addition to creating your own error models to evaluate pre-trained
models, error models can also be built alongside a model.

.. code:: ipython3

    import olorenchemengine as oce
    import pandas as pd
    import numpy as np
    import json
    import tqdm
    
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    #lipo_dataset = oce.DatasetFromCSV("Lipophilicity.csv", structure_col = "smiles", property_col = "exp")
    #splitter = oce.RandomSplit(split_proportions=[0.8,0.1,0.1])
    #lipo_dataset = splitter.transform(lipo_dataset)
    #oce.save(lipo_dataset, 'lipophilicity_dataset.oce')
    
    dataset = oce.load('lipophilicity_dataset.oce')
    model = oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000)

To build an error model during model training, simply input the error
model you wish to use. Here, we will use the ``oce.SDC`` error model.

.. code:: ipython3

    error_model = oce.SDC()
    model.fit(dataset.train_dataset[0], dataset.train_dataset[1], error_model=error_model)

The error model is now built and stored in ``model.error_model``. From
here, any error model methods, such as ``.train()`` and ``.train_cv()``
for aggregate error models, or ``.fit()`` and ``.fit_cv()`` for all
error models, can be run. Note that by default, ``.train`` is not run
for aggregate error models, and must be run individually before model
fitting.

Fitting can also be done when running ``model.test()`` by setting
``fit_error_model=True``.

.. code:: ipython3

    model.test(dataset.valid_dataset[0], dataset.valid_dataset[1], fit_error_model=True)

Finally, if a model contains a fitted error model, setting
``return_ci=True`` when running ``model.predict()`` will return the
confidence intervals. Setting ``return_vis=True`` will in turn return
``VisualizeError`` objects.

.. code:: ipython3

    df = model.predict(dataset.test_dataset[0], return_ci=True, return_vis=True)

.. code:: ipython3

    df.head()

.. code:: ipython3

    df.vis[0].render_ipynb()

Production Level Models
=======================

Production level models use the entire dataset to train the model. As
such, metrics and error model training and fitting are done via cross
validation. The entire process can be done by calling the ``.fit_cv()``
function.

.. code:: ipython3

    model = oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000)
    error_model = oce.SDC()
    
    model.fit_cv(dataset.entire_dataset[0], dataset.entire_dataset[1], error_model=error_model, scoring = "r2")

The trained error model will be stored in ``model.error_model``
