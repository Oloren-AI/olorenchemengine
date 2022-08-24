Predicting Confidence Intervals
===============================

We will go over the basics of using the BaseErrorModel class for
predicting confidence intervals, including both basic and more
sophisticated predictors.

.. code:: ipython3

    import olorenchemengine as oce
    import pandas as pd
    import numpy as np
    import json
    import tqdm
    
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    dataset = oce.DatasetFromCSV("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv", structure_col = "smiles", property_col = "exp")
    splitter = oce.RandomSplit(split_proportions=[0.8,0.1,0.1])
    dataset = dataset + splitter
    oce.save(dataset, 'lipophilicity_dataset.oce')
    
    model = oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000)
    model.fit(dataset.train_dataset[0], dataset.train_dataset[1])
    oce.save(model, 'lipophilicity_model_rf.oce')

SDC
---

We will demonstrate the basics of the class with SDC. We start by
creating an instance of the SDC class with our trained model and the
dataset used for training.

.. code:: ipython3

    testSDC = oce.SDC()
    testSDC.build(model, dataset.train_dataset[0], dataset.train_dataset[1])

We can now fit the estimator with a dataset used for validation. Fitting
our estimator will display a graph of the residual versus the confidence
score. The graph can be turned off by setting ``plot = False``. Blue
points are our validation datapoints, red points are the confidence
intervals for each bin, and the red line is the fitted linear model.

.. code:: ipython3

    testSDC.fit(dataset.valid_dataset[0], dataset.valid_dataset[1])


.. parsed-literal::

    420it [00:02, 177.36it/s]



.. image:: 2A_Confidence_Intervals_files/2A_Confidence_Intervals_7_1.png


We can also fit the estimator by running cross validation on the
training dataset.

.. code:: ipython3

    testSDC.fit_cv()

Finally, we can predict confidence scores for any input, including the
test data of the original dataset.

.. code:: ipython3

    testSDC.score(dataset.test_dataset[0])

ADAN
----

ADAN is a more sophisticated model with six different parameters. We
will now demonstrate running the ADAN model with the BaseConfidenceScore
class. Like before, we start by creating the BaseADAN object.

.. code:: ipython3

    testADAN = oce.BaseADAN(criteria='E_raw')
    testADAN.build(model, dataset.train_dataset[0], dataset.train_dataset[1])

Fitting the model is the same except for the input of the ADAN criteria
we wish to use. ``criteria`` must be assigned to a subset of
``['A','B','C','D','E','F','A_raw','B_raw','C_raw','D_raw','E_raw','F_raw']``,
or optionally, set ``criteria='Category'`` to use the original ADAN
category criterion. Raw values are standardized based on the testing
data.

.. code:: ipython3

    testADAN.fit(dataset.valid_dataset[0], dataset.valid_dataset[1], method = 'bin', quantile=0.8)



.. image:: 2A_Confidence_Intervals_files/2A_Confidence_Intervals_16_0.png


Predicting confidence scores is the same as before.

.. code:: ipython3

    testADAN.score(dataset.test_dataset[0])

Aggergate Error Models: Random Forest
-------------------------------------

We demonstrate our AggregateError class by running a random forest model
on several different confident scores. SDC is a measure of distance to
model, wRMSD1 and wRMSD2 are measures of local model performance, and
PREDICTED is the output of the model.

.. code:: ipython3

    models = [oce.SDC(), oce.wRMSD1(), oce.wRMSD2(), oce.PREDICTED()]
    testrf = oce.RandomForestErrorModel(models)
    testrf.build(model, dataset.train_dataset[0], dataset.train_dataset[1])

Like fitting, training the aggregate model can also be done with an
external dataset via the ``.train`` method, or with cross validation of
the training dataset via the ``.train_cv`` method. We recommend training
via cross validation.

.. code:: ipython3

    testrf.train_cv()

Just like in the BaseErrorModel class, we can now fit the error model.
We will do this on the validation dataset.

.. code:: ipython3

    testrf.fit(dataset.valid_dataset[0], dataset.valid_dataset[1], method='qbin')


.. parsed-literal::

    420it [00:07, 54.26it/s]
    420it [00:08, 52.43it/s] 
    420it [00:07, 54.24it/s]



.. image:: 2A_Confidence_Intervals_files/2A_Confidence_Intervals_25_1.png


Analysis
--------

We can analyze what fraction of the test data is within is predicted
confidence interval. If our datasets were chosen properly, that fraction
should be very similar to the confidence interval we chose during
fitting (0.8). We can also compare the predicted confidence intervals to
the confidence intervals calculated given the standard deviation of the
validation dataset.

.. code:: ipython3

    in_interval = np.abs(dataset.test_dataset[1] - model.predict(dataset.test_dataset[0]['smiles'])) < testSDC.score(dataset.test_dataset[0])


.. parsed-literal::

    420it [00:02, 175.30it/s]


.. code:: ipython3

    sum(in_interval) / len(in_interval)




.. parsed-literal::

    0.9285714285714286


