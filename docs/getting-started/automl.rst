AutoML Approaches
===========================

When optimizing for initial results, our framework allows you to skip some of the initial design decisions related to
selecting the best model for a given dataset by using our framework of AutoML approaches. In this framework, AutoML allows
models to be automatically generated and selected. The full list of AutoML approaches is available at :py:mod:`olorenchemengine.automl`.

We can get started with the AutoML framework as follows:
::

    automl = oce.NaiveSelection()

    for _ in range(10): # try 10 different AutoML models
        model = automl.get_model()
        model.fit(...)

Every AutoML method implements the get_model function, but some approaches take into account various factors
(such as performance of the previously returned model) in selecting the model.