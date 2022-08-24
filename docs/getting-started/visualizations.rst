Visualizing Models and Data
===========================

Let's revisit the starter snippet from the homepage of our documentation.

Line by line, we'll walk through how Visualizing your Models and Datasets works!

First, let's import the necessary libraries:
::
    from olorenchemengine.visualizations import *
    from olorenchemengine import *
    import olorenchemengine as oce
::

Second, we'll use a pandas dataframe of csv data given by the line
::
    df = oce.ExampleDataset
::
to train the model, but visualizations do not
allow us to pass in a pandas dataframe. Instead, it must be of type
'BaseDataset'

with this line,
::
    test_dataset = oce.BaseDataset(name='purple', data=df.to_csv(), structure_col='Smiles', property_col='pChEMBL Value')
::

See how it all comes together below.
::
    ## Loading in a dataset

    # df is a Pandas Dataframe with the following columns:
    # "Smiles" (structure)
    # "pChEMBL Value" (property to be predicted)
    df = oce.ExampleDataset
    test = oce.BaseDataset(name='purple', data=df.to_csv(), structure_col='Smiles', property_col='pChEMBL Value')
::

Next, we'll train, fit, save, and load our model in the same as before:
::
    model = oce.BaseBoosting([
                oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"),n_estimators=1000),
                oce.BaseTorchGeometricModel(oce.TLFromCheckpoint("default"), preinitialized=True),
                oce.RandomForestModel(oce.OlorenCheckpoint("default"),n_estimators=1000)])


    ## Training the model
    model.fit(df["Smiles"], df["pChEMBL Value"])

    ## Predicting property of new compounds
    y_pred = model.predict(["CC(=O)OC1=CC=CC=C1C(=O)O"])

    ## Saving the model
    oce.save(model, "model.oce")

    ## Loading the model
    loaded_model = oce.load("model.oce")
::

Now, we can use the loaded_model to Visualize our with VisualizeModelSim, a method that
" Visualizes a model's predicted vs true plot on given dataset, where each point is colored
by a compound's similarity to the train set"

Note: notice how we use test_dataset, which is oce.ExampleDataset converted into a BaseDataset
object.
::

  vis =  oce.VisualizeModelSim(model=loaded_model,dataset=test_dataset)
::

Finally, every Visualization has a render_ipynb method that can be used to render the visualization. So, we can do the following:
::
    vis.render_ipynb()
::
And your visualization is viewable in a jupyter notebook.

"""