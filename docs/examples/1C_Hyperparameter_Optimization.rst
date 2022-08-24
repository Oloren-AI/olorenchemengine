Hyperparameter Optimization
===========================

We’ll be using oce’s hyperparameter optimization engine, powered by
hyperopt, to create a stronger model of Caco-2 permeability

.. code:: ipython3

    # We'll first be downloading the data as described in 1A.
    
    import olorenchemengine as oce
    
    import requests
    from sklearn import metrics
    r = requests.get("https://ndownloader.figstatic.com/files/4917022")
    open("caco2_data.xlsx" , 'wb').write(r.content)
    
    import pandas as pd
    import numpy as np
    
    df = pd.read_excel("caco2_data.xlsx")
    
    df["split"] = df["Dataset"].replace({"Tr": "train", "Te": "test"})
    df = df[["smi", "split", "logPapp"]].dropna()
    
    import random
    p = 0.8
    train_size = int(len(df[df["split"]=="train"]) * p)
    val_size = len(df[df["split"]=="train"]) - train_size
    l = ["valid"]*val_size + ["train"]*train_size
    random.shuffle(l)
    df.loc[df["split"] == "train", "split"] = l
    
    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False
    df = df[df["logPapp"].apply(isfloat)]
    
    df["logPapp"] = df["logPapp"].astype('float')
    
    dataset = oce.BaseDataset(data = df.to_csv(), structure_col="smi", property_col="logPapp")
    manager = oce.ModelManager(dataset, metrics="Root Mean Squared Error", file_path="1C_manager.oce")

.. code:: ipython3

    # Here we will be using a short-list of good model architectures and hyperparameter
    # values, which were found using the Therapeutic Data Commons ADMET dataset collection.
    
    manager.run(oce.TOP_MODELS_ADMET())

.. code:: ipython3

    model = oce.BaseBoosting(
        [oce.RandomForestModel(oce.OptChoice("descriptor1", [oce.GobbiPharma2D(), oce.Mol2Vec(), oce.DescriptastorusDescriptor("rdkit2dnormalized"), oce.DescriptastorusDescriptor("morgan3counts"), oce.OlorenCheckpoint("default")]),
            n_estimators=oce.OptChoice("n_estimators1", [10, 500, 1000, 2000]),
            max_features = oce.OptChoice("max_features1", ["log2", "auto"]),),
        oce.RandomForestModel(oce.OptChoice("descriptor2", [oce.GobbiPharma2D(), oce.Mol2Vec(), oce.DescriptastorusDescriptor("rdkit2dnormalized"), oce.DescriptastorusDescriptor("morgan3counts"), oce.OlorenCheckpoint("default")]),
            n_estimators=oce.OptChoice("n_estimators2", [10, 500, 1000, 2000]),
            max_features = oce.OptChoice("max_features2", ["log2", "auto"]),),
        oce.RandomForestModel(oce.OptChoice("descriptor3", [oce.GobbiPharma2D(), oce.Mol2Vec(), oce.DescriptastorusDescriptor("rdkit2dnormalized"), oce.DescriptastorusDescriptor("morgan3counts"), oce.OlorenCheckpoint("default")]),
            n_estimators=oce.OptChoice("n_estimators3", [10, 500, 1000, 2000]),
            max_features = oce.OptChoice("max_features3", ["log2", "auto"]),)]
    )

.. code:: ipython3

    best = oce.optimize(model, manager, max_evals=100)

.. code:: ipython3

    manager.get_model_database().sort_values(by="Root Mean Squared Error", ascending=True)

.. code:: ipython3

    manager.get_model_database().sort_values(by="Root Mean Squared Error", ascending=True)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Model Name</th>
          <th>Model Parameters</th>
          <th>Fitting Time</th>
          <th>Root Mean Squared Error</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>94</th>
          <td>BaseBoosting 8K36OOLO</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>91.027555</td>
          <td>0.335641</td>
        </tr>
        <tr>
          <th>95</th>
          <td>BaseBoosting k8-gHshR</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>677.343201</td>
          <td>0.339610</td>
        </tr>
        <tr>
          <th>10</th>
          <td>BaseBoosting cZHDmmMV</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>262.673683</td>
          <td>0.341740</td>
        </tr>
        <tr>
          <th>93</th>
          <td>BaseBoosting K4L62AtM</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>59.878670</td>
          <td>0.343320</td>
        </tr>
        <tr>
          <th>97</th>
          <td>BaseBoosting 6v_E8N2p</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>66.435989</td>
          <td>0.349018</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>8</th>
          <td>KNN 20fz7vhA</td>
          <td>{'BC_class_name': 'KNN', 'args': [{'BC_class_n...</td>
          <td>1.294361</td>
          <td>0.525758</td>
        </tr>
        <tr>
          <th>21</th>
          <td>BaseBoosting m_RXV5dC</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>703.818117</td>
          <td>0.533155</td>
        </tr>
        <tr>
          <th>92</th>
          <td>ChemPropModel TuX4IIuT</td>
          <td>{'BC_class_name': 'ChemPropModel', 'args': [],...</td>
          <td>13.275340</td>
          <td>0.538600</td>
        </tr>
        <tr>
          <th>91</th>
          <td>ChemPropModel TuX4IIuT</td>
          <td>{'BC_class_name': 'ChemPropModel', 'args': [],...</td>
          <td>17.817980</td>
          <td>5.149165</td>
        </tr>
        <tr>
          <th>89</th>
          <td>ChemPropModel 7l6982xF</td>
          <td>{'BC_class_name': 'ChemPropModel', 'args': [],...</td>
          <td>4.267098</td>
          <td>5.150783</td>
        </tr>
      </tbody>
    </table>
    <p>98 rows × 4 columns</p>
    </div>



.. code:: ipython3

    manager.best_model.test(*dataset.test_dataset)


.. parsed-literal::

    255it [00:00, 439.32it/s]
    100%|██████████| 6/6 [00:00<00:00, 72.42it/s]




.. parsed-literal::

    {'r2': 0.7555024600133741,
     'Explained Variance': 0.7555024603158613,
     'Max Error': 1.3791490625619556,
     'Mean Absolute Error': 0.28752496358544677,
     'Mean Squared Error': 0.14721778747961853,
     'Root Mean Squared Error': 0.383689702076585}


