PyG ChemProp Implementation
===========================

Background
----------

"ChemProp" is a simple but effective Graph Neural Network (GNN) for
Molecular Property Prediction, first used successfully in anti-biotic
discovery in 2019 by `Yang et
al. <https://doi.org/10.1021/acs.jcim.9b00237>`__ Like SPGNNs, it
provides an alternative way to represent molecules as 3D graphs with
nodes (atoms) and edges (bonds) instead of a 1D string representation
("SMILES"), which can provide added functionality.

Here, we will briefly overview the implementation of the original
ChemProp model oce uses which is adapted from Takigawa’s `Github
repository <https://github.com/itakigawa/pyg_chemprop>`__. We will
discuss its functionality with oce’s BaseModel structure and compare our
results to the original ChemProp’s results.

ChemProp Model Training
-----------------------

In this example, we will train a ChemProp model on the HIV dataset from
`Stanford OGB <https://ogb.stanford.edu/docs/graphprop/>`__.

.. code:: ipython3

    import io
    import sys
    import zipfile
    
    import pandas as pd
    import requests
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    
    import olorenchemengine as oce

Next, we create the dataset, check our model’s definition, and fit it to
training data. oce’s backend takes care of all of this in just a few
lines of code, from train-test splits to preprocessing to
SMILES-to-graph conversions.

.. code:: ipython3

    data_dir = "./data"
    data_url = "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip"
    r = requests.get(data_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(data_dir)
    df = pd.read_csv(f"{data_dir}/hiv/mapping/mol.csv.gz")
    
    X_train, X_test, y_train, y_test = train_test_split(df["smiles"], df["HIV_active"], test_size=0.2, random_state=42)
    
    model = oce.ChemPropModel()
    model.fit(X_train, y_train)

Now, we’ll evaluate the results of our training/fitting on the test set,
and see if we can achieve a better accuracy than the original example
dataset, which has an ROC_AUC score (auc) of 0.679 and an Accuracy score
(acc) of 0.968.

.. code:: ipython3

    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, (y_pred > 0.5).astype(int))
    print(f"test auc={auc:.6} acc={acc:.6}", file=sys.stderr)


.. parsed-literal::

    8226it [00:22, 364.44it/s]
    100%|██████████| 165/165 [00:10<00:00, 16.49it/s]
    test auc=0.704605 acc=0.963895

