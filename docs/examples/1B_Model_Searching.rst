Model Searching
===============

In this example, we will use the permeability data in 1A, “ADME
Properties Evaluation in Drug Discovery: Prediction of Caco-2 Cell
Permeability Using a Combination of NSGA-II and Boosting”
https://pubs.acs.org/doi/10.1021/acs.jcim.5b00642, and instead of
defining a model ourselves, we will search over a list of top model
architectures and find the best one.

.. code:: ipython3

    # We'll first load in the data, same as in 1A
    
    import requests
    import os
    
    if not os.path.exists("caco2_data.xlsx"):
        r = requests.get("https://ndownloader.figstatic.com/files/4917022")
        open("caco2_data.xlsx" , 'wb').write(r.content)
    
    # Reading the data into a dataframe
    # Subsetting the data into molecule, split, and property
    # Converting property values to floats
    # Creating splits
    
    import pandas as pd
    import numpy as np
    
    df = pd.read_excel("caco2_data.xlsx")
    
    df["split"] = df["Dataset"].replace({"Tr": "train", "Te": "test"})
    df = df[["smi", "split", "logPapp"]].dropna()
    
    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False
    df = df[df["logPapp"].apply(isfloat)]
    
    df["logPapp"] = df["logPapp"].astype('float')
    
    # Now we use the dataframe to create a BaseDataset object.
    # We will generate it from the pd.DataFrame object.
    # We have defined our own split column, which will be used by the dataset object.
    
    import olorenchemengine as oce
    
    dataset = oce.BaseDataset(data = df.to_csv(), structure_col="smi", property_col="logPapp")

.. code:: ipython3

    # We'll now get our list of top model architectures.
    # Each of these models has certain situations where it outperforms the others,
    # so we test all of them to see which model is best for this specific situation.
    models = oce.TOP_MODELS_ADMET()
    
    # We'll also create a ModelManager object to keep track of our experiments
    mm = oce.ModelManager(dataset, metrics = ["Root Mean Squared Error"], file_path="mm_1B_results.oce")

.. code:: ipython3

    # This will now use our model manager to test our top models
    # and will take around 1-4 hours to run in total for this dataset,
    # though it will take more or less time depending on the machine.
    
    mm.run(models)

.. code:: ipython3

    # Get the list of models and sort by their RMSE performance metrics
    # We see that the best model now outperforms the published models.
    
    mm.get_model_database().sort_values(by="Root Mean Squared Error")




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
          <th>0</th>
          <td>ZWK_XGBoostModel 8t4Lbm1C</td>
          <td>{'BC_class_name': 'ZWK_XGBoostModel', 'args': ...</td>
          <td>583.755889</td>
          <td>0.306506</td>
        </tr>
        <tr>
          <th>4</th>
          <td>RFStacker ZObB1n2V</td>
          <td>{'BC_class_name': 'RFStacker', 'args': [[{'BC_...</td>
          <td>1370.884091</td>
          <td>0.326439</td>
        </tr>
        <tr>
          <th>2</th>
          <td>BaseBoosting sSOI0-2O</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>80.049373</td>
          <td>0.332818</td>
        </tr>
        <tr>
          <th>6</th>
          <td>RFStacker Dg3XrFow</td>
          <td>{'BC_class_name': 'RFStacker', 'args': [[{'BC_...</td>
          <td>1077.793208</td>
          <td>0.344461</td>
        </tr>
        <tr>
          <th>9</th>
          <td>ZWK_XGBoostModel u3zq9AAV</td>
          <td>{'BC_class_name': 'ZWK_XGBoostModel', 'args': ...</td>
          <td>583.068085</td>
          <td>0.350327</td>
        </tr>
        <tr>
          <th>1</th>
          <td>BaseBoosting GDDXgNxr</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>98.140028</td>
          <td>0.354108</td>
        </tr>
        <tr>
          <th>13</th>
          <td>RFStacker J-KhwR5S</td>
          <td>{'BC_class_name': 'RFStacker', 'args': [[{'BC_...</td>
          <td>2523.654553</td>
          <td>0.378332</td>
        </tr>
        <tr>
          <th>10</th>
          <td>BaseBoosting 1zpI0dIb</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>31.493627</td>
          <td>0.390516</td>
        </tr>
        <tr>
          <th>11</th>
          <td>BaseBoosting ADkCCrwJ</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>4.215762</td>
          <td>0.402029</td>
        </tr>
        <tr>
          <th>15</th>
          <td>RFStacker kHyqmLCI</td>
          <td>{'BC_class_name': 'RFStacker', 'args': [[{'BC_...</td>
          <td>4003.144378</td>
          <td>0.450928</td>
        </tr>
        <tr>
          <th>7</th>
          <td>ResampleAdaboost rw2YnX2a</td>
          <td>{'BC_class_name': 'ResampleAdaboost', 'args': ...</td>
          <td>150.905021</td>
          <td>0.452858</td>
        </tr>
        <tr>
          <th>14</th>
          <td>BaseBoosting Q-ko4Uuj</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>70.730023</td>
          <td>0.460666</td>
        </tr>
        <tr>
          <th>16</th>
          <td>ResampleAdaboost rw2YnX2a</td>
          <td>{'BC_class_name': 'ResampleAdaboost', 'args': ...</td>
          <td>432.079354</td>
          <td>0.474882</td>
        </tr>
        <tr>
          <th>12</th>
          <td>SPGNN TWy3l_kb</td>
          <td>{'BC_class_name': 'SPGNN', 'args': [], 'kwargs...</td>
          <td>9.815435</td>
          <td>0.487068</td>
        </tr>
        <tr>
          <th>5</th>
          <td>BaseBoosting Px-cadEt</td>
          <td>{'BC_class_name': 'BaseBoosting', 'args': [[{'...</td>
          <td>54.437475</td>
          <td>0.492785</td>
        </tr>
        <tr>
          <th>8</th>
          <td>KNN 20fz7vhA</td>
          <td>{'BC_class_name': 'KNN', 'args': [{'BC_class_n...</td>
          <td>1.745354</td>
          <td>0.526132</td>
        </tr>
        <tr>
          <th>17</th>
          <td>KNN 20fz7vhA</td>
          <td>{'BC_class_name': 'KNN', 'args': [{'BC_class_n...</td>
          <td>0.016418</td>
          <td>0.926369</td>
        </tr>
        <tr>
          <th>3</th>
          <td>SPGNN 8PvbRqPX</td>
          <td>{'BC_class_name': 'SPGNN', 'args': [], 'kwargs...</td>
          <td>10.181282</td>
          <td>2.647019</td>
        </tr>
      </tbody>
    </table>
    </div>


