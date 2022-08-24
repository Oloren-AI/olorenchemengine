"""
Model comparisons is a set of visualizations that compare various models or
parameters thereof. This can be used for hyperparameter tuning, benchmarking
visualizations, or model selections.
"""

from olorenchemengine.base_class import *
from olorenchemengine.representations import *
from olorenchemengine.dataset import *
from olorenchemengine.uncertainty import *
from olorenchemengine.interpret import *
from olorenchemengine.internal import *
from olorenchemengine.manager import *
from olorenchemengine.visualizations import *

class VisualizationModelManager_Bar(BaseVisualization):
    """ Visualize the database of a model manager in a bar plot, with the model
    parameters and name displayed on hover

    Parameters:
        mm (ModelManager): The model manager to visualize
        top (int): The number of models to display
    """

    @log_arguments
    def __init__(self, mm: BaseModelManager, top = 5, log=True, **kwargs):
        self.df = mm.get_model_database()
        self.metrics = mm.metrics
        self.mm = mm
        self.top = top
        super().__init__(log=False)
        self.packages += ["plotly"]

    @property
    def JS_NAME(self) -> str:
        return "ModelManagerBar"

    def get_data(self, metric = None) -> dict:
        """
        Get the data for the visualization

        Parameters:
            metric (str): The metrics to use for the visualization

        Returns:
            dict: The data for the visualization
        """
        df2 = self.df.sort_values(by=metric, ascending=(True if self.mm.metric_direction[metric] == "lower" else False))
        df2 = df2.iloc[:self.top]
        d =  df2.to_dict("l")
        if not metric is None:
            assert metric in self.metrics, f"Designated metric, {metric}, not in model manager metrics: {self.metrics}"
            d["metric"] = metric
        else:
            d["metric"] = self.metrics[0]
        d["title"] = "Bar Plot of Best Models"
        return d

    def render_ipynb(self, metric=None, *args, print_html_js=False, **kwargs) -> str:
        """
        Render the visualization in an ipython notebook

        Parameters:
            metric (str): The metric to use for the visualization
            *args: Additional arguments to pass to the visualization
            print_html_js (bool): Whether to print the javascript code for the visualization
            **kwargs: Additional arguments to pass to the visualization

        Returns:
            str: The html code for the visualization
        """
        d = self.get_data(metric)
        return self.render_ipynb_from_dict(d, *args, print_html_js=print_html_js, **kwargs)

class ModelOverTime(CompoundScatterPlot):
    """ Visualizes the performance of a given model architecture over a length of time.

    This visualization will use a date split and for each date i, will train a model
    on the data on dates 1, ..., i-1, evaluating on date i. Note, any existing splits
    in the dataset will be ignored

    Parameters:
        dataset (BaseDataset): The dataset to use for the visualization
        model (BaseModel): The model to use for the visualization
    """

    @log_arguments
    def __init__(self, dataset: BaseDataset, model: BaseModel, *args, log=True,
        title="Model Performance over Time", yaxis_title="Error", xaxis_title="Date", **kwargs):
        self.dataset = dataset
        self.model = model
        self.title = title
        self.yaxis_title = yaxis_title
        self.xaxis_title = xaxis_title

        self.dataset.data = self.dataset.data.sort_values(by=[self.dataset.date_col])
        self.dates = self.dataset.data[self.dataset.date_col].unique()
        results = pd.DataFrame()
        for i in tqdm(range(len(self.dates)-1)):
            # try:
            train_date = self.dates[i]
            test_date = self.dates[i+1]
            train = self.dataset.data[self.dataset.data[self.dataset.date_col] <= train_date]
            train["split"] = "train"
            test = self.dataset.data[self.dataset.data[self.dataset.date_col] == test_date]
            test["split"] = "test"
            DATA = pd.concat([train, test])
            dataset = (oce.BaseDataset(data = DATA.to_csv(), structure_col="SMILES", property_col = "log AUC") + oce.CleanStructures())
            print(f"Train size is {len(train)} and test size is {len(test)}")
            self.model.fit(*dataset.train_dataset)

            test_set = dataset.data[dataset.data["split"] == "test"]
            test_set["pred"] = self.model.predict(test_set["SMILES"])

            results = pd.concat([results, test_set])

            if self.model.setting == "regression":
                results["Error (|Predicted - True|)"] = np.abs(results["pred"] - results[self.dataset.property_col])
                self.yaxis_title = "Error (|Predicted - True|)"
            elif self.model.setting == "classification":
                results["Error (p or 1-p)"] = np.abs(results[self.dataset.property_col]+(-2*results[self.dataset.property_col]+1)*results["pred"])
                self.yaxis_title = "Error (p or 1-p)"

        results["X"] = results[self.dataset.date_col]
        if self.model.setting == "regression":
            results["Y"] = results["Error (|Predicted - True|)"]
            self.xaxis_title = xaxis_title
        elif self.model.setting == "classification":
            results["Y"] = results["Error (p or 1-p)"]
            self.yaxis_title = yaxis_title
        results["SMILES"] = results[self.dataset.structure_col]
        super().__init__(*args,df = results, xaxis_type="date",
            title=title, xaxis_title=self.xaxis_title, yaxis_title=self.yaxis_title, log=False, **kwargs)

    def get_data(self):
        d = super().get_data()
        window = 10
        quantile = 0.80
        results = list(zip(pd.to_datetime(self.df["X"]).apply(lambda x: x.timestamp()), self.df["Y"]))
        scores, residuals = zip(*sorted(results))
        X = pd.Series(scores).rolling(window).mean()
        y = pd.Series(residuals).rolling(window).quantile(quantile)
        X = np.array(X[~np.isnan(X)])
        y = np.array(y[~np.isnan(y)])

        d["trace_update"] = {'x':self.df["X"].tolist(),
            'y': y.tolist(),
            'mode': 'markers',
            'type': 'scatter',
            'marker': {'color': 'red'}}
        return d