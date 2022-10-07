from cmath import e
from dataclasses import dataclass
import dataclasses
from lib2to3.pgen2.literals import simple_escapes
import os
from re import escape
import urllib.parse

from IPython.display import IFrame
from pandas.core.indexes.accessors import NoNewAttributesMixin

from sklearn.model_selection import PredefinedSplit

from olorenchemengine.base_class import *
from olorenchemengine.representations import *
from olorenchemengine.dataset import *
from olorenchemengine.uncertainty import *
from olorenchemengine.interpret import *
from olorenchemengine.internal import *
from olorenchemengine.ensemble import *

from rdkit import Chem
from rdkit.Chem import Draw

def get_oas_compatible_visualizations():
    """ Returns a list of all available visualizations that are compatible with OAS,
    meaning that they can be created and specified in OAS. All visualizations
    can be shared via OAS."""

    # Get list of all visualizations
    all_visualizations = list(all_subclasses(BaseVisualization))

    # Remove all visualizations do not have a specified get_attributes
    return [
        vis.__name__
        for vis in all_visualizations
        if hasattr(vis, "get_attributes")
        and (not hasattr(vis.__bases__[0], "get_attributes") or vis.get_attributes != vis.__bases__[0].get_attributes)
    ]


class BaseVisualization(BaseClass):
    """Base class for all visualizations. Each visualization should implement a
    `get_data` method that returns a JSON-like dictionary of data to be
    used in the visualization. Each visualization should also have a corresponding
    JavaScript file (or one of a parent class) specified in `JS_NAME` that
    renders the visualization.

    Each visualization class should also implement a get_attributes method that
    that will return for OAS to use in the visualization page. Through this method,
    the visualization will specify what sections, attributes, and component types
    it needs for the user to specify.
    """

    # A list of packages that can be specified in self.packages to be loaded
    package_urls = {
        "d3": "https://d3js.org/d3.v4.js",
        "plotly": "https://cdn.plot.ly/plotly-2.14.0.min.js",
        "olorenrenderer": "https://unpkg.com/olorenrenderer@1.0.0-c/dist/oloren-renderer.min.js",
        "rdkit": "https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js",
    }

    @log_arguments
    def __init__(self, log=True, **kwargs):
        self.packages = []

    def get_data() -> dict:
        """Get data for visualization in JSON-like dictionary.
        """

        return {}

    @staticmethod
    def get_attributes() -> list:
        """Send the attributes for OAS visualization through an ordered list in which they will displayed in order.

        Returns:
            list: List of attributes for OAS to display."""
        raise NotImplementedError("get_attributes must be implemented for this visualization")

    @classmethod
    def from_attributes(cls, attributes: dict) -> "BaseVisualization":
        """Create visualization from attributes.

        Parameters:
            attributes (dict): Attributes to be used in visualization.
            cls (BaseVisualization): Class of visualization to be created.
        """
        try:
            vis = cls(**attributes)
            return vis
        except Exception as e:
            raise ValueError(f"Error creating visualization {cls.__name__} using {attributes}: {e}")

    @property
    def JS_NAME(self) -> str:
        """Name of JavaScript file for visualization, needs to be in the scripts
        folder.

        Returns:
            str: Name of JavaScript file.
        """
        return self.__class__.__name__

    def get_html(self, data_str: str, packages: str, **kwargs):
        """ Returns the HTML code for the visualization.

        Parameters:
            data_str (str): JSON string of data to be used in visualization.
            packages (str): HTML code which loades requisite JavaScript packages
                specified in `self.packages`."""

        # The template html file for the visualization, the data is passed via a
        # div with attribute 'data-visdata' and id 'basevis-entry'.
        html = f"""
        <!DOCTYPE html>
        <head>
        <meta charset="utf-8">
        </head>

        <html style="height: 100%;">

        <!-- Load packages -->
        {packages}
        <body style="background-color:#FFFFFF; height: 100%;">
        <!-- Create a div where the graph will take place -->
        <div id="basevis-entry" data-visdata="{data_str}" style="width: 100%; height: 100%; display:inline-block;"></div>
        </body>
        </meta>
        """

        return html

    def get_js(self, *args, **kwargs) -> str:
        """Render visualization to JavaScript string.

        Parameters:
            args (list): Arguments to be passed to visualization.
            kwargs (dict): Keyword arguments to be passed to visualization.

        Returns:
            str: JavaScript code for visualization.
        """

        with open(os.path.join(os.path.dirname(__file__), "scripts/", self.JS_NAME + ".js"), "r") as f:
            js = f.read()

        resize = (
            'window.addEventListener("resize", function(){{'
            'if (document.getElementById("{id}")) {{'
            '{plotly_root}.Plots.resize(document.getElementById("{id}"));'
            "}};}})"
        ).format(plotly_root="Plotly", id="basevis-entry")

        return f"<script>{js}</script>" + f"<script> {resize} </script>"

    def render(self, data: dict = None, print_html_js=False, **kwargs) -> str:
        """Render visualization to generate a data_url string.

        Parameters:
            data (dict): Data to be used in visualization. Optional, if not provided,
                data will be retrieved from `get_data` method.
            print_html_js (bool): Whether or not to print the JavaScript code for the
                visualization.
        """
        # Formats the data to be properly JSON
        if data is None:
            data_str = (
                json.dumps(self.get_data(**kwargs), separators=(',', ':'))
                .replace('"', "&quot;")
                .replace("None", "null")
                .replace("NaN", "null")
                .replace("True", "true")
                .replace("False", "false")
            )
        else:
            data_str = (
                json.dumps(data, separators=(',', ':'))
                .replace('"', "&quot;")
                .replace("None", "null")
                .replace("NaN", "null")
                .replace("True", "true")
                .replace("False", "false")
            )

        # Compiles package imports into HTML
        packages = "".join([f'<script src = "{self.package_urls[package]}"></script>' for package in self.packages])
        # Get base HTML code for visualization
        html = self.get_html(data_str, packages)

        # Get JavaScript code for visualization
        js = self.get_js()

        if print_html_js:
            print(html)
            print(js)

        # Return HTML with embeded JavaScript for visualization
        return html + js

    def save_html(self, path):
        with open(path, "w+") as f:
            f.write(self.render())

    def render_ipynb(self, data: dict = None, print_html_js=False, **kwargs) -> str:
        """Render visualization to IPython notebook IFrame.

        Parameters:
            data (dict): Data to be used in visualization. Optional, if not provided,
                data will be retrieved from `get_data` method.
            print_html_js (bool): Whether or not to print the JavaScript code for the
                visualization.
        """
        return IFrame(self.render_data_url(data, print_html_js, **kwargs), width=800, height=600)

    def render_data_url(self, data: dict = None, print_html_js=False, **kwargs) -> str:
        """Render visualization to data_url string.

        Parameters:
            data (dict): Data to be used in visualization. Optional, if not provided,
                data will be retrieved from `get_data` method.
            print_html_js (bool): Whether or not to print the JavaScript code for the
                visualization.
        """
        data_url = "data:text/html," + urllib.parse.quote(self.render(data, print_html_js, **kwargs), safe="")
        return data_url

    def upload_oas(self):
        oce.oas_connector.upload_vis(self)

    def get_link(self):
        """Get link to visualization."""
        pass

    def render_oas(self):
        """Render for OAS"""
        return self.render_data_url()

    def _save(self) -> dict:
        return {}

    def _load(self, d: dict):
        pass

class VisualizeError(BaseVisualization):
    """Visualizes property values and errors for a compound

    Parameters:
        dataset (Union[BaseDataset, list, pd.Series, np.ndarray]): reference data to be
            plotted in the visualization
        value (Union[int, float, np.ndarray]): single property value for the target compound
        error (Union[int, float, np.ndarray]): single error value for the target compound
        box (bool): whether or not to include a boxplot in the visualization
        points (bool): whether or not to include the reference points in the visualization
        xaxis_title (str): Title for x-axis. If not set, will try to use the property column name
            if dataset is a BaseDataset.
        yaxis_title (str): Title for y-axis.
    """
    @log_arguments
    def __init__(
        self,
        dataset: Union[BaseDataset, list, pd.Series, np.ndarray],
        value: Union[int, float, np.ndarray],
        error: Union[int, float, np.ndarray],
        ci=None,
        box=False,
        points=True,
        title="Error Plot",
        xaxis_title=None,
        yaxis_title=None,
        width=800,
        height=600,
        **kwargs
    ):
        assert error >= 0, "error must be nonnegative"

        if box:
            self.box = box
        if points:
            self.points = points

        if issubclass(type(dataset), BaseDataset):
            self.reference = dataset.data[dataset.property_col].tolist()
        elif isinstance(dataset, pd.Series):
            self.reference = dataset.tolist()
        elif isinstance(dataset, np.ndarray):
            self.reference = dataset.tolist()
        elif isinstance(dataset, list):
            self.reference = dataset

        if isinstance(value, np.ndarray):
            assert len(value) == 1, "value length must be 1"
            value = value[0]
        self.value = value
        if isinstance(error, np.ndarray):
            assert len(error) == 1, "error length must be 1"
            error = error[0]
        self.error = error

        if xaxis_title is None:
            if hasattr(dataset, "property_col"):
                xaxis_title = dataset.property_col
            else:
                xaxis_title = "Property Value"
        self.xaxis_title = xaxis_title

        if yaxis_title is None:
            yaxis_title = "Density"
        self.yaxis_title = yaxis_title

        if not ci is None:
            self.ci = int(ci * 100)

        self.title = title
        self.width=width
        self.height=height
        super().__init__(**kwargs)
        self.packages = ["plotly"]

    def get_data(self):
        d =  {
            "reference": self.reference,
            "value": self.value,
            "error": self.error,
            "title": self.title,
            "xaxis_title": self.xaxis_title,
            "yaxis_title": self.yaxis_title,
            "width": self.width,
            "height": self.height,
        }
        if hasattr(self, "box"):
            d["box"] = " "
        if hasattr(self, "points"):
            d["points"] = " "
        if hasattr(self, "ci"):
            d["ci"] = self.ci

        return d

class VisualizeCompounds(BaseVisualization):
    """Visualizes the compounds in a set.

    Parameters:
        dataset (Union[BaseDataset, list, pd.Series]): Compounds to be visualized.
            BaseDataset is preferred but list and pd.Series will be converted to
            lists of structures to be rendered.
        table_width (int, optional): Number of compounds in table row. Defaults to 2.
        table_height (int): Number of rows in table
        compound_width (int): Width of each compound image in the table
        compound_height = 250 (int): Height of each compound image in the table
        annotations = None (str): Name of columns in dataset to be used to annotated
            the table. Will only be considered if dataset is a BaseDataset.
        kekulize = True (bool): Whether or not to kekulize molecules for rendering.
            Default is True.
        box = True (bool):
        shuffle = True (bool): Whether or not to shuffle the compounds."""

    @log_arguments
    def __init__(self, dataset: Union[BaseDataset, list, pd.Series, str, Chem.Mol], table_width: int = 1, table_height: int = 5,
        compound_width: int = 500, compound_height: int = 500, annotations = None, kekulize = True,
        box=False, shuffle=False,log=True, **kwargs):
        self.dataset = dataset
        if issubclass(type(dataset), BaseDataset):
            self.compounds = dataset.data[dataset.structure_col].head(table_width * table_height).tolist()
        elif isinstance(dataset, pd.Series):
            self.compounds = dataset.tolist()
        elif isinstance(dataset, list):
            if isinstance(dataset[0], str):
                self.compounds = dataset
            if isinstance(dataset[0], Chem.Mol):
                self.compounds = [Chem.MolToSmiles(m) for m in dataset]
        elif isinstance(dataset, str):
            self.compounds = [dataset]
        elif issubclass(type(dataset), Chem.Mol):
            self.compounds = [Chem.MolToSmiles(dataset)]
        if shuffle:
            import random
            random.shuffle(self.compounds)

        if not issubclass(type(dataset), BaseDataset):
            self.annotations = None
        else:
            self.annotations = annotations
            if not isinstance(self.annotations, list):
                self.annotations = [self.annotations]

        self.table_width = table_width
        self.table_height = table_height
        self.compound_width = compound_width
        self.compound_height = compound_height
        self.annotations = annotations
        self.kekulize = kekulize
        self.box = box
        super().__init__(**kwargs)
        self.packages = ["olorenrenderer", "plotly"]

    def get_data(self):
        if self.kekulize:
            self.compounds = [Chem.MolToSmiles(Chem.MolFromSmiles(s, sanitize=False), kekuleSmiles = True) for s in self.compounds]
            d =  {"smiles": self.compounds,
                "table_width": self.table_width,
                "table_height": self.table_height,
                "compound_width": self.compound_width,
                "compound_height": self.compound_height,
                "box": self.box}

        if not self.annotations is None:
            d.update({"annotations": {col: self.dataset.data[col].tolist() for col in self.annotations}})

        return d

class VisualizeDatasetCompounds(VisualizeCompounds):
    """Alias for VisualizeCompounds"""
    pass

class ScatterPlot(BaseVisualization):
    """Scatter plot visualization.

    Parameters:
        df (pd.DataFrame): Dataframe to be used in visualization.
    """

    @log_arguments
    def __init__(self, df, log=True, **kwargs):
        self.df = df
        super().__init__(log=False)
        self.packages += ["plotly"]

    def get_data(self) -> dict:
        """Get data for visualization in JSON-like dictionary.

        Returns:
            dict: Data for visualization."""
        return self.df.to_dict("l")


class CompoundScatterPlot(BaseVisualization):
    """ Scatter plot visualization, where molecules are displayed on hover on
    an x-y plane.

    Parameters:
        df (pd.DataFrame): Dataframe to be used in visualization. Needs to have
            X, Y, and SMILES columns.
        title (str): Title of the visualization.
        xaxis_title (str, optional): Title for x-axis. Default is 'X axis'.
        yaxis_title (str, optional): Title for y-axis. Default is 'y axis'.
        x_col (str, optional): If specified, uses value as the column name for the
            x-axis value. Default is None.
        y_col (str, optional): If specified, uses value as the column name for the
            y-axis value. Default is None.
        smiles_col (str, optional): If specified, uses value as the column name for the
            molecule smiles. Default is None.
        kekulize (bool, optional): Whether or not to kekulize molecules for display.
            Default is True.
        color_col (str, optional): If specified, uses value as the column name for the
            color of the markers. Default is None.
        xaxis_type (str, optional): Type of x-axis. Default is 'linear', other
            options are 'log' or 'date'.
        yaxis_type (str, optional): Type of y-axis. Default is 'linear', other
            options are 'log' or 'date'.
        axesratio (float, optional): ratio of x-axis to y-axis lengths.
            Default is None, which allows it to be auto chosen by Plotly.
        xdomain (float, optional): domain for x-axis.
            Default is None, which allows it to be auto chosen by Plotly.
        ydomain (float, optional): domain for y-axis.
            Default is None, which allows it to be auto chosen by Plotly.
        xdtick (float, optional): tick interval for the x-axis.
            Default is None, which allows it to be auto chosen by Plotly.
        ydtick (float, optional): tick interval for the y-axis.
            Default is None, which allows it to be auto chosen by Plotly.
        xrange (float, optional): tick interval for the x-axis.
            Default is None, which allows it to be auto chosen by Plotly.
        ydtick (float, optional): tick interval for the y-axis.
            Default is None, which allows it to be auto chosen by Plotly.
        xrange (float, optional): tick interval for the x-axis.
            Default is None, which allows it to be auto chosen by Plotly.
        opacity (float, optional): opacity of the markers. Useful for seeing the
            distribution of dense data. Default is 1.0
        width (int, optional): width of the plot. Default is 600.
        height (int, optional): height of the plot. Default is 600.
    """

    @log_arguments
    def __init__(
        self,
        df,
        title: str = "Compound Scatter Plot",
        xaxis_title: str = "X axis",
        yaxis_title: str = "y axis",
        x_col: str =    None,
        y_col: str =    None,
        smiles_col: str = None,
        kekulize: bool = True,
        color_col: str = None,
        colorscale: str = None,
        xaxis_type: str = "linear",
        yaxis_type: str = "linear",
        axesratio: float = None,
        xdomain: float = None,
        ydomain: float = None,
        xdtick: float = None,
        ydtick: float = None,
        xrange: float = None,
        yrange: float = None,
        opacity: float = 1,
        width: int = 800,
        height: int = 600,
        log: bool = True,
        **kwargs,
    ):
        # needs to have columns "X", "Y", "SMILES", or have them specified by
        # x_col, y_col, and smiles_col
        self.df = df

        # maps columns appropriately
        if x_col is not None:
            self.df["X"] = self.df[x_col]
            if x_col != "X":
                self.df = self.df.drop(x_col, axis=1)
        if y_col is not None:
            self.df["Y"] = self.df[y_col]
            if y_col != "Y":
                self.df = self.df.drop(y_col, axis=1)
        if smiles_col is not None:
            self.df["SMILES"] = self.df[smiles_col]
            if smiles_col != "SMILES":
                self.df = self.df.drop(smiles_col, axis=1)

        if color_col == "property_col":
            self.color_col = self.dataset.property_col
        else:
            self.color_col = color_col
        if color_col is not None:
            self.df["color"] = self.df[color_col]
            if color_col != "color":
                self.df = self.df.drop(color_col, axis=1)
        self.colorscale = colorscale

        # Sets up axes titling using column names as defaults if available
        if xaxis_title is None and x_col is not None:
            self.xaxis_title = x_col
        else:
            self.xaxis_title = xaxis_title
        if yaxis_title is None and y_col is not None:
            self.yaxis_title = y_col
        else:
            self.yaxis_title = yaxis_title

        # Saves aesthetic variables
        self.kekulize = kekulize
        self.xaxis_type = xaxis_type
        self.yaxis_type = yaxis_type
        self.axesratio = axesratio
        self.xdomain = xdomain
        self.ydomain = ydomain
        self.xrange = xrange
        self.yrange = yrange
        self.xdtick = xdtick
        self.ydtick = ydtick
        self.width = width
        self.height = height
        self.opacity = opacity
        self.title = title

        # Create DataFrame
        self.df = self.df.dropna(subset=["X", "Y"])
        cols = ["SMILES", "X", "Y"]
        if "color" in self.df.columns:
            cols += ["color"]
        self.df = self.df[cols]

        super().__init__(log=False, **kwargs)

        # Add packages to import for JavaScript
        self.packages += ["plotly", "olorenrenderer"]

    @property
    def JS_NAME(self) -> str:
        return "CompoundScatterPlot"

    def get_data(self, include_data=True) -> dict:
        """Get data for visualization in JSON-like dictionary.

        Parameters:
            include_data (bool): Whether or not to include data in the visualization.
                If False, only the attributes will be returned.
        Returns:
            dict: Data for visualization."""

        if include_data:
            d = self.df.to_dict("l")
            d["SMILES"] = self.df["SMILES"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), kekuleSmiles=self.kekulize)).tolist()
        else:
            d = dict()
        d["title"] = self.title
        d["xaxis_title"] = self.xaxis_title
        d["yaxis_title"] = self.yaxis_title
        d["hover_size"] = 0.4
        d["xaxis_type"] = self.xaxis_type
        d["yaxis_type"] = self.yaxis_type
        d["width"] = self.width
        d["height"] = self.height
        d["opacity"] = self.opacity
        if not self.colorscale is None:
            d["colorscale"] = self.colorscale

        if self.axesratio is not None:
            d["axesratio"] = self.axesratio
        if self.xdomain is not None:
            d["xdomain"] = self.xdomain
        if self.ydomain is not None:
            d["ydomain"] = self.ydomain
        if self.xrange is not None:
            d["xrange"] = self.xrange
        if self.yrange is not None:
            d["yrange"] = self.yrange

        if self.xdtick is not None:
            d["xdtick"] = self.xdtick
        if self.ydtick is not None:
            d["ydtick"] = self.ydtick

        return d

    @staticmethod
    def get_attributes():
        """List of Customizable attributes for Visualization to display.

        Returns:
            list: List of attributes for OAS to display.
        """
        return [{"Labels": [["title", "inputString"], ["xaxis_title", "inputString"], ["yaxis_title", "inputString"]]}]


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class ChemicalSpacePlot(CompoundScatterPlot):
    """ Visualize chemical space by calculating a vector representation and
    performing dimensionality reduction to 2 dimensions.

    Parameters:
        dataset (BaseDataset, pd.Seriess, list): BaseDataset to be used in visualization. Alternatively
            can be a list or pd.Series where then this object will be treated as a list
            of structures.
        rep (BaseCompoundVecRepresentation): Representation to use for dimensionality reduction.
        dim_reduction (str, optional): Dimensionality reduction method to use. Default is
            'tsne' other options are 'pca'.
        color (str, optional): Column name to use for color. If it is 'property_col' the
            visualization will use the property_col of the dataset to color the
            markers. Default is None, meaning no variable coloring for the
            chemical space plot
        colorscale (str, optional): Color scale to use for coloring the chemical space plot.
            Other options can be found
        """

    @log_arguments
    def __init__(self, dataset: Union[BaseDataset, list, pd.Series, pd.DataFrame], rep: BaseCompoundVecRepresentation,
            *args, dim_reduction="tsne",
            smiles_col=None, title ="Chemical Space Plot", log=True, **kwargs):

        # Sets visualization instance variables
        if issubclass(type(dataset), BaseDataset):
            self.structures = dataset.data[dataset.structure_col]
        elif isinstance(dataset, pd.DataFrame):
            assert smiles_col is not None, "smiles_col must be defined if `dataset` parameter is pd.DataFrame"
            self.structures = dataset[smiles_col]
        else:
            self.structures = dataset
        self.rep = rep
        self.dim_reduction = dim_reduction

        # Converts the molecules in the dataset to the desired representation
        chem_rep_list = self.rep.convert(self.structures)

        # Does dimensionality reduction on the given representation to get the
        # 2D coordinates of the chemical space plot
        if self.dim_reduction == "tsne":
            df = self.tsne_df(chem_rep_list)
        elif self.dim_reduction == "pca":
            df = self.pca_df(chem_rep_list)

        # Sets the dataframe up to be used by the parent class CompoundScatterPlot
        if issubclass(type(dataset), BaseDataset):
            self.df = dataset.data
        elif issubclass(type(dataset), pd.DataFrame):
            self.df = dataset
        else:
            self.df = pd.DataFrame()

        self.df["X"] = df["Component 1"]
        self.df["Y"] = df["Component 2"]
        self.df["SMILES"] = self.structures

        self.title = title

        super().__init__(self.df, *args, title=self.title,
            xaxis_title="Component 1", yaxis_title="Component 2", smiles_col = smiles_col,
            log=False, **kwargs)

    def tsne_df(self, chem_rep_list):
        """
        Takes in a list containing the chemical representation of a collection of
        molecules and returns a dataframe containing the t-SNE reduction to 2 components.

        Parameters:
            chem_rep_list (list): List of chemical representations of molecules.
        Returns:
            pandas.DataFrame: Dataframe containing the t-SNE reduction to 2 components.
        """
        pca_50 = PCA(n_components=min(50, len(chem_rep_list[0]), len(chem_rep_list)))
        pca_result_50 = pca_50.fit_transform(chem_rep_list)
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(pca_result_50)
        return pd.DataFrame(tsne_results, columns=["Component 1", "Component 2"])

    def pca_df(self, chem_rep_list):
        """
        Takes in a list containing the chemical representation of a collection of
        molecules and returns a dataframe containing the PCA reduction to 2 components.

        Parameters:
            chem_rep_list (list): List of chemical representations of molecules.
        Returns:
            pandas.DataFrame: Dataframe containing the PCA reduction to 2 components.
        """
        pca = PCA(n_components=2)
        pca_arr = pca.fit_transform(chem_rep_list)
        return pd.DataFrame(pca_arr, columns=["Component 1", "Component 2"])

    def get_data(self, color_col: str = None, size_col: str = None, SMILES: str = None) -> dict:
        d = super().get_data(include_data=True)

        self.color_col = color_col
        if not self.color_col is None:
            assert self.color_col in self.df.columns, f"specified color column, {self.color_col}, not in columns"
            d["color"] = self.df[self.color_col].tolist()

        self.size_col = size_col
        if not self.size_col is None:
            assert self.size_col in self.df.columns, f"specified size column, {self.size_col}, not in columns"
            d["size"] = self.df[self.size_col].tolist()

        return d

class VisualizeMoleculePerturbations(ChemicalSpacePlot):
    """ Visualize perturbations of a single molecule given from a PerturbationEngine
    in a ChemicalSpacePlot.

    Parameters:
        smiles (str): SMILES of molecule to perturb.
        perturbation_engine (PerturbationEngine): Perturbation engine, which has
            the underlying algorithm for perturbing molecules. Default is `SwapMutations(radius = 0)`
        rep (BaseVecRepresentation): Molecular vector representation to use for
            dimensionality reduction"""

    @log_arguments
    def __init__(self, smiles: str,
            perturbation_engine: PerturbationEngine = None,
            rep: BaseVecRepresentation = None,
            idx: int = None,
            n: int = None):
        self.smiles = smiles
        if perturbation_engine is None:
            self.perturbation_engine = SwapMutations(radius = 0)
        else:
            self.perturbation_engine = perturbation_engine
        if rep is None:
            self.rep = DescriptastorusDescriptor("morgan3counts")
        else:
            self.rep = rep

        if n is None:
            self.n = 100
        else:
            self.n = n

        from rdkit import Chem
        from rdkit.Chem import AllChem

        df = pd.DataFrame()
        if idx is None:
            df["SMILES"] = self.perturbation_engine.get_compound_list(smiles) + [smiles]
        else:
            df["SMILES"] = self.perturbation_engine.get_compound_list(smiles, idx = idx)+ [smiles]
        df["mols"] = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
        df = df.dropna(subset = ["mols"])

        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048, useChirality=False) for m in df["mols"]]

        df["sim"] = DataStructs.BulkTanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(self.smiles), 2, nBits=2048, useChirality=False) , fps)

        super().__init__(df,
                self.rep,
                title = "Chemical Space Plot of Molecular Perturbations<br><sub>Points colored by tanimoto similarity to the reference compund</sub>",
                color_col = "sim",
                smiles_col = "SMILES",
                colorscale = "YlOrRd")

class VisualizeDatasetSplit(ChemicalSpacePlot):
    """Visualize a dataset by seeing where train/test compounds are in a dimensionality
    reduced space by coloring the compounds by whether or not they are in
    train/test.

    Parameters:
        dataset (BaseDataset): Dataset to visualize.
        rep (BaseStructVecRepresentation): Representation to use for dimensionality reduction.
        model (BaseModel, optional): Model to use for prediction in order to color
            the dataset split by residual value.
        colorscale (str): Color scale to use for coloring the compounds.
        res_lim (int): Capping the visualized residual size to the specified value.
        opacity (float): Opacity of the markers.
    """

    @log_arguments
    def __init__(
        self,
        dataset: BaseDataset,
        rep: BaseCompoundVecRepresentation,
        model: BaseModel = None,
        res_lim: float = None,
        opacity: float = 0.4,
        *args,
        **kwargs,
    ):
        # Set visualization instance variables
        self.dataset = dataset
        self.model = model
        self.res_lim = res_lim
        self.opacity = opacity
        if not type(dataset) is BaseDataset:
            raise TypeError(
                "dataset must be a BaseDataset. Consider using df.to_csv() in the"
                + " oce.BaseDataset class to convert a pandas.DataFrame of data to a BaseDataset."
            )

        # Either visualize residual magnitudes or not
        if model is None:
            super().__init__(
                dataset,
                rep,
                *args,
                title="Dataset Visualization<br><sub>Red outline is train; Blue outline is test</sub>",
                dim_reduction="tsne",
                opacity = self.opacity,
                log=False,
                **kwargs,
            )
        else:
            super().__init__(
                dataset,
                rep,
                *args,
                title="Model Residual Visualization<br><sub>Red outline is train; Blue outline is test; Marker size is residual magnitude</sub>",
                dim_reduction="tsne",
                opacity = self.opacity,
                log=False,
                **kwargs,
            )

    def get_data(self) -> dict:
        d = super().get_data(SMILES=self.dataset.structure_col)

        if self.model is None:
            d["color"] = ["red" if x == "train" else "blue" for x in self.dataset.entire_dataset_split]
        else:
            d["outline"] = ["red" if x == "train" else "blue" for x in self.dataset.entire_dataset_split]

            residuals = (self.dataset.entire_dataset[1] - self.model.predict(self.dataset.entire_dataset[0])) ** 2
            if self.res_lim is None:
                self.res_lim = np.percentile(residuals, 90)
            residuals = residuals / self.res_lim

            d["Z"] = residuals.tolist()
            d["color"] = residuals.tolist()
            d["opacity"] = self.opacity
            d["size"] = (residuals * 18).tolist()

        return d

    @staticmethod
    def get_attributes():
        return [
            {"Dataset Selection": [["dataset", "datasetSelector"],]},
            {"Configure Chemical Space": [["rep", "selector", ["oce", "rdkit2d-normed", "morgan"]],]},
        ]

    @classmethod
    def from_attributes(cls, attributes: dict) -> "BaseVisualization":
        """Replace the dataset with a dataframe with a SMILES column and construct the rep object"""

        oas_reps = {
            "oce": oce.OlorenCheckpoint("default"),
            "rdkit2d-normed": oce.DescriptastorusDescriptor("rdkit2dnormalized"),
            "morgan": oce.DescriptastorusDescriptor("morgan3counts"),
        }

        attributes["rep"] = oas_reps[attributes["rep"]]

        return super(ChemicalSpacePlot, cls).from_attributes(attributes)

from rdkit.Chem import Draw

class MorganContributions(BaseVisualization):
    """Visualize a morgan fingerprint bit contributions of a molecule.

    Each bit of a morgan fingerprint maps to a specific substructure so this visualization
    takes in a given compound and a dataset to calibrate on, and visualizes which
    substructure in the compound most contribute to the predicted property value
    of the compound.

    This is done by looking at the bits which are activated for the given compound,
    and then systematically switching those bits off and on examining how each
    bit effects the output of the model.

    Parameters:
        smiles (str): smiles string of the molecule to visualize.
        dataset (BaseDataset): Dataset to calibrate the MorganContributions model on.
        compound_id (str, optional). Optional identifier for the compound.
        radius (int, optional): Parameter for the Morgan representation.
            Default is 2.
        nbits(int, optional): Parameter for the Morgan representation.
            Default is 1024.
        """

    @log_arguments
    def __init__(
            self,
            smiles: str,
            dataset: BaseDataset,
            compound_id: str = "The compound",
            radius=2,
            nbits=1024,
            *args,
            **kwargs
        ):
        self.radius = radius
        self.nbits = nbits
        self.dataset = dataset
        self.property = dataset.property_col

        self.smiles = smiles
        self.compound_id = compound_id,
        self.model = self._train_model()
        self.original_prediction, self.predictions = self._make_predictions(self.smiles)
        self.args = args
        self.kwargs = kwargs
        self.packages = ["plotly", "rdkit", "olorenrenderer"]

    def _train_model(self):
        """ Train random forest model based on the morgan vec representation"""
        model = oce.RandomForestModel(MorganVecRepresentation(scale=None, collinear_thresh=1.01), n_estimators=1000)
        model.fit(*self.dataset.entire_dataset)
        return model

    def _make_predictions(self, smiles):
        """ Make predictions on the dataset with systematic experiments to test
        the effect of bit flips."""
        morgan_rep = self._to_morgan(smiles)
        original_prediction = self.model.predict([morgan_rep], skip_preprocess=True)

        morgan_variants = self._switch_bits()
        predictions = self.model.predict(morgan_variants, skip_preprocess=True)

        return original_prediction, predictions


    def _to_morgan(self, smiles):
        """ Return morgan representation of given compound (represented by
        smiles)."""
        return MorganVecRepresentation._convert(self, smiles)

    def _get_on_bits(self, smiles):
        """
        Returns a list of bits that are on in the Morgan vector.
        """
        fp = self._to_morgan(smiles)
        idxs = np.where(fp == 1)[0]
        return idxs

    def _switch_bits(self):
        """
        Returns a list of lists of on bits, where each list has a new bit
        switched off.

        for use as the argument to _predict.
        """
        fp = self._to_morgan(self.smiles)
        idxs = self._get_on_bits(self.smiles)
        fp_repeating = np.repeat(fp[np.newaxis, :], idxs.shape[0], 0)
        fp_repeating[np.arange(idxs.shape[0]), idxs] = 0
        return fp_repeating

    def _top_3_predictions(self):
        """
        Returns the lists of the top 3 predictions associated with the
        3 most beneficial/detrimental bits.

        tuple: (bottom_3_bits, top_3_bits)
        """
        top_3 = np.argsort(self.predictions)[-3:]
        bottom_3 = np.argsort(self.predictions)[:3]

        idxs = self._get_on_bits(self.smiles)

        bottom_3_bits = idxs[bottom_3]
        top_3_bits = idxs[top_3]

        return bottom_3_bits, top_3_bits

    def _top_3_min_max(self):
        self.predictions = self._calculate_effect()
        max_idxs = np.argsort(self.predictions)[-3:]
        min_idxs = np.argsort(self.predictions)[:3]

        bottom_3 = self.predictions[min_idxs]
        top_3 = self.predictions[max_idxs]
        return bottom_3, top_3

    def _calculate_effect(self):
        """
        Calculates the effect of switching a bit on or off.
        """
        effect = self.original_prediction - self.predictions
        return effect

    def visualize_morganfp(self):
        """
        Takes a list of predictions and returns the list of images
        of the minimum and maximum substructures of switching a bit on or off.

        Returns:
            Tuple of two lists of images. (minimum effect, maximum effect)
        """

        mol = Chem.MolFromSmiles(self.smiles)
        info = MorganVecRepresentation.info(self, self.smiles)

        mfp_svgs_min = []
        mfp_svgs_max = []

        self.predictions = self._calculate_effect()
        bottom_3_bits, top_3_bits = self._top_3_predictions()

        mfp_svgs_min = {str(i): Draw.DrawMorganBit(mol, i, info, useSVG=True) for i in bottom_3_bits}
        mfp_svgs_max = {str(i): Draw.DrawMorganBit(mol, i, info, useSVG=True) for i in top_3_bits}

        return mfp_svgs_min, mfp_svgs_max

    def _filter(self, min_val, max_val):

        min_delete = np.delete(min_val, np.where(min_val > 0))
        min_filtered_val = np.where(min_delete != 0)[0]

        max_delete = np.delete(max_val, np.where(max_val < 0))
        max_filtered_val = np.where(max_delete != 0)[0]

        min_filtered = min_delete[min_filtered_val]
        max_filtered = max_delete[max_filtered_val]
        return min_filtered, max_filtered

    def get_data(self):
        """
        Returns a dictionary of the data to be visualized.
        """
        bottom_3_bits, top_3_bits = self._top_3_predictions()

        bottom_3_min, bottom_3_max = self._top_3_min_max()

        bottom_3_min, bottom_3_max = self._filter(bottom_3_min, bottom_3_max)
        print(bottom_3_min, bottom_3_max)
        bottom_bits, top_bits = bottom_3_bits[0:len(bottom_3_min)], top_3_bits[0:len(bottom_3_max)]
        substructures = self.get_substructures(bottom_bits, top_bits)
        print(substructures)

        return {
            "original_molecule": self.smiles,
            "original_prediction": self.original_prediction[0],
            "compound_id": self.compound_id,
            "maxbits": top_bits.tolist(),
            "minbits": bottom_bits.tolist(),
            "maxeffect": bottom_3_max.tolist(),
            "mineffect": bottom_3_min.tolist(),
            "substructures": substructures,
            "property": self.property,
        }

    def get_html(self, data_str: str, packages):
        html = f"""
        <!DOCTYPE html>
        <head>
        <meta charset="utf-8">
        </head>

        <html style="height: 100%;">

        <!-- Load packages -->
        {packages}

        <body id="start" style="background-color:#FFFFFF; height: 100%; width: 100%; display: flex; flex-direction: row; ">
        <!-- Create a div where the graph will take place -->
        <div id="basevis-entry" data-visdata="{data_str}" style="width:65%; height:100%"></div>
        <div id="basevis-molecule" style="width: 35%; height: 100%;"></div>
        </body>
        </meta>
        <script>
        Plotly.newPlot('basevis-entry', plot_data, layout);</script><script> window.addEventListener("resize", function()
        {{if (document.getElementById("basevis-entry")) {{Plotly.Plots.resize(document.getElementById("basevis-entry"));}};}}) </script>
        """
        return html

    def get_substructures(self, bottom_3_bits, top_3_bits):
        """
        Returns a list of the substructures of the molecule.
        """
        mol = Chem.MolFromSmiles(self.smiles)
        info = MorganVecRepresentation.info(self, self.smiles)

        mol = Chem.MolFromSmiles(self.smiles)
        sub_structures = {}
        for i in bottom_3_bits:
            atom_tuples = info[i]
            atomID = atom_tuples[0][0]
            radius = atom_tuples[0][1]
            sub_structures[str(i)] = str(self.getSubstructSmi(mol, int(atomID), int(radius))[0])

        for i in top_3_bits:
            atom_tuples = info[i]
            atomID = atom_tuples[0][0]
            radius = atom_tuples[0][1]
            sub_structures[str(i)] = str(self.getSubstructSmi(mol, int(atomID), int(radius))[0])
        return sub_structures

    # Functions for providing detailed descriptions of MFP bits from Nadine Schneider
    #  It's probably better to do this using the atomSymbols argument but this does work.
    #
    # The following functions are used to generate the SMILES of the MFP bits as provided by Greg Landrum.
    # The link to the source: http://rdkit.blogspot.com/2016/02/morgan-fingerprint-bit-statistics.html

    def _includeRingMembership(self, s, n):
        r=';R]'
        d="]"
        return r.join([d.join(s.split(d)[:n]),d.join(s.split(d)[n:])])

    def _includeDegree(self, s, n, d):
        r=';D'+str(d)+']'
        d="]"
        return r.join([d.join(s.split(d)[:n]),d.join(s.split(d)[n:])])

    def _writePropsToSmiles(self, mol,smi,order):
        #finalsmi = copy.deepcopy(smi)
        finalsmi = smi
        for i,a in enumerate(order):
            atom = mol.GetAtomWithIdx(a)
            if atom.IsInRing():
                finalsmi = self._includeRingMembership(finalsmi,i+1)
            finalsmi = self._includeDegree(finalsmi, i+1, atom.GetDegree())
        return finalsmi

    def getSubstructSmi(self, mol,atomID,radius):
        if radius>0:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomID)
            atomsToUse=[]
            for b in env:
                atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
                atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
            atomsToUse = list(set(atomsToUse))
        else:
            atomsToUse = [atomID]
            env=None

        smi = Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,allHsExplicit=True, allBondsExplicit=True, rootedAtAtom=atomID)
        order = eval(mol.GetProp("_smilesAtomOutputOrder"))
        smi2 = self._writePropsToSmiles(mol,smi,order)
        return smi,smi2

    @staticmethod
    def get_attributes():
        return [
            {
                "Select a dataset to visualize": [
                    ["dataset", "datasetSelector"],
                ]
            }
        ]

    @property
    def JS_NAME(self) -> str:
        """Name of JavaScript file for visualization, needs to be in the scripts
        folder.

        Returns:
            str: Name of JavaScript file.
        """
        return self.__class__.__name__

class VisualizeADAN(CompoundScatterPlot):
    """Visualize a model trained on a dataset, by seeing predicted vs true
    colored by ADAN criteria.

    Parameters:
        dataset (Dataset): Dataset to visualize.
        model (BaseModel): Model to use for prediction.
        rep (BaseStructVecRepresentation): Representation to use for dimensionality reduction.
        threshold (float): Threshold for ADAN visualization between 0.0 and 1.0.
            Roughly a higher value corresponds to a more strict threshold resulting
            in more compounds being marked as out-of-distribution for the ADAN
            criteria."""

    @log_arguments
    def __init__(
        self,
        dataset: BaseDataset,
        model: BaseModel,
        *args,
        rep: BaseCompoundVecRepresentation = None,
        dim_reduction: str = "pls",
        explvar: float = 0.8,
        threshold: float = 0.95,
        **kwargs,
    ):
        if not type(dataset) is BaseDataset:
            raise TypeError(
                "dataset must be a BaseDataset. If you're using a Pandas Dataframe consider using df.to_csv() in the"
                + " oce.BaseDataset class to convert a pandas.DataFrame of data to a BaseDataset."
            )
        self.dataset = dataset
        self.model = model
        self.ADAN = oce.ADAN(None)
        self.ADAN.build(
            self.model, 
            X=dataset.train_dataset[0], 
            y=np.array(dataset.train_dataset[1]), 
            rep=rep, 
            dim_reduction=dim_reduction,
            explvar=explvar,
            threshold=threshold
        )
        self.ADAN.calculate_full(dataset.test_dataset[0])

        self.df_ = pd.DataFrame(
            {
                "X": self.dataset.test_dataset[1],
                "Y": self.model.predict(self.dataset.test_dataset[0]),
                "SMILES": self.dataset.test_dataset[0][self.dataset.structure_col],
            }
        )

        super().__init__(
            self.df_,
            *args,
            title="ADAN Visualization",
            xaxis_title="Experimental value",
            yaxis_title="Predicted value",
            log=False,
            **kwargs,
        )

    def get_data(self, criterion="B") -> dict:
        self.title = f"ADAN Visualization<br><sub>Points colored by ADAN criterion {criterion}</sub>"

        self.df = self.df_.copy().reset_index(drop=True)
        self.df["color"] = self.adan.results[criterion]
        self.df["Z"] = (self.df["X"] - self.df["Y"]) ** 2
        if "raw" in criterion:
            from sklearn.metrics import r2_score

            print(
                f"ADAN Criterion {criterion} correlation with squared error is {r2_score(self.adan.results[criterion], self.df['Z'])}"
            )
            import matplotlib.pyplot as plt
            plt.scatter(self.adan.results[criterion], self.df["Z"])
            plt.show()
        else:
            print(f"ADAN Criterion {criterion}; RMSE by class")
            for i in sorted(self.adan.results[criterion].unique()):
                print(f"Class {i}: RMSE {np.sqrt(np.mean(self.df[self.adan.results[criterion] == i]['Z']))}")
        return super().get_data()

    @staticmethod
    def get_attributes():
        return [
            {
                "Select a model and dataset for ADAN visualization": [
                    ["model", "modelSelector"],
                    ["dataset", "datasetSelector"],
                ]
            },
            {"ADAN Parameters": [["threshold", "inputNumberRange", [0, 100]]]},
        ]


class VisualizeModelSim(CompoundScatterPlot):
    """ Visualize a model's predicted vs true plot on given dataset, where each
    point is colored by a compounds similarity to the train set

    Parameters:
        dataset (BaseDataset): Dataset to visualize model performance on, the
            visualization will only select the set specified by eval_set.
        model (BaseModel): Model to visualize.
        eval_set (str): Subset of dataset to visualize, either 'train', 'test',
            or 'valid'."""

    @log_arguments
    def __init__(self, dataset: BaseDataset, model: BaseModel, eval_set = "test", *args, log=True, **kwargs):
        if not type(dataset) is BaseDataset:
            raise TypeError(
                "dataset must be a BaseDataset. If you're using a Pandas Dataframe consider using df.to_csv() in the"
                + " oce.BaseDataset class to convert a pandas.DataFrame of data to a BaseDataset."
            )
        self.dataset = dataset
        self.model = model

        from rdkit.Chem import AllChem
        from rdkit import DataStructs

        fingerprinter = lambda s: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(s), 4)
        train_fps = [fingerprinter(s) for s in self.dataset.train_dataset[0][self.dataset.structure_col]]

        if eval_set == "test":
            self.eval_set = self.dataset.test_dataset
        elif eval_set == "train":
            self.eval_set = self.dataset.train_dataset
        elif eval_set == "valid":
            self.eval_set = self.dataset.valid_dataset

        eval_fps = [fingerprinter(s) for s in self.eval_set[0][self.dataset.structure_col]]
        self.sim = np.array([max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in eval_fps])

        self.df = pd.DataFrame(
            {
                "X": self.eval_set[1],
                "Y": self.model.predict(self.eval_set[0]),
                "SMILES": self.eval_set[0][self.dataset.structure_col],
                "color": self.sim,
            }
        )

        super().__init__(
            self.df,
            *args,
            title="Predicted vs True Plot<br><sub>Points colored by similarity to train set</sub>",
            xaxis_title="Experimental value",
            yaxis_title="Predicted value",
            log=False,
            **kwargs,
        )

    def get_data(self) -> dict:
        d = super().get_data()
        d["color"] = self.sim.tolist()
        d["trace_update"] = {'x':[0.9*min(self.df["X"].min(), self.df["Y"].min()), 1.1*max(self.df["X"].max(), self.df["Y"].max())],
            'y':[0.9*min(self.df["X"].min(), self.df["Y"].min()), 1.1*max(self.df["X"].max(), self.df["Y"].max())],
            'mode': 'line',
            'type': 'scatter'}
        return d

    @staticmethod
    def get_attributes():
        return [{"Select a Model/Dataset": [["model", "modelSelector"], ["dataset", "datasetSelector"],]}]


class VisualizeCounterfactual(CompoundScatterPlot):
    """Visualize a model's counterfactuals in the chemical space around a given compound,
    plotted in Tanimoto similarity space

    Parameters:
        smiles (str): Given compound to visualize counterfactuals for.
        model (BaseModel): Model to evaluate counterfactuals on.
        perturbation_engine (PerturbationEngine): Engine to generate counterfactuals
            with. Defauult is SwapMutations with radius 1.
        delta (int, float, or tuple, option): Margin that defines counterfactuals
            for regression models. Default is (-1,1).
        n (int, optional): Number of counterfactuals to generate.
            Default is 40."""

    @log_arguments
    def __init__(
        self,
        smiles: str,
        model: BaseModel,
        perturbation_engine: PerturbationEngine = None,
        delta: Union[int, float, Tuple] = (-1, 1),
        n: int = 40,
        pca: bool = False,
        **kwargs,
    ):
        if perturbation_engine is None:
            perturbation_engine = SwapMutations(radius=1)

        self.smiles = smiles
        self.model = model
        self.perturbation_engine = perturbation_engine
        if type(delta) in (float, int):
            delta = (-delta, delta)
        self.delta = delta
        self.cf_engine = CounterfactualEngine(model=self.model, perturbation_engine=self.perturbation_engine)
        self.cf_engine.generate_samples(self.smiles)

        self.cf_engine.generate_cfs(delta=self.delta, n=n // 2)
        self.base = self.cf_engine.samples[0]["Value"]

        if self.model.setting == "classification":
            self.factuals = self.cf_engine._select_cfs(lambda s: s["Value"] == self.base, n // 2)
        else:
            self.factuals = self.cf_engine._select_cfs(
                lambda s: s["Value"] - self.delta[0] > self.base > s["Value"] - self.delta[1], n // 2
            )

        samples = self.cf_engine.cfs + self.factuals

        if pca:
            df = pd.DataFrame(
                {
                    "X": [s["Coordinate"][0] for s in samples],
                    "Y": [s["Coordinate"][1] for s in samples],
                    "SMILES": [s["SMILES"] for s in samples],
                }
            )
            xaxis_title = "Component 1"
            yaxis_title = "Component 2"
        else:
            df = pd.DataFrame(
                {
                    "X": [s["Similarity"] for s in samples],
                    "Y": [s["Value"] for s in samples],
                    "SMILES": [s["SMILES"] for s in samples],
                }
            )
            xaxis_title = "Similarity"
            yaxis_title = "f(x)"

        super().__init__(
            df,
            title="Counterfactual Visualization",
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            log=False,
            **kwargs,
        )

    def get_data(self) -> dict:
        if self.model.setting == "classification":
            get_color = lambda s: "#AB63FA"
            get_cf_type = lambda s: "Counterfactual"
        else:
            get_color = lambda s: "#EF553B" if s["Value"] > self.base else "#636EFA"
            get_cf_type = lambda s: "High Counterfactual" if s["Value"] > self.base else "Low Counterfactual"
        self.df["color"] = ["#00CC96"] + [get_color(s) for s in self.cf_engine.cfs[1:]] + ["#FECB52"] * len(self.factuals)

        format_label = lambda s: "Similarity = {}<br>f(x) = {}".format(
            np.round(s["Similarity"], 2), np.round(s["Value"], 3)
        )
        base_template = ["f(x) = {}".format(np.round(self.base, 3))]
        cf_template = [format_label(s) for s in self.cf_engine.cfs[1:]]
        f_template = [format_label(s) for s in self.factuals]
        self.df["hovertemplate"] = base_template + cf_template + f_template

        self.df["group"] = ["Base"] + [get_cf_type(s) for s in self.cf_engine.cfs[1:]] + ["Factual"] * len(self.factuals)
        self.df["size"] = 12
        return super().get_data()

class VisualizeModelSim2(CompoundScatterPlot):
    """ Visualize the connection between a model's error on a given compound and
    specific variables.

    Parameters:
        dataset (Dataset): Dataset to evaluate model on.
        model (BaseModel): Model to evaluate on.
        eval_set (str): Subset of dataset to visualize, either 'train', 'test',
            or 'valid'. Default is 'test'
        var_name (str): Which variable to put on the x-axis. Options are 'sim'
            the similarity of a compound to the train set, 'prop' the true
            property value, and 'pred' the predicted property value. Default
            is 'sim'."""

    @log_arguments
    def __init__(self, dataset: BaseDataset, model: BaseModel, eval_set = "test", var_name = "sim", *args, log=True, **kwargs):

        if not type(dataset) is BaseDataset:
            raise TypeError(
                "dataset must be a BaseDataset. If you're using a Pandas Dataframe consider using df.to_csv() in the"
                + " oce.BaseDataset class to convert a pandas.DataFrame of data to a BaseDataset."
            )
        self.dataset = dataset
        self.model = model

        from rdkit.Chem import AllChem
        from rdkit import DataStructs

        fingerprinter = lambda s: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(s), 4)
        train_fps = [fingerprinter(s) for s in self.dataset.train_dataset[0][self.dataset.structure_col]]

        if eval_set == "test":
            self.eval_set = self.dataset.test_dataset
        elif eval_set == "train":
            self.eval_set = self.dataset.train_dataset
        elif eval_set == "valid":
            self.eval_set = self.dataset.valid_dataset

        eval_fps = [fingerprinter(s) for s in self.eval_set[0][self.dataset.structure_col]]
        sim = np.array([max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in eval_fps])

        preds = self.model.predict(self.eval_set[0])
        error = preds - self.eval_set[1]

        if var_name == "sim":
            self.var = sim
        elif var_name == "prop":
            self.var = self.eval_set[1]
        elif var_name == "pred":
            self.var = preds

        self.df = pd.DataFrame(
            {
                "X": self.var,
                "Y": error,
                "SMILES": self.eval_set[0][self.dataset.structure_col],
                "color": self.eval_set[1],
            }
        )

        super().__init__(
            self.df,
            *args,
            title=f"Residual vs {var_name} Plot<br><sub>Points colored by property value</sub>",
            xaxis_title=var_name,
            yaxis_title="Error",
            log=False,
            **kwargs,
        )

    def get_data(self) -> dict:
        d = super().get_data()
        d["colorscale"] = "Portland"
        return d

    @staticmethod
    def get_attributes():
        return [{"Select a Model/Dataset": [["model", "modelSelector"], ["dataset", "datasetSelector"],]}]

class ModelROC(BaseVisualization):
    """ Visualize the ROC curve for a model.

    Parameters:
        dataset (Dataset): Dataset to evaluate model on
        model (BaseModel): Model to evaluate on
        model_name (str, optional): name of model. Default is None.
        eval_set (str, optional): Subset of dataset to visualize, either 'train', 'test',
            or 'valid'. Default is 'test'."""

    @log_arguments
    def __init__(self, dataset: BaseDataset, model: BaseModel, *args, eval_set="test",
        log=True, model_name=None, **kwargs):

        if not type(dataset) is BaseDataset:
            raise TypeError(
                "dataset must be a BaseDataset. If you're using a Pandas Dataframe consider using df.to_csv() in the"
                + " oce.BaseDataset class to convert a pandas.DataFrame of data to a BaseDataset."
            )
        self.dataset = dataset
        self.model = model

        if eval_set == "test":
            self.eval_set = self.dataset.test_dataset
        elif eval_set == "train":
            self.eval_set = self.dataset.train_dataset
        elif eval_set == "valid":
            self.eval_set = self.dataset.valid_dataset
        else:
            self.eval_set = self.dataset.entire_dataset

        preds = self.model.predict(self.eval_set[0])

        from sklearn import metrics

        y_true = self.eval_set[1]
        if isinstance(y_true, pd.Series):
            y_true = y_true.tolist()

        fpr, tpr, thresholds = metrics.roc_curve(y_true, preds)
        ix = np.sort(np.random.choice(np.arange(len(fpr)), size=1000, replace=True))
        self.df = pd.DataFrame({"X": fpr[ix], "Y": tpr[ix],})

        self.score = metrics.roc_auc_score(y_true, preds)

        if model_name is None:
            self.model_name = model_name_from_params(oce.parameterize(model))
        else:
            self.model_name = model_name

        self.packages = ["plotly"]

    @staticmethod
    def get_attributes():
        return [{"Select a Model/Dataset": [["model", "modelSelector"], ["dataset", "datasetSelector"],]}]

    @property
    def JS_NAME(self) -> str:
        return "ModelROC"

    def get_data(self) -> dict:
        d = self.df.to_dict("l")
        d["model_name"] = self.model_name
        d["score"] = np.around(self.score, decimals=2)
        return d

class ModelROCThreshold(BaseVisualization):
    """ Visualize the ROC curve for a model.

    Parameters:
        dataset (Dataset): Dataset to evaluate model on
        model (BaseModel): Model to evaluate on
        model_name (str, optional): name of model. Default is None.
        eval_set (str, optional): Subset of dataset to visualize, either 'train', 'test',
            or 'valid'. Default is 'test'."""

    @log_arguments
    def __init__(self, dataset: BaseDataset, model: BaseModel, *args, eval_set="test",
        log=True, model_name=None, **kwargs):

        if not type(dataset) is BaseDataset:
            raise TypeError(
                "dataset must be a BaseDataset. If you're using a Pandas Dataframe consider using df.to_csv() in the"
                + " oce.BaseDataset class to convert a pandas.DataFrame of data to a BaseDataset."
            )
        self.dataset = dataset
        self.model = model

        if eval_set == "test":
            self.eval_set = self.dataset.test_dataset
        elif eval_set == "train":
            self.eval_set = self.dataset.train_dataset
        elif eval_set == "valid":
            self.eval_set = self.dataset.valid_dataset
        else:
            self.eval_set = self.dataset.entire_dataset

        preds = self.model.predict(self.eval_set[0])

        from sklearn import metrics

        y_true = self.eval_set[1]
        if isinstance(y_true, pd.Series):
            y_true = y_true.tolist()

        fpr, tpr, thresholds = metrics.roc_curve(y_true, preds)
        ix = np.sort(np.random.choice(np.arange(len(fpr)), size=1000, replace=True))
        self.df = pd.DataFrame({"X": fpr[ix], "Y": tpr[ix],
            "thresholds": thresholds[ix]},)

        self.score = metrics.roc_auc_score(y_true, preds)

        if model_name is None:
            self.model_name = model_name_from_params(oce.parameterize(model))
        else:
            self.model_name = model_name

        self.packages = ["plotly"]

    @staticmethod
    def get_attributes():
        return [{"Select a Model/Dataset": [["model", "modelSelector"], ["dataset", "datasetSelector"],]}]

    @property
    def JS_NAME(self) -> str:
            return "ModelROCThreshold"

    def get_data(self) -> dict:
        d = self.df.to_dict("l")
        d["model_name"] = self.model_name
        d["score"] = np.around(self.score, decimals=2)
        d["P"] = (sum(self.dataset.test_dataset[1]))/len(self.dataset.test_dataset[1])
        d["N"] = (len(self.dataset.test_dataset[1]) - sum(self.dataset.test_dataset[1]))/len(self.dataset.test_dataset[1])
        return d

class ModelPR(BaseVisualization):
    """ Visualize the Precision-Recall curve for a model.

    Parameters:
        dataset (Dataset): Dataset to evaluate model on
        model (BaseModel): Model to evaluate on
        model_name (str, optional): Name of model. Default is None.
        eval_set (str, optional): Subset of dataset to visualize, either 'train', 'test',
            or 'valid'. Default is 'test'."""

    @log_arguments
    def __init__(self, dataset: BaseDataset, model: BaseModel, *args, log=True, model_name=None, **kwargs):
        self.dataset = dataset
        self.model = model

        preds = self.model.predict(self.dataset.test_dataset[0])

        from sklearn import metrics

        y_true = self.dataset.test_dataset[1]
        if isinstance(y_true, pd.Series):
            y_true = y_true.tolist()
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, preds)
        ix = np.sort(np.random.choice(np.arange(len(precision)), size=1000, replace=True))
        self.df = pd.DataFrame({"X": recall[ix], "Y": precision[ix],})

        self.score = metrics.auc(recall, precision)
        self.baseline = sum(self.dataset.test_dataset[1])/len(self.dataset.test_dataset[1])

        if model_name is None:
            self.model_name = model_name_from_params(oce.parameterize(model))
        else:
            self.model_name = model_name

        self.packages = ["plotly"]

    @staticmethod
    def get_attributes():
        return [{"Select a Model/Dataset": [["model", "modelSelector"], ["dataset", "datasetSelector"],]}]

    @property
    def JS_NAME(self) -> str:
        return "ModelPR"

    def get_data(self) -> dict:
        d = self.df.to_dict("l")
        d["model_name"] = self.model_name
        d["score"] = np.around(self.score, decimals=2)
        d["baseline"] = self.baseline
        return d

class BaseErrorWaterfall(BaseVisualization):
    """ Visualize the error waterfall for a base boosting model.

   Args:
        model (BaseBoosting): Model to evaluate on. must be base boosting model.
        x_data (Union[pd.DataFrame, np.ndarray]): Data to predict on using the model
        y_data (Union[pd.Series, list, np.ndarray], optional): True values to compare to. Defaults to None. If None, then the waterfall plot will be for residuals.
        normalization (bool, optional): If the data is normalized. Defaults to False.
    """

    @log_arguments
    def __init__(
        self, 
        model: BaseBoosting, 
        x_data: Union[pd.DataFrame, np.ndarray], 
        *y_data: Union[pd.Series, list, np.ndarray], 
        normalization = False, 
        log=True, 
        **kwargs):

        self.model = model
        self.x_data = x_data
        if not y_data:
            self.y_data = None
        else:
            self.y_data = y_data
        self.normalization = normalization
        super().__init__(log=False)
        self.packages += ["plotly"]
        
        self.df = pd.DataFrame({})

    def get_data(self) -> dict:
        """Get data for visualization in JSON-like dictionary.

        Returns:
            dict: Data for visualization."""
        predictions = self.model.predict(self.x_data, waterfall = True, normalize = self.normalization)
        data = self.model._waterfall(y_data = self.y_data, normalize = self.normalization)        
 
        diffs = [j - i for i, j in zip(data, data[1:])]
        diffs.insert(0, data[0])
        diffs.append(0)

        if self.y_data is not None:
            model_names = ["Model " + str(i) for i in range(1, len(data))]
            model_names.insert(0, "Dataset Baseline")
        else:
            model_names = ["Model " + str(i+1) for i in range(1, len(data))]
            model_names.insert(0, "Model 1 Baseline")
        model_names.append("Boosted Model")

        self.df['diffs'] = diffs
        self.df['model_names'] = model_names

        text = [f'{val:.2f}' for val in diffs]
        text[-1] = f'{data[-1]:.2f}'
        self.df['text'] = text

        w_type = ['relative' for val in diffs]
        w_type[-1] = 'total'
        self.df['w_type'] = w_type

        return self.df.to_dict("l")