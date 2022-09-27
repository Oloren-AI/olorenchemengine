"""
These visualizations are intended for exploratory analysis.
"""

from olorenchemengine.base_class import *
from olorenchemengine.representations import *
from olorenchemengine.dataset import *
from olorenchemengine.uncertainty import *
from olorenchemengine.interpret import *
from olorenchemengine.internal import *
from olorenchemengine.manager import *
from olorenchemengine.visualizations import *

class VisualizeFeatures(BaseVisualization):
    """
    Visualize how the selected features are correlated with the dataset.

    Parameters:
        dataset (BaseDataset): the dataset to analyze
        features (list): the features to visualize. For the elements of the feature
            list, strings will be treated as a column of the dataset and BaseVecRepresentations
            are used to convert structures to features, the provided representations
            should have the names property defined for better interpretability.
            In addition to features thefeature_cols in the dataset are also visualized."""

    @log_arguments
    def __init__(self, dataset: BaseDataset, features: list = [], log=True, **kwargs):
        self.dataset = dataset
        self.features = dataset.feature_cols + features
        self.packages = ["plotly", "olorenrenderer"]
        self.data = {
            "SMILES": self.dataset.data[self.dataset.structure_col].tolist(),
            self.dataset.property_col: self.dataset.data[self.dataset.property_col].tolist(),
        }
        self.d = {}
        self.feature_cols = []
        self.spearman_corr = []
        from scipy.stats import spearmanr
        for feature in self.features:
            if isinstance(feature, str):
                self.feature_cols.append(features)
                self.d[feature] = self.dataset.data[feature].tolist()
                self.spearman_corr.append(spearmanr(self.d[feature], self.dataset.data[self.dataset.property_col].tolist()))
            elif issubclass(type(feature), BaseRepresentation):
                rep_features = feature.convert(self.dataset.entire_dataset[0])
                rep_features = np.array(rep_features).transpose()
                try:
                    names = feature.names
                except:
                    names = [f"{feature.__class__.__name__}-{i}" for i in range(rep_features.shape[0])]
                for i, name in enumerate(names):
                    self.feature_cols.append(name)
                    self.d[name] = rep_features[i].tolist()
                    self.spearman_corr.append(spearmanr(self.d[name], self.dataset.data[self.dataset.property_col]))

    def get_data(self):
        d = {"datacols": self.d.copy(),
            "SMILES": self.data["SMILES"],
            "PROPERTY": self.dataset.property_col,
            "PROPERTY_VALUES": self.data[self.dataset.property_col],
            "FEATURE_COLS": self.feature_cols,
            "SPEARMAN_COEF": [np.around(x[0], decimals=2) for x in self.spearman_corr],
            "SPEARMAN_PVAL": [np.around(x[1], decimals=2) for x in self.spearman_corr],
            "hoverSize": 0.8}
        return d



