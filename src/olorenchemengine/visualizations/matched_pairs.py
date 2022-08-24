from typing import *

from olorenchemengine.base_class import *
from olorenchemengine.dataset import *
from olorenchemengine.visualizations.visualization import *


class MatchedPairsTable(BaseVisualization):
    @log_arguments
    def __init__(self, dataset: BaseDataset, mode: str = "features", annotations: Union[str, List[str]] = []):

        self.packages = ["d3"]

        self.dataset = dataset

        if not isinstance(annotations, list):
            annotations = [annotations]
        self.annotations = annotations

        self.mps = self.matched_pairs(dataset.data, dataset.feature_cols)
        if mode == "features":
            self.mp_diffs = pd.DataFrame()

            for mp in self.mps:
                r1 = self.dataset.data.iloc[mp[0]]
                r2 = self.dataset.data.iloc[mp[1]]
                diff = mp[2][0]
                self.mp_diffs = self.mp_diffs.append(
                    {
                        "col": diff[0],
                        "initial": diff[1],
                        "final": diff[2],
                        **{
                            f"Delta (Final - Initial) {property_col}": r2[property_col + " mean"]
                            - r1[property_col + " mean"]
                            for property_col in dataset.property_col
                        },
                        **{
                            f"Fold (Final/Initial) {property_col}": r2[property_col + " mean"]
                            / r1[property_col + " mean"]
                            for property_col in dataset.property_col
                        },
                        **{f"Initial {annotation}": r1[annotation] for annotation in self.annotations},
                        **{f"Final {annotation}": r2[annotation] for annotation in self.annotations},
                    },
                    ignore_index=True,
                )
            self.mp_diffs = self.mp_diffs.sort_values(by="col", ascending=True).reset_index().drop(columns=["index"])
        elif mode == "property":
            self.mp_diffs = pd.DataFrame()
            for mp in self.mps:
                r1 = self.dataset.data.iloc[mp[0]]
                r2 = self.dataset.data.iloc[mp[1]]
                diff = mp[2][0]

                for property in self.dataset.property_col:
                    if not r1[property] == r2[property]:
                        self.mp_diffs = self.mp_diffs.append(
                            {
                                "col": diff[0],
                                "initial": diff[1],
                                "final": diff[2],
                                f"Initial {property}": r1[property],
                                f"Final {property}": r2[property],
                                **{f"Initial {annotation}": r1[annotation] for annotation in annotations},
                                **{f"Final {annotation}": r2[annotation] for annotation in annotations},
                            },
                            ignore_index=True,
                        )
            self.mp_diffs = (
                self.mp_diffs.sort_values(by=["col", "initial", "final"], ascending=True)
                .reset_index()
                .drop(columns=["index"])
            )

    def matched_pairs(self, df, cols, dist=1):
        out = []
        for i_ in tqdm(range(len(df.index))):
            i = df.index[i_]
            for j_ in range(len(df.index)):
                j = df.index[j_]
                if not i_ == j_:
                    count = []
                    close = True
                    for col in cols:
                        if str(df.iloc[i][col]) != str(df.iloc[j][col]):
                            count.append((col, df.iloc[i][col], df.iloc[j][col]))
                        if len(count) > dist:
                            close = False
                            break
                    if close:
                        out.append((i, j, count))
        return out

    def get_data(self):
        data = []
        for i, col in enumerate(self.mp_diffs["col"].unique()):
            data.append({"col": col, "i": i, "col_data": self.mp_diffs.loc[self.mp_diffs["col"] == col].to_dict("r")})
        return data

    @property
    def JS_NAME(self):
        return "MatchedPairsTable"


class MatchedPairsHeatmap(MatchedPairsTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.packages = ["d3", "plotly"]

    def get_data(self):
        data = []
        for property_col in self.dataset.property_col:
            property_col = f"Delta (Final - Initial) {property_col}"
            for i, col in enumerate(self.mp_diffs["col"].unique()):
                col_data = self.mp_diffs[self.mp_diffs["col"] == col]
                mapping = {str(x): i for i, x in enumerate(col_data["initial"].unique())}
                col_heatmap = np.zeros((len(mapping), len(mapping)))
                for i, col_row in col_data.iterrows():
                    try:
                        x = float(col_row[property_col])
                        if not np.isnan(x):
                            col_heatmap[mapping[str(col_row["final"])], mapping[str(col_row["initial"])]] += col_row[
                                property_col
                            ]
                    except:
                        pass

                data.append(
                    {
                        "z": col_heatmap.tolist(),
                        "x": list(mapping.keys()),
                        "y": list(mapping.keys()),
                        "xlabel": "Initial",
                        "ylabel": "Final",
                        "title": f"Residue number {col}, colored by sum of {property_col}",
                    }
                )
        return data

    @property
    def JS_NAME(self):
        return "MatchedPairsHeatmap"
