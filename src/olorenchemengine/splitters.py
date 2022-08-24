""" For creating splits on the data"""

from abc import abstractmethod

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import itertools
import random

from .base_class import *
from .dataset import *
from .representations import DescriptastorusDescriptor


class BaseSplitter(BaseDatasetTransform):
    """Base class for all splitters.

    Parameters:
        split_proportions (float): Proportion of the data to be used for training.
        log (bool): Whether to log the data or not.
    """

    @log_arguments
    def __init__(self, split_proportions=[0.8, 0.1, 0.1], log=True):
        assert (
            len(split_proportions) == 3
        ), "split_proportions must be a list of length 3 for train/val/test, if val set is to be empty, leave the second entry as 0"
        self.split_proportions = split_proportions
        self.split_proportions = np.array(self.split_proportions)
        self.split_proportions /= np.sum(self.split_proportions)

    @abstractmethod
    def split(self, data, *args, **kwargs):
        """Split data into train/val/test sets.

        Parameters:
            data (pandas.DataFrame): Dataset to split, must have a structure column.

        Returns:
            (tuple): Tuple of training, validation, and testing dataframes.
        """
        pass

    def transform(self, dataset: BaseDataset, *args, **kwargs) -> BaseDataset:
        train, val, test = self.split(dataset.data, *args,
            structure_col=dataset.structure_col, date_col=dataset.date_col, **kwargs)
        train["split"] = "train"
        val["split"] = "valid"
        test["split"] = "test"
        dataset.data = pd.concat([train, val, test])
        return dataset

    def _save(self):
        pass

    def _load(self, d):
        pass


class RandomSplit(BaseSplitter):
    """Split data randomly into train/val/test sets.

    Parameters:
        data (pandas.DataFrame): Dataset to split.
        split_proportions (tuple[int]): Tuple of train/val/test proportions of data to split into.
        log (bool): Whether to log the data or not.

    Methods:
        split(data): Return array of train/val/test dataframes in format [train, val, test].

    Example
    ------------------------------
    import olorenchemengine as oce

    df = pd.read_csv("Your Dataset")
    dataset = (
        oce.BaseDataset(data = df.to_csv(),
        structure_col = "SMILES COLUMN", property_col = "PROPERTY COLUMN") +
        oce.RandomSplit(split_proportions = [0.8, 0.1, 0.1])
    )
    #OR
    train, val, test = oce.RandomSplit(split_proportions = [0.8, 0.1, 0.1]).split(df)
    ------------------------------
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):
        super().__init__(log=False, **kwargs)

    def split(self, data, *args, **kwargs):
        out = []
        for i in range(len(self.split_proportions)):
            p = self.split_proportions[i] / np.sum(self.split_proportions[i:])
            if p == 0:
                out.append(pd.DataFrame(columns=data.columns))
            elif p == 1:
                out.append(data)
            else:
                split, remaining = train_test_split(data, train_size=p)
                out.append(split)
                data = remaining
        return out


class StratifiedSplitter(BaseSplitter):
    """Split data into train/val/test sets stratified by a value column (generally the label).

    Parameters:
        value_col (str): Name of the column to stratify by.
        log (bool): Whether to log the data or not.

    Methods:
        split(data): Return array of train/val/test dataframes in format [train, val, test].

    Example
    ------------------------------
    import olorenchemengine as oce

    df = pd.read_csv("Your Dataset")
    dataset = (
        oce.BaseDataset(data = df.to_csv(),
        structure_col = "SMILES COLUMN", property_col = "PROPERTY COLUMN") +
        oce.StratifiedSplitter(split_proportions = [0.8, 0.1, 0.1], value_col="PROPERTY COLUMN")
    )
    #OR
    train, val, test = oce.StratifiedSplitter(split_proportions = [0.8, 0.1, 0.1], value_col="PROPERTY COLUMN").split(df)
    ------------------------------
    """

    @log_arguments
    def __init__(self, value_col, log=True, **kwargs):
        self.value_col = value_col
        super().__init__(log=False, **kwargs)

    def split(self, data, *args, **kwargs):
        out = []
        for i in range(len(self.split_proportions) - 1):
            train, remaining = train_test_split(
                data,
                train_size=np.sum([self.split_proportions[j] for j in range(i + 1)]),
                stratify=data[self.value_col],
            )
            out.append(train)
            data = remaining
        out.append(data)

        return out


class DateSplitter(BaseSplitter):
    """Split data into train/val/test sets by date range.

    Parameters:
        date_col (str): Name of the column to split by.
        log (bool): Whether to log the data or not.

    Methods:
        split(data, date_col): Return array of train/val/test dataframes in format [train, val, test].

    Example
    ------------------------------
    import olorenchemengine as oce

    df = pd.read_csv("Your Dataset")
    dataset = (
        oce.BaseDataset(data = df.to_csv(),
        structure_col = "SMILES COLUMN", property_col = "PROPERTY COLUMN") +
        oce.DateSplitter(split_proportions = [0.8, 0.1, 0.1], date_col="DATE COLUMN")
    )
    #OR
    train, val, test = oce.DateSplitter(split_proportions = [0.8, 0.1, 0.1], date_col="DATE COLUMN").split(df)
    ------------------------------
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):
        super().__init__(log=False, **kwargs)

    def split(self, data, date_col, *args, **kwargs):
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values(by=date_col)

        splits = [0] + [int(sum(self.split_proportions[:i+1])*len(data)) for i in range(len(self.split_proportions))]

        out = []
        for i in range(len(splits)-1):
            if splits[i] == splits[i+1]:
                out.append(pd.DataFrame())
            else:
                out.append(data.iloc[splits[i]:splits[i+1]])
        return out


class ScaffoldSplit(BaseSplitter):
    """Split data into train/val/test sets by scaffold. Makes sure that the same Bemis-Murcko scaffold is not used in both train and test.

    Parameters:
        scaffold_filter_threshold (float): Threshold for minimum number of compounds per scaffold class for a scaffold class to be included.
        split_proportions (tuple[int]): Tuple of train/val/test proportions of data to split into.
        split_type (string): type of split
            murcko: split data by bemis-murcko scaffold
            kmeans_murcko: split data by kmeans clustering murcko scaffolds

    Methods:
        split(data, structure_col): Return array of train/val/test dataframes in format [train, val, test].

    Example
    ------------------------------
    import olorenchemengine as oce

    df = pd.read_csv("Your Dataset")
    dataset = (
        oce.BaseDataset(data = df.to_csv(),
        structure_col = "SMILES COLUMN", property_col = "PROPERTY COLUMN") +
        oce.ScaffoldSplit(split_proportions = [0.8, 0.1, 0.1], scaffold_filter_threshold = 5, split_type = "murcko")
    )
    #OR
    train, val, test = oce.ScaffoldSplitter(split_proportions = [0.8, 0.1, 0.1], scaffold_filter_threshold = 5, split_type = "murcko").split(df, structure_col = "SMILES COLUMN")
    ------------------------------
    """

    @log_arguments
    def __init__(self, scaffold_filter_threshold: int = 0, split_type="murcko", log=True, **kwargs):
        self.scaffold_filter_threshold = scaffold_filter_threshold
        self.split_type = split_type
        super().__init__(log=False, **kwargs)

    def _generate_scaffold(self, smiles):
        """(Private) Return murcko scaffold string of target molecule
        Parameters:
            smiles (string): smiles representation of molecule to calculate scaffold for

        Returns:
            (string): string representation of molecule's scaffold
        """
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold

    def _generate_scaffold_column(self, data, structure_col):
        """(Private) Generate scaffold column for data

        Parameters:
            data (pandas.DataFrame): data to generate scaffold column for
            structure_col (string): column name for structure column

        Returns:
            (pandas.DataFrame): data with scaffold column added
        """

        data["murcko"] = ""
        for i, row in data.iterrows():
            data.at[i, "murcko"] = self._generate_scaffold(data.at[i, structure_col])
        return data

    def _kmeans_murcko_split(self, data):
        """Create a list of filtered scaffolds and clusters the scaffolds by their Morgan fingerprint.

        Parameters:
            data (pandas.DataFrame): data to split

        Returns:
            (list): list of filtered scaffolds and clusters"""
        out = [pd.DataFrame(columns=data.columns.values)] * 3
        scaffold_classes = data["murcko"].value_counts()
        filtered_scaffolds = [
            scaffold
            for scaffold in scaffold_classes.index
            if scaffold_classes[scaffold] >= self.scaffold_filter_threshold
        ]
        data = data.loc[data["murcko"].isin(filtered_scaffolds)]
        scaff_size = len(filtered_scaffolds)
        data_size = len(data.index)
        cluster_count = np.count_nonzero(self.split_proportions)
        split_reached = [False] * cluster_count

        mtc = DescriptastorusDescriptor("morgan3counts").convert(filtered_scaffolds)

        from sklearn.cluster import KMeans
        cluster = KMeans(n_clusters=cluster_count).fit_predict(mtc)

        """This loops through the array of cluster labels. Checks if the train/val/test split is full, if not then adds the scaffold corresponding to cluster label.
        """

        if cluster_count == 2:
            cluster = [2 if i == 1 else 0 for i in cluster]
        for i in range(len(cluster)):
            cluster_id = cluster[i]
            if len(out[cluster_id].index) / data_size <= self.split_proportions[cluster_id]:
                out[cluster_id] = pd.concat(
                    [out[cluster_id], data.loc[data["murcko"] == filtered_scaffolds[i]]], ignore_index=True, sort=False
                )
            else:
                if cluster_count == 2 and cluster_id == 2:
                    split_reached[1] = True
                else:
                    split_reached[cluster_id] = True
            if split_reached == [True] * cluster_count or i >= scaff_size:
                break
        data = data.drop("murcko", 1)
        return out

    def _murcko_split(self, data):
        """Create a list of filtered scaffolds and clusters the scaffolds by their Morgan fingerprint.

        Parameters:
            data (pandas.DataFrame): data to split

        Returns:
            (list): list of filtered scaffolds and clusters"""
        out = [pd.DataFrame(columns=data.columns.values)] * 3
        scaffold_classes = data["murcko"].value_counts()
        filtered_scaffolds = [
            scaffold
            for scaffold in scaffold_classes.index
            if scaffold_classes[scaffold] >= self.scaffold_filter_threshold
        ]
        data = data.loc[data["murcko"].isin(filtered_scaffolds)]
        scaff_size = len(filtered_scaffolds)
        scaff_index = 0
        data_size = len(data.index)
        split_reached = [False] * 3

        if scaff_size == 0:
            raise Exception("No scaffolds with enough molecules to meet filter threshold. Please lower filter threshold.")

        '''
        Cycle through train/val/test sets.
        If split proportion is not met, append data with the next most common scaffold.
        If all split proportions are met or scaffolds are fully cycled return split.'''
        for i in itertools.cycle([0, 1, 2]):
            if self.split_proportions[i] == 0:
                split_reached[i] = True
            elif len(out[i].index) / data_size <= self.split_proportions[i]:
                out[i] = pd.concat(
                    [out[i], data.loc[data["murcko"] == filtered_scaffolds[scaff_index]]], ignore_index=True, sort=False
                )
                scaff_index += 1
            else:
                split_reached[i] = True
            if split_reached == [True] * 3 or scaff_index >= scaff_size:
                break
        data = data.drop("murcko", 1)
        return out

    def split(self, data: pd.DataFrame, *args, structure_col: str = "Smiles", **kwargs):
        data = self._generate_scaffold_column(data, structure_col)
        if self.split_type == "murcko":
            return self._murcko_split(data)
        elif self.split_type == "kmeans_murcko":
            return self._kmeans_murcko_split(data)
        else:
            return "Invalid split_type initializer."

class dc_ScaffoldSplit(BaseSplitter):
    """
    Split data into train/val/test sets by scaffold using DeepChem implementation. https://deepchem.readthedocs.io/en/latest/api_reference/splitters.html#scaffoldsplitter

    Parameters:
        split_proportions (tuple[int]): Tuple of train/val/test proportions of data to split into.

    Methods:
        split(data, structure_col): Return array of train/val/test dataframes in format [train, val, test].

    Example
    ------------------------------
    import olorenchemengine as oce

    df = pd.read_csv("Your Dataset")
    dataset = (
        oce.BaseDataset(data = df.to_csv(),
        structure_col = "SMILES COLUMN", property_col = "PROPERTY COLUMN") +
        oce.dc_ScaffoldSplit(split_proportions = [0.8, 0.1, 0.1])
    )
    #OR
    train, val, test = oce.dc_ScaffoldSplitter(split_proportions = [0.8, 0.1, 0.1]).split(df, structure_col = "SMILES COLUMN")
    ------------------------------
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):
        super().__init__(log=False, **kwargs)

    def split(self, data: pd.DataFrame, *args, structure_col: str = "Smiles", **kwargs):
        try:
            import deepchem as dc
        except:
            print("Error, deepchem not installed. Please use ScaffoldSplit instead or install deepchem")
        dc_dataset = dc.data.DiskDataset.from_numpy(X=data.index, ids=data[structure_col])
        scaffoldsplitter = dc.splits.ScaffoldSplitter()
        train,valid,test = scaffoldsplitter.train_valid_test_split(dc_dataset, frac_train=self.split_proportions[0], frac_valid=self.split_proportions[1], frac_test=self.split_proportions[2])
        return data.loc[train.X], data.loc[valid.X], data.loc[test.X]

class gg_ScaffoldSplit(BaseSplitter):
    """
    Split data into train/val/test sets by scaffold using implementation from https://www.nature.com/articles/s42256-021-00438-4, https://github.com/PaddlePaddle/PaddleHelix

    Parameters:
        split_proportions (tuple[int]): Tuple of train/val/test proportions of data to split into.

    Methods:
        split(data, structure_col): Return array of train/val/test dataframes in format [train, val, test].

    Example
    ------------------------------
    import olorenchemengine as oce

    df = pd.read_csv("Your Dataset")
    dataset = (
        oce.BaseDataset(data = df.to_csv(),
        structure_col = "SMILES COLUMN", property_col = "PROPERTY COLUMN") +
        oce.gg_ScaffoldSplit(split_proportions = [0.8, 0.1, 0.1])
    )
    #OR
    train, val, test = oce.gg_ScaffoldSplitter(split_proportions = [0.8, 0.1, 0.1]).split(df, structure_col = "SMILES COLUMN")
    ------------------------------
    """

    @log_arguments
    def __init__(self, log=True, **kwargs):
        super().__init__(log=False, **kwargs)

    def split(self, data: pd.DataFrame, *args, structure_col: str = "smiles", **kwargs):
        return self.gg_split(data,
            frac_train=self.split_proportions[0],
            frac_valid=self.split_proportions[1],
            frac_test=self.split_proportions[2],
            structure_col = structure_col)

    def gg_split(self,
            dataset,
            frac_train=None,
            frac_valid=None,
            frac_test=None,
            structure_col = "smiles"):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split. Make sure each element in
                the dataset has key "smiles" which will be used to calculate the
                scaffold.
            frac_train(float): the fraction of data to be used for the train split.
            frac_valid(float): the fraction of data to be used for the valid split.
            frac_test(float): the fraction of data to be used for the test split.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}
        for i in range(N):
            scaffold = self.generate_scaffold(dataset.iloc[i][structure_col], include_chirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        train_dataset = dataset.iloc[train_idx]
        valid_dataset = dataset.iloc[valid_idx]
        test_dataset = dataset.iloc[test_idx]
        return train_dataset, valid_dataset, test_dataset

    def generate_scaffold(self, smiles, include_chirality=False):
        """
        Obtain Bemis-Murcko scaffold from smiles.

        Args:
            smiles: smiles sequence
            include_chirality: Default=False

        Return:
            the scaffold of the given smiles.
        """
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality)
        return scaffold

class PropertySplit(BaseSplitter):
    """Split molecules into train/val/test based on user-defined property.

    Parameters:
        property_col (string): column in dataset with property values to split data on
        threshold (int) (optional): user-defined value to split data. If set to None (default), threshold will be determined based on split_proportions. User defines a single threshold for train/test split.
        noise (int): random noise to add to dataset before splitting. Note: data is minmax scaled to [0, 1] range before noise is introduced.
        categorical (bool): Set True to convert property values to categorical format ([0, 1, 2]) based on threshold.

        Methods:
            split(data): Return array of train/val/test dataframes in format [train, val, test].

        Example
        ------------------------------
        import olorenchemengine as oce

        df = pd.read_csv("Your Dataset")
        dataset = (
            oce.BaseDataset(data = df.to_csv(),
            structure_col = "SMILES COLUMN", property_col = "PROPERTY COLUMN") +
            oce.PropertySplit(split_proportions = [0.8, 0.1, 0.1], property_col = "PROPERTY COLUMN", threshold = 0.5, noise = 0.1, categorical = False)
        )
        #OR
        train, val, test = oce.PropertySplit(split_proportions = [0.8, 0.1, 0.1], property_col = "PROPERTY COLUMN", threshold = 0.5, noise = 0.1, categorical = False).split(df)
    ------------------------------
    """

    @log_arguments
    def __init__(self, property_col, threshold=None, noise=0.1, categorical=False, log=True, **kwargs):
        self.property_col = property_col
        self.threshold = threshold
        self.noise = noise
        self.categorical = categorical
        super().__init__(log=False, **kwargs)

    def _quantile_uniform(self, data, noise):
        """Quantile transform to uniform distribution, minmax scaled to 0-1 range.

        Parameters:
            data: data that is to be scaled to 0-1 range

        Returns:
            (pandas.DataFrame): scaled data
        """
        property = data[self.property_col]

        data["prop_norm"] = QuantileTransformer(output_distribution="uniform").fit_transform(property)
        data["prop_norm"] = (
            (data["prop_norm"] + [random.uniform(-noise, noise) for i in range(len(data))])
            .clip(lower=0, upper=1)
            .sample(frac=1)
        )

        out = []
        out.append(data.loc[data["prop_norm"] <= self.split_proportions[0]])
        out.append(
            data.loc[
                (data["prop_norm"] > self.split_proportions[0])
                & (data["prop_norm"] <= self.split_proportions[0] + self.split_proportions[1])
            ]
        )
        out.append(data.loc[data["prop_norm"] > (self.split_proportions[0] + self.split_proportions[1])])

        if self.categorical:
            thresh1 = np.percentile(data[self.property_col], self.split_proportions[0] * 10)
            thresh2 = np.percentile(
                data[self.property_col], (self.split_proportions[0] + self.split_proportions[1]) * 10
            )
            for df in out:
                df.loc[df[self.property_col] <= thresh1, self.property_col] = int(0)
                df.loc[
                    (df[self.property_col] > thresh1) & (data[self.property_col] <= thresh2), self.property_col
                ] = int(1)
                df.loc[df[self.property_col] > thresh2, self.property_col] = int(2)

        data = data.drop("prop_norm", 1)
        return out

    def _quantile_uniform_thresh(self, data, threshold, noise):
        property = data[self.property_col]

        transformer = QuantileTransformer(output_distribution="uniform").fit(property)
        data["prop_norm"] = transformer.transform(property.to_numpy().reshape(-1, 1))

        threshold_scaled = (
            (transformer.transform(np.array([threshold]).reshape(1, -1)) - data["prop_norm"].min())
            / (data["prop_norm"].max() - data["prop_norm"].min())
        )[0][0]

        data["prop_norm"] = (
            (data["prop_norm"] + [random.uniform(-noise, noise) for i in range(len(data))])
            .clip(lower=0, upper=1)
            .sample(frac=1)
        )

        out = [pd.DataFrame(columns=data.columns.values)] * 3
        out[0] = data.loc[data["prop_norm"] <= threshold_scaled]
        out[2] = data.loc[data["prop_norm"] > threshold_scaled]

        if self.categorical:
            for df in out:
                df[self.property_col] = np.where(df[self.property_col] <= self.threshold, int(0), int(1))

        data = data.drop("prop_norm", 1)
        return out

    def split(self, data: pd.DataFrame, *args, **kwargs):
        if self.threshold == None:
            return self._quantile_uniform(data, self.noise)
        else:
            return self._quantile_uniform_thresh(data, self.threshold, self.noise)
