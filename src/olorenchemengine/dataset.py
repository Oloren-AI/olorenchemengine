from typing_extensions import Self
import warnings

warnings.filterwarnings("ignore")

from time import sleep

import olorenchemengine as oce

from .base_class import *

class BaseDataset(BaseClass):
    """ BaseDataset for all dataset objects

    BaseDataset holds its data in a Pandas DataFrame.

    Parameters:
        name (str): Name of the dataset
        data (str): The output when df.to_csv() where df is the pd.DataFrame containing the dataset.
        structure_col (str): Name of column containing structure information, e.g. "smiles"
        feature_cols (list[str]): List of names of columns containing features, e.g. ["X1", "X2"]
        property_col (str): Name of property of interest, e.g. "Y"
    """

    @log_arguments
    def __init__(
        self,
        name: str =None,
        data: str =None,
        structure_col: str = None,
        property_col: str =None,
        feature_cols: list =[],
        date_col: str = None,
        log=True,
        **kwargs
    ):

        if name is None:
            code = base64.urlsafe_b64encode(hashlib.md5(str(data).encode("utf-8")).digest())[:16].decode("utf-8")
            self.name = f"Dataset-{code}"
        else:
            self.name = name

        self.data = pd.read_csv(io.StringIO(data))

        # saving column names
        self.structure_col = structure_col
        self.feature_cols = feature_cols

        if not feature_cols is None:
            self.input_cols = [self.structure_col] + self.feature_cols
        else:
            self.input_cols = [self.structure_col]

        self.property_col = property_col
        self.date_col = date_col
        if not self.date_col is None:
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])

    @property
    def entire_dataset(self):
        """ Returns the entire dataset

        Returns:
            pd.DataFrame: The entire dataset
            """

        return self.data[self.input_cols], self.data[self.property_col]

    @property
    def entire_dataset_split(self):
        """ Returns a tuple of three elements where the first is the input train data,
        the second is the input validation data, and the third is the input test data

        Returns:
            tuple: (train_data, val_data, test_data)
        """
        assert "split" in self.data.columns, "Dataset not split yet, please use a splitter or define the split column"
        return self.data["split"]

    @property
    def train_dataset(self):
        """ Returns the train dataset
        """
        assert "split" in self.data.columns, "Dataset not split yet, please use a splitter or define the split column"
        train = self.data[self.data["split"] == "train"]
        return train[self.input_cols], train[self.property_col]

    @property
    def valid_dataset(self):
        """ Gives a tuple of two elements where the first is the input val data and
        the second is the property of interest

        Returns:
            pd.DataFrame: The validation data

        """
        assert "split" in self.data.columns, "Dataset not split yet, please use a splitter or define the split column"
        valid = self.data[self.data["split"] == "valid"]
        return valid[self.input_cols], valid[self.property_col]

    @property
    def trainval_dataset(self):
        """ Returns the train and validation dataset
        """
        assert "split" in self.data.columns, "Dataset not split yet, please use a splitter or define the split column"
        train = self.data[self.data["split"] == "train"]
        val = self.data[self.data["split"] == "valid"]
        trainval = pd.concat([train, val])
        return trainval[self.input_cols], trainval[self.property_col]

    @property
    def test_dataset(self):
        """ Gives a tuple of two elements where the first is the input test data and
        the second is the property of interest

        Returns:
            pd.DataFrame: The test data

        """
        assert "split" in self.data.columns, "Dataset not split yet, please use a splitter or define the split column"
        test = self.data[self.data["split"] == "test"]
        return test[self.input_cols], test[self.property_col]

    @property
    def size(self):
        return (len(self.train_dataset[0]), len(self.valid_dataset[0]), len(self.test_dataset[0]))

    def transform(self, dataset: Self):
        """ Combines this dataset with the passed dataset object"""
        # Rename passed dataset to homogenize with this dataset

    def _save(self):
        d = {"data_save": self.data.to_csv()}
        return d

    def _load(self, d):
        from io import StringIO

        self.data = pd.read_csv(StringIO(d["data_save"]))

class BaseDatasetTransform(BaseClass):
    """ Applies a transformation onto the inputted BaseDataset.

    Transformation applied as defined in the abstract method transform.

    Parameters:
        dataset (BaseDataset): The dataset to transform.
    """

    @abstractmethod
    def transform(self, dataset: BaseDataset) -> BaseDataset:
        """ Applies a transformation onto the inputted BaseDataset.

        Parameters:
        dataset (BaseDataset): The dataset to transform.
        """
        pass

    def _save(self):
        return {}

    def _load(self, d):
        pass

def func(self: BaseDataset, other: BaseDatasetTransform) -> BaseDataset:
    return other.transform(self)

BaseDataset.__add__ = func

class DatasetFromCSV(BaseDataset):
    """ DatasetFromFile for all dataset objects

    Parameters:
        file_path (str): Relative or absolute to a local CSV file
    """

    @log_arguments
    def __init__(self, file_path, log=True, **kwargs):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        super().__init__(data=self.data.to_csv(), log=False, **kwargs)


import requests
import io


class DatasetFromCDDSearch(BaseDataset):
    """ Dataset for retreiving data from CDD via a saved search.

    Requires a CDD Token to be set.

    Parameters:
        search_id (str): The ID of the saved CDD search to use.
        cache_file_path (str): The path to the file to cache the dataset.
        update (bool): Whether or no to update the cached dataset by redoing the CDD search"""

    @log_arguments
    def __init__(self, search_id, cache_file_path=None, update=True, log=True, **kwargs):

        if cache_file_path is None:
            cache_file_path = f"{search_id}.csv"

        if not update and os.path.exists(cache_file_path):
            self.data = pd.read_csv(cache_file_path)
            self.data.to_csv(cache_file_path)
        else:
            self.data = self.get_dataset_cdd_saved_search(search_id)
        self.data.to_csv(cache_file_path)

        super().__init__(data=self.data.to_csv(), log=False, **kwargs)

    def run_saved_search(self, search_id):
        """  Uses a CDD Token (passed as search_id) to search saved datasets to find and return its
        related dataset export id.

        Parameters:
        search_id (str): The ID of the saved CDD search to use.
        """
        base_url = f"https://app.collaborativedrug.com/api/v1/vaults/{oce.CONFIG['VAULT_ID']}/"
        headers = {"X-CDD-token": f"{oce.CONFIG['CDD_TOKEN']}"}
        url = base_url + f"searches/{search_id}"

        response = requests.request("GET", url, headers=headers).json()
        return response["id"]

    def check_export_status(self, export_id):
        """Uses the export_id passed as a parameter to find the pertinent dataset and return
        its export status

        Parameters:
        export_id (str): The unique export ID of the dataset searched for

        """
        base_url = f"https://app.collaborativedrug.com/api/v1/vaults/{oce.CONFIG['VAULT_ID']}/"
        headers = {"X-CDD-token": f"{oce.CONFIG['CDD_TOKEN']}"}
        url = base_url + f"export_progress/{export_id}"

        response = requests.request("GET", url, headers=headers).json()
        return response["status"]

    def get_export(self, export_id):
        """Uses the export_id passed as a parameter to find the pertinent dataset and return
        the dataset's data in CSV format.

        Parameters:
        export_id (str): The unique export ID of the dataset searched for

        """
        base_url = f"https://app.collaborativedrug.com/api/v1/vaults/{oce.CONFIG['VAULT_ID']}/"
        headers = {"X-CDD-token": f"{oce.CONFIG['CDD_TOKEN']}"}
        url = base_url + f"exports/{export_id}"

        response = requests.request("GET", url, headers=headers)
        data_stream = io.StringIO(response.text)
        return pd.read_csv(data_stream)

    def get_dataset_cdd_saved_search(self, search_id):
        """ Uses a CDD Token (passed as search_id) to search saved datasets to find and return its
        related dataset export id. Using the export id, it then checks the export status and
        returns the dataset's data in CSV format.

        Parameters:
        search_id (str): The ID of the saved CDD search to use.
        """
        export_id = self.run_saved_search(search_id)
        i = 0
        status = "new"
        while True:
            print(f"Export status is {status}, checking in {2**i} seconds...")
            sleep(2 ** i)
            status = self.check_export_status(export_id)
            if status == "finished":
                print("Export ready!")
                break
            i += 1
        return self.get_export(export_id)

from rdkit import Chem

class CleanStructures(BaseDatasetTransform):
    """ CleanStructures creates a new dataset from the original dataset by removing
    structures that are not valid.

    Parameters:
        dataset (BaseDataset): The dataset to clean.
    """

    def transform(self, dataset: BaseDataset, dropna_property: bool = True, **kwargs):
        cols = dataset.data.columns.tolist()
        def try_clean(s):
            try:
                return Chem.MolToSmiles(Chem.MolFromSmiles(s))
            except:
                return None
        start_num = len(dataset.data)
        dataset.data[dataset.structure_col] = dataset.data[dataset.structure_col].apply(lambda x: try_clean(x))
        dataset.data = dataset.data.dropna(subset=[dataset.structure_col])
        if hasattr(dataset, "property_col") and not dataset.property_col is None:
            dataset.data = dataset.data.dropna(subset=[dataset.property_col])
        dataset.data = dataset.data[cols]
        print(f"{start_num - len(dataset.data)} structure(s) were removed.")
        return dataset

class Discretize(BaseDatasetTransform):
    """ Discretize creates a new dataset from the original dataset by discretizing
    the property column.

    Parameters:
        prop_cutoff (float): where to threshold the property column.
        dir (str): Whether to have the 1 class be "smaller" o "larger" then the
            property value. Default, "larger".
    """

    @log_arguments
    def __init__(self, prop_cutoff: float, dir: str = "larger", log=True, **kwargs):
        self.prop_cutoff = prop_cutoff

    def transform(self, dataset: BaseDataset, **kwargs):
        dataset.data[dataset.property_col] = [1 if x > self.prop_cutoff else 0 for x in dataset.data[dataset.property_col]]
        return dataset

class OneHotEncode(BaseDatasetTransform):
    """ This one hot encodes a given feature column

    Parameters:
        feature_col (str): The feature column to one hot encode.
    """

    @log_arguments
    def __init__(self, feature_col: str, log=True, **kwargs):
        self.feature_col = feature_col

    def transform(self, dataset: BaseDataset, **kwargs):
        one_hot = pd.get_dummies(dataset.data[self.feature_col])
        dataset.data = dataset.data.join(one_hot)
        dataset.feature_cols += one_hot.columns.tolist()
        return dataset