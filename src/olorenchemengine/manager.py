from timeit import default_timer as timer
from typing import List

import olorenchemengine as oce
from .base_class import *
from .dataset import *

from tqdm import tqdm


def TOP_MODELS_ADMET() -> List[BaseModel]:
    """
    Returns a list of the top models from the ADMET dataset.

    Returns:
        List[BaseModel]: A list of the top models from the ADMET dataset.
    """
    df = pd.read_csv(download_public_file("ModelDatabase/ModelDatabase_small.csv"))

    return [oce.create_BC(mp) for mp in df["Parameters"]]


class BaseModelManager(BaseClass):
    """BaseModelManager is the base class for all model managers.

    Parameters:
        dataset (Dataset): The dataset to use for training and testing.
        metrics (List[str]): A list of metrics to use.
        verbose (bool): Whether or not to print progress.
        file_path (str): The path to the model_database.
        log (bool): Whether or not to log to the model_database.
    """

    @log_arguments
    def __init__(
        self,
        dataset: BaseDataset,
        metrics: List[str],
        file_path: str = None,
        primary_metric: str = None,
        verbose=True,
        log=True,
    ):
        self.dataset = dataset

        if not isinstance(metrics, list):
            metrics = [metrics]
        assert sum([m in metric_functions.keys() for m in metrics]) == len(
            metrics
        ), "metrics must be either defined classification or regression metrics"
        self.metrics = metrics

        # Parameters for saving and storing the best model
        if primary_metric is None:
            self.primary_metric = metrics[0]
        else:
            self.primary_metric = primary_metric
        self.best_model = None
        self.best_primary_metric = float("inf")

        self.model_database = pd.DataFrame(columns=["Model Name", "Model Parameters", "Fitting Time"] + self.metrics)

        self.file_path = file_path
        if not self.file_path is None and os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                d = pickle.load(f)
                self._load(d["instance_save"])

        self.verbose = verbose

    def get_dataset(self):
        return self.dataset

    def primary_metric(self):
        return self.primary_metric

    def run(self, models: Union[BaseModel, List[BaseModel]], return_models: bool = False):
        """Runs the model on the dataset and saves the results to the model_database.

        Parameters:
            models (Union[BaseModel,List[BaseModel]]): The model(s) to run.
            return_models (bool): Whether or not to return the trained models.
        """
        import joblib

        if not isinstance(models, list):
            models = [models]

        model_outputs = []
        primary_metric_outputs = []
        print(models)
        for model in tqdm(models) if self.verbose else models:
            model = model.copy()
            model_parameters = parameterize(model)
            model_name = model_name_from_params(model_parameters)

            print("Fitting model")
            start = timer()
            model.fit(*self.dataset.train_dataset)
            end = timer()
            print("Evaluating model")

            if len(self.dataset.valid_dataset[0]) > 0:
                eval_set = self.dataset.valid_dataset
            else:
                eval_set = self.dataset.test_dataset

            y_pred = model.predict(eval_set[0])
            y = eval_set[1]

            stats = {metric: metric_functions[metric](y, y_pred) for metric in self.metrics}

            if not self.file_path is None and os.path.exists(self.file_path):
                with open(self.file_path, "rb") as f:
                    d = pickle.load(f)
                    self._load(d["instance_save"])

            self.model_database = self.model_database.append(
                {
                    "Model Name": model_name,
                    "Model Parameters": str(model_parameters),
                    "Fitting Time": end - start,
                    **stats,
                },
                ignore_index=True,
            )

            train_dataset = self.dataset.train_dataset
            dataset_hash = joblib.hash(train_dataset[0]) + joblib.hash(train_dataset[1])

            # LOGGING START

            import requests

            requests.get(
                f"https://api.oloren.ai/firestore/log_model_performance",
                params={
                    "name": model.name,
                    "dataset_hash": dataset_hash,
                    "metrics": json.dumps(stats),
                    "metric_direction": json.dumps(metric_direction),
                    "params": json.dumps(parameterize(model)),
                },
            )

            # LOGGING END

            # Using metric direction,
            if metric_direction[self.primary_metric] == "higher":
                primary_metric_outputs.append(-1 * metric_functions[self.primary_metric](y, y_pred))
            else:
                primary_metric_outputs.append(metric_functions[self.primary_metric](y, y_pred))
            if primary_metric_outputs[-1] < self.best_primary_metric:
                self.best_model = model
                self.best_primary_metric = primary_metric_outputs[-1]

            if not self.file_path is None:
                with open(self.file_path, "w+") as f:
                    oce.save(self, self.file_path)

            if return_models:
                model_outputs.append(model)

        if return_models:
            if len(model_outputs) == 1:
                return primary_metric_outputs[0], model_outputs[0]
            else:
                return primary_metric_outputs, model_outputs
        else:
            if len(primary_metric_outputs) == 1:
                return primary_metric_outputs[0]
            else:
                return primary_metric_outputs

    def get_model_database(self):

        return self.model_database


class ModelManager(BaseModelManager):
    """ModelManager is the class that tracks model development against a specified
    dataset. It is responsible for saving parameter settings and metrics.

    Parameters:
        dataset (BaseDataset): The dataset to use for model development.
        metrics (list[Str]): The metrics to track e.g. ROC AUC, Root Mean Squared Error.
        autosave (str): The path to save the model manager to. Optional.
    """

    @log_arguments
    def __init__(
        self,
        dataset: BaseDataset,
        metrics: List[str],
        file_path: str = None,
        primary_metric: str = None,
        verbose=True,
        log=True,
    ):
        self.name = oce.model_name_from_model(self)
        super().__init__(
            dataset, metrics, file_path=file_path, primary_metric=primary_metric, verbose=verbose, log=False
        )

    def _save(self) -> dict:
        return {
            "model_database": self.model_database.to_dict(orient="records"),
            "best_model_params": parameterize(self.best_model),
            "best_model": self.best_model._save(),
            "best_primary_metric": self.best_primary_metric,
        }

    def _load(self, d: dict) -> None:
        self.model_database = pd.DataFrame(d["model_database"])
        if "best_model_params" in d:
            self.best_model = create_BC(d["best_model_params"])
            self.best_model._load(d["best_model"])
            self.best_primary_metric = d["best_primary_metric"]
        else:
            self.best_model = None
            self.best_primary_metric = float("inf")


class SheetsModelManager(BaseModelManager):
    """SheetsModelManager is the class that tracks model development against a specified
    dataset on Google Sheets. It is responsible for saving parameter settings and metrics.

    Parameters:
        dataset (BaseDataset): The dataset to use for model development.
        metrics (list[Str]): The metrics to track e.g. ROC AUC, Root Mean Squared Error.
        name (str): The name of the Google Sheets to save this to. Optional.
        email (str): The email to share the results to. Optional, Default is
            share to anyone with the link.
    """

    @log_arguments
    def __init__(
        self,
        dataset: BaseDataset,
        metrics: List[str],
        file_path: str = None,
        primary_metric: str = None,
        name: str = "SheetsModelManager",
        email: str = "",
        log=True,
    ):

        import gspread

        if "GOOGLE_CREDENTIALS_FILENAME" in oce.CONFIG:
            self.gc = gspread.oauth(credentials_filename=oce.CONFIG["GOOGLE_CREDENTIALS_FILENAME"],)
        elif "GOOGLE_SERVICE_ACC_FILENAME" in oce.CONFIG:
            self.gc = gspread.service_account(filename=oce.CONFIG["GOOGLE_SERVICE_ACC_FILENAME"])
        else:
            raise Exception(
                "GOOGLE_CREDENTIALS_FILENAME and GOOGLE_SERVICE_ACC_FILENAME not found in CONFIG, please set one of them.\
                https://docs.f.org/en/latest/oauth2.html#oauth-client-id"
            )

        try:
            self.wb = self.gc.open(name)
        except:
            self.wb = self.gc.create(name)

        try:
            self.ws = self.wb.worksheet(dataset.name)
        except:
            self.ws = self.wb.add_worksheet(title=dataset.name, rows=1, cols=1)

        self.email = email

        super().__init__(dataset, metrics, file_path=file_path, primary_metric=primary_metric, log=False)

    def _save(self) -> dict:
        self.wb.share(self.email, perm_type="anyone", role="writer")
        self.ws.update([self.model_database.columns.values.tolist()] + self.model_database.values.tolist())
        self.url = f"https://docs.google.com/spreadsheets/d/{self.wb._properties['id']}/edit#gid=0"
        print(f"Saved to Google Sheets. Please visit {self.url} to view the results.")

    def _load(self, d: dict) -> None:
        self.model_database = pd.DataFrame(self.ws.get_all_records())


class FirebaseModelManager(BaseModelManager):
    """FirebaseModelManager is a ModelManager that saves model parameters
    and performances to a Firebase database.

     A Firebase service account key in oce.CONFIG is required for database access.

    Model information is saved to a collection called 'models' in the database.
    For each document, the following is saved:
        - uid: the user id of the user associated with the model
        - did: the dataset_id of on which the model was trained
        - model_parameters: parameters of the BaseModel oce object
        - model_name
        - model_status
        - fit_time
        - metrics: model training metrics

    Dataset information is saved to a collection called 'datasets' in the database.
    For each document, the following is saved:
        - dataset: map representation of the BaseDataset oce object
        - hashed_dataset: md5 hash of the dataset data
        - uid: the user id of the user associated with the dataset

    Parameters:
        dataset (BaseDataset): The dataset to use for model development.
        metrics (list[Str]): The metrics to track e.g. ROC AUC, Root Mean Squared Error.
        file_path (str): The path to save the model manager to.
        uid (str): The user id associated with the model manager
    """

    @log_arguments
    def __init__(
        self,
        dataset: BaseDataset,
        metrics: List[str],
        uid: str,
        primary_metric: str = None,
        file_path: str = None,
        log=True,
    ):
        import firebase_admin
        from firebase_admin import credentials
        from firebase_admin import firestore

        if "FIREBASE_CREDENTIALS_CERTIFICATE" in oce.CONFIG:
            cred = credentials.Certificate(oce.CONFIG["FIREBASE_CREDENTIALS_CERTIFICATE"])
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
        else:
            raise Exception("Firebase credentials in CONFIG were not found or not properly set. ")

        super().__init__(dataset, metrics, file_path=file_path, primary_metric=primary_metric, log=False)
        self.uid = uid

    def run(self, models: Union[BaseModel, List[BaseModel]]):
        """
        Run the model manager on the given model(s).

        Parameters:
            models (BaseModel or list[BaseModel]): The model(s) to run.
        """
        super().run(models)

        model_database = self.get_model_database()
        dataset = self.dataset
        dataset_data = dataset.data
        hashed_dataset = joblib.hash(dataset_data)
        dataset_params = oce.pretty_params(dataset)
        data_map = {
            "name": dataset.name,
            "file_path": dataset.file_path,
            "structure_col": dataset.structure_col,
            "property_col": dataset.property_col,
            "feature_cols": dataset.feature_cols,
            "hashed_dataset": hashed_dataset,
            "dataset_params": dataset_params,
        }

        # create a new document in datasets
        dataset_ref = self.db.collection("datasets").document()
        dataset_ref.set(
            {"uid": self.uid, "data": data_map,}
        )

        # check if userID is already in users, if not, create a new document in users
        user_ref = self.db.collection("users").document(self.uid)
        if not user_ref.get().exists:
            user_ref.set(
                {"uid": self.uid,}
            )

        for model in model_database.to_dict("index").values():
            name: str = model["Model Name"]
            fit_time: float = model["Fitting Time"]
            parameters: dict = model["Model Parameters"]
            metric_results: dict = {
                key: value
                for key, value in model.items()
                if key not in ["Model Name", "Fitting Time", "Model Parameters"]
            }

            model_map = {
                "status": "Trained",
                "name": name,
                "parameters": json.dumps(parameters),
                "metrics": metric_results,
                "fit_time": fit_time,
            }

            # create a new document in models
            model_ref = self.db.collection("models").document()
            model_ref.set(
                {"uid": self.uid, "did": dataset_ref.id, "data": model_map,}
            )

    def _save(self) -> dict:
        pass

    def _load(self, d: dict):
        pass
