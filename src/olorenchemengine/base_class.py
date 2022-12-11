""" base_class consists of the building blocks of models: the base classes models should extend from and relevant utility functions.
"""
from __future__ import annotations

import inspect
import io
import logging
import os
import sys
from abc import abstractmethod
from typing import Any, Callable, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

import olorenchemengine as oce
from olorenchemengine.internal import *

# List of metrics that can be used for evaluating classification models
classification_metrics = {
    "Average Precision": average_precision_score,
    "ROC-AUC": roc_auc_score,
}

# List of metrics that can be used for evaluating regression models
from scipy.stats import spearmanr

regression_metrics = {
    "r2": r2_score,
    "Spearman": lambda y, y_pred: spearmanr(y, y_pred)[0],
    "Explained Variance": explained_variance_score,
    "Max Error": max_error,
    "Mean Absolute Error": mean_absolute_error,
    "Mean Squared Error": mean_squared_error,
    "Root Mean Squared Error": lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
}

# Entire list of metrics that can be used for evaluation models
metric_functions = {**classification_metrics, **regression_metrics}

# Dictionary indicating whether a better model has a 'higher' metric value or a 'lower' metric value for a given materic
metric_direction = {
    "Average Precision": "higher",
    "ROC-AUC": "higher",
    "r2": "higher",
    "Explained Variance": "higher",
    "Max Error": "lower",
    "Mean Absolute Error": "lower",
    "Mean Squared Error": "lower",
    "Root Mean Squared Error": "lower",
    "Spearman": "higher",
}


class BaseObject(BaseClass):
    """BaseObject is the parent class for all classes which directly wrap some object to be saved via joblib.

    Attributes:
        obj (object): the object which is wrapped by the BaseObject
    """

    @log_arguments
    def __init__(self, obj=None):
        self.obj = obj

    def _save(self):
        import joblib

        b = io.BytesIO()
        joblib.dump(self.obj, b)
        return {"obj": b.getvalue()}

    def _load(self, d):
        import joblib

        super()._load(d)
        self.obj = joblib.load(io.BytesIO(d["obj"]))


class BaseEstimator(BaseObject):

    """Utility class used to wrap any object with a fit and predict method"""

    def fit(self, X, y):
        """Fit the estimator to the data

        Parameters:
            X (np.array): The data to fit the estimator to
            y (np.array): The target data to fit the estimator to
        Returns:
            self (object): The estimator object fit to the data
        """

        return self.obj.fit(X, y)

    def predict(self, X):
        """Predict the output of the estimator

        Parameters:
            X (np.array): The data to predict the output of the estimator on
        Returns:
            y (np.array): The predicted output of the estimator
        """
        pred = self.obj.predict(X)
        pred = np.array(pred)
        if pred.ndim == 1:
            pred = pred
        else:
            pred = np.array([pred])
        return pred


class LinearRegression(BaseEstimator):

    """Wrapper for sklearn LinearRegression"""

    @log_arguments
    def __init__(self, *args, **kwargs):
        from sklearn.linear_model import LinearRegression as LinearRegression_

        self.obj = LinearRegression_(*args, **kwargs)


class BasePreprocessor(BaseObject):
    """BasePreprocessor is the parent class for all preprocessors which transform the features or properties of a dataset.

    Methods:
        fit: fit the preprocessor to the dataset
        fit_transform: fit the preprocessor to the dataset return the transformed values
        transform: return the transformed values
        inverse_transform: return the original values from the transformed values
    """

    def fit(self, X):
        """Fits the preprocessor to the dataset.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The fit preprocessor instance
        """
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.obj.fit(X)

    def fit_transform(self, X):
        """Fits the preprocessor to the dataset and returns the transformed values.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The transformed values of the dataset as a numpy array
        """
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = self.obj.fit_transform(X)
        return X

    def transform(self, X):
        """Returns the transformed values of the dataset as a numpy array.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The transformed values of the dataset as a numpy array
        """
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = self.obj.transform(X)
        return X

    def inverse_transform(self, X):
        """Returns the original values from the transformed values.

        Parameters:
            X (np.ndarray): the transformed values

        Returns:
            The original values from the transformed values
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = np.array(X)
        return self.obj.inverse_transform(X)


class QuantileTransformer(BasePreprocessor):
    """QuantileTransformer is a BasePreprocessor which transforms a dataset by quantile transformation to specified distribution.

    Attributes:
        obj (sklearn.preprocessing.QuantileTransformer): the object which is wrapped by the BasePreprocessor
    """

    @log_arguments
    def __init__(
        self,
        n_quantiles=1000,
        output_distribution="normal",
        subsample=1e5,
        random_state=None,
    ):

        from sklearn.preprocessing import QuantileTransformer

        self.obj = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            subsample=subsample,
            random_state=random_state,
        )


class StandardScaler(BasePreprocessor):
    """StandardScaler is a BasePreprocessor which standardizes the data by removing the mean and scaling to unit variance.

    Attributes:
        obj (sklearn.preprocessing.StandardScaler): the object which is wrapped by the BasePreprocessor
    """

    @log_arguments
    def __init__(self, with_mean=True, with_std=True):

        from sklearn.preprocessing import StandardScaler

        self.obj = StandardScaler(with_mean=with_mean, with_std=with_std)


class LogScaler(BasePreprocessor):
    """LogScaler is a BasePreprocessor which standardizes the data by taking the log and then removing the mean and scaling to unit variance."""

    @log_arguments
    def __init__(self, with_mean=True, with_std=True):

        from sklearn.preprocessing import StandardScaler

        self.obj = StandardScaler(with_mean=with_mean, with_std=with_std)

    def fit(self, X):
        """Fits the preprocessor to the dataset.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The fit preprocessor instance
        """
        X = np.array(X)
        self.mean = np.mean(X)
        X = np.log10(X + self.mean + 1e-3)
        return self.obj.fit(X.reshape(-1, 1))

    def fit_transform(self, X):
        """Fits the preprocessor to the dataset and returns the transformed values.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The transformed values of the dataset as a numpy array
        """
        X = np.array(X)
        self.mean = np.mean(X)
        X = np.log10(X + self.mean + 1e-3)
        return self.obj.fit_transform(X.reshape(-1, 1)).reshape(-1)

    def transform(self, X):
        """Returns the transformed values of the dataset as a numpy array.

        Parameters:
            X (np.ndarray): the dataset

        Returns:
            The transformed values of the dataset as a numpy array
        """
        X = np.log10(np.array(X) + self.mean + 1e-3)
        return self.obj.transform(X.reshape(-1, 1)).reshape(-1)

    def inverse_transform(self, X):
        """Returns the original values from the transformed values.

        Parameters:
            X (np.ndarray): the transformed values

        Returns:
            The original values from the transformed values
        """
        X = np.array(X)
        return (
            10 ** (self.obj.inverse_transform(X.reshape(-1, 1)).reshape(-1))
            - self.mean
            - 1e-3
        )

    def _save(self):
        d = super()._save()
        d["mean"] = self.mean
        return d

    def _load(self, d):
        super()._load(d)
        self.mean = d["mean"]
        return self


def detect_setting(data):
    values, _ = np.unique(data, return_counts=True)
    if len(values) <= 2:
        return "classification"
    else:
        return "regression"


class BaseModel(BaseClass):
    """
    BaseModel for training and evaluating different models

    Parameters:
        normalization (BasePreprocessor or str): the normalization to be used for the data
        setting (str): whether the model is a "classification" model or a "regression" model.
            Default is "auto" which automatically detects the setting from the dataset.
        model_name (str): the name of the model. Default is None, which instructs
            BaseModel to use `model_name_from_model` to select the name of the model.

    Attributes:
        setting (str): whether the model is a "classification" model or a "regression" model
        model_name (str): the name of the model, either assigned or a hashed name

    Methods:
        preprocess: preprocess the inputted data into the appropriate format
        _fit: fit the model to the preprocessed data, to be used internally implemented by child classes
        fit: fit the model to the inputted data, user can specify if they want regression or classification using the setting parameter.
        _predict: predict the properties of the inputted data, to be used internally implemented by child classes
        predict: predict the properties of the inputted data
        test: test the model on the inputted data, output metrics and optionally predicted values
        copy: returns a copy of the model (internal state not copied)
    """

    def __init__(self, normalization="zscore", setting="auto", name=None, **kwargs):

        self.normalization = normalization
        self.setting = setting
        if name is None:
            self.name = model_name_from_model(self)
        else:
            self.name = name

        self.calibrator = None

    def preprocess(self, X, y, fit=False):
        """

        Args:
            X (list of smiles)

        Returns:
            Processed list converted into whatever input for the model
        """
        return X

    def visualize_parameters_ipynb(self):
        from urllib.parse import quote

        from IPython.display import IFrame, display

        parameters = parameterize(self)
        display(
            IFrame(
                f"https://oas.oloren.ai/jsonvis?json={quote(json.dumps(parameters))}",
                width=800,
                height=500,
            )
        )

    @abstractmethod
    def _fit(self, X_train, y_train: np.ndarray) -> None:
        """To be implemented by the child class; fits the model on the provided dataset given preprocessed features.

         Note: _fit shouldn't be directly called by the user, rather it should be called indirectly via the fit method.

        Args:
            X_train: preprocessed features
            y_train (np.ndarray): values to predict
        """
        pass

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, list, np.ndarray],
        valid: Tuple[
            Union[pd.DataFrame, np.ndarray], Union[pd.Series, list, np.ndarray]
        ] = None,
        error_model: BaseErrorModel = None,
    ):
        """Calls the _fit method of the model to fit the model on the provided dataset.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): Input data to be fit on (structures + optionally features) e.g. a pd.DataFrame containing a "smiles" column or a list of experimental data
            y_train (Union[pd.Series, list, np.ndarray]): Values to predict from the input data
            valid (Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, list, np.ndarray]]): Optional validation data, which can be used as with methods like early stopping and model averaging.
            error_model (BaseErrorModel): Optional error model, which can be used to predict confidence intervals.
        """
        y_train = np.array(y_train)

        X_train_original = X_train
        y_train_original = y_train

        if not valid is None:
            X_valid, y_valid = valid

        # Pick setting and normalize y-values as appropriate
        if self.setting == "auto":
            self.setting = detect_setting(y_train)
        if self.setting == "classification":
            y_train = np.array(y_train).astype(int)
            if not valid is None:
                y_valid = np.array(y_valid).astype(int)
            logging.info("Classification dataset detected")
        elif self.setting == "regression":
            if self.normalization == "zscore":
                self.ymean = np.mean(y_train)
                self.ystd = np.std(y_train)
                y_train = (y_train - self.ymean) / self.ystd
                if not valid is None:
                    y_valid = (y_valid - self.ymean) / self.ystd
            elif issubclass(type(self.normalization), BasePreprocessor):
                self.normalization = self.normalization.copy()
                y_train = self.normalization.fit_transform(y_train)
                y_train = np.array(y_train)
                if not valid is None:
                    y_valid = self.normalization.transform(y_valid)
                    y_valid = np.array(y_valid)
            if (np.isreal(y_train.any()) == False) & (str(y_train).find("%") == -1):
                logging.warning(
                    " Dataset contains non-numeric values, which is incompatible with regression."
                )
                y_train.str.replace(r"[^0-9.%]", "", regex=True)
            logging.info("Regression dataset detected")
        else:
            logging.error(
                "Setting was not defined as either auto, regression, or classification."
            )
            sys.exit(1)

        # Train model
        X_train = self.preprocess(X_train, y_train, fit=True)
        if not valid is None:
            X_valid = self.preprocess(X_valid, y_valid, fit=False)

        self._fit(X_train, y_train)

        # Calibrate model
        if not valid is None and self.setting == "classification":
            self.calibrate_model(X_valid, y_valid)

        # Build error model
        if not error_model is None:
            error_model.build(self, X_train_original, y_train_original)
            self.error_model = error_model

    def calibrate(self, X_valid, y_valid):
        y_pred_valid = self.predict(X_valid)

        from sklearn.calibration import calibration_curve

        prob_pred, prob_true = calibration_curve(
            y_valid, y_pred_valid, n_bins=min(len(y_valid), 10)
        )
        prob_pred, prob_true = zip(
            *[x for x in zip(prob_pred, prob_true) if 0 < x[1] < 1]
        )
        prob_true = np.array(prob_true)
        prob_true = np.log(prob_true / (1 - prob_true))
        prob_pred = np.array(prob_pred)
        if np.unique(prob_true).shape[0] > 1 and np.unique(prob_pred).shape[0] > 1:
            self.calibrator = LinearRegression()
            self.calibrator.fit(prob_pred.reshape(-1, 1), prob_true.reshape(-1, 1))

    def fit_class(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, list, np.ndarray],
        valid: Tuple[
            Union[pd.DataFrame, np.ndarray], Union[pd.Series, list, np.ndarray]
        ] = None,
    ):
        from sklearn.model_selection import KFold
        from tqdm import tqdm

        kf = KFold(n_splits=5)
        y_pred = list()
        y_true = list()
        for train, test in tqdm(kf.split(X_train)):
            X_fold_train = X_train.loc[train]
            y_fold_train = y_train.loc[train]
            X_fold_test = X_train.loc[test]
            y_fold_test = y_train.loc[test]
            self.fit(X_fold_train, y_fold_train)
            preds = self.predict(X_fold_test)
            if not isinstance(preds, list):
                preds = preds.tolist()
            y_pred += preds
            y_true += y_fold_test.tolist()

        from sklearn.calibration import calibration_curve

        prob_pred, prob_true = calibration_curve(
            y_true, y_pred, n_bins=min(len(y_true), 10)
        )
        prob_pred, prob_true = zip(
            *[x for x in zip(prob_pred, prob_true) if 0 < x[1] < 1]
        )
        prob_true = np.array(prob_true)
        prob_true = np.log(prob_true / (1 - prob_true))
        prob_pred = np.array(prob_pred)
        self.calibrator = LinearRegression()
        self.calibrator.fit(prob_pred.reshape(-1, 1), prob_true.reshape(-1, 1))

        self.fit(X_train, y_train)

    def _unnormalize(self, Y):
        if (
            self.normalization == "zscore"
            and hasattr(self, "ymean")
            and hasattr(self, "ystd")
        ):
            result = Y * self.ystd + self.ymean
        elif issubclass(type(self.normalization), BasePreprocessor):
            result = self.normalization.inverse_transform(Y)
        else:
            result = Y
        return result

    @abstractmethod
    def _predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """To be implemented by the child class; returns the predicted values for provided dataset given preprocessed features.

        Note: _predict shouldn't be directly called by the user, rather it should be called indirectly via the predict method.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data to be predicted (structures + optionally features), will be preprocessed by the preprocess method.

        Returns:
            np.ndarray: Predicted values for the provided dataset.
            Shape: (number of samples, number of predicted values)
        """
        pass

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, list, pd.Series],
        return_ci=False,
        return_vis=False,
        skip_preprocess=False,
        **kwargs,
    ) -> np.ndarray:
        """Calls the _predict method of the model and returns the predicted values for provided dataset.

        Args:
            X (Union[pd.DataFrame, np.ndarray, list, pd.Series]): Input data to be predicted (structures + optionally features), will be preprocessed by the preprocess method.
            return_ci (bool): If error model fitted, whether or not to return the confidence intervals.
            return_vis (bool): If error model fitted, whether or not to return BaseVisualization objects.

        Returns:
            If return_ci and return_vis are False:
                np.ndarray: Predicted values for the provided dataset.
                Shape: (number of samples, number of predicted values)
            If return_ci or return_vis are true:
                pd.DataFrame: Predicted values, confidence intervals, and/or error plots for the provided dataset.
        """
        X_original = X
        if not skip_preprocess:
            X = self.preprocess(X, None, fit=False)
        Y = self._predict(X, **kwargs)
        if self.setting == "regression":
            result = self._unnormalize(Y)
            if self.calibrator is not None:
                result = self.calibrator.predict(result.reshape(-1, 1)).reshape(-1)
            if return_ci or return_vis:
                assert hasattr(self.error_model, "reg"), "Error model is not fitted yet."
                result = pd.DataFrame({"predicted": result})
                errors = self.error_model.score(X_original)
                if return_ci:
                    result["ci"] = errors
                if return_vis:
                    from olorenchemengine.visualizations.visualization import (
                        VisualizeError,
                    )

                    ci = self.error_model.ci
                    result["vis"] = [
                        VisualizeError(
                            self.error_model.y_train,
                            float(result["predicted"][i]),
                            errors[i],
                            ci=ci,
                        )
                        for i in range(len(errors))
                    ]
            return result
        else:
            if self.calibrator is not None:
                Y = np.array(Y)
                Y = Y.reshape(-1, 1)
                Y = self.calibrator.predict(Y)
                return 1 / (1 + np.exp(-Y.flatten()))
            else:
                return Y

    def fit_cv(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, list, np.ndarray],
        n_splits: int = 5,
        error_model: BaseErrorModel = None,
        ci: float = 0.8,
        scoring: str = None,
        **kwargs
    ):
        """Trains a production-ready model.

        This method trains the model on the entire dataset. It also performs an
        intermediate cross-validation step over dataset to both generate test
        metrics across the entire dataset, as well as to generate information
        which is used to calibrate the trained model.

        Calibration means to ensure that the probabilities outputted by
        classifiers reflect true distributions and to create appropriate
        confidence intervals for regression data.

        Args:
            X_train (Union[pd.DataFrame, np.ndarray]): Input data to be fit on (structures + optionally features) e.g. a pd.DataFrame containing a "smiles" column or a list of experimental data
            y_train (Union[pd.Series, list, np.ndarray]): Values to predict from the input data
            n_splits (int): Number of cross validation splits, default 5
            error_model (BaseErrorModel): Optional error model, which can be used to predict confidence intervals.
            ci (float): the confidence interval predicted by the error model
            scoring (str): Metric function to use for scoring cross validation splits; must be in `metric_functions`

        Returns:
            list: Cross validation metrics for each split
        """

        self.fit(X, y, error_model=error_model)

        residuals = None
        scores = None
        pred = None
        true = None

        cross_val_metrics = []
        X = np.array(X).flatten()
        y = np.array(y).flatten()

        if scoring is None:
            if self.setting == "regression":
                scoring = "Explained Variance"
            elif self.setting == "classification":
                scoring = "Average Precision"

        from sklearn.calibration import calibration_curve
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_splits)

        split = 1
        for train_index, test_index in kf.split(X):
            print('evaluating split {} of {}'.format(split, n_splits))
            split += 1

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = self.copy()
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)

            if self.setting == "regression":
                y_pred_test = np.array(y_pred_test).flatten()
                if pred is None:
                    pred = y_pred_test
                    true = y_test
                else:
                    pred = np.concatenate((pred, y_pred_test))
                    true = np.concatenate((true, y_test))
                
                if hasattr(self, "error_model"):
                    em = type(self.error_model)(
                        *self.error_model.args, **self.error_model.kwargs
                    )
                    em.build(model, X_train, y_train)
                    scores_fold = em.calculate(X_test, y_pred_test)
                    if scores is None:
                        scores = scores_fold
                    else:
                        scores = np.concatenate((scores, scores_fold))
            elif self.setting == "classification":
                prob_pred, prob_true = calibration_curve(
                    y_test, y_pred_test, n_bins=min(len(y_test), 10)
                )
                prob_pred, prob_true = zip(
                    *[x for x in zip(prob_pred, prob_true) if 0 < x[1] < 1]
                )
                prob_true = np.array(prob_true)
                prob_true = np.log(prob_true / (1 - prob_true))
                prob_pred = np.array(prob_pred)
                if pred is None:
                    pred = prob_pred
                    true = prob_true
                else:
                    pred = np.concatenate((pred, prob_pred))
                    true = np.concatenate((true, prob_true))

            cross_val_metrics.append(metric_functions[scoring](y_test, y_pred_test))

        if self.setting == "regression":
            self.calibrator = LinearRegression()
            self.calibrator.fit(pred.reshape(-1, 1), true.reshape(-1, 1))
            pred = self.calibrator.predict(pred.reshape(-1, 1)).reshape(-1)
            true = true.reshape(-1)
            residuals = np.abs(pred - true)
            if hasattr(self, "error_model"):
                self.error_model._fit(residuals, scores, **kwargs)
        elif self.setting == "classification":
            self.calibrator = LinearRegression()
            self.calibrator.fit(pred.reshape(-1, 1), true.reshape(-1, 1))

        return cross_val_metrics

    def test(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, list, np.ndarray],
        values: bool = False,
        fit_error_model: bool = False,
    ) -> dict:
        """Tests the model on the provided dataset returning a dictionary of metrics and optionally the predicted values.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input test data to be predicted (structures + optionally features)
            y (Union[pd.Series, list, np.ndarray]): True values for the properties
            values (bool, optional): Whether or not to return the predicted values for the test data. Defaults to False.
            fit_error_model (bool): If present, whether or not to fit the error model on the test data.

        Returns:
            dict: Dictionary of metrics and optionally the predicted values
        """
        if fit_error_model and hasattr(self, "error_model"):
            self.error_model.fit(X, y)

        import json

        import joblib

        import joblib

        dataset_hash = joblib.hash(X) + joblib.hash(y)

        y_pred = self.predict(X)

        import pandas as pd

        if isinstance(y, pd.Series):
            y = y.tolist()

        d = {}

        if self.setting == "classification":
            d.update(
                {k: float(v(y, y_pred)) for k, v in classification_metrics.items()}
            )
        elif self.setting == "regression":
            d.update({k: float(v(y, y_pred)) for k, v in regression_metrics.items()})

        # LOGGING START
        import requests

        try:
            response = requests.get(
                f"https://api.oloren.ai/firestore/log_model_performance",
                params={
                    "name": self.name,
                    "dataset_hash": dataset_hash,
                    "metrics": json.dumps(d),
                    "params": json.dumps(parameterize(self)),
                },
            )
        except:
            pass
        # LOGGING END

        if values:
            d.update({"values": y_pred})

        return d

    def create_error_model(
        self, 
        error_model: BaseErrorModel, 
        X_train: Union[pd.DataFrame, np.ndarray, list, pd.Series], 
        y_train: Union[np.ndarray, list, pd.Series], 
        X_valid: Union[pd.DataFrame, np.ndarray, list, pd.Series] = None, 
        y_valid: Union[np.ndarray, list, pd.Series] = None,
        **kwargs
    ):
        """Initializes, builds, and fits an error model on the input value.

        The error model is built with the training dataset and fit via either a validation dataset
        or cross validation. The error model is stored in model.error_model.

        Args:
            error_model (BaseErrorModel): Error model type to be created
            X_train (array-like): Input data for model training
            y_train (array-like): Values for model training
            X_valid (array-like): Input data for error model fitting. If no value passed in, the 
                error model is fit via cross validation on the training dataset.
            y_valid (array-like): Values for error model fitting. If no value passed in, the 
                error model is fit via cross validation on the training dataset.
        
        Example
        ------------------------------
        import olorenchemengine as oce

        model = oce.RandomForestModel(representation = oce.MorganVecRepresentation(radius=2, nbits=2048), n_estimators = 1000)
        model.fit(train["Drug"], train["Y"])
        oce.create_error_model(model, oce.SDC(), train["Drug"], train["Y"], valid["Drug"], valid["Y"], ci = 0.95, method = "roll")
        model.error_model.score(test["Drug"])
        ------------------------------
        """
        self.error_model = type(error_model)(**kwargs)
        self.error_model.build(self, X_train, y_train)
        if X_valid is None or y_valid is None:
            self.error_model.fit_cv()
        else:
            self.error_model.fit(X_valid, y_valid)

    def _save(self) -> dict:
        d = {}
        if hasattr(self, "ymean") and hasattr(self, "ystd"):
            d.update({"ymean": self.ymean})
            d.update({"ystd": self.ystd})
        if hasattr(self, "setting"):
            d.update({"setting": self.setting})
        if not self.calibrator is None:
            d.update({"calibrator": self.calibrator._save()})
        if hasattr(self, "error_model"):
            d.update({"error_model": saves(self.error_model)})
        if hasattr(self, "normalization") and issubclass(
            type(self.normalization), BasePreprocessor
        ):
            d.update({"normalization": self.normalization._save()})
        return d

    def _load(self, d) -> None:
        if "ymean" in d.keys() and "ystd" in d.keys():
            self.ymean = d["ymean"]
            self.ystd = d["ystd"]
        if "setting" in d.keys():
            self.setting = d["setting"]
        if "calibrator" in d.keys():
            self.calibrator = LinearRegression()
            self.calibrator._load(d["calibrator"])
        if "error_model" in d.keys():
            error_model = loads(d["error_model"])
            error_model.model = self
            print("Rebuilding error model...")
            error_model._build()
            print("Refitting error model...")
            error_model._fit(error_model.residuals, error_model.scores)
            self.error_model = error_model
        if "normalization" in d.keys():
            self.normalization._load(d["normalization"])

    def copy(self) -> BaseModel:
        """returns a copy of itself

        Returns:
            BaseModel: copied instance of itself
        """
        if hasattr(self, "args"):
            return type(self)(*self.args, **self.kwargs)
        else:
            return type(self)()

    def upload_oas(self, fname: str = None):
        """uploads the BaseClass object to cloud for OAS access. Model must be trained.

        Args:
            fname (str) (optional): the file name to name the uploaded model file. If left empty/None, names the file with default name associated with the BaseClass object.
        """
        try:
            self.predict("C")
        except:
            raise Exception("Model for upload must be trained.")

        if not fname:
            fname = self.name

        out = oas_connector.upload_model(self, fname)
        print(f"Access your model at: https://oas.oloren.ai/endpoint/?mid={out[1].id}")


class MakeMultiClassModel(BaseModel):
    """
    Base class for extending the classification capabilities of BaseModel to more than two classes, e.g. classes {W,X,Y,Z}.
    Uses the commonly-implemented One-vs-Rest (OvR) strategy. For each classifier, the class is fitted against all the other classes. The probabilities are then normalized and compared for each class.

    Parameters:
        individual_classifier (BaseModel): Model for binary classification tasks,
            which is to be turned into a multi-class model.
    """

    @log_arguments
    def __init__(self, individual_classifier: BaseModel) -> None:
        super().__init__()
        self.classifier = individual_classifier
        self.classifiers = []

    def _fit(self, X_train, y_train: np.ndarray) -> None:
        """Empty method to correctly override the fit method of the parent BaseModel class."""
        pass

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, list, np.ndarray],
        valid: Tuple[
            Union[pd.DataFrame, np.ndarray], Union[pd.Series, list, np.ndarray]
        ] = None,
    ):

        # Outputs a warning if, at a glance, it doesn't make sense for the dataset to be a classification dataset. The 0.1 can be adjusted.
        if ((np.unique(y_train)).size / (y_train.size)) > 0.1:
            logging.warning(
                " # of unique values / # of total datapoints is above the default threshold of 0.1."
            )

        self.sorted_classes = np.unique(
            y_train
        )  # Sorts the unique values in the array. Ex. [Z,X,Y,Z,X,X,Y] -> [X,Y,Z]
        for classes in self.sorted_classes:
            y_train_class = (y_train == classes).astype(
                int
            )  # Sets the currently iterating class (ex. X) to 1 and all others (ex. Y,Z) to 0, converting the dataset to a binary classification problem.
            self.model = (
                self.classifier.copy()
            )  # Creates a copy of the user-specified model (ex. Random Forest)
            self.model.fit(
                X_train, y_train_class
            )  # Calls the BaseModel fit method, since we now just have a binary classification dataset.
            self.classifiers.append(
                self.model
            )  # Adds the results of the fitted model to an array for later analysis.

    def _predict(self, X: Union[pd.DataFrame, np.ndarray, list, pd.Series]):
        """Empty method to correctly override the predict method of the parent BaseModel class."""
        pass

    def predict(self, X: Union[pd.DataFrame, np.ndarray, list, pd.Series]):
        """
        Args:
            X (Union[pd.DataFrame, np.ndarray, list, pd.Series]): Input data to be predicted (structures + optionally features).

        Returns:
            pd.DataFrame: Predicted values for the provided dataset, with multiple columns for the 2+ different classes and each row representing a different prediction.
        """
        predictions = {}

        # Iterates through two arrays at once in order to build the predictions dictionary
        # The key is the class and the value is the binary classification prediction result for that class vs. all the others.
        for y_class, classifier in zip(self.sorted_classes, self.classifiers):
            predictions[y_class] = classifier.predict(X)

        predictions = pd.DataFrame(predictions)
        predictions = (
            predictions.T / predictions.sum(axis=1)
        ).T  # normalizes outputs between 0 and 1

        return predictions

    def _save(self):
        d = super()._save()
        d.update(
            {
                "classifiers": [saves(classifier) for classifier in self.classifiers],
                "sorted_classes": self.sorted_classes,
            }
        )

    def _load(self, d):
        super()._load(d)
        self.classifiers = [loads(classifier) for classifier in d["classifiers"]]
        self.sorted_classes = d["sorted_classes"]


class BaseSKLearnModel(BaseModel):
    """Base class for creating sklearn-type models, e.g. with a sklearn RandomForestRegressor and RandomForestClassifier.

    Attributes:
        representation (BaseRepresentation): Representation to be used to preprocess the input data.
        regression_model (BaseModel or BaseEstimator): Model to be used for regression tasks.
        classification_model (BaseModel or BaseEstimator): Model to be used for classification tasks.
    """

    @log_arguments
    def __init__(
        self, representation, regression_model, classification_model, log=True, **kwargs
    ):
        self.representation = representation
        self.regression_model = regression_model
        self.classification_model = classification_model
        super().__init__(log=False, **kwargs)

    def preprocess(self, X, y, fit=False):
        if self.representation is None:
            return X
        else:
            return self.representation.convert(X, fit=fit)

    def _fit(self, X_train, y_train):
        values, counts = np.unique(y_train, return_counts=True)
        if len(values) == 2:
            self.setting = "classification"
            self.classification_model.fit(X_train, y_train)
            self.model = self.classification_model
        else:
            self.setting = "regression"
            self.regression_model.fit(X_train, y_train)
            self.model = self.regression_model

    def _predict(self, X):
        return self.model.predict(X)

    def _save(self):
        d = super()._save()
        if self.setting == "classification":
            d.update(
                {"model": self.classification_model._save(), "setting": self.setting}
            )
        else:
            d.update({"model": self.regression_model._save(), "setting": self.setting})
        return d

    def _load(self, d):
        super()._load(d)
        if d["setting"] == "classification":
            self.classification_model._load(d["model"])
            self.model = self.classification_model
        else:
            self.regression_model._load(d["model"])
            self.model = self.regression_model

class BaseReduction(BaseClass):
    """
    BaseReduction for applying dimensionality reduction on high-dimensional data

    Parameters:
        n_components (int): the number of components to keep

    Methods:
        fit: fit the model with input data
        fit_transform: fit the model with and apply dimensionality reduction to input data
        transform: apply dimensionality reduction to input data
    """
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X):
        pass
    
    @abstractmethod
    def fit_transform(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass
    

class BaseSKLearnReduction(BaseReduction):
    """
    Base class for creating sklearn dimensionality reduction

    """
    def fit(self, X):
        return self.obj.fit(X)
    
    def fit_transform(self, X):
        return self.obj.fit_transform(X)

    def transform(self, X):
        return self.obj.transform(X)

# List of regression functions for fitting error models.
regression_functions = {
    "logistic":     lambda x, a, b, c, d, e: (a - b) / (1 + np.exp(-c * (x - d))) ** e + b,
    "exponential":  lambda x, a, b, c: a * np.exp(b * x) + c,
    "logarithmic":  lambda x, a, b, c: a * np.log(b * x + c),
    "power":        lambda x, a, b, c: a * x ** b + c,
    "linear":       lambda x, a, b: a * x + b,
}

# List of regression function initializations.
regression_init = {
    "logistic":     lambda X, y: [np.max(y), np.min(y), 1, (np.max(X) + np.min(X)) / 2, 1],
    "exponential":  lambda X, y: [1, 1, np.min(y)],
    "logarithmic":  lambda X, y: [1, 1, 1],
    "power":        lambda X, y: [1, 1, np.min(y)],
    "linear":       lambda X, y: [1, 1],
}

class BaseErrorModel(BaseClass):
    """Base class for error models.

    Estimates confidence intervals for trained oce models.

    Args:
        ci (float): desired confidence interval
        method ({'bin','qbin','roll'}): whether to fit the error model via binning, 
            quantile binning, or rolling quantile
        bins (int): number of bins for binned quantiles. If None, selects the number
            of points per bins as n^(2/3) / 2.
        window (int): number of points per window for rolling quantiles. If None,
            selects the number of points per window as n^(2/3) / 2.
        curvetype (str): function used for regression. If `auto`, the function is 
            chosen automatically to minimize the mse.

    Methods:
        build: builds the error model from a trained BaseModel and dataset
        _build: optionally implemented, error model-specific computations
        fit: fits confidence scores to a trained model and external dataset
        fit_cv: fits confidence scores to k-fold cross validation on the training dataset
        _fit: fits confidence scores to residuals
        calculate: calculates confidence scores from inputs
        score: returns confidence intervals on a dataset
    """

    @log_arguments
    def __init__(self, 
        ci: float = 0.95, 
        method: str = "qbin",
        curvetype: str = "auto",
        window: int = None,
        bins: int = None,
        log=True, 
        **kwargs):

        assert curvetype == "auto" or curvetype in regression_functions, "Regression function `{}` is invalid. Please choose one of `{}`, or `auto`.".format(curvetype, "`, `".join(regression_functions.keys()))

        self.ci = ci
        self.method = method
        self.window = window
        self.bins = bins
        self.curvetype = curvetype
        
    def build(
        self,
        model: BaseModel,
        X: Union[pd.DataFrame, np.ndarray, list, pd.Series],
        y: Union[np.ndarray, list, pd.Series],
        **kwargs
    ):
        """Builds the error model with a trained model and training dataset

        Parameters:
            model (BaseModel): trained model
            X (array-like): training features, list of SMILES
            y (array-like): training values
        """
        self.model = model
        self.X_train = X
        self.y_train = np.array(y).flatten()
        self.y_pred_train = np.array(self.model.predict(self.X_train)).flatten()
        self._build(**kwargs)
    
    def _build(self, **kwargs):
        """Optionally implemented by child class; builds the error model to prepare it for fitting.
        """
        pass

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, list, pd.Series],
        y: Union[np.ndarray, list, pd.Series]
    ) -> plotly.graph_objects.Figure:
        """Fits confidence scores to an external dataset

        Args:
            X (array-like): features, smiles
            y (array-like): true values
        
        Returns:
            plotly figure of fitted model against validation dataset
        """ 
        y_pred = np.array(self.model.predict(X)).flatten()
        residuals = np.array(y) - y_pred
        scores = self.calculate(X, y_pred)

        return self._fit(residuals, scores)

    def fit_cv(
        self,
        n_splits: int = 10
    ) -> plotly.graph_objects.Figure:
        """Fits confidence scores to the training dataset via cross validation.

        Args:
            n_splits (int): Number of cross validation splits, default 5

        Returns:
            plotly figure of fitted model against validation dataset
        """
        from sklearn.model_selection import KFold
        
        self.X_train = np.array(self.X_train).flatten()

        residuals = None
        scores = None
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        split = 1
        for train_index, test_index in kf.split(self.X_train):
            print('evaluating split {} of {}'.format(split, n_splits))
            split += 1

            X_train, X_test = self.X_train[train_index], self.X_train[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]
            model = self.model.copy()
            model.fit(X_train, y_train)
            
            y_pred_test = np.array(model.predict(X_test)).flatten()
            pred_error = np.abs(y_test - y_pred_test)
            if residuals is None:
                residuals = pred_error
            else:
                residuals = np.concatenate((residuals, pred_error))

            em = type(self)(*self.args, **self.kwargs)
            em.build(model, X_train, y_train)
            scores_fold = em.calculate(X_test, y_pred_test)
            if scores is None:
                scores = scores_fold
            else:
                scores = np.concatenate((scores, scores_fold))

        return self._fit(residuals, scores)

    def _fit(
        self,
        residuals: np.ndarray,
        scores: np.ndarray
    ) -> plotly.graph_objects.Figure:
        """Fits confidence scores to residuals.

        Args:
            residuals (1-dimensional np.ndarray): array of residuals
            scores (1-dimensional np.ndarray): array of confidence scores

        Returns:
            plotly figure of fitted model against validation dataset
        """
        if self.bins is None:
            self.bins = int(np.ceil(2 * len(residuals) ** (1/3)))
        if self.window is None:
            self.window = int(np.floor(0.5 * len(residuals) ** (2/3)))
        self.residuals = residuals
        self.scores = scores

        self._upper_reg, X_upper, y_upper = self._fit_regression(0.5 + self.ci / 2)
        self._lower_reg, X_lower, y_lower = self._fit_regression(0.5 - self.ci / 2)
        self.reg = lambda X: list(zip(self._lower_reg(X), self._upper_reg(X)))
        
        import plotly.express as px
        import plotly.graph_objects as go

        t = np.linspace(min(self.scores), max(self.scores), 100)

        upper_points = px.scatter(x = X_upper, y = y_upper)
        lower_points = px.scatter(x = X_lower, y = y_lower)
        upper_line = px.line(x=t, y=self._upper_reg(t))
        lower_line = px.line(x=t, y=self._lower_reg(t))
        all_points = px.scatter(x = scores, y = residuals)

        upper_points.update_traces(hovertemplate=None, hoverinfo="x+y", marker=dict(color="#0072B2", size=7))
        lower_points.update_traces(hovertemplate=None, hoverinfo="x+y", marker=dict(color="#0072B2", size=7))
        upper_line.update_traces(hovertemplate=None, hoverinfo="x+y", line=dict(color="#D55E00"))
        lower_line.update_traces(hovertemplate=None, hoverinfo="x+y", line=dict(color="#D55E00"))
        all_points.update_traces(hovertemplate=None, hoverinfo="skip", marker=dict(color="#009E73", size=2))

        fig = go.Figure(data=all_points.data + upper_line.data + lower_line.data + upper_points.data + lower_points.data)

        return fig

    def _fit_regression(self, quantile):
        """Fits a regression curve to a quantile of the fitting data.

        Args:
            quantile (float): fraction of data below the regression function

        Returns:
            X and y coordinates for regression and regression function
        """
        if self.method == "bin":
            bins = min(len(np.unique(self.scores)), self.bins)
            bin_labels = pd.cut(self.scores, bins, labels=False)
            labels = np.unique(bin_labels)
            X = [np.mean(self.scores[bin_labels == i]) for i in labels]
            y = [np.quantile(self.residuals[bin_labels == i], quantile) for i in labels]
        elif self.method == "qbin":
            bins = min(len(np.unique(self.scores)), self.bins)
            bin_labels = pd.qcut(self.scores, bins, labels=False, duplicates="drop")
            labels = np.unique(bin_labels)
            X = [np.mean(self.scores[bin_labels == i]) for i in labels]
            y = [np.quantile(self.residuals[bin_labels == i], quantile) for i in labels]
        elif self.method == "roll":
            window = min(len(self.scores), self.window)
            idxs = np.argsort(self.scores)
            X = pd.Series(self.scores[idxs]).rolling(window).mean()[window-1:]
            y = pd.Series(self.residuals[idxs]).rolling(window).quantile(quantile)[window-1:]
        else:
            raise NameError("Method {} is not recognized. Valid inputs are \"bin\", \"qbin\", and \"roll\".".format(self.method))
        
        X = np.array(X)
        y = np.array(y)

        if len(X) == 1:
            print("WARNING: Only one unique score computed. Please decrease bin/window size or use a larger dataset.")
            return np.vectorize(lambda x: y[0] if (x == X[0]) else np.nan), X, y

        from scipy.optimize import curve_fit
        
        if self.curvetype == "auto":
            min_mse = None
            for reg in regression_functions:
                func = regression_functions[reg]
                init = regression_init[reg](X, y)
                try:
                    params = curve_fit(func, X, y, p0=init, maxfev = 500 * len(init))[0]
                    y_pred = func(X, *params)
                    mse = mean_squared_error(y, y_pred)
                except (RuntimeError, TypeError, ValueError):
                    continue
                if min_mse is None or mse < min_mse:
                    opt_func = func
                    opt_params = params
                    min_mse = mse
            assert min_mse is not None, "Curve fit failed to find a suitable regression model."
        else:
            opt_func = regression_functions[self.curvetype]
            init = regression_init[self.curvetype](X, y)
            try:
                opt_params = curve_fit(opt_func, X, y, p0=init, maxfev = 500 * len(init))[0]
            except (RuntimeError, TypeError, ValueError):
                raise RuntimeError("Curve fit failed to fit regression model.")
        
        reg = lambda x: np.maximum(np.minimum(opt_func(x, *opt_params), np.max(self.residuals)), np.min(self.residuals))
        return reg, X, y

    @abstractmethod
    def calculate(
        self,
        X: Union[pd.DataFrame, np.ndarray, list, pd.Series],
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """To be implemented by the child class; calculates confidence scores from inputs.

        Args:
            X: features, list of SMILES
            y_pred (1-dimensional np.ndarray): predicted values

        Returns:
            scores (1-dimensional np.ndarray)
        """
        pass

    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray, list, pd.Series],
    ) -> np.ndarray:
        """Calculates confidence scores on a dataset.

        Args:
            X (array-like): target dataset, list of SMILES

        Returns:
            a list of confidence intervals as tuples for each input
        """
        assert hasattr(self, "reg"), "Error model not fitted yet."

        y_pred = np.array(self.model.predict(X)).flatten()
        scores = self.calculate(X, y_pred)

        return self.reg(scores)

    def copy(self) -> BaseErrorModel:
        """returns a copy of itself

        Returns:
            BaseErrorModel: copied instance of itself
        """
        if hasattr(self, "args"):
            return type(self)(*self.args, **self.kwargs)
        else:
            return type(self)()

    def _save(self) -> dict:
        d = {}
        if hasattr(self, "model") and not self is getattr(
            self.model, "error_model", None
        ):
            d.update({"model": saves(self.model)})
        if hasattr(self, "X_train"):
            d.update({"X_train": self.X_train})
            d.update({"y_train": self.y_train})
            d.update({"y_pred_train": self.y_pred_train})
        if hasattr(self, "residuals"):
            d.update({"residuals": self.residuals})
            d.update({"scores": self.scores})
        return d

    def _load(self, d) -> None:
        if "model" in d.keys():
            self.model = loads(d["model"])
        if "X_train" in d.keys():
            self.X_train = d["X_train"]
            self.y_train = d["y_train"]
            self.y_pred_train = d["y_pred_train"]
            if hasattr(self, "model"):
                print("Rebuilding error model...")
                self._build()
        if "residuals" in d.keys():
            self.residuals = d["residuals"]
            self.scores = d["scores"]
            if hasattr(self, "model"):
                print("Refitting error model...")
                self._fit(self.residuals, self.scores)