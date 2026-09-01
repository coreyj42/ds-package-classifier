from sklearn.pipeline import Pipeline as Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import Optional, Union, Protocol
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from ds_package_classifier.base import BaseClassifier


class ModelProtocol(Protocol):
    """
    Protocol defining the required interface for model objects used in MLClassifier.

    __DESCRIPTION__
    This protocol enforces that any model passed to MLClassifier must implement
    the `fit` and `predict` methods. It enables structural typing, allowing flexibility
    without requiring inheritance from a specific base class.

    __REQUIRED METHODS__
    - fit(X, y): Trains the model on the provided features and labels.
    - predict(X): Generates predictions for the given input features.
    """

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray, list]
    ) -> None:
        pass

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[np.ndarray, list]:
        pass


class MLClassifier(BaseClassifier):
    """
    Classifier based on scikit-learn fit/predict format.

    __DESCRIPTION__
    This class wraps a model and optional preprocessing steps into a unified pipeline.
    It supports training, prediction, probability estimation, and cross-validation prediction.
    If a ColumnTransformer is provided, it is combined with the model into a scikit-learn Pipeline.

    __PARAMETERS__
    - df: pd.DataFrame
        A pandas DataFrame containing the full dataset.
    - target_column: str
        The name of the column in `df` representing the target variable.
    - model: ModelProtocol
        A model object conforming to ModelProtocol (must implement `fit` and `predict`).
    - column_transformer: ColumnTransformer
        Optional scikit-learn ColumnTransformer for preprocessing features.
    - preprocessing_step: str
        Name of the preprocessing step in the pipeline (default: "preprocessing_step").
    - classifier_step: str
        Name of the classifier step in the pipeline (default: "classifier_step").
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        model: ModelProtocol,
        column_transformer: Optional[ColumnTransformer] = None,
        preprocessing_step: str = "preprocessing_step",
        classifier_step: str = "classifier_step",
    ):
        super().__init__(
            df,
            target_column,
            model,
            column_transformer=column_transformer,
            preprocessing_step=preprocessing_step,
            classifier_step=classifier_step,
        )

        if self.column_transformer:
            self.model = self._build_column_transformer_model_pipeline()

    def train(self):
        """
        Trains the model using the training data stored in `self.X` and `self.y`.
        """
        self.model.fit(self.X, self.y)

    def predict(
        self, X_pred: Union[pd.DataFrame, np.ndarray]
    ) -> Union[list, np.ndarray]:
        """
        Generates class label predictions for the given input features.

        __PARAMETERS__
        - X_pred: Union[pd.DataFrame, np.ndarray]
            Feature matrix for prediction.

        __RETURNS__
        - y_pred: Union[list, np.ndarray]
            Predicted class labels.
        """

        y_pred = self.model.predict(X_pred)
        return y_pred

    def predict_proba(
        self, X_pred: Union[pd.DataFrame, np.ndarray]
    ) -> Union[list, np.ndarray]:
        """
        Generates class probability predictions for the given input features.

        __PARAMETERS__
        - X_pred: Union[pd.DataFrame, np.ndarray]
            Feature matrix for prediction.

        __RETURNS__
        - y_pred_proba: Union[list, np.ndarray]
            Predicted class probabilities.
        """
        y_pred_proba = self.model.predict_proba(X_pred)
        return y_pred_proba

    def cross_val_predict(self, n_splits: int) -> Union[list, np.ndarray]:
        """
        Performs out-of-fold prediction using StratifiedKFold cross-validation.

        __PARAMETERS__
        - n_splits: int
            Number of folds for cross-validation.

        __RETURNS__
        - y_pred: Union[list, np.ndarray]
            Predicted class labels for each sample.
        """
        stratified_cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
        y_pred_cross_val = cross_val_predict(
            self.model, self.X, self.y, cv=stratified_cv
        )
        return y_pred_cross_val

    def cross_val_predict_proba(self, n_splits: int) -> Union[list, np.ndarray]:
        """
        Performs out-of-fold probability prediction using StratifiedKFold cross-validation.

        __PARAMETERS__
        - n_splits: int
            Number of folds for cross-validation.

        __RETURNS__
        - y_pred_proba: Union[list, np.ndarray]
            Predicted class probabilities for each sample.
        """
        stratified_cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
        y_pred_cross_val_proba = cross_val_predict(
            self.model, self.X, self.y, cv=stratified_cv, method="predict_proba"
        )
        return y_pred_cross_val_proba

    def predict_with_threshold(
        self, X_pred: Union[pd.DataFrame, np.ndarray], threshold: float
    ) -> Union[list, np.ndarray]:
        """
        Generates class label predictions for the given input features using a manually set threshold. Requires binary classification.

        __PARAMETERS__
        - X_pred: Union[pd.DataFrame, np.ndarray]
            Feature matrix for prediction.
        - threshold: float
            Threshold for which prediction probability must be greater than in order to be considered positive

        __RETURNS__
        - y_pred: Union[list, np.ndarray]
            Predicted class for each sample.
        """
        # Check for two classes
        unique_classes = np.unique(self.y)
        if len(unique_classes) != 2:
            raise ValueError(
                f"Binary classification expected, but found {len(unique_classes)} unique classes in training targets: {unique_classes}"
            )

        # Calculate predictions based on probability predictions which are greater than threshold
        y_pred_proba = self.predict_proba(X_pred)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        return y_pred

    # Combine preprocessing column transformer and model into a Pipeline
    def _build_column_transformer_model_pipeline(self) -> Pipeline:
        """
        Constructs a scikit-learn Pipeline that chains together a preprocessing step and a classifier model.

        __RETURNS__
        - pipeline: sklearn.pipeline.Pipeline
            A pipeline object that sequentially applies the preprocessing and classification steps.
        """

        # Check column_transformer is defined and not None
        if self.column_transformer is None:
            raise ValueError(
                'Child class must assign a non-null value to "self.column_transformer"'
            )

        return Pipeline(
            steps=[
                (self.preprocessing_step, self.column_transformer),
                (self.classifier_step, self.model),
            ]
        )
