from abc import ABC, abstractmethod
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import Optional
from copy import deepcopy


class BaseClassifier(ABC):
    """
    Abstract base class for building classifier models with optional preprocessing.

    __DESCRIPTION__:
    This class defines the core structure and interface for classifier implementations.
    It handles data preparation, stores model and preprocessing configuration, and enforces
    implementation of key methods such as training, prediction.

    __PARAMETERS__
    - df: pd.DataFrame:
        A pandas DataFrame containing the full dataset.
    - target_column: str
        The name of the column in `df` representing the target variable.
    - model
        A model object.
    - column_transformer: ColumnTransformer
        Optional scikit-learn ColumnTransformer for preprocessing features.
    - preprocessing_step: str
        Name of the preprocessing step in a pipeline (default: "preprocessing_step").
    - classifier_step: str
        Name of the classifier step in a pipeline (default: "classifier_step").
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        model,
        column_transformer: Optional[ColumnTransformer] = None,
        preprocessing_step: str = "preprocessing_step",
        classifier_step: str = "classifier_step",
    ):
        self.df = deepcopy(df)
        self.target_column = target_column
        self.model = model
        self.column_transformer = column_transformer
        self.preprocessing_step = preprocessing_step
        self.classifier_step = classifier_step

        # Define training features and target
        self.X = self.df.drop(columns=[self.target_column])
        self.y = list(self.df[self.target_column])

    @abstractmethod
    def train(self):
        """
        Trains the model using the training data stored in `self.X` and `self.y`.
        """
        pass

    @abstractmethod
    def predict(self, X_pred):
        """
        Generates class label predictions for the given input features.

        __PARAMETERS__
        - X_pred: A pandas DataFrame or NumPy array of input features.
        """
        pass

    @abstractmethod
    def predict_proba(self, X_pred):
        """
        Generates class probability estimates for the given input features.

        __PARAMETERS__:
        - X_pred: input features.
        """
        pass

    @abstractmethod
    def cross_val_predict(self, n_splits):
        """
        Performs out-of-fold prediction on the input features.

        __PARAMETERS__
        - n_splits: Number of folds for StratifiedKFold cross-validation.
        """
        pass

    @abstractmethod
    def cross_val_predict_proba(self, n_splits):
        """
        Performs out-of-fold prediction and returns predicted class probabilities.

        __PARAMETERS__
        - n_splits: Number of folds for StratifiedKFold cross-validation.
        """
        pass

    @abstractmethod
    def predict_with_threshold(self, X_pred, threshold):
        """Generates class label predictions for the given input features using a manually set threshold. Requires binary classification.

        __PARAMETERS__
        - X_pred
            Feature matrix for prediction.
        - threshold
            Threshold for which prediction probability must be greater than in order to be considered positive
        """
        pass
