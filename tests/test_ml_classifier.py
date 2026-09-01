import pytest
import pandas as pd
import numpy as np
import re
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as Pipeline
from sklearn.compose import ColumnTransformer

from ds_package_classifier.ml_classifier import MLClassifier


@pytest.fixture
def sample_target_column():
    return "target"


@pytest.fixture
def sample_df():
    df_test = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2, 2, 2],
            "b": [0, 0, 12, 10, 100, 130, 99, 122],
            "c": [1.3, 4.55, 2.11, 0.0, 2043.2, 222, 113, 922],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    return df_test


@pytest.fixture
def xgb_classifier(sample_df, sample_target_column):
    xgb_model = XGBClassifier()
    xgb_classifier = MLClassifier(sample_df, sample_target_column, xgb_model)
    return xgb_classifier


def test_pipeline_without_column_transformer_defined(xgb_classifier):
    with pytest.raises(
        ValueError,
        match='Child class must assign a non-null value to "self.column_transformer"',
    ):
        xgb_classifier._build_column_transformer_model_pipeline()


@pytest.fixture
def xgb_classifier_trained(xgb_classifier):
    xgb_classifier.train()
    return xgb_classifier


# Test the prediction function in MLClassifier
def test_predict_class_function(xgb_classifier_trained, sample_df):
    y_pred = xgb_classifier_trained.predict(xgb_classifier_trained.X)
    assert len(y_pred) == len(sample_df)
    assert np.all(np.isin(y_pred, [0, 1]))


# Test the predict_proba function in MLClassifier
def test_predict_proba_class_functions(xgb_classifier_trained, sample_df):
    y_pred_proba = xgb_classifier_trained.predict_proba(xgb_classifier_trained.X)

    # Check shape: (n_samples, n_classes)
    n_classes = len(np.unique(sample_df["target"]))
    assert y_pred_proba.shape == (len(sample_df), n_classes)

    # Check all values are floats
    assert np.issubdtype(y_pred_proba.dtype, np.floating)


# Test cross_val_predict function in MLClassifier
def test_cross_val_predict_class_functions(xgb_classifier_trained, sample_df):
    y_pred_cross_val = xgb_classifier_trained.cross_val_predict(2)
    assert len(y_pred_cross_val) == len(sample_df)
    assert np.all(np.isin(y_pred_cross_val, [0, 1]))


# Test the cross_val_predict_proba function in MLClassifier
def test_cross_val_predict_proba_class_functions(xgb_classifier_trained, sample_df):
    y_pred_cross_val_proba = xgb_classifier_trained.cross_val_predict_proba(2)

    # Check shape: (n_samples, n_classes)
    n_classes = len(np.unique(sample_df["target"]))
    assert y_pred_cross_val_proba.shape == (len(sample_df), n_classes)

    # Check all values are floats
    assert np.issubdtype(y_pred_cross_val_proba.dtype, np.floating)


# Test prediction with threshold for binary classification
def test_predict_with_threshold(xgb_classifier_trained):
    xgb_classifier_trained.train()
    y_pred = xgb_classifier_trained.predict_with_threshold(
        xgb_classifier_trained.X, threshold=0.0
    )

    # assert with threshold=0 all predictions are 1
    assert np.all(y_pred == 1)


# Sample DataFrame a containing categorical feature column. Used to test one-hot encoding.
@pytest.fixture
def sample_df_with_strings():
    df_test = pd.DataFrame(
        {
            "a": [
                "france",
                "spain",
                "spain",
                "spain",
                "germany",
                "germany",
                "germany",
                "germany",
            ],
            "b": [0, 0, 12, 10, 100, 130, 99, 122],
            "c": [1.3, 4.55, 2.11, 0.0, 2043.2, 222, 113, 922],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    return df_test


# Column transformer to one-hot encode categorical column
@pytest.fixture
def column_transformer():
    # Define numeric feature columns
    numeric_features = ["b", "c"]

    # Define categorical feature columns
    categorical_features = ["a"]

    # Build Column Transformer Steps
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Build column transformer
    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )
    column_transformer.set_output(transform="pandas")
    return column_transformer


# Xgb classifier pipeline trained with column containing categorical features
@pytest.fixture
def xgb_classifier_one_hot_pipeline_trained(
    sample_df, sample_target_column, column_transformer
):
    xgb_model = XGBClassifier()
    xgb_classifier = MLClassifier(
        sample_df,
        sample_target_column,
        xgb_model,
        column_transformer=column_transformer,
    )
    xgb_classifier.train()
    return xgb_classifier


# Test train and predict of model pipeline with one-hot-encoding column transformer
def test_one_hot_encoding_model_pipeline(xgb_classifier_one_hot_pipeline_trained):
    y_pred = xgb_classifier_one_hot_pipeline_trained.predict(
        xgb_classifier_one_hot_pipeline_trained.X
    )
    assert len(y_pred) == len(xgb_classifier_one_hot_pipeline_trained.X)


# Sample DataFrame containining three distinct class labels
@pytest.fixture
def sample_df_multiclass():
    df_test = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2, 2, 2],
            "b": [0, 0, 12, 10, 100, 130, 99, 122],
            "c": [1.3, 4.55, 2.11, 0.0, 2043.2, 222, 113, 922],
            "target": [0, 0, 0, 1, 1, 1, 2, 2],
        }
    )
    return df_test


# XGB classifier trained on multiclass data
@pytest.fixture
def xgb_classifier_multiclass_trained(sample_df_multiclass, sample_target_column):
    xgb_model = XGBClassifier()
    xgb_classifier = MLClassifier(sample_df_multiclass, sample_target_column, xgb_model)
    xgb_classifier.train()
    return xgb_classifier


# test threshold prediction fails with non-binary classification
def test_predict_with_threshold_with_multiclass_classifier(
    xgb_classifier_multiclass_trained,
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Binary classification expected, but found 3 unique classes in training targets: [0 1 2]"
        ),
    ):
        xgb_classifier_multiclass_trained.predict_with_threshold(
            xgb_classifier_multiclass_trained.X, threshold=0.0
        )
