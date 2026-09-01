import pytest

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
import re

from ds_package_classifier.utils import metrics
from ds_package_classifier.ml_classifier import MLClassifier


@pytest.fixture
def sample_target_column():
    return "target"


# Multiclass data sample from sklearn iris dataset
@pytest.fixture
def sample_df_multiclass():
    data = load_iris()
    df_test = pd.DataFrame(data.data, columns=data.feature_names)
    df_test["target"] = data.target
    return df_test


# Classifier initialised and trained on multiclass iris data
@pytest.fixture
def trained_classifier_multiclass(sample_df_multiclass, sample_target_column):
    xgb_model = XGBClassifier()
    classifier = MLClassifier(sample_df_multiclass, sample_target_column, xgb_model)
    classifier.train()
    return classifier


# Predictions from multiclass classifier
@pytest.fixture
def predictions_multiclass(
    trained_classifier_multiclass, sample_df_multiclass, sample_target_column
):
    X = sample_df_multiclass.drop(columns=[sample_target_column])
    return trained_classifier_multiclass.predict(X)


# True labels from iris data
@pytest.fixture
def true_labels_multiclass(sample_df_multiclass, sample_target_column):
    return sample_df_multiclass[sample_target_column].values


# Binary class data sample from sklearn breast_cancer dataset
@pytest.fixture
def sample_df_binary_class():
    data = load_breast_cancer()
    df_test = pd.DataFrame(data.data, columns=data.feature_names)
    df_test["target"] = data.target
    return df_test


# Untrained classifier initialised with binary class data sample
@pytest.fixture
def classifier_binary_class(sample_df_binary_class, sample_target_column):
    xgb_model = XGBClassifier()
    classifier = MLClassifier(sample_df_binary_class, sample_target_column, xgb_model)
    return classifier


# Test calculation of evaluation metrics using predictions of multiclass classifier
def test_evaluation_metrics_calculations(
    predictions_multiclass, true_labels_multiclass
):
    accuracy, precision, recall, f1, conf_matrix = metrics.calculate_metrics(
        true_labels_multiclass, predictions_multiclass, average="macro"
    )

    assert isinstance(accuracy, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    assert conf_matrix.shape == (3, 3)


# Test calculation of evaluation metrics using cross-valuations predictions of multiclass classifier
def test_cross_validation_evaluation_metrics_calculations(
    trained_classifier_multiclass, true_labels_multiclass, sample_df_multiclass
):
    accuracy, precision, recall, f1, conf_matrix, y_pred = (
        metrics.get_cross_val_prediction_metrics(
            trained_classifier_multiclass,
            true_labels_multiclass,
            n_splits=5,
            average="macro",
        )
    )

    assert isinstance(accuracy, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    assert conf_matrix.shape == (3, 3)
    assert len(y_pred) == len(sample_df_multiclass)


# Test threshold scores calculation raises error when used with multiclass data sample
def test_get_top_binary_thresholds_with_non_binary_class(trained_classifier_multiclass):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Binary classification expected, but found 3 unique classes: [0 1 2]"
        ),
    ):
        metrics.get_top_binary_thresholds(
            trained_classifier_multiclass, beta_values=[2, 3], n_splits=5
        )


# Test threshold score calculations on binary class data sample
def test_get_top_binary_thresholds_with_binary_class(classifier_binary_class):
    best_thresholds = metrics.get_top_binary_thresholds(
        classifier_binary_class, beta_values=[2, 3], n_splits=5
    )
    assert len(best_thresholds) == 3
    assert "f1" in best_thresholds
    assert "fbeta_2" in best_thresholds
    assert "fbeta_3" in best_thresholds


# Test threshold evaluation metrics raises error when used with multiclass data sample
def test_get_binary_thresholds_cross_val_prediction_metrics_with_non_binary_class(
    trained_classifier_multiclass,
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Binary classification expected, but found 3 unique classes: [0 1 2]"
        ),
    ):
        thresholds = [0.25, 0.5, 0.75, 1.0]
        metrics.get_binary_thresholds_cross_val_prediction_metrics(
            trained_classifier_multiclass, thresholds=thresholds, n_splits=5
        )


#  Test threshold evaluation metrics on binary class data sample
def test_get_binary_thresholds_cross_val_prediction_metrics_with_binary_class(
    classifier_binary_class,
):
    thresholds = [0.1, 0.5, 0.9]
    df_threshold_metrics = metrics.get_binary_thresholds_cross_val_prediction_metrics(
        classifier_binary_class, thresholds=thresholds, n_splits=5
    )
    assert len(df_threshold_metrics) == 3

    # Check expected columns
    expected_columns = {"threshold", "accuracy", "precision", "recall", "f1"}
    assert len(df_threshold_metrics.columns) == len(expected_columns)
    assert set(df_threshold_metrics.columns) == expected_columns

    # Check all values are numeric and not null
    for column in df_threshold_metrics.columns:
        assert pd.api.types.is_numeric_dtype(df_threshold_metrics[column])
        assert df_threshold_metrics[column].notnull().all()
