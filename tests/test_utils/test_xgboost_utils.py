import pytest

import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline as Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml

from ds_package_classifier.utils import xgboost_utils
from ds_package_classifier.ml_classifier import MLClassifier
from ds_package_classifier.utils import preprocessing


@pytest.fixture
def sample_target_column():
    return "target"


@pytest.fixture
def sample_df():
    df_test = pd.DataFrame(
        {
            "a": [1, 1, 0],
            "b": [42, 189, 0],
            "c": [1.3, 2.11, 2043.2],
            "target": [1, 1, 0],
        }
    )
    return df_test


@pytest.fixture
def xgb_classifier(sample_df, sample_target_column):
    xgb_model = XGBClassifier()
    xgb_classifier = MLClassifier(sample_df, sample_target_column, xgb_model)
    return xgb_classifier


# Test that error is raised when extract_most_used_categorical_features_from_pipeline() is called using a BaseClassifier that is missing a pipeline
def test_extract_most_used_categorical_features_without_column_transformer(
    xgb_classifier,
):
    with pytest.raises(
        TypeError,
        match="Expected ml_classifier's 'model' to be a scikit-learn Pipeline object",
    ):
        xgboost_utils.extract_most_used_categorical_features_from_pipeline(
            xgb_classifier, ["a", "b"], 5, "cat"
        )


# DataFrame containing census income data. Used as a sample dataframe that contains categorical features
@pytest.fixture
def sample_df_census_income():
    X, y = fetch_openml(
        name="adult", version=2, as_frame=True, return_X_y=True, n_retries=5
    )
    df_test = X[["workclass", "sex", "hours-per-week"]].copy()

    # encode y to contain numeric values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    df_test["target"] = y_encoded

    return df_test


# Numeric census data features to use in test
@pytest.fixture
def numeric_features_census_income():
    return ["hours-per-week"]


# Categorical census data features to use in test
@pytest.fixture
def categorical_features_census_income():
    return ["workclass", "sex"]


# Define a column transformer to be used in a model pipeline for predictions on census data
@pytest.fixture
def column_transformer_census_income(
    numeric_features_census_income, categorical_features_census_income
):

    # Preprocessing
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    column_transformer_census_income = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features_census_income),
            ("num", numeric_transformer, numeric_features_census_income),
        ]
    )
    column_transformer_census_income.set_output(transform="pandas")
    return column_transformer_census_income


# Define a logistic regression model pipeline to test xhboost utils ability to detect invalid models
@pytest.fixture
def lr_classifier_pipeline(
    sample_df_census_income, sample_target_column, column_transformer_census_income
):
    lr_model = LogisticRegression()
    lr_classifier = MLClassifier(
        sample_df_census_income,
        sample_target_column,
        lr_model,
        column_transformer=column_transformer_census_income,
    )
    return lr_classifier


# Test using logistic regression model pipeline raises error
def test_extract_most_used_categorical_features_from_non_xgboost_pipeline(
    lr_classifier_pipeline,
):
    with pytest.raises(
        TypeError,
        match="Expected ml_classifier's model to be an instance of XGBClassifier, but got LogisticRegression instead",
    ):
        xgboost_utils.extract_most_used_categorical_features_from_pipeline(
            lr_classifier_pipeline, ["a", "b"], 5, "cat"
        )


# Define an XGBoost classifier pipeline using the census_income sample DataFrame. Use to test XGBoost utils when applied to a valid model.
@pytest.fixture
def xgb_classifier_pipeline_census_income(
    sample_df_census_income, sample_target_column, column_transformer_census_income
):
    xgb_model = XGBClassifier()
    xgb_classifier = MLClassifier(
        sample_df_census_income,
        sample_target_column,
        xgb_model,
        column_transformer=column_transformer_census_income,
    )
    return xgb_classifier


# Test the extraction of most used categorical features from xgboost model pipeline
def test_extract_most_used_categorical_features_from_pipeline(
    xgb_classifier_pipeline_census_income, categorical_features_census_income
):
    xgb_classifier_pipeline_census_income.train()
    top_categories = xgboost_utils.extract_most_used_categorical_features_from_pipeline(
        xgb_classifier_pipeline_census_income, categorical_features_census_income, 4
    )

    assert len(top_categories) == 2
    assert len(top_categories[0]) == 3
    assert len(top_categories[1]) == 1


# Define a DataFrame containing list values (multi-categorical values)
@pytest.fixture
def sample_df_multi_categorical():
    df_test = pd.DataFrame(
        {
            "a": [1, 3, 5, 2, 3, 2, 5, 2, 1, 1],
            "b": [
                ["france", "spain"],
                ["france", "spain"],
                ["france"],
                ["france"],
                ["france", "ireland"],
                ["germany", "spain"],
                ["germany", "spain"],
                ["germany"],
                ["germany", "ireland"],
                ["germany"],
            ],
            "c": [1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            "target": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
    )
    return df_test


# Test extraction of most useful binarized list features (multi-categorical features) from xgboost model pipeline
def test_extract_most_used_multi_categorical_features_from_pipeline(
    sample_df_multi_categorical, sample_target_column
):

    # Binarize list features to create encoded columns containing 1s and 0s
    df_binarized, _ = preprocessing.binarize_list_features(
        sample_df_multi_categorical, ["b"]
    )

    # Column names containing numeric values
    numeric_features = "a", "c"

    # Column names of binarized list value columns
    multi_categorical_features = "b_france", "b_germany", "b_spain", "b_ireland"

    # Build column Transformer step of model pipeline
    column_transformer = ColumnTransformer(
        transformers=[
            ("multi_cat", "passthrough", multi_categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )
    column_transformer.set_output(transform="pandas")

    # Initialize classifier
    xgb_model = XGBClassifier()
    xgb_classifier = MLClassifier(
        df_binarized,
        sample_target_column,
        xgb_model,
        column_transformer=column_transformer,
    )

    # Fit the classfier
    xgb_classifier.train()

    # Extract the most used multi-categorical features
    top_categories = (
        xgboost_utils.extract_most_used_multi_categorical_features_from_pipeline(
            xgb_classifier, 2, multi_categorical_transformer_name="multi_cat"
        )
    )

    assert len(top_categories) == 2
    assert "b_france" in top_categories
    assert "b_germany" in top_categories
