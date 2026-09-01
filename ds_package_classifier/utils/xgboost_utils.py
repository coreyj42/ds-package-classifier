import pandas as pd
from sklearn.pipeline import Pipeline as Pipeline
from xgboost import XGBClassifier

from ds_package_classifier.ml_classifier import MLClassifier


def extract_feature_importance_from_pipeline(
    ml_classifier: MLClassifier,
) -> pd.DataFrame:
    """
    Returns the most useful features during XGBoost training

    __PARAMETERS__
    ml_classifier : MLClassifier
        An instance of MLClassifier that contains a trained scikit-learn Pipeline
        with preprocessing and an XGBClassifier model.

    __RETURNS__
    pd.DataFrame
        A DataFrame of feature importance
    """

    model_pipeline = ml_classifier.model

    # Check that model is a scikit-learn Pipeline
    if not isinstance(model_pipeline, Pipeline):
        raise TypeError(
            "Expected ml_classifier's 'model' to be a scikit-learn Pipeline object"
        )

    # Extract pipeline steps
    column_transformer = model_pipeline.named_steps[ml_classifier.preprocessing_step]
    xgb_model = model_pipeline.named_steps[ml_classifier.classifier_step]

    # Check that ml_classifier uses XGBClassifier for xgb_model
    if not isinstance(xgb_model, XGBClassifier):
        raise TypeError(
            f"Expected ml_classifier's model to be an instance of XGBClassifier, but got {type(xgb_model).__name__} instead"
        )

    # Get feature names from the preprocessor
    feature_names = column_transformer.get_feature_names_out()

    # Get feature importances from the XGBoost model
    importances = xgb_model.feature_importances_

    # Create a DataFrame to pair names and importances
    df_importances = pd.DataFrame({"feature": feature_names, "importance": importances})

    return df_importances


def extract_most_used_categorical_features_from_pipeline(
    ml_classifier: MLClassifier,
    original_features: list[str],
    top_n: int,
    categorical_transformer_name: str = "cat",
):
    """
    Identifies the most influential one-hot encoded categorical features used by an XGBoost model
    and returns them in a format suitable for feature selection via a column transformer.

    __PARAMETERS__
    ml_classifier : MLClassifier
        An instance of MLClassifier that contains a trained scikit-learn Pipeline
        with preprocessing and an XGBClassifier model.
    original_features : list[str]
        List of original categorical column names before encoding.
    top_n : int
        Number of top categorical features to extract based on model importance.
    categorical_transformer_name : str
        Prefix used in the pipeline for the categorical transformer (e.g., 'cat' for 'cat__feature_name_value').

    __RETURNS__
    list[list[str]]
        A list of lists, where each sublist contains the most important encoded values for each original categorical feature.
    """

    df_importances = extract_feature_importance_from_pipeline(ml_classifier)

    categorical_transformer_name = f"{categorical_transformer_name}__"

    # Filter for categorical features
    df_categorical_features = df_importances[
        df_importances["feature"].str.startswith(categorical_transformer_name)
    ]

    # Sort by importance and get top n
    df_top_n_categotrcial_features = df_categorical_features.sort_values(
        by="importance", ascending=False
    ).head(top_n)

    # Build a mapping from column name to its values
    column_value_map = {col: [] for col in original_features}

    for feature in df_top_n_categotrcial_features["feature"]:
        # Remove 'cat__' prefix
        encoded_feature = feature.replace(categorical_transformer_name, "")

        # Match the longest column name from categorical_features
        matched_column = max(
            (col for col in original_features if encoded_feature.startswith(col + "_")),
            key=len,
            default=None,
        )

        if matched_column:
            value = encoded_feature[len(matched_column) + 1 :]  # +1 for the underscore
            column_value_map[matched_column].append(value)

    # convert to list for encoder
    top_categories = [column_value_map[col] for col in original_features]

    return top_categories


def extract_most_used_multi_categorical_features_from_pipeline(
    ml_classifier: MLClassifier,
    top_n: int,
    multi_categorical_transformer_name: str = "multi_cat",
):
    """
    Identifies the most influential binarized multi-categorical features used by an XGBoost model.

    __PARAMETERS__
    ml_classifier : MLClassifier
        An instance of MLClassifier that contains a trained scikit-learn Pipeline
        with preprocessing and an XGBClassifier model.
    top_n : int
        Number of top multi-categorical features to extract based on model importance.
    multi_categorical_transformer_name : str
        Prefix used in the pipeline for the multi-categorical transformer (e.g., 'multi_cat' for 'multi_cat__feature_name_value').

    __RETURNS__
    list[list[str]]
        A list of lists, where each sublist contains the most important encoded values for each original categorical feature.
    """

    df_importances = extract_feature_importance_from_pipeline(ml_classifier)

    multi_categorical_transformer_name = f"{multi_categorical_transformer_name}__"

    df_multi_cat_features = df_importances[
        df_importances["feature"].str.startswith(multi_categorical_transformer_name)
    ]
    df_multi_cat_features

    # Sort by importance and get top n
    top_n_multi_cat = df_multi_cat_features.sort_values(
        by="importance", ascending=False
    ).head(top_n)
    top_binarized_multi_category_features = list(top_n_multi_cat.feature.values)

    # Remove the 'multi_cat__' prefix from each string
    top_binarized_multi_category_features = [
        name.replace("multi_cat__", "")
        for name in top_binarized_multi_category_features
    ]

    return top_binarized_multi_category_features
