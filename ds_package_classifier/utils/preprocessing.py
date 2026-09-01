import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from copy import deepcopy

from ds_package_classifier.utils.utils import get_invalid_columns_for_binarization


def binarize_list_features(
    df: pd.DataFrame, multi_categorical_features: list, build_if_missing: list = []
):
    """
    Apply multi-label binary encoding to multi-categorical features (features with list values)

    __PARAMETERS__
    df: pd.DataFrame
        The input dataframe
    multi_categorical_features: list
        The columns to encode
    build_if_missing: list
        A list of column names to be guaranteed in the final dataframe. If the the encoding does not produce any columns in the list a column will be generated containing all zero values

    __RETURNS__
    pd.DataFrame
        A dataframe with multi-label binary encoding applied to specified columns
    list[str]
        Column names of new encoded features
    """

    # check for valid input
    invalid_columns = get_invalid_columns_for_binarization(
        df, multi_categorical_features
    )
    if len(invalid_columns) > 0:
        raise ValueError(
            f"invalid values detected in columns to binarize: {invalid_columns}"
        )

    # reset index to avoid issues with concat of binarized features
    df_tmp = deepcopy(df)
    df_tmp.reset_index(inplace=True, drop=True)

    mlb = MultiLabelBinarizer()
    binarized_dfs = []
    expanded_multi_categorical_features = []
    for multi_categorical_feature in multi_categorical_features:
        df_tmp_single_feature = deepcopy(df_tmp[[multi_categorical_feature]])
        tags_encoded = mlb.fit_transform(
            df_tmp_single_feature[multi_categorical_feature]
        )
        column_names = [f"{multi_categorical_feature}_{x}" for x in mlb.classes_]

        # convert binarized features to DataFrame
        df_binarized = pd.DataFrame(tags_encoded, columns=column_names)
        df_binarized.columns = [
            col.lower().replace(" ", "_") for col in df_binarized.columns
        ]
        binarized_dfs.append(df_binarized)

        # save new feature names
        expanded_multi_categorical_features += list(df_binarized.columns)

    df_tmp = pd.concat([df_tmp] + binarized_dfs, axis=1)
    df_tmp.drop(columns=multi_categorical_features, inplace=True)

    # check for missing multicategorical features
    for feature in build_if_missing:
        if feature not in df_tmp.columns:
            df_tmp[feature] = 0

    return df_tmp, expanded_multi_categorical_features
