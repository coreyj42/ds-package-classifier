import pandas as pd
from copy import deepcopy

from ds_package_classifier.utils import utils


# drop outliers from majority class
def drop_outliers_iqr(
    df: pd.DataFrame,
    target_column: str,
    outlier_columns: list = [],
    multiplier: float = 1.5,
    withhold_classes: list = [],
):
    """
    Drops rows from the dataframe where any of the specified columns contain outliers. Outliers are detected using the IQR method.

    __PARAMETERS__
    df: pd.DataFrame
        The input dataframe
    target_column: str
        Defines the target column of the DataFrame
    outlier_columns: list
        List of column names to check for outliers
    multiplier: float
        Controls the aggressiveness of the outlier filtering.
    withhold_classes: list
        Values in the target column to withhold from outlier filtering

    __RETURNS__
    pd.DataFrame
        A dataframe with outliers removed
    """

    df_tmp = deepcopy(df)

    # if no outlier columns defined, check all columns
    if len(outlier_columns) == 0:
        outlier_columns = df_tmp.drop(columns=target_column).columns

    # check df values of outlier columns are all numeric (allowing null values)
    if not utils.check_numeric_or_null(df_tmp[outlier_columns]):
        raise ValueError(
            "The DataFrame contains invalid values. Must be numeric or null"
        )

    # check target column is not in list of columns to check for outliers
    if target_column in outlier_columns:
        raise ValueError(
            "The target column should not be present in the list of columns to search for outliers"
        )

    # Separate withheld classes
    df_withheld = df_tmp[df_tmp[target_column].isin(withhold_classes)]
    df_not_withheld = df_tmp[~(df_tmp[target_column].isin(withhold_classes))]

    # Apply IQR filtering to specified rows
    outlier_indices = set()
    for column in outlier_columns:
        if column in df_not_withheld.columns:
            Q1 = df_not_withheld[df_not_withheld[column].notna()][column].quantile(0.25)
            Q3 = df_not_withheld[df_not_withheld[column].notna()][column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            # get row indices of outliers
            df_outliers = df_not_withheld[
                (df_not_withheld[column] < lower_bound)
                | (df_not_withheld[column] > upper_bound)
            ]
            outlier_indices = outlier_indices.union(set(df_outliers.index))

    df_not_withheld = df_not_withheld.drop(index=outlier_indices)

    # Combine filtered rows with withheld rows
    df_result = pd.concat([df_withheld, df_not_withheld], ignore_index=True)

    return df_result
