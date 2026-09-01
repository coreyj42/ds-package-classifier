import numpy as np
import pandas as pd


# Check that all values in a DataFrame are numeric (allowing null values)
def check_numeric_or_null(df: pd.DataFrame) -> bool:
    def is_valid_value(x):
        try:
            # Accept nulls
            if pd.isnull(x):
                return True
            # Accept only scalar numeric types
            if isinstance(x, (int, float, np.integer, np.floating)):
                return True
            # Reject lists and all other types
            if isinstance(x, list):
                return False
            return False
        except Exception:
            return False

    return df.map(is_valid_value).all().all()


# Calculate imbalance ratio needed to give more weight to minority class
def calculate_imbalance_ratio(df: pd.DataFrame, target_column: str):
    imbalance_ratio = len(df[df[target_column] == 0]) / len(df[df[target_column] == 1])
    return imbalance_ratio


def get_invalid_columns_for_binarization(
    df: pd.DataFrame, columns_to_check: list
) -> list:
    """
    Checks whether all elements in each list within the specified columns are strings.

    __PARAMETERS__
    df: pd.DataFrame
        The input DataFrame
    columns_to_check: list
        List of column names to check.

    __RETURNS__
    list
        A list with column names of all columns which contain invalid values for binarization
    """
    invalid_columns = []
    for column in columns_to_check:
        for cell in df[column]:
            if isinstance(cell, (list, np.ndarray)):
                if not all(isinstance(item, str) for item in cell):
                    invalid_columns.append(column)
                    break
            else:
                invalid_columns.append(column)
                break

    return invalid_columns
