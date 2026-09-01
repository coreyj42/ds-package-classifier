import pytest

import pandas as pd

from ds_package_classifier.utils import cleaning


@pytest.fixture
def sample_target_column():
    return "target"


@pytest.fixture
def sample_df_df():
    df_test = pd.DataFrame(
        {
            "a": [1.1, 2, 0.9, 1, 1.1, 0.9, 1, 1.2, 1, 0.9, 1, 99],
            "b": [99, 0, 2, 1, 2, 1, 2, 1, 1, 0, 2, 0],
            "c": [0, 1, 1, 1, 0, 99, 2, 0, 2, 2, 1, 1],
            "target": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        }
    )
    return df_test


def test_outlier_filtering_with_target_in_outlier_columns(
    sample_df_df, sample_target_column
):
    with pytest.raises(
        ValueError,
        match="The target column should not be present in the list of columns to search for outliers",
    ):
        cleaning.drop_outliers_iqr(
            sample_df_df,
            sample_target_column,
            outlier_columns=["a", sample_target_column],
        )


def test_basic_oultier_filtering(sample_df_df, sample_target_column):
    df_test = cleaning.drop_outliers_iqr(sample_df_df, sample_target_column)
    assert df_test.shape == (8, 4)


def test_outlier_filtering_specify_columns(sample_df_df, sample_target_column):
    df_test = cleaning.drop_outliers_iqr(
        sample_df_df, sample_target_column, outlier_columns=["a", "b"]
    )
    assert df_test.shape == (9, 4)


def test_outlier_filtering_with_withhold_class(sample_df_df, sample_target_column):
    df_test = cleaning.drop_outliers_iqr(
        sample_df_df, sample_target_column, withhold_classes=[2]
    )
    assert df_test.shape == (9, 4)


def test_outlier_filtering_with_increased_multiplier(
    sample_df_df, sample_target_column
):
    df_test = cleaning.drop_outliers_iqr(
        sample_df_df, sample_target_column, multiplier=1000
    )
    assert df_test.shape == (12, 4)


@pytest.fixture
def sample_df_outlier_with_strings():
    df_test = pd.DataFrame(
        {
            "a": [1.1, 2, 0.9, 1, 1.1, 0.9, 1, "1.2", 1, 0.9, 1, 99999],
            "b": [99999, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "c": [1, 1, 1, 1, 1, 99999, 1, 1, 1, 1, 1, 1],
            "target": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        }
    )
    return df_test


def test_outlier_filtering_with_string_value_in_outlier_data(
    sample_df_outlier_with_strings, sample_target_column
):
    with pytest.raises(
        ValueError,
        match="The DataFrame contains invalid values. Must be numeric or null",
    ):
        cleaning.drop_outliers_iqr(sample_df_outlier_with_strings, sample_target_column)


@pytest.fixture
def sample_df_outlier_with_lists():
    df_test = pd.DataFrame(
        {
            "a": [
                [1.1],
                [2],
                [0.9],
                [1],
                [1.1],
                [0.9],
                [1],
                [],
                [1],
                [0.9],
                [1],
                [99999],
            ],
            "b": [99999, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "c": [1, 1, 1, 1, 1, 99999, 1, 1, 1, 1, 1, 1],
            "target": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        }
    )
    return df_test


def test_outlier_filtering_with_list_value_in_outlier_data(
    sample_df_outlier_with_lists, sample_target_column
):
    with pytest.raises(
        ValueError,
        match="The DataFrame contains invalid values. Must be numeric or null",
    ):
        cleaning.drop_outliers_iqr(sample_df_outlier_with_lists, sample_target_column)
