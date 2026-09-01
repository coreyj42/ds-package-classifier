import pytest

import pandas as pd

from ds_package_classifier.utils import preprocessing


@pytest.fixture
def sample_target_column():
    return "target"


@pytest.fixture
def sample_invalid_multicategory_df():
    df_test = pd.DataFrame(
        {
            "a": [1, 3, 5, 2],
            "b": [["france", "spain", "germany"], ["germany", "spain"], [], ["spain"]],
            "c": [["visa", "mastercard"], ["paypal"], ["mastercard"], [1]],
        }
    )
    return df_test


def test_invalid_values_in_columns_to_binarize(sample_invalid_multicategory_df):
    with pytest.raises(
        ValueError,
        match=r"invalid values detected in columns to binarize: \['a', 'c'\]",
    ):
        preprocessing.binarize_list_features(
            sample_invalid_multicategory_df, ["a", "b", "c"]
        )


@pytest.fixture
def sample_multicategory_df():
    df_test = pd.DataFrame(
        {
            "a": [1, 3, 5, 2],
            "b": [["france", "spain", "germany"], ["germany", "spain"], [], ["spain"]],
            "c": [["visa", "mastercard"], ["paypal"], ["mastercard"], ["paypal"]],
        }
    )
    return df_test


def test_binarized_columns(sample_multicategory_df):
    df_binarized, expanded_multi_categorical_features = (
        preprocessing.binarize_list_features(sample_multicategory_df, ["b", "c"])
    )
    assert "a" in df_binarized.columns
    assert "b" not in df_binarized.columns
    assert "b_france" in df_binarized.columns
    assert df_binarized.shape == (4, 7)

    assert len(expanded_multi_categorical_features) == 6
    assert "b_france" in expanded_multi_categorical_features
    assert "c_visa" in expanded_multi_categorical_features
