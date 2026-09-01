import pytest
import pandas as pd

from ds_package_classifier.base import BaseClassifier


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


class MockModel:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class MockClassifier(BaseClassifier):
    def __init__(self, df, target_column, model):
        super().__init__(df, target_column, model)

    def train(self):
        return self

    def predict(self, y):
        return list(self.df[self.target])

    def predict_proba(self, X_pred):
        y_pred = list(self.df[self.target])
        return [float(x) for x in y_pred]

    def cross_val_predict(self, n_splits):
        return list(self.df[self.target])

    def cross_val_predict_proba(self, n_splits):
        y_pred = list(self.df[self.target])
        return [float(x) for x in y_pred]

    def predict_with_threshold(self, X_pred, threshold):
        return list(self.df[self.target])


def test_X_y_shape(sample_df, sample_target_column):
    mock_clkassifier = MockClassifier(sample_df, sample_target_column, MockModel())
    assert mock_clkassifier.X.shape[0] == sample_df.shape[0]
    assert mock_clkassifier.X.shape[1] == sample_df.shape[1] - 1
    assert len(mock_clkassifier.y) == mock_clkassifier.X.shape[0]
