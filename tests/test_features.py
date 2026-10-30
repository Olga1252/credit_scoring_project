import pandas as pd
import pytest

from src.features.build_features import feature_engineering


@pytest.fixture
def client_data():
    return pd.DataFrame(
        {
            "LIMIT_BAL": [10000],
            "BILL_AMT1": [100],
            "BILL_AMT2": [200],
            "BILL_AMT3": [300],
            "BILL_AMT4": [400],
            "BILL_AMT5": [500],
            "BILL_AMT6": [600],
            "PAY_AMT1": [10],
            "PAY_AMT2": [20],
            "PAY_AMT3": [30],
            "PAY_AMT4": [40],
            "PAY_AMT5": [50],
            "PAY_AMT6": [60],
            "PAY_0": [2],
            "PAY_2": [0],
            "PAY_3": [-1],
            "PAY_4": [1],
            "PAY_5": [-2],
            "PAY_6": [0],
        },
        index=[42],
    )


def test_engineered_feature_values(client_data):
    result = feature_engineering(client_data)
    row = result.loc[42]

    assert row["BILL_AMT_SUM"] == 2100
    assert row["PAY_AMT_SUM"] == 210
    assert row["BILL_AMT_MEAN"] == pytest.approx(350)
    assert row["PAY_AMT_MEAN"] == pytest.approx(35)
    assert row["NUM_DELAYED_PAYMENTS"] == 2
    assert row["BILL_TO_LIMIT_RATIO"] == pytest.approx(0.21)


def test_original_data_is_preserved(client_data):
    original = client_data.copy(deep=True)

    result = feature_engineering(client_data)

    pd.testing.assert_frame_equal(client_data, original)
    pd.testing.assert_frame_equal(result[original.columns], original)
    assert result.shape[1] == original.shape[1] + 6


@pytest.mark.parametrize("limit", [0, -100])
def test_nonpositive_limit_produces_missing_ratio(client_data, limit):
    client_data["LIMIT_BAL"] = limit

    result = feature_engineering(client_data)

    assert pd.isna(result.loc[42, "BILL_TO_LIMIT_RATIO"])