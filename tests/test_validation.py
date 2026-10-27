import pandas as pd
import pytest
from src.data.validation import validate_data


@pytest.fixture
def valid_df():
    df = pd.DataFrame(
        {
            "ID": [1],
            "LIMIT_BAL": [20000.0],
            "SEX": [2],
            "EDUCATION": [2],
            "MARRIAGE": [1],
            "AGE": [24],
            "PAY_0": [0],
            **{f"PAY_{i}": [0] for i in range(2, 7)},
            **{f"BILL_AMT{i}": [1000.0] for i in range(1, 7)},
            **{f"PAY_AMT{i}": [100.0] for i in range(1, 7)},
            "default.payment.next.month": [0],
        }
    )
    return df


def test_validate_data_passes(valid_df):
    validate_data(valid_df)


def test_missing_column_is_rejected():
    df = pd.DataFrame(
        {
            "AGE": [24],
        }
    )
    with pytest.raises(ValueError, match="Missing required columns:"):
        validate_data(df)


def test_negative_age_is_rejected(valid_df):
    valid_df.loc[0, "AGE"] = -5

    with pytest.raises(ValueError, match="AGE"):
        validate_data(valid_df)


def test_missing_values_are_rejected(valid_df):
    valid_df.loc[0, "LIMIT_BAL"] = float("nan")

    with pytest.raises(ValueError, match="LIMIT_BAL"):
        validate_data(valid_df)


def test_text_type_is_rejected(valid_df):
    valid_df["AGE"] = ["wrong"]

    with pytest.raises(ValueError, match="AGE"):
        validate_data(valid_df)
