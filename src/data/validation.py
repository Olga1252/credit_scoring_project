from pathlib import Path

import great_expectations as gx
import pandas as pd


def validate_data(df):
    required_columns = [
        "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        *[f"BILL_AMT{i}" for i in range(1, 7)],
        *[f"PAY_AMT{i}" for i in range(1, 7)],
        "default.payment.next.month",
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    
    for column in required_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must contain numeric data."
            )
    
        
    context = gx.get_context(mode="ephemeral")
    source_name = context.data_sources.add_pandas(name="credit_scoring_source")
    asset_name = source_name.add_dataframe_asset(name="credit_scoring_data")
    batch_definition = asset_name.add_batch_definition_whole_dataframe("all_rows")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    
    expectations = [
        gx.expectations.ExpectColumnValuesToBeBetween(column="AGE", min_value=18, max_value=100),
        gx.expectations.ExpectColumnValuesToBeInSet(column="default.payment.next.month", value_set=[0, 1]),
        
    ]
    
    for column in df.columns:
        expectations.append(gx.expectations.ExpectColumnValuesToNotBeNull(column=column))
    for expectation in expectations:
        results = batch.validate(expectation)
        
        if not results["success"]:
            raise ValueError(f"Data validation failed for expectation: {expectation}")
    print("Data validation passed successfully.")
    
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / 'data' / 'raw' / 'UCI_Credit_Card.csv'
    df = pd.read_csv(data_path)
    validate_data(df)