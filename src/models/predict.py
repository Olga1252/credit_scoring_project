from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import feature_engineering


def predict_samples():
    project_root = Path(__file__).resolve().parents[2]
    model = joblib.load(
        project_root/ 'models'/ 'trained' / 'model.joblib'
                
    )
    
    test_df = pd.read_csv(
        project_root/ 'data'/ 'processed'/ 'test_data.csv'
    )
    
    sample = test_df.head(3)
    features = feature_engineering(
        sample.drop(columns=['ID', 'default.payment.next.month'])
    )
    
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    result = pd.DataFrame({
        "ID": sample["ID"].to_numpy(),
        "predicted_class": predictions,
        "default_probability": probabilities,
    })
    print(result.to_string(index=False))


if __name__ == "__main__":
    predict_samples()