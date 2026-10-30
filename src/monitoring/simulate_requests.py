from pathlib import Path

import httpx
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[2]
    test_path = project_root / "data" / "processed" / "test_data.csv"

    sample = pd.read_csv(test_path).sample(n=500, random_state=42)
    features = sample.drop(columns=["ID", "default.payment.next.month"])

    predictions = []

    with httpx.Client(
        base_url="http://127.0.0.1:8001",
        timeout=30.0,
        trust_env=False,
    ) as client:
        health = client.get("/health")
        health.raise_for_status()

        for index, payload in enumerate(features.to_dict(orient="records"), start=1):
            response = client.post("/predict", json=payload)

            if response.status_code != 200:
                raise RuntimeError(
                    f"Request {index} failed: "
                    f"{response.status_code} {response.text}"
                )

            predictions.append(response.json())

            if index % 100 == 0:
                print(f"Processed {index}/500 requests")

    report_dir = project_root / "reports" / "monitoring"
    report_dir.mkdir(parents=True, exist_ok=True)

    result = pd.concat(
        [
            sample.reset_index(drop=True),
            pd.DataFrame(predictions),
        ],
        axis=1,
    )

    output_path = report_dir / "api_predictions.csv"
    result.to_csv(output_path, index=False)

    print("Predictions saved to:", output_path)
    print(result[["ID", "predicted_class", "default_probability"]].head())


if __name__ == "__main__":
    main()
