from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = {
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
}


def calculate_psi(reference, current, categorical=False):
    if reference.empty or current.empty:
        raise ValueError("Cannot calculate PSI for empty data.")

    if reference.isna().any() or current.isna().any():
        raise ValueError("Missing values must be handled before calculating PSI.")

    if categorical or reference.nunique() == 1:
        categories = pd.Index(reference.unique()).union(pd.Index(current.unique()))
        reference_counts = (
            reference.value_counts().reindex(categories, fill_value=0).to_numpy()
        )
        current_counts = (
            current.value_counts().reindex(categories, fill_value=0).to_numpy()
        )
    else:

        inner_edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)[1:-1]))
        edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))

        reference_counts = np.histogram(reference, bins=edges)[0]
        current_counts = np.histogram(current, bins=edges)[0]

    reference_share = (reference_counts + 0.5) / (
        reference_counts.sum() + 0.5 * len(reference_counts)
    )
    current_share = (current_counts + 0.5) / (
        current_counts.sum() + 0.5 * len(current_counts)
    )

    return float(
        np.sum(
            (current_share - reference_share) * np.log(current_share / reference_share)
        )
    )


def main():
    project_root = Path(__file__).resolve().parents[2]

    reference = pd.read_csv(project_root / "data" / "processed" / "train_data.csv")
    report_dir = project_root / "reports" / "monitoring"
    current = pd.read_csv(report_dir / "api_predictions.csv")

    feature_columns = reference.drop(
        columns=["ID", "default.payment.next.month"]
    ).columns

    results = []

    for column in feature_columns:
        psi = calculate_psi(
            reference[column],
            current[column],
            categorical=column in CATEGORICAL_FEATURES,
        )

        if psi < 0.1:
            status = "stable"
        elif psi < 0.25:
            status = "check"
        else:
            status = "drift"

        results.append({"feature": column, "psi": psi, "status": status})

    report = pd.DataFrame(results).sort_values("psi", ascending=False)
    output_path = report_dir / "drift_report.csv"
    report.to_csv(output_path, index=False)

    print(f"Reference rows: {len(reference)}")
    print(f"Current rows: {len(current)}")
    print(report.to_string(index=False))
    print("Drift report saved to:", output_path)


if __name__ == "__main__":
    main()
