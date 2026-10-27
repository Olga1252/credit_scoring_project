from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

import json

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.features.build_features import feature_engineering
from src.models.pipeline import build_pipeline
import mlflow
import mlflow.sklearn
import joblib


def train():
    project_root = Path(__file__).resolve().parents[2]
    train_path = project_root / "data" / "processed" / "train_data.csv"

    train_df = pd.read_csv(train_path)

    target = "default.payment.next.month"
    X_train = feature_engineering(train_df.drop(columns=["ID", target]))
    y_train = train_df[target]

    categorical_features = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]
    numeric_features = [
        column for column in X_train.columns if column not in categorical_features
    ]

    model_pipeline = build_pipeline(numeric_features, categorical_features)

    print("Feature matrix shape:", X_train.shape)
    print("Target shape:", y_train.shape)
    print("Pipeline created. Steps:", list(model_pipeline.named_steps))

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring=["roc_auc", "precision", "recall", "f1"],
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    print("Best parameters:", search.best_params_)
    print("Best CV ROC-AUC:", search.best_score_)

    test_path = project_root / "data" / "processed" / "test_data.csv"
    test_df = pd.read_csv(test_path)

    X_test = feature_engineering(test_df.drop(columns=["ID", target]))
    y_test = test_df[target]

    best_model = search.best_estimator_

    model_dir = project_root / "models" / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(best_model, model_path)
    print("Model saved to:", model_path)

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    report_dir = project_root / "reports" / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    pd.DataFrame(search.cv_results_).to_csv(
        report_dir / "grid_search_results.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax)

    ax.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve test set")
    fig.tight_layout()
    fig.savefig(report_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Reports saved to:", report_dir)

    database_path = (project_root / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{database_path}")
    mlflow.set_experiment("credit_scoring_project")

    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_params(search.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metric("cv_roc_auc", search.best_score_)
        mlflow.log_metrics({f"test_{name}": value for name, value in metrics.items()})
        for filename in [
            "metrics.json",
            "grid_search_results.csv",
            "roc_curve.png",
        ]:
            mlflow.log_artifact(str(report_dir / filename))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            input_example=X_train.head(3),
            skops_trusted_types=["numpy.dtype"],
        )

    for index, params in enumerate(search.cv_results_["params"]):
        with mlflow.start_run(run_name=f"logistic_cv_{index+1}"):
            mlflow.log_params(params)
            mlflow.log_param("cv_flods", cv.get_n_splits())

            for metric in ["roc_auc", "precision", "recall", "f1"]:
                value = search.cv_results_[f"mean_test_{metric}"][index]
                mlflow.log_metric(f"cv_{metric}", float(value))
    print("Mlflow runs saved. ")


if __name__ == "__main__":
    train()
