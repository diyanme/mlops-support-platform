from pathlib import Path

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/customers.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "churn_model.joblib"

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "customer-churn"


FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "contract_type",
]

TARGET_COLUMN = "churn"


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def train_model():
    with mlflow.start_run():

        # Load dataset
        df = pd.read_csv(DATA_PATH)

        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # Define feature types
        numeric_features = [
            "tenure_months",
            "monthly_charges",
            "support_tickets",
        ]

        categorical_features = [
            "contract_type",
        ]

        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
            ],
            remainder="passthrough",
        )

        # ML pipeline
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.5,
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

        # Log training parameters
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "C": 0.5,
                "max_iter": 1000,
                "random_state": 42,
                "test_size": 0.2,
            }
        )

        # Train
        model_pipeline.fit(X_train, y_train)

        # Predict
        predictions = model_pipeline.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        print("\n=== Model Evaluation ===")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\n=== Confusion Matrix ===")
        print(confusion_matrix(y_test, predictions))

        print("\n=== Classification Report ===")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        # Save local model artifact
        MODEL_DIR.mkdir(exist_ok=True)

        joblib.dump(model_pipeline, MODEL_PATH)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model_pipeline,
            name="churn_model",
        )

        print(f"\nModel saved to: {MODEL_PATH}")

        return model_pipeline


if __name__ == "__main__":
    train_model()