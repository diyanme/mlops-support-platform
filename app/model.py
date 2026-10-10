import os

import mlflow
import mlflow.sklearn


class ModelLoader:
    def __init__(self):
        self.model_name = os.getenv(
            "MLFLOW_MODEL_NAME",
            "customer-churn-model",
        )

        self.model_alias = os.getenv(
            "MLFLOW_MODEL_ALIAS",
            "champion",
        )

        self.tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        )

        mlflow.set_tracking_uri(self.tracking_uri)

        self.model_uri = (
            f"models:/{self.model_name}@{self.model_alias}"
        )

        self.model = None

    def load(self):
        print(f"Loading model from: {self.model_uri}")

        self.model = mlflow.sklearn.load_model(
            self.model_uri
        )

        print("Model loaded successfully.")

        return self.model

    def predict(self, data):
        if self.model is None:
            raise RuntimeError(
                "Model has not been loaded."
            )

        return self.model.predict(data)

    def predict_proba(self, data):
        if self.model is None:
            raise RuntimeError(
                "Model has not been loaded."
            )

        return self.model.predict_proba(data)