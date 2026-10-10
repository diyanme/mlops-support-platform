from pathlib import Path

import pandas as pd

from app.metrics import MODEL_DRIFT_DETECTED


TRAINING_DATA_PATH = Path("data/customers.csv")

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
]

DEFAULT_DRIFT_THRESHOLD = 0.20


def calculate_baseline(data: pd.DataFrame) -> dict:
    baseline = {}

    for feature in NUMERIC_FEATURES:
        baseline[feature] = {
            "mean": data[feature].mean(),
            "std": data[feature].std(),
        }

    return baseline


def check_drift(
    baseline: dict,
    current_data: pd.DataFrame,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> dict:
    results = {}

    for feature in NUMERIC_FEATURES:
        baseline_mean = baseline[feature]["mean"]
        current_mean = current_data[feature].mean()

        if baseline_mean == 0:
            difference_ratio = 0
        else:
            difference_ratio = (
                abs(current_mean - baseline_mean)
                / abs(baseline_mean)
            )

        results[feature] = {
            "baseline_mean": baseline_mean,
            "current_mean": current_mean,
            "difference_ratio": difference_ratio,
            "drift_detected": bool(difference_ratio > threshold),
        }

    return results


def generate_drift_report(
    results: dict,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> dict:
    drifted_features = [
        feature
        for feature, result in results.items()
        if result["drift_detected"]
    ]

    return {
        "status": "DRIFT_DETECTED" if drifted_features else "NORMAL",
        "threshold": threshold,
        "drifted_features": drifted_features,
        "drift_count": len(drifted_features),
        "total_features_checked": len(results),
    }

def update_drift_metric(report: dict) -> None:
    if report["status"] == "DRIFT_DETECTED":
        MODEL_DRIFT_DETECTED.set(1)
    else:
        MODEL_DRIFT_DETECTED.set(0)
        

def load_training_data() -> pd.DataFrame:
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAINING_DATA_PATH}"
        )

    return pd.read_csv(TRAINING_DATA_PATH)


if __name__ == "__main__":
    training_data = load_training_data()

    baseline = calculate_baseline(training_data)

    print("=== Training Data Baseline ===")

    for feature, values in baseline.items():
        print(
            f"{feature}: "
            f"mean={values['mean']:.2f}, "
            f"std={values['std']:.2f}"
        )