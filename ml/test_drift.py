import pandas as pd
import pytest

from app.metrics import MODEL_DRIFT_DETECTED
from ml.drift import (
    calculate_baseline,
    check_drift,
    generate_drift_report,
    load_training_data,
    update_drift_metric,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "tenure_months": [10, 20, 30],
            "monthly_charges": [50.0, 100.0, 150.0],
            "support_tickets": [1, 2, 3],
        }
    )


def test_load_training_data():
    data = load_training_data()

    assert not data.empty
    assert "tenure_months" in data.columns
    assert "monthly_charges" in data.columns
    assert "support_tickets" in data.columns


def test_calculate_baseline(sample_data):
    baseline = calculate_baseline(sample_data)

    assert baseline["tenure_months"]["mean"] == pytest.approx(20)
    assert baseline["monthly_charges"]["mean"] == pytest.approx(100)
    assert baseline["support_tickets"]["mean"] == pytest.approx(2)


def test_check_drift_when_data_is_unchanged(sample_data):
    baseline = calculate_baseline(sample_data)

    results = check_drift(baseline, sample_data)

    assert all(
        result["drift_detected"] is False
        for result in results.values()
    )


def test_check_drift_when_feature_changes_significantly(sample_data):
    baseline = calculate_baseline(sample_data)

    changed_data = sample_data.copy()
    changed_data["monthly_charges"] = [200.0, 250.0, 300.0]

    results = check_drift(baseline, changed_data)

    assert results["monthly_charges"]["drift_detected"] is True
    assert results["tenure_months"]["drift_detected"] is False
    assert results["support_tickets"]["drift_detected"] is False


def test_generate_drift_report_detected():
    results = {
        "tenure_months": {"drift_detected": False},
        "monthly_charges": {"drift_detected": True},
        "support_tickets": {"drift_detected": False},
    }

    report = generate_drift_report(results)

    assert report["status"] == "DRIFT_DETECTED"
    assert report["drift_count"] == 1
    assert report["total_features_checked"] == 3
    assert report["drifted_features"] == ["monthly_charges"]


def test_generate_normal_drift_report():
    results = {
        "tenure_months": {"drift_detected": False},
        "monthly_charges": {"drift_detected": False},
        "support_tickets": {"drift_detected": False},
    }

    report = generate_drift_report(results)

    assert report["status"] == "NORMAL"
    assert report["drift_count"] == 0
    assert report["drifted_features"] == []


def test_update_drift_metric():
    update_drift_metric({"status": "DRIFT_DETECTED"})
    assert MODEL_DRIFT_DETECTED._value.get() == 1

    update_drift_metric({"status": "NORMAL"})
    assert MODEL_DRIFT_DETECTED._value.get() == 0
