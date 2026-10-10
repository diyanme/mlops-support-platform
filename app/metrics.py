from prometheus_client import Counter, Histogram, Gauge


PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of predictions by predicted class",
    ["prediction"],

)

MODEL_DRIFT_DETECTED = Gauge(
    "model_drift_detected",
    "Whether model input drift has been detected",
)