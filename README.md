# MLOps Support Platform — Project Guide

## 1. Project Overview

This project demonstrates an end-to-end MLOps platform for predicting customer churn.

It combines machine learning, model tracking, API development, containerization, CI/CD, monitoring, and Kubernetes deployment.

**GitHub Repository:** https://github.com/diyanme/mlops-support-platform

**Container Registry:** https://github.com/diyanme/mlops-support-platform/pkgs/container/mlops-support-platform

### Technologies Used

- Python 3.12
- scikit-learn
- MLflow
- FastAPI
- Docker
- Docker Compose
- GitHub Actions
- GitHub Container Registry (GHCR)
- Prometheus
- Grafana
- Kubernetes
- Minikube
- Helm
- pytest

---

## 2. Architecture

```text
Customer Dataset
      |
      v
Data Validation
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
MLflow Experiment Tracking
      |
      v
MLflow Model Registry
      |
      v
FastAPI Prediction Service
      |
      +------> Health and Readiness Endpoints
      |
      +------> Prediction Metrics
                    |
                    v
                Prometheus
                    |
                    v
                 Grafana


Application Deployment:

Application Code
      |
      v
Docker Image
      |
      +------> Docker Compose
      |
      +------> Kubernetes Deployment
                    |
                    v
                   Helm


CI/CD:

Git Push
   |
   v
GitHub Actions
   |
   v
Automated Tests
   |
   v
Docker Build
   |
   v
Publish Image to GHCR
```

**Important:** GitHub Actions tests, builds, and publishes the image. Kubernetes deployment is performed separately; the current workflow does not automatically deploy to Kubernetes.

---

## 3. Machine Learning

### Dataset

The sample dataset is stored in:

`data/customers.csv`

It contains customer information such as:

- Customer tenure
- Monthly charges
- Support tickets
- Contract type
- Churn label

### Model

The project uses Logistic Regression with a preprocessing pipeline.

### Recorded Evaluation Results

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.9375 |
| Precision | 0.8000 |
| Recall    | 1.0000 |
| F1 Score  | 0.8889 |

These results are from the sample dataset and do not guarantee performance on real-world customer data.

### Quality Gate

The evaluation process checks whether the F1 score meets the configured threshold.

- Minimum F1 score: `0.80`
- Recorded model F1 score: `0.8889`
- Result: Pass

### MLflow Model Registry

Registered model:

`customer-churn-model`

Model aliases include:

- `champion`
- `challenger`

The API loads the model using the `champion` alias.

### Drift Detection

The project checks for changes in numeric feature distributions against a baseline.

The drift detection module includes tests for:

- Loading training data
- Calculating the baseline
- Detecting unchanged data
- Detecting significant feature changes
- Generating drift reports
- Updating the drift metric

---

## 4. FastAPI Prediction Service

The API serves predictions from the registered model.

### Main Endpoints

| Endpoint    | Purpose                         |
| ----------- | ------------------------------- |
| `/health`   | Checks API health               |
| `/ready`    | Checks whether the API is ready |
| `/model`    | Returns model information       |
| `/predict`  | Generates a churn prediction    |
| `/metrics/` | Exposes Prometheus metrics      |

The exact endpoint behavior depends on the application implementation.

### Example Prediction

Request:

```bash
curl -sS -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "tenure_months": 12,
    "monthly_charges": 75.5,
    "support_tickets": 3,
    "contract_type": "Month-to-month"
  }'
```

Recorded response:

```json
{
  "prediction": 1,
  "churn_probability": 0.6926
}
```

Here, `prediction: 1` represents the churn class in this dataset.

The probability is a model estimate, not a guarantee that the customer will churn.

---

## 5. Docker and Docker Compose

Docker packages the application and its dependencies into a container image.

Docker Compose runs multiple services together.

The project includes an API container and monitoring services, with MLflow running on the host in the tested local configuration.

### Check Running Containers

```bash
docker compose ps
```

### View Logs

```bash
docker compose logs --tail=100
```

For the API service:

```bash
docker compose logs --tail=100 api
```

Use the actual Compose service name if it differs from `api`.

### Restart Services

```bash
docker compose restart
```

### Stop Services

```bash
docker compose down
```

### Rebuild and Start

```bash
docker compose up --build -d
```

---

## 6. MLflow

MLflow tracks experiments and stores model registry information.

In this project, the MLflow server uses a SQLite backend database.

### Local MLflow URL

http://localhost:5000

### Responsibilities

- Experiment tracking
- Recording model metrics
- Storing model artifacts
- Registering models
- Managing model aliases
- Providing the registered model to the API

### Important Concept

The API depends on the MLflow server and the registered model being available.

If the model cannot load, investigate:

1. Whether MLflow is running.
2. Whether the API can reach MLflow.
3. Whether the registered model exists.
4. Whether the `champion` alias exists.
5. Whether the model artifact can be downloaded.

---

## 7. Prometheus and Grafana

### Prometheus

Prometheus scrapes the API's metrics endpoint and stores time-series measurements.

Local URL:

http://localhost:9090

### Grafana

Grafana displays monitoring metrics in dashboards.

Local URL:

http://localhost:3000

Use the credentials configured in your local environment.

### Important Metrics

| Metric                       | Purpose                                       |
| ---------------------------- | --------------------------------------------- |
| `prediction_requests_total`  | Counts prediction requests                    |
| `prediction_errors_total`    | Counts prediction errors                      |
| `prediction_latency_seconds` | Measures prediction latency                   |
| `predictions_total`          | Counts predictions by category, if configured |
| `model_drift_detected`       | Reports drift-detection state                 |

### Verify Prometheus Is Scraping the API

```bash
curl -sG http://localhost:9090/api/v1/query \
  --data-urlencode 'query=up{job="mlops-api"}'
```

Expected result:

`1` means the scrape target is up.

### Check Prediction Requests

```bash
curl -sG http://localhost:9090/api/v1/query \
  --data-urlencode 'query=prediction_requests_total'
```

### Check Request Rate

```bash
curl -sG http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(prediction_requests_total[5m])'
```

The request rate is an average over the five-minute window.

---

## 8. Kubernetes

The project contains Kubernetes manifests in the `k8s/` directory.

They define resources such as:

- Deployment
- Service
- Ingress

The application was tested in a local Minikube cluster.

### Check Cluster

```bash
kubectl get nodes
```

### Check Deployments

```bash
kubectl get deployments
```

### Check Pods

```bash
kubectl get pods
```

### Check Services

```bash
kubectl get services
```

### Check Ingress

```bash
kubectl get ingress
```

### Check Deployment Status

```bash
kubectl rollout status deployment/mlops-api
```

For the Helm deployment:

```bash
kubectl rollout status deployment/mlops-api-helm
```

### Inspect Pod Logs

```bash
kubectl logs deployment/mlops-api-helm
```

### Describe a Pod

```bash
kubectl describe pod <pod-name>
```

Replace `<pod-name>` with the actual pod name.

### Important Concept

A Kubernetes Deployment manages the desired number of application replicas.

A Service provides stable network access to pods.

An Ingress routes HTTP traffic to a Service when an Ingress controller is configured.

---

## 9. Helm

Helm packages Kubernetes resources into a reusable chart.

The chart is located in:

`mlops-api-chart/`

### Validate the Chart

```bash
helm lint ./mlops-api-chart
```

### Render Templates

```bash
helm template mlops-api-helm ./mlops-api-chart
```

### List Releases

```bash
helm list
```

### Check Release Status

```bash
helm status mlops-api-helm
```

### Inspect Release History

```bash
helm history mlops-api-helm
```

### Upgrade a Release

```bash
helm upgrade mlops-api-helm ./mlops-api-chart
```

Only run an upgrade when you intend to apply the current chart configuration to the cluster.

### Important Concept

Helm uses chart templates and values to generate Kubernetes resources.

This makes application configuration easier to reuse across environments.

---

## 10. GitHub Actions CI/CD

Workflow file:

`.github/workflows/ci.yml`

### Current Pipeline

1. Trigger on pushes to `main`.
2. Trigger on pull requests targeting `main`.
3. Check out the repository.
4. Set up Python 3.12.
5. Install locked dependencies.
6. Run automated tests.
7. Build the Docker image.
8. On pushes to `main`, authenticate to GHCR.
9. Publish the image with `latest` and short-SHA tags.

### Verify Tests Locally

```bash
python -m pytest -v
```

### Verify the Docker Build Locally

```bash
docker build -t mlops-churn-api:test .
```

### Check GitHub Actions

https://github.com/diyanme/mlops-support-platform/actions

Both the test/build job and the publishing job passed after the project changes were pushed.

### Important Limitation

The pipeline does not currently perform automated Kubernetes deployment.

---

## 11. GitHub Container Registry

The project publishes its image to:

`ghcr.io/diyanme/mlops-support-platform`

The workflow publishes:

- `latest`
- A short commit-SHA tag

### List the Local Image

```bash
docker image ls ghcr.io/diyanme/mlops-support-platform
```

### Pull the Published Image

```bash
docker pull ghcr.io/diyanme/mlops-support-platform:latest
```

The `latest` tag can change when new commits are published. For reproducible deployments, prefer a specific image tag or digest.

---

## 12. Automated Tests

The current automated suite contains seven drift-detection tests.

### Run the Tests

```bash
python -m pytest -v
```

### Verified Result

```text
7 passed
```

These tests cover drift detection and related functionality.

They do not constitute comprehensive coverage of every API endpoint, model-training path, or deployment scenario.

---

## 13. Git Workflow

### Check Current Branch

```bash
git branch --show-current
```

### Check Working Tree

```bash
git status --short
```

### Review Recent Commits

```bash
git log --oneline -5
```

### Stage a Specific File

```bash
git add README.md
```

### Commit Changes

```bash
git commit -m "Describe the change"
```

### Push to GitHub

```bash
git push origin main
```

### Important

Do not accidentally commit local backups, secrets, `.env` files, or unrelated documentation.

Review `git status` before committing.

---

## 14. Troubleshooting

### API Is Not Responding

Check:

```bash
docker compose ps
docker compose logs --tail=100
curl -sS http://localhost:8001/health
```

Check that the API container is running and the host port is correctly mapped.

### Prometheus Shows the API as Down

Check:

```bash
curl -sS http://localhost:9090/api/v1/targets
```

Investigate:

- Whether the API container is running.
- Whether the metrics endpoint responds.
- Whether the Prometheus scrape target is correct.
- Whether the containers can communicate.

### Model Loading Fails

Check:

- MLflow server availability.
- Model registry name.
- Model alias.
- Artifact availability.
- API-to-MLflow network connectivity.

### Kubernetes Pod Is Not Ready

Check:

```bash
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

Look for image-pull errors, readiness probe failures, configuration errors, and application startup failures.

### Helm Upgrade Fails

Check:

```bash
helm status mlops-api-helm
helm history mlops-api-helm
kubectl get events --sort-by=.metadata.creationTimestamp
```

Inspect the actual error before attempting a rollback or another upgrade.

### GitHub Actions Fails

Open:

https://github.com/diyanme/mlops-support-platform/actions

Inspect the failed step and its logs. Reproduce the issue locally when practical.

---

## 15. Security and Production Considerations

This is a portfolio project, not a production-certified platform.

Before exposing it to real users, address:

- Authentication and authorization.
- Secret management.
- Network restrictions and TLS.
- Container image and dependency scanning.
- Resource requests and limits.
- Monitoring alerts and incident response.
- Data privacy and retention.
- Model validation and retraining controls.
- Backup and recovery.
- Automated API and integration tests.

Never expose credentials, tokens, or sensitive customer information in the repository.

---

## 16. Current Project Status

| Component                       | Status          |
| ------------------------------- | --------------- |
| Model training and evaluation   | Implemented     |
| MLflow tracking and registry    | Implemented     |
| FastAPI prediction service      | Implemented     |
| Docker and Compose              | Implemented     |
| GitHub Actions tests and build  | Passed          |
| GHCR image publishing           | Passed          |
| Prometheus monitoring           | Verified        |
| Grafana dashboard               | Created         |
| Kubernetes manifests            | Implemented     |
| Helm chart                      | Validated       |
| Local Kubernetes deployment     | Verified        |
| Drift-detection tests           | 7 passed        |
| README                          | Added           |
| Automated Kubernetes deployment | Not implemented |

---

## 17. How to Present This Project in an Interview

A concise explanation:

"I built an MLOps support platform that trains and tracks a customer-churn model using MLflow, serves predictions through FastAPI, and packages the application using Docker. I configured GitHub Actions to run tests, build the image, and publish it to GitHub Container Registry. I also implemented Prometheus metrics, a Grafana dashboard, drift detection, and Kubernetes deployment using Helm. I tested the main components locally and verified the CI pipeline."

Be prepared to explain:

1. How the model is trained and registered.
2. How the API loads the champion model.
3. How Docker packages the application.
4. How GitHub Actions builds and publishes the image.
5. How Prometheus scrapes API metrics.
6. How to troubleshoot a failing pod.
7. How Helm templates generate Kubernetes resources.
8. How drift detection works.
9. What happens when MLflow becomes unavailable.
10. What you would improve before production deployment.

**Be clear about the difference between a portfolio implementation and professional production experience.**

---

## Final Recommendation

The core project is complete for a junior DevOps/MLOps Support portfolio.

Do not keep adding tools just to make the project larger.

Focus on understanding the architecture, practising troubleshooting, and explaining the decisions you made.
