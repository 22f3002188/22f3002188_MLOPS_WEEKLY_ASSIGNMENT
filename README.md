# Week 6 – MLOps Assignment
**IITM BS Roll Number:** 22f3002188

## Project Overview

This project demonstrates a complete MLOps deployment pipeline for an IRIS Classification API using Docker, GitHub Actions, Google Artifact Registry, and Google Kubernetes Engine (GKE).

The API predicts the Iris flower species from the following four features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## Technologies Used

- Python
- FastAPI
- Docker
- GitHub Actions
- Google Cloud Platform (GCP)
- Google Artifact Registry
- Google Kubernetes Engine (GKE)
- Kubernetes
- MLflow
- Feast

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── cd.yml
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── Dockerfile
├── iris_fastapi.py
├── model.joblib
├── requirements.txt
├── train.py
└── README.md
```

---

## Task 1 – Pod vs Container

### Docker Container

A Docker container packages an application together with all required libraries and dependencies.

### Kubernetes Pod

A Pod is the smallest deployable unit in Kubernetes.

A Pod can contain one or more containers that:
- share storage
- share networking
- share lifecycle

Kubernetes deploys Pods instead of individual containers because Pods provide better scheduling, networking and scalability.

---

## Task 2 – Dockerization

The IRIS FastAPI application was containerized using Docker.

Docker image contains:

- FastAPI application
- Trained ML model
- Python dependencies
- Uvicorn server

Build image:

```bash
docker build -t iris-api .
```

Run locally:

```bash
docker run -d -p 8200:8200 iris-api
```

---

## Task 3 – GCP Service Account

A dedicated service account was created.

Required IAM roles:

- Artifact Registry Writer
- Container Developer
- Storage Admin
- Container Cluster Viewer

Credentials were securely stored as GitHub Secrets.

---

## Task 4 – GitHub Actions

GitHub Actions automatically:

- Builds Docker image
- Pushes image to Artifact Registry
- Authenticates with GCP
- Connects to GKE
- Updates Kubernetes Deployment

Workflow file:

```
.github/workflows/cd.yml
```

---

## Task 5 – Google Kubernetes Engine Deployment

Deployment steps:

- Create GKE Cluster
- Deploy Kubernetes Deployment
- Create LoadBalancer Service
- Expose FastAPI publicly

Verify deployment:

```bash
kubectl get pods
```

```bash
kubectl get services
```

API Test:

```bash
curl http://<EXTERNAL-IP>
```

Prediction:

```bash
curl -X POST http://<EXTERNAL-IP>/predict/ \
-H "Content-Type: application/json" \
-d '{
"sepal_length":5.1,
"sepal_width":3.5,
"petal_length":1.4,
"petal_width":0.2
}'
```

Example Output

```json
{
  "predicted_class":"setosa"
}
```

---

## Optional Task

Currently the Docker image contains the trained model (`model.joblib`).

Future improvement:

- Pull latest model automatically from MLflow Model Registry during Docker build.

---

## Author

Harsh Jayswal

Roll Number: 22f3002188

IIT Madras BS Degree Program