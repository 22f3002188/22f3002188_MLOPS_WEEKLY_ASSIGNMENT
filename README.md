# Week 7 – Stress Testing, Observability & Scaling the IRIS Pipeline

## Student Details

- **Name:** Harsh Jayswal
- **Roll Number:** 22F3002188
- **Course:** MLOps
- **Assignment:** Week 7 – Stress Testing, Observability & Scaling

---

# Objective

The objective of this assignment is to evaluate the performance of the deployed IRIS FastAPI application under high concurrent traffic, observe Kubernetes Horizontal Pod Autoscaler (HPA) behavior, monitor the application using Google Cloud Monitoring and Cloud Logging, and analyze system bottlenecks when autoscaling is constrained.

---

# Technologies Used

- Google Cloud Platform (GCP)
- Google Kubernetes Engine (GKE)
- Kubernetes
- FastAPI
- Docker
- Artifact Registry
- Horizontal Pod Autoscaler (HPA)
- Cloud Monitoring
- Cloud Logging
- wrk
- GitHub
- GitHub Actions

---

# Project Structure

```
.
├── Dockerfile
├── iris_fastapi.py
├── model.joblib
├── post.lua
├── requirements.txt
├── k8s
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
└── tests
```

---

# Deployment Architecture

```
GitHub Repository
        │
        ▼
Docker Image
        │
        ▼
Artifact Registry
        │
        ▼
Google Kubernetes Engine
        │
        ▼
FastAPI IRIS API
        │
        ▼
wrk Stress Testing
        │
        ▼
Cloud Monitoring + Cloud Logging
```

---

# Task 1

The application was deployed on Google Kubernetes Engine using the existing Kubernetes manifests.

Deployment includes:

- Deployment
- LoadBalancer Service
- Horizontal Pod Autoscaler

---

# Task 2 – Stress Testing

The application was stress tested using wrk.

Example command:

```bash
wrk -t4 -c1000 -d30s -s post.lua http://<EXTERNAL_IP>/predict/
```

Observed metrics:

- Requests per second
- Average latency
- Socket timeout count

---

# Task 3 – Horizontal Pod Autoscaler

Configured HPA:

```
Min Replicas : 1
Max Replicas : 3
Target CPU   : 50%
```

During stress testing:

- HPA increased desired replicas
- Additional pods were created
- Scheduler attempted to place new pods

---

# Task 4 – Observability

## Cloud Monitoring

Observed:

- CPU Usage
- Memory Usage
- Pod Activity

CPU usage increased significantly during stress testing.

---

## Cloud Logging

Logs Explorer was used to inspect container logs.

Filtered by:

```
resource.type="k8s_container"
cluster_name="iris-cluster"
container_name="iris-api"
```

Application logs were successfully captured.

---

# Task 5 – Bottleneck Analysis

Autoscaling was restricted by setting

```
maxReplicas: 1
```

Stress test:

```
wrk -t4 -c2000 -d30s -s post.lua http://<EXTERNAL_IP>/predict/
```

Observed:

- CPU utilization exceeded target
- HPA could not scale beyond one pod
- Higher latency
- Increased request timeouts

---

# Bottleneck Identified

While allowing three replicas, Kubernetes attempted to create additional pods.

However, two pods remained in Pending state due to:

```
Insufficient CPU
```

Scheduler Event:

```
0/2 nodes are available:
2 Insufficient cpu
```

This prevented the application from scaling successfully under heavy load.

---

# Learning Outcomes

Through this assignment I learned:

- Stress testing using wrk
- Kubernetes Horizontal Pod Autoscaler
- Kubernetes scheduling behavior
- Cloud Monitoring dashboards
- Cloud Logging
- Resource bottleneck analysis
- Performance evaluation under high concurrency

---

# Conclusion

The IRIS API was successfully stress tested on GKE.

Cloud Monitoring and Cloud Logging were used to observe system behavior.

The HPA responded to increased load, while scheduler resource constraints demonstrated practical bottlenecks during autoscaling.

This assignment provided hands-on experience with production-grade monitoring and scalability concepts in Kubernetes.
