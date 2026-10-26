# Deployment Guide

> **See also**: [README.md](../README.md) · [ARCHITECTURE.md](ARCHITECTURE.md) ·
> [DEVELOPMENT.md](DEVELOPMENT.md) · [SECURITY.md](SECURITY.md)

This guide covers the three supported deployment targets:

1. **Local dev** with `docker-compose.yml`
2. **Single-node production** with `docker-compose.prod.yml`
3. **Kubernetes** with plain `kubectl apply` (release YAML manifests — no Helm)

Plus: GPU node configuration, TLS setup, and Prometheus + Grafana
monitoring.

## Table of contents

- [Prerequisites](#prerequisites)
- [Docker Compose (local dev)](#docker-compose-local-dev)
- [Production](#production)
- [Kubernetes (plain YAML manifests)](#kubernetes-plain-yaml-manifests)
- [GPU node configuration](#gpu-node-configuration)
- [TLS setup](#tls-setup)
- [Monitoring (Prometheus + Grafana)](#monitoring-prometheus--grafana)

## Prerequisites

| Component   | Min version | Notes                                            |
|-------------|-------------|--------------------------------------------------|
| Docker      | 24.0        | With Compose v2 (`docker compose`).              |
| NVIDIA GPU  | optional    | For GPU image; CUDA 12.x driver + nvidia-container-toolkit. |
| kubectl     | 1.28        | For Kubernetes deployment.                       |
| TLS cert    | optional    | Let's Encrypt or self-signed for testing.        |

## Docker Compose (local dev)

The dev compose file lives at `docker-compose.yml` (project root) and
runs the CPU image, Postgres, Redis, Prometheus, Grafana, and the
dashboard on a single host.

```bash
cp .env.example .env
# Edit .env — at minimum set POSTGRES_PASSWORD and JWT_SECRET.

docker compose up --build
```

| Service        | Port  | Notes                                |
|----------------|-------|--------------------------------------|
| `api`          | 8000  | FastAPI app + WebSocket stream.      |
| `nginx`        | 80    | Reverse proxy + dashboard.           |
| `postgres`     | 5432  | Postgres 16 + pgvector.              |
| `redis`        | 6379  | Cooldown cache + event pub/sub.      |
| `prometheus`   | 9090  | Metrics scraper.                     |
| `grafana`      | 3000  | Dashboard (admin/admin).             |

Stop and wipe state:

```bash
docker compose down -v
```

## Production

`docker-compose.prod.yml` differs from dev in:

- Per-service resource limits and restart policies.
- GPU image used by default (`facial-attendance:latest-gpu`).
- Log rotation (json-file driver, 10 files × 50 MB).
- Only the `nginx` service exposes ports (80/443); all internal services
  stay on the private `attendance-net` bridge network.

```bash
# Generate or obtain TLS certs (example with self-signed for testing)
mkdir -p deploy/docker/certs
openssl req -x509 -newkey rsa:4096 -nodes -days 365 \
  -keyout deploy/docker/certs/key.pem \
  -out deploy/docker/certs/cert.pem \
  -subj "/CN=facial-attendance.local"

# Launch
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Healthcheck the production stack:

```bash
curl -fsS https://facial-attendance.local/health
curl -fsS https://facial-attendance.local/readyz
```

## Kubernetes (plain YAML manifests)

All release manifests live in `deploy/k8s/`. **No Helm required.** Apply
them in order:

```bash
# 1. Namespace
kubectl apply -f deploy/k8s/namespace.yaml

# 2. Config + secrets (edit secret.yaml with real base64 values first,
#    or use `kubectl create secret generic facial-attendance-secrets
#    --from-env-file=.env -n facial-attendance`)
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/secret.yaml

# 3. Data layer
kubectl apply -f deploy/k8s/postgres.yaml
kubectl apply -f deploy/k8s/redis.yaml          # optional, for cooldown cache

# 4. Network policies (lock down before app pods start)
kubectl apply -f deploy/k8s/network-policy.yaml

# 5. API (deployment + service + HPA + PDB)
kubectl apply -f deploy/k8s/api-deployment.yaml
kubectl apply -f deploy/k8s/api-service.yaml
kubectl apply -f deploy/k8s/api-hpa.yaml
kubectl apply -f deploy/k8s/api-pdb.yaml

# 6. Worker (background processor for multi-camera ingestion)
kubectl apply -f deploy/k8s/worker-deployment.yaml

# 7. Monitoring
kubectl apply -f deploy/k8s/prometheus.yaml
kubectl apply -f deploy/k8s/grafana.yaml

# 8. Ingress (edit host in ingress.yaml first)
kubectl apply -f deploy/k8s/ingress.yaml
```

Or apply everything in one shot:

```bash
kubectl apply -f deploy/k8s/
```

Verify:

```bash
kubectl -n facial-attendance get pods
kubectl -n facial-attendance port-forward svc/api 8000:8000
curl http://localhost:8000/readyz
```

### Customising the manifests

The YAML files are plain — just edit them directly:

| File                         | What to edit before applying                          |
|------------------------------|-------------------------------------------------------|
| `secret.yaml`                | Replace base64 placeholders with real DB password, JWT secret, Sentry DSN. (For production use sealed-secrets / external-secrets / Vault.) |
| `configmap.yaml`             | Tweak `SIMILARITY_THRESHOLD`, `COOLDOWN_SECONDS`, `LOG_LEVEL`. |
| `api-deployment.yaml`        | Image repo + tag, replica count, resource requests/limits, GPU resource limit (uncomment the `nvidia.com/gpu: 1` line). |
| `worker-deployment.yaml`     | Same as above; tune replicas for camera parallelism.   |
| `ingress.yaml`               | `spec.rules[].host`, TLS secret name.                  |
| `postgres.yaml`              | PVC storage size, storage class.                       |

### Rolling out a new image

```bash
# Edit api-deployment.yaml → bump image tag, then:
kubectl apply -f deploy/k8s/api-deployment.yaml

# Or patch in-place:
kubectl -n facial-attendance set image deployment/api \
  api=ghcr.io/your-org/facial-attendance:sha-abc1234

# Watch the rollout
kubectl -n facial-attendance rollout status deployment/api
```

### Scaling

```bash
# Manual scale
kubectl -n facial-attendance scale deployment/api --replicas=4

# HPA (already applied above) auto-scales on CPU > 70%, min 2 max 10
kubectl -n facial-attendance get hpa
```

### Uninstall

```bash
kubectl delete namespace facial-attendance
```

## GPU node configuration

### Docker Compose (GPU)

Use `docker-compose.prod.yml` (it references the GPU image by default).
Requirements:

- Host has NVIDIA driver 535+.
- [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
  installed.
- `/usr/bin/nvidia-container-runtime` exists.

Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Kubernetes (GPU)

1. Install the NVIDIA GPU operator (one-time per cluster — uses Helm
   just for the operator itself, **not** for our app):

   ```bash
   helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
   helm install gpu-operator nvidia/gpu-operator \
     --namespace gpu-operator --create-namespace
   ```

2. Label your GPU nodes:

   ```bash
   kubectl label node <node-name> accelerator=nvidia
   ```

3. Edit `deploy/k8s/api-deployment.yaml`:
   - Change image to the GPU tag: `ghcr.io/your-org/facial-attendance:latest-gpu`
   - Uncomment the GPU resource limit:
     ```yaml
     resources:
       limits:
         nvidia.com/gpu: 1
     ```
   - Add a node selector:
     ```yaml
     nodeSelector:
       accelerator: nvidia
     ```

4. Apply:

   ```bash
   kubectl apply -f deploy/k8s/api-deployment.yaml
   ```

## TLS setup

### With cert-manager (recommended for k8s)

Install cert-manager (one-time per cluster):

```bash
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace --set installCRDs=true
```

Create a `ClusterIssuer` (save as `deploy/k8s/cluster-issuer.yaml`):

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: you@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
```

```bash
kubectl apply -f deploy/k8s/cluster-issuer.yaml
```

Then edit `deploy/k8s/ingress.yaml`:

```yaml
spec:
  tls:
    - hosts:
        - facial-attendance.example.com
      secretName: facial-attendance-tls
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
```

```bash
kubectl apply -f deploy/k8s/ingress.yaml
```

### With Docker Compose

Mount certs into `nginx` via a volume in `docker-compose.prod.yml` and
update `configs/nginx.conf` to listen on 443 with `ssl_certificate`.

## Monitoring (Prometheus + Grafana)

Already wired in both Docker Compose and Kubernetes manifests.

**Docker Compose:**

- Prometheus → http://localhost:9090
- Grafana → http://localhost:3000 (admin / admin)

**Kubernetes:**

```bash
kubectl -n facial-attendance port-forward svc/prometheus 9090:9090
kubectl -n facial-attendance port-forward svc/grafana 3000:3000
```

The pre-built dashboard `configs/grafana/dashboards/attendance-overview.json`
is auto-provisioned and shows: FPS, faces detected/s, recognitions/s,
spoof rejection rate, active tracks, DB connection pool, request latency
p50/p95.

Prometheus scrapes `/metrics` on both `api` and `worker` services. The
following custom metrics are exposed by the app:

| Metric                          | Type      | Labels                       |
|---------------------------------|-----------|------------------------------|
| `frames_processed_total`        | Counter   | `strategy`, `camera_id`      |
| `faces_detected_total`          | Counter   | `strategy`                   |
| `recognitions_total`            | Counter   | `strategy`, `result`         |
| `spoof_rejected_total`          | Counter   | `reason`                     |
| `attendance_logged_total`       | Counter   | `department`                 |
| `frame_processing_seconds`      | Histogram | `strategy`                   |
