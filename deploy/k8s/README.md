# Kubernetes manifests — Facial Attendance & Motion Tracking System

Target namespace: **`facial-attendance`**

## Apply order (recommended)

```bash
# 1. Namespace first
kubectl apply -f deploy/k8s/namespace.yaml

# 2. Config + secrets (replace placeholders in secret.yaml first!)
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/secret.yaml

# 3. Data layer
kubectl apply -f deploy/k8s/postgres.yaml
kubectl apply -f deploy/k8s/redis.yaml         # optional Redis for cooldown cache & event pub/sub

# 4. Network policies (lock down before app pods start)
kubectl apply -f deploy/k8s/network-policy.yaml

# 5. Observability
kubectl apply -f deploy/k8s/prometheus.yaml
kubectl apply -f deploy/k8s/grafana.yaml

# 6. App workloads
kubectl apply -f deploy/k8s/api-deployment.yaml
kubectl apply -f deploy/k8s/api-service.yaml
kubectl apply -f deploy/k8s/api-pdb.yaml
kubectl apply -f deploy/k8s/api-hpa.yaml
kubectl apply -f deploy/k8s/worker-deployment.yaml

# 7. Edge
kubectl apply -f deploy/k8s/ingress.yaml
```

Or one-liner (sorted by file prefix):

```bash
kubectl apply -f deploy/k8s/
```

## Pre-requisites

- Cluster 1.28+ with a CNI that supports `NetworkPolicy` (Calico / Cilium).
- Default `StorageClass` (`standard`) for PVCs — change in `postgres.yaml` /
  `api-deployment.yaml` as needed.
- nginx ingress controller (`ingress-nginx`) for the Ingress resource.
- (GPU) NVIDIA GPU Operator + node taints for the `nvidia.com/gpu` resource —
  uncomment the limit in `api-deployment.yaml`.
- (TLS) `cert-manager` + a `ClusterIssuer` named `letsencrypt-prod`.

## Secrets

`secret.yaml` ships with **placeholder base64 values**. For production use:

- **Sealed Secrets** — `kubeseal -f secret.yaml -w sealed-secret.yaml -n facial-attendance`
- **External Secrets** — bind to AWS Secrets Manager / GCP Secret Manager / Vault
- **HashiCorp Vault** — `vault-injector` sidecar with `vault.hashicorp.com/agent-inject: "true"`

Never commit real secrets to git.

## Verifying

```bash
kubectl -n facial-attendance get pods,svc,ingress
kubectl -n facial-attendance logs deploy/api -f
kubectl -n facial-attendance port-forward svc/api 8000:8000
curl http://localhost:8000/health
curl http://localhost:8000/readyz
curl http://localhost:8000/metrics
```

## Teardown

```bash
kubectl delete -f deploy/k8s/
kubectl delete namespace facial-attendance
```
