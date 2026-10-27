# Security Policy

> **See also**: [README.md](../README.md) · [ARCHITECTURE.md](ARCHITECTURE.md) ·
> [DEPLOYMENT.md](DEPLOYMENT.md) · [CONTRIBUTING.md](../CONTRIBUTING.md)

## Table of contents

- [Reporting a vulnerability](#reporting-a-vulnerability)
- [Supported versions](#supported-versions)
- [Threat model](#threat-model)
- [Deployment hardening](#deployment-hardening)
- [Biometric data handling](#biometric-data-handling)
- [Secrets management](#secrets-management)
- [Network policies](#network-policies)
- [RBAC](#rbac)
- [Audit logging](#audit-logging)

## Reporting a vulnerability

We take security vulnerabilities seriously, especially given that this
system processes biometric data.

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead, report privately via one of:

1. **GitHub Security Advisories** (preferred) —
   https://github.com/your-org/facial-attendance-system/security/advisories/new
2. **Email** — security@your-org.example (PGP key fingerprint:
   `0xABCD 1234 5678 9ABC DEF0 1234 5678 9ABC DEF0 1234`)

Please include:

- Affected version (or commit SHA).
- Steps to reproduce (proof-of-concept).
- Impact assessment (what an attacker could do).
- Suggested fix (optional).

We will acknowledge receipt within **2 business days** and aim to ship a
fix or mitigation within **30 days** for high-severity issues. We
request a 90-day embargo before public disclosure to allow downstream
users time to patch.

## Supported versions

| Version | Supported          | Notes                                       |
|---------|--------------------|---------------------------------------------|
| 0.1.x   | ✅ security only    | Initial release. Bug fixes may require upgrade. |
| < 0.1   | ❌ not supported   | Pre-release; do not deploy in production.   |

We backport security fixes to the latest minor release only. Upgrade
promptly when a new patch is released.

## Threat model

| Threat                                            | Mitigation in this system                                      |
|---------------------------------------------------|----------------------------------------------------------------|
| Static photo held to camera (presentation attack)  | EAR-based blink detection + head-pose variance check          |
| Screen-replay (phone playing video)               | Static-frame rejection + head-pose variance check             |
| Enrollment of a non-consenting person             | Out-of-band consent workflow (see Biometric data handling)    |
| Stolen embedding → replay                         | Embeddings never leave the server; JWT-gated endpoints        |
| DoS via high-FPS video                            | `MAX_STREAM_FPS` rate cap + per-IP connection limit           |
| SQL injection                                     | SQLAlchemy parameterised queries everywhere (no raw strings)  |
| XSS in dashboard                                  | Vanilla JS, no `innerHTML` on user-controlled data; CSP       |
| Token theft                                       | Short-lived JWT (1h), refresh via httpOnly cookie             |
| Supply-chain (typosquatted deps)                  | Dependabot + pip-audit + Trivy image scan in CI               |

### Out of scope

The current liveness verifier is **heuristic**. It does NOT defend
against:

- High-quality 3D masks.
- Real-time deepfake video replay.
- Adversarial patches designed to fool RetinaFace/ArcFace.

For high-security deployments, add a learned anti-spoofing model
(roadmap v0.2+) and a hardware authenticator (e.g. FIDO2) for
enrollment.

## Deployment hardening

### TLS

- Terminate TLS at the ingress (cert-manager + Let's Encrypt in k8s, or
  nginx + Let's Encrypt for docker-compose).
- Minimum TLS version: **1.2** (prefer 1.3). Disable TLS 1.0/1.1.
- Use modern cipher suites (Mozilla "intermediate" profile is fine).
- Enable HSTS (`Strict-Transport-Security: max-age=63072000;
  includeSubDomains; preload`).

### Cookies

- `HttpOnly`, `Secure`, `SameSite=Strict` on all session cookies.
- Refresh token cookie: short-lived (≤ 24h), rotating.

### Headers

The FastAPI app sets the following headers by default (via
`app/main.py` middleware):

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 0           # modern browsers; rely on CSP instead
Content-Security-Policy: default-src 'self'; img-src 'self' data: blob:; connect-src 'self' wss:; script-src 'self'; style-src 'self' 'unsafe-inline'
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: camera=(self), microphone=(), geolocation=()
```

### Resource limits

| Layer        | Limit                                   |
|--------------|-----------------------------------------|
| API          | 2 GiB memory, 2 CPU per pod (default).  |
| Postgres     | `max_connections=100`, `shared_buffers=1GB`. |
| WebSocket    | 30 FPS per connection, max 10 concurrent per IP. |
| Disk         | Read-only root FS in prod; logs to `emptyDir`. |

## Biometric data handling

This system processes **biometric data** (face embeddings). Treat
embeddings as PII — they are irreversible but not anonymous (an
embedding can be matched back to the original face).

### Collection

- Enroll only employees who have **signed consent** on file (out-of-band
  workflow — this repo does not implement the consent form itself).
- Document the lawful basis under applicable law (GDPR Art. 6 + 9,
  BIPA, CCPA, etc.).

### Storage

- Embeddings live in `employees.embedding` (pgvector `vector(512)` when
  enabled, otherwise float32 bytes in PostgreSQL `BYTEA`).
- Postgres data-at-rest encryption: enable on the EBS / PVC level
  (LUKS or cloud KMS).
- Backups **must** be encrypted. Never pipe `pg_dump` to an unencrypted
  bucket.

### Retention

- Attendance logs: retain per company policy (default 1 year).
- Embeddings: deleted immediately on `DELETE /api/employees/{id}`.
- Right-to-be-forgotten: `DELETE /api/employees/{id}` cascades to
  `attendance_log` soft-delete (records retained for audit but
  pseudonymised — `employee_id` replaced with `"[REDACTED]"`).

### Test fixtures

- **Never** commit real face images to this repository.
- Use only the synthetic fixtures under
  `tests/fixtures/sample_faces/` (see
  [`tests/fixtures/sample_faces/README.md`](../tests/fixtures/sample_faces/README.md)).
- For local testing with real faces, place them in
  `tests/fixtures/sample_faces/local/` (gitignored) and **never** push.

## Secrets management

### Environment variables

All secrets are passed via environment variables (see
`configs/.env.example`). **Never** commit a real `.env` file —
`.gitignore` excludes it.

### Kubernetes

Use the **External Secrets Operator** + a managed secret store (AWS
Secrets Manager, GCP Secret Manager, HashiCorp Vault):

```yaml
# deploy/k8s/externalsecret.yaml
apiVersion: external-secrets.io/v1
kind: ExternalSecret
metadata:
  name: facial-attendance-secrets
  namespace: facial-attendance
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: facial-attendance-secrets
  data:
    - secretKey: POSTGRES_PASSWORD
      remoteRef:
        key: secret/facial-attendance/prod
        property: postgres_password
    - secretKey: JWT_SECRET
      remoteRef:
        key: secret/facial-attendance/prod
        property: jwt_secret
```

### Rotation

- `JWT_SECRET`: rotate quarterly. Old tokens invalidated immediately.
- `POSTGRES_PASSWORD`: rotate every 90 days. Update the secret, then
  restart the API pods (no Postgres restart needed — asyncpg reconnects).

## Network policies

Restrict ingress/egress at the namespace level. Example
`NetworkPolicy` for the `facial-attendance` namespace:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: facial-attendance-default-deny
  namespace: facial-attendance
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: facial-attendance
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
    # Deny all other egress by default; punch holes as needed.
```

The API should **not** have direct outbound internet access in
production — InsightFace models are pre-downloaded at build time.

## RBAC

### Kubernetes

The API runs as a `ServiceAccount` with **no** cluster-role bindings. It
does not need to talk to the Kubernetes API. The `deploy/k8s/` manifests
create only the bare-minimum `ServiceAccount` (defined inline in
`api-deployment.yaml`).

### Application

Two built-in roles:

| Role    | Permissions                                              |
|---------|----------------------------------------------------------|
| `admin` | Enroll, delete, export, view all attendance.            |
| `viewer`| View attendance (own department only), no enroll/delete. |

Roles are assigned via JWT claims (`"role": "admin"`). Issue admin
tokens sparingly; prefer `viewer` for read-only dashboards.

## Audit logging

Every state-changing API call (`POST /api/employees`,
`DELETE /api/employees/{id}`, `POST /api/auth/token`) emits an audit
log entry to:

1. The `audit_log` table (Postgres) — `actor`, `action`, `resource`,
   `timestamp`, `request_id`.
2. `stdout` as structured JSON (captured by your log aggregator:
   Loki, Datadog, CloudWatch, etc.).

Audit logs are retained for **1 year** minimum (regulatory best
practice). Export monthly to cold storage.
