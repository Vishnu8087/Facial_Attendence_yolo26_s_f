###############################################################################
#  Facial Attendance & Motion Tracking System — Makefile
#
#  Conventions:
#    - Targets are silent unless they fail. Use `make <target> V=1` for verbose.
#    - All paths are relative to the project root.
#    - Works on Linux + macOS. Requires docker, docker compose, kubectl.
###############################################################################

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# ---- Variables ------------------------------------------------------------
PROJECT      := facial-attendance
ROOT         := $(shell pwd)
COMPOSE      := docker compose
COMPOSE_PROD := $(COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml
COMPOSE_DEV  := $(COMPOSE) -f docker-compose.yml -f docker-compose.override.yml
NAMESPACE    := facial-attendance
RELEASE      := facial-attendance
IMAGE_REPO   ?= facial-attendance/api
IMAGE_TAG    ?= latest-cpu
GPU_TAG      ?= latest-gpu
PYTEST       := pytest
MODELS_DIR   ?= ./models
MODEL_NAME   ?= buffalo_l

# ---- Pretty colors (disabled if not a TTY) --------------------------------
ifneq (,$(findstring xterm,$(TERM)))
        BLUE   := \033[34m
        GREEN  := \033[32m
        YELLOW := \033[33m
        RED    := \033[31m
        NC     := \033[0m
else
        BLUE   :=
        GREEN  :=
        YELLOW :=
        RED    :=
        NC     :=
endif

# ---- Help -----------------------------------------------------------------
.PHONY: help
help: ## Show this help
        @echo -e "$(BLUE)Facial Attendance System - Makefile targets$(NC)"
        @echo ""
        @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
          | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(NC) %s\n", $$1, $$2}' \
          | sort

# ===========================================================================
#  Docker build
# ===========================================================================
.PHONY: build build-gpu build-cpu build-db pull

build: build-cpu ## Build default (CPU) image

build-cpu: ## Build CPU image
        @echo -e "$(BLUE)==> Building CPU image $(IMAGE_REPO):$(IMAGE_TAG)$(NC)"
        docker build \
          -f deploy/docker/Dockerfile.cpu \
          -t $(IMAGE_REPO):$(IMAGE_TAG) \
          .

build-gpu: ## Build GPU image (CUDA 12.1)
        @echo -e "$(BLUE)==> Building GPU image $(IMAGE_REPO):$(GPU_TAG)$(NC)"
        docker build \
          -f deploy/docker/Dockerfile.gpu \
          -t $(IMAGE_REPO):$(GPU_TAG) \
          .

build-db: ## Build custom Postgres+pgvector image (optional)
        docker build -f deploy/docker/Dockerfile.db -t $(PROJECT)/db:latest .

pull: ## Pull all upstream images (db, redis, prometheus, grafana, nginx)
        @echo -e "$(BLUE)==> Pulling upstream images$(NC)"
        docker pull pgvector/pgvector:pg16
        docker pull redis:7-alpine
        docker pull prom/prometheus:latest
        docker pull grafana/grafana:latest
        docker pull nginx:alpine

# ===========================================================================
#  Docker Compose lifecycle
# ===========================================================================
.PHONY: up up-prod up-dev down down-clean restart ps logs logs-api logs-worker

up: ## Start full stack (dev with hot reload)
        $(COMPOSE_DEV) up -d --build

up-prod: ## Start full stack (prod-like, GPU image, resource limits)
        $(COMPOSE_PROD) up -d --build

up-dev: ## Alias for `up` (dev)
        $(COMPOSE_DEV) up -d --build

down: ## Stop and remove containers (keep volumes)
        $(COMPOSE) down --remove-orphans

down-clean: ## Stop and remove containers AND volumes (data loss!)
        $(COMPOSE) down -v --remove-orphans

restart: ## Restart all services
        $(COMPOSE) restart

ps: ## List containers
        $(COMPOSE) ps

logs: ## Tail logs from all services
        $(COMPOSE) logs -f --tail=200

logs-api: ## Tail API logs
        $(COMPOSE) logs -f --tail=200 api

logs-worker: ## Tail worker logs
        $(COMPOSE) logs -f --tail=200 worker

# ===========================================================================
#  DB / migrations
# ===========================================================================
.PHONY: migrate migrate-create migrate-down psql db-shell redis-shell

migrate: ## Run alembic upgrade head inside the API container
        $(COMPOSE) exec api alembic upgrade head

migrate-create: ## Create a new alembic revision: make migrate-create MSG="add x"
        @test -n "$(MSG)" || (echo "Usage: make migrate-create MSG=\"description\"" && exit 1)
        $(COMPOSE) exec api alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Roll back the latest migration
        $(COMPOSE) exec api alembic downgrade -1

psql: db-shell ## Open psql shell on the db container

db-shell:
        $(COMPOSE) exec db psql -U $$POSTGRES_USER -d $$POSTGRES_DB

redis-shell:
        $(COMPOSE) exec redis redis-cli

# ===========================================================================
#  Enrollment / CLI helpers
# ===========================================================================
.PHONY: enroll enroll-list download-models download-yolo26

enroll: ## Enroll a person: make enroll NAME=Alice FILE=./photo.jpg
        @test -n "$(NAME)" -a -n "$(FILE)" || (echo "Usage: make enroll NAME=Alice FILE=./photo.jpg" && exit 1)
        curl -s -X POST http://localhost:8000/api/enroll \
          -F "name=$(NAME)" \
          -F "file=@$(FILE)" | jq .

enroll-list: ## List enrolled people
        curl -s http://localhost:8000/api/persons | jq .

download-models: ## Download InsightFace model pack (default: buffalo_l)
        python scripts/download_models.py --pack $(MODEL_NAME) --dest $(MODELS_DIR)

download-yolo26: ## Download YOLO26-face weights: make download-yolo26 [VARIANT=yolo26n-face.pt]
        python scripts/download_models.py --yolo26 --no-insightface \
          --yolo26-variant $(or $(VARIANT),yolo26n-face.pt) --dest $(MODELS_DIR)

# ===========================================================================
#  Tests / lint / type-check
# ===========================================================================
.PHONY: test test-cov lint format type-check

test: ## Run pytest (local venv if present, else inside api container)
        @if [ -d .venv ]; then \
          .venv/bin/pytest -q; \
        else \
          $(COMPOSE) exec api $(PYTEST) -q || echo "Run 'make up' first"; \
        fi

test-cov: ## Run pytest with coverage report
        $(COMPOSE) exec api $(PYTEST) --cov=app --cov-report=term-missing

lint: ## Lint with ruff
        @if [ -d .venv ]; then .venv/bin/ruff check app tests; \
        else $(COMPOSE) exec api ruff check app tests; fi

format: ## Format with black + ruff
        @if [ -d .venv ]; then \
          .venv/bin/black app tests; \
          .venv/bin/ruff format app tests; \
        else $(COMPOSE) exec api sh -c 'black app tests && ruff format app tests'; fi

type-check: ## Type-check with mypy
        @if [ -d .venv ]; then .venv/bin/mypy app; \
        else $(COMPOSE) exec api mypy app; fi

# ===========================================================================
#  Kubernetes
# ===========================================================================
.PHONY: k8s-apply k8s-delete k8s-pods k8s-status k8s-logs k8s-port-forward

k8s-apply: ## Apply all k8s manifests in order
        @echo -e "$(BLUE)==> Applying k8s manifests to namespace $(NAMESPACE)$(NC)"
        kubectl apply -f deploy/k8s/namespace.yaml
        kubectl apply -f deploy/k8s/configmap.yaml
        kubectl apply -f deploy/k8s/secret.yaml
        kubectl apply -f deploy/k8s/postgres.yaml
        kubectl apply -f deploy/k8s/network-policy.yaml
        kubectl apply -f deploy/k8s/prometheus.yaml
        kubectl apply -f deploy/k8s/grafana.yaml
        kubectl apply -f deploy/k8s/api-deployment.yaml
        kubectl apply -f deploy/k8s/api-service.yaml
        kubectl apply -f deploy/k8s/api-pdb.yaml
        kubectl apply -f deploy/k8s/api-hpa.yaml
        kubectl apply -f deploy/k8s/worker-deployment.yaml
        kubectl apply -f deploy/k8s/ingress.yaml

k8s-delete: ## Delete all k8s resources
        kubectl delete -f deploy/k8s/ --ignore-not-found
        kubectl delete namespace $(NAMESPACE) --ignore-not-found

k8s-pods: ## List pods, services, ingress
        kubectl -n $(NAMESPACE) get pods,svc,ingress

k8s-status: ## Show deployment + HPA status
        kubectl -n $(NAMESPACE) describe deploy api
        kubectl -n $(NAMESPACE) describe hpa api

k8s-logs: ## Tail logs: make k8s-logs S=api (or worker)
        kubectl -n $(NAMESPACE) logs -f deploy/$(or $(S),api)

k8s-port-forward: ## Port-forward: make k8s-port-forward S=api P=8000
        kubectl -n $(NAMESPACE) port-forward svc/$(or $(S),api) $(or $(P),8000):8000

# ===========================================================================
#  Misc
# ===========================================================================
.PHONY: clean clean-models clean-docker health

clean: ## Remove build artifacts, caches, and Python bytecode
        find . -type d -name __pycache__ -prune -exec rm -rf {} +
        find . -type d -name .pytest_cache -prune -exec rm -rf {} +
        find . -type d -name .mypy_cache -prune -exec rm -rf {} +
        find . -type d -name .ruff_cache -prune -exec rm -rf {} +
        rm -rf build dist *.egg-info .coverage htmlcov coverage.xml

clean-models: ## Delete cached model weights (forces re-download on next start)
        rm -rf models/* models/.models-ready 2>/dev/null || true

clean-docker: ## Remove dangling docker images and stopped containers
        docker container prune -f
        docker image prune -f
        docker volume ls -qf dangling=true | xargs -r docker volume rm

health: ## Hit health endpoint of locally-running api
        curl -sS http://localhost:8000/health && echo "" || echo "API not reachable"
        curl -sS http://localhost:8000/readyz && echo "" || true
