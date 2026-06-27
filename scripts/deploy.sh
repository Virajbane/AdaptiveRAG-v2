#!/bin/bash
# scripts/deploy.sh - Manual production deployment

set -euo pipefail

COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"
LOG_FILE="logs/deploy_$(date +%Y%m%d_%H%M%S).log"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()   { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1" | tee -a "$LOG_FILE"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"              | tee -a "$LOG_FILE"; }
error() { echo -e "${RED}[ERROR]${NC} $1"                | tee -a "$LOG_FILE"; exit 1; }

mkdir -p logs
log "=== RAG 2.0 Production Deployment ==="

[[ -f "$ENV_FILE" ]] || error ".env file not found"
source "$ENV_FILE"
[[ -z "${MONGO_PASSWORD:-}" ]]  && error "MONGO_PASSWORD not set"
[[ -z "${REDIS_PASSWORD:-}" ]]  && error "REDIS_PASSWORD not set"
[[ -z "${JWT_SECRET_KEY:-}" ]]  && error "JWT_SECRET_KEY not set"
[[ -z "${QDRANT_API_KEY:-}" ]]  && error "QDRANT_API_KEY not set"
log "✅ Environment validated"

docker info &>/dev/null || error "Docker not running"
log "✅ Docker running"

log "Backup before deploy..."
bash scripts/backup.sh || warn "Backup failed — continuing"

log "Pulling latest images..."
docker compose -f "$COMPOSE_FILE" pull 2>&1 | tee -a "$LOG_FILE"

log "Starting databases..."
docker compose -f "$COMPOSE_FILE" up -d mongodb redis qdrant ollama
sleep 10

log "Deploying backend..."
docker compose -f "$COMPOSE_FILE" up -d --no-deps backend
sleep 5
curl --silent --fail http://localhost:8000/health &>/dev/null || error "Backend health check failed"
log "✅ Backend healthy"

log "Deploying frontend..."
docker compose -f "$COMPOSE_FILE" up -d --no-deps frontend
sleep 5

docker compose -f "$COMPOSE_FILE" up -d --no-deps nginx

log "Final health check..."
for i in $(seq 1 5); do
  curl --silent --fail http://localhost:8000/health &>/dev/null && { log "✅ API healthy"; break; }
  [[ $i -eq 5 ]] && error "Health check failed after 5 retries"
  warn "Retry $i/5..."; sleep 5
done

docker image prune -f &>/dev/null
log "=== Deployment complete ==="
docker compose -f "$COMPOSE_FILE" ps
