# ============================================================
# Phase 12 Setup Script
# Run this from your project ROOT folder in PowerShell
# e.g.  cd C:\Users\Viraj\your-rag-project
#        .\setup-phase12.ps1
# ============================================================

Write-Host "=== Phase 12: Creating all files ===" -ForegroundColor Cyan

# ── Create folders ──────────────────────────────────────────
New-Item -ItemType Directory -Force -Path ".github\workflows" | Out-Null
New-Item -ItemType Directory -Force -Path "nginx"             | Out-Null
New-Item -ItemType Directory -Force -Path "scripts"           | Out-Null
New-Item -ItemType Directory -Force -Path "nginx\ssl"         | Out-Null

Write-Host "✅ Folders created" -ForegroundColor Green

# ============================================================
# FILE 1: docker-compose.prod.yml
# ============================================================
@'
version: '3.8'

services:
  # MongoDB
  mongodb:
    image: mongo:7.0
    container_name: rag-mongodb
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD}
      MONGO_INITDB_DATABASE: rag_db
    volumes:
      - mongodb_data:/data/db
      - mongodb_config:/data/configdb
    networks:
      - rag-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          memory: 512M
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongosh localhost:27017/test --quiet
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Redis
  redis:
    image: redis:7-alpine
    container_name: rag-redis
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --appendonly yes
      --maxmemory 256mb
      --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - rag-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          memory: 256M
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Qdrant
  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag-qdrant
    environment:
      QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY}
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - rag-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Ollama
  ollama:
    image: ollama/ollama:latest
    container_name: rag-ollama
    environment:
      OLLAMA_HOST: 0.0.0.0:11434
      OLLAMA_KEEP_ALIVE: 24h
    volumes:
      - ollama_data:/root/.ollama
    networks:
      - rag-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Backend
  backend:
    image: ${DOCKER_REGISTRY}/rag-backend:${IMAGE_TAG:-latest}
    container_name: rag-backend
    environment:
      MONGODB_URL: mongodb://admin:${MONGO_PASSWORD}@mongodb:27017/rag_db?authSource=admin
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379
      QDRANT_URL: http://qdrant:6333
      QDRANT_API_KEY: ${QDRANT_API_KEY}
      OLLAMA_BASE_URL: http://ollama:11434
      ENVIRONMENT: production
      TAVILY_API_KEY: ${TAVILY_API_KEY}
      JWT_SECRET_KEY: ${JWT_SECRET_KEY}
      LOG_LEVEL: INFO
      DEBUG: "false"
    depends_on:
      mongodb:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    networks:
      - rag-network
    restart: always
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          memory: 512M
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      rollback_config:
        parallelism: 1
        delay: 5s
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"

  # Frontend
  frontend:
    image: ${DOCKER_REGISTRY}/rag-frontend:${IMAGE_TAG:-latest}
    container_name: rag-frontend
    environment:
      NEXT_PUBLIC_API_URL: ${API_URL}
      NODE_ENV: production
    depends_on:
      - backend
    networks:
      - rag-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          memory: 256M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: rag-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - frontend
      - backend
    networks:
      - rag-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  mongodb_data:
    driver: local
  mongodb_config:
    driver: local
  redis_data:
    driver: local
  qdrant_data:
    driver: local
  ollama_data:
    driver: local
  nginx_logs:
    driver: local

networks:
  rag-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
'@ | Set-Content "docker-compose.prod.yml" -Encoding UTF8

Write-Host "✅ docker-compose.prod.yml" -ForegroundColor Green

# ============================================================
# FILE 2: nginx/nginx.prod.conf
# ============================================================
@'
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;

    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";

    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=2r/s;

    upstream backend {
        least_conn;
        server backend:8000;
        keepalive 32;
    }

    upstream frontend {
        server frontend:3000;
        keepalive 16;
    }

    server {
        listen 80;
        server_name _;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name _;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_session_cache shared:SSL:10m;

        client_max_body_size 55M;

        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
            proxy_read_timeout 180s;
            proxy_connect_timeout 10s;
        }

        location /api/v1/auth/ {
            limit_req zone=auth burst=5 nodelay;
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            access_log off;
        }

        location / {
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
'@ | Set-Content "nginx\nginx.prod.conf" -Encoding UTF8

Write-Host "✅ nginx/nginx.prod.conf" -ForegroundColor Green

# ============================================================
# FILE 3: .github/workflows/deploy.yml
# ============================================================
@'
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  DOCKER_REGISTRY: ghcr.io/${{ github.repository_owner }}
  IMAGE_TAG: ${{ github.sha }}

jobs:
  test-backend:
    name: Backend Tests
    runs-on: ubuntu-latest
    services:
      mongodb:
        image: mongo:7.0
        env:
          MONGO_INITDB_ROOT_USERNAME: admin
          MONGO_INITDB_ROOT_PASSWORD: testpassword
        ports:
          - 27017:27017
        options: >-
          --health-cmd "echo 'db.runCommand(\"ping\").ok' | mongosh localhost/test --quiet"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'
          cache-dependency-path: backend/requirements.txt
      - name: Install dependencies
        working-directory: backend
        run: pip install -r requirements.txt
      - name: Run unit tests
        working-directory: backend
        env:
          MONGODB_URL: mongodb://admin:testpassword@localhost:27017/rag_test?authSource=admin
          REDIS_URL: redis://localhost:6379
          QDRANT_URL: http://localhost:6333
          JWT_SECRET_KEY: test-secret-key-for-ci-minimum-32-chars
          SECRET_KEY: test-secret-key-for-ci-minimum-32-chars
          ENVIRONMENT: test
        run: |
          pytest tests/unit/ -v --tb=short
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: backend-test-results
          path: backend/test-results/

  test-frontend:
    name: Frontend Build Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      - name: Install dependencies
        working-directory: frontend
        run: npm ci
      - name: Lint
        working-directory: frontend
        run: npm run lint
      - name: Build
        working-directory: frontend
        env:
          NEXT_PUBLIC_API_URL: http://localhost:8000
        run: npm run build

  build-images:
    name: Build Docker Images
    runs-on: ubuntu-latest
    needs: [test-backend, test-frontend]
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/setup-buildx-action@v3
      - name: Build & push backend
        uses: docker/build-push-action@v5
        with:
          context: ./backend
          push: true
          tags: |
            ${{ env.DOCKER_REGISTRY }}/rag-backend:${{ env.IMAGE_TAG }}
            ${{ env.DOCKER_REGISTRY }}/rag-backend:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
      - name: Build & push frontend
        uses: docker/build-push-action@v5
        with:
          context: ./frontend
          push: true
          tags: |
            ${{ env.DOCKER_REGISTRY }}/rag-frontend:${{ env.IMAGE_TAG }}
            ${{ env.DOCKER_REGISTRY }}/rag-frontend:latest
          build-args: |
            NEXT_PUBLIC_API_URL=${{ secrets.API_URL }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build-images]
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Deploy via SSH
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.PROD_HOST }}
          username: ${{ secrets.PROD_USER }}
          key: ${{ secrets.PROD_SSH_KEY }}
          script: |
            cd /opt/rag-system
            git pull origin main
            export IMAGE_TAG=${{ env.IMAGE_TAG }}
            export DOCKER_REGISTRY=${{ env.DOCKER_REGISTRY }}
            echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
            docker compose -f docker-compose.prod.yml pull backend frontend
            docker compose -f docker-compose.prod.yml up -d --no-deps --scale backend=2 backend frontend
            docker image prune -f
      - name: Smoke test
        run: |
          sleep 15
          curl --fail --retry 5 --retry-delay 5 https://${{ secrets.PROD_HOST }}/health
'@ | Set-Content ".github\workflows\deploy.yml" -Encoding UTF8

Write-Host "✅ .github/workflows/deploy.yml" -ForegroundColor Green

# ============================================================
# FILE 4: scripts/deploy.sh
# ============================================================
@'
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
'@ | Set-Content "scripts\deploy.sh" -Encoding UTF8

Write-Host "✅ scripts/deploy.sh" -ForegroundColor Green

# ============================================================
# FILE 5: scripts/backup.sh
# ============================================================
@'
#!/bin/bash
# scripts/backup.sh - Backup MongoDB, Qdrant, Redis

set -euo pipefail

BACKUP_DIR="backups/$(date +%Y-%m-%d_%H-%M-%S)"
RETENTION_DAYS=7
ENV_FILE=".env"

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
log()   { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
mkdir -p "$BACKUP_DIR"
log "=== Backup to $BACKUP_DIR ==="

# MongoDB
log "Backing up MongoDB..."
docker exec rag-mongodb mongodump \
  --username admin --password "${MONGO_PASSWORD}" \
  --authenticationDatabase admin --db rag_db \
  --out /tmp/mongodump --quiet 2>&1 || error "mongodump failed"
docker cp rag-mongodb:/tmp/mongodump "$BACKUP_DIR/mongodb"
docker exec rag-mongodb rm -rf /tmp/mongodump
tar -czf "$BACKUP_DIR/mongodb.tar.gz" -C "$BACKUP_DIR" mongodb
rm -rf "$BACKUP_DIR/mongodb"
log "✅ MongoDB done ($(du -sh "$BACKUP_DIR/mongodb.tar.gz" | cut -f1))"

# Qdrant
log "Backing up Qdrant..."
docker run --rm --volumes-from rag-qdrant \
  -v "$(pwd)/$BACKUP_DIR:/backup" alpine:latest \
  tar -czf /backup/qdrant.tar.gz /qdrant/storage 2>/dev/null || \
  log "⚠️  Qdrant skipped (not running)"
log "✅ Qdrant done"

# Redis
log "Backing up Redis..."
docker exec rag-redis redis-cli -a "${REDIS_PASSWORD}" SAVE &>/dev/null || true
docker cp rag-redis:/data/dump.rdb "$BACKUP_DIR/redis.rdb" 2>/dev/null || \
  log "⚠️  Redis skipped (not running)"
log "✅ Redis done"

# Manifest
cat > "$BACKUP_DIR/manifest.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname": "$(hostname)"
}
EOF

# Cleanup old backups
find backups/ -maxdepth 1 -type d -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
log "=== Backup complete: $BACKUP_DIR ($(du -sh "$BACKUP_DIR" | cut -f1)) ==="
'@ | Set-Content "scripts\backup.sh" -Encoding UTF8

Write-Host "✅ scripts/backup.sh" -ForegroundColor Green

# ============================================================
# FILE 6: scripts/health-check.sh
# ============================================================
@'
#!/bin/bash
# scripts/health-check.sh

BASE_URL="${1:-http://localhost:8000}"
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0

check() {
  local name="$1" cmd="$2"
  if eval "$cmd" &>/dev/null; then
    echo -e "${GREEN}✅ $name${NC}"; ((PASS++))
  else
    echo -e "${RED}❌ $name${NC}"; ((FAIL++))
  fi
}

echo "=== RAG 2.0 Health Check === Target: $BASE_URL"
echo ""
echo "── API ──"
check "Backend reachable"   "curl -sf $BASE_URL/health"
check "Returns healthy"     "curl -sf $BASE_URL/health | grep -q 'healthy\|ok'"
check "API docs up"         "curl -sf $BASE_URL/docs"

echo ""
echo "── Services ──"
check "Redis"   "docker exec rag-redis redis-cli ping 2>/dev/null | grep -q PONG"
check "Qdrant"  "curl -sf http://localhost:6333/health"
check "Ollama"  "curl -sf http://localhost:11434/api/tags"

echo ""
echo "── Containers ──"
check "rag-backend"   "docker ps --filter name=rag-backend  --filter status=running | grep -q rag-backend"
check "rag-frontend"  "docker ps --filter name=rag-frontend --filter status=running | grep -q rag-frontend"
check "rag-mongodb"   "docker ps --filter name=rag-mongodb  --filter status=running | grep -q rag-mongodb"
check "rag-redis"     "docker ps --filter name=rag-redis    --filter status=running | grep -q rag-redis"
check "rag-qdrant"    "docker ps --filter name=rag-qdrant   --filter status=running | grep -q rag-qdrant"

echo ""
DISK=$(df / | awk 'NR==2{print $5}' | tr -d '%')
[[ $DISK -lt 80 ]] && { echo -e "${GREEN}✅ Disk: ${DISK}%${NC}"; ((PASS++)); } || \
                       { echo -e "${RED}❌ Disk: ${DISK}% (critical)${NC}"; ((FAIL++)); }

echo ""
echo "Passed: $PASS  Failed: $FAIL"
[[ $FAIL -eq 0 ]] && echo -e "${GREEN}🎉 All checks passed${NC}" && exit 0
echo -e "${RED}⚠️  $FAIL check(s) failed${NC}" && exit 1
'@ | Set-Content "scripts\health-check.sh" -Encoding UTF8

Write-Host "✅ scripts/health-check.sh" -ForegroundColor Green

# ============================================================
# FILE 7: .env.prod.example
# ============================================================
@'
# Copy this to .env and fill in real values
# NEVER commit .env to git

MONGO_PASSWORD=change_me_strong_password_here
REDIS_PASSWORD=change_me_strong_redis_password
QDRANT_API_KEY=change_me_qdrant_api_key
JWT_SECRET_KEY=change_me_at_least_32_chars_random_string_here
TAVILY_API_KEY=tvly-your_key_here
DOCKER_REGISTRY=ghcr.io/your-github-username
IMAGE_TAG=latest
API_URL=https://your-domain.com
ENVIRONMENT=production
LOG_LEVEL=INFO
'@ | Set-Content ".env.prod.example" -Encoding UTF8

Write-Host "✅ .env.prod.example" -ForegroundColor Green

# ============================================================
# FILE 8: DEPLOYMENT.md
# ============================================================
@'
# Deployment Guide - RAG 2.0

## Methods
| Method | Use case |
|--------|----------|
| docker-compose.yml | Local dev |
| docker-compose.prod.yml | Production manual |
| GitHub Actions | Auto CI/CD on push to main |

## Quick Start (Production)

```bash
cp .env.prod.example .env
# Fill in all values in .env
bash scripts/deploy.sh
bash scripts/health-check.sh
```

## GitHub Actions CI/CD

Add these secrets in GitHub repo Settings > Secrets > Actions:
- PROD_HOST  — your server IP
- PROD_USER  — SSH username
- PROD_SSH_KEY — private SSH key
- API_URL    — https://your-domain.com

Pipeline: push to main -> tests -> build images -> deploy -> smoke test

## Backups
```bash
bash scripts/backup.sh          # manual
# Cron: 0 2 * * * /path/scripts/backup.sh
```
Kept for 7 days, then auto-deleted.

## Scaling
```bash
docker compose -f docker-compose.prod.yml up -d --scale backend=3
```

## Logs
```bash
docker compose -f docker-compose.prod.yml logs -f backend
docker compose -f docker-compose.prod.yml logs -f frontend
```
'@ | Set-Content "DEPLOYMENT.md" -Encoding UTF8

Write-Host "✅ DEPLOYMENT.md" -ForegroundColor Green

# ── Git commit ──────────────────────────────────────────────
Write-Host ""
Write-Host "=== Committing to git ===" -ForegroundColor Cyan

git add .
git commit -m "Phase 12: Deployment & DevOps - CI/CD pipeline, prod docker-compose, nginx, backup scripts"
git push origin main

Write-Host ""
Write-Host "=== Phase 12 Complete! ===" -ForegroundColor Green
Write-Host "Files created:" -ForegroundColor White
Write-Host "  docker-compose.prod.yml"
Write-Host "  nginx/nginx.prod.conf"
Write-Host "  .github/workflows/deploy.yml"
Write-Host "  scripts/deploy.sh"
Write-Host "  scripts/backup.sh"
Write-Host "  scripts/health-check.sh"
Write-Host "  .env.prod.example"
Write-Host "  DEPLOYMENT.md"
Write-Host ""
Write-Host "Check GitHub Actions tab to see the pipeline run!" -ForegroundColor Cyan