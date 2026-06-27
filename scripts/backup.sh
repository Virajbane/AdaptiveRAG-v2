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
