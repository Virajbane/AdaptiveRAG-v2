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
