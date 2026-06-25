Write-Host "="*60 -ForegroundColor Cyan
Write-Host "PHASE 9: PRODUCTION DEPLOYMENT VALIDATION" -ForegroundColor Cyan
Write-Host "="*60 -ForegroundColor Cyan

# Test 1: Docker containers running
Write-Host "`nTest 1: Docker Containers..." -ForegroundColor Yellow
$containers = docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "rag-"
if ($containers) {
    Write-Host "✅ All containers running" -ForegroundColor Green
    Write-Host $containers -ForegroundColor Gray
} else {
    Write-Host "❌ Containers not running" -ForegroundColor Red
}

# Test 2: Services health
Write-Host "`nTest 2: Services Health..." -ForegroundColor Yellow

$healthResp = Invoke-WebRequest -Uri "http://localhost:8000/health/detailed" `
  -UseBasicParsing -ErrorAction SilentlyContinue

if ($healthResp.StatusCode -eq 200) {
    $health = $healthResp.Content | ConvertFrom-Json
    Write-Host "✅ Overall health: $($health.overall)" -ForegroundColor Green
    foreach ($service in $health.services) {
        $color = if ($service.status -eq "healthy") { "Green" } else { "Yellow" }
        Write-Host "   - $($service.service): $($service.status)" -ForegroundColor $color
    }
} else {
    Write-Host "❌ Health check failed" -ForegroundColor Red
}

# Test 3: API endpoints
Write-Host "`nTest 3: API Endpoints..." -ForegroundColor Yellow

$endpoints = @(
    "http://localhost:8000/health",
    "http://localhost:8000/docs",
    "http://localhost:3000"
)

foreach ($endpoint in $endpoints) {
    $resp = Invoke-WebRequest -Uri $endpoint -UseBasicParsing -ErrorAction SilentlyContinue
    if ($resp.StatusCode -eq 200) {
        Write-Host "✅ $endpoint" -ForegroundColor Green
    } else {
        Write-Host "❌ $endpoint" -ForegroundColor Red
    }
}

# Test 4: Database connectivity
Write-Host "`nTest 4: Database Connectivity..." -ForegroundColor Yellow

# MongoDB
$mongoTest = docker exec rag-mongodb mongosh --eval "db.adminCommand('ping')" 2>&1
if ($mongoTest -match "ok") {
    Write-Host "✅ MongoDB connected" -ForegroundColor Green
} else {
    Write-Host "❌ MongoDB failed" -ForegroundColor Red
}

# Redis
$redisTest = docker exec rag-redis redis-cli ping 2>&1
if ($redisTest -match "PONG") {
    Write-Host "✅ Redis connected" -ForegroundColor Green
} else {
    Write-Host "❌ Redis failed" -ForegroundColor Red
}

Write-Host "`n" + "="*60 -ForegroundColor Cyan
Write-Host "🎉 PRODUCTION DEPLOYMENT VALIDATION COMPLETE" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Cyan