# Performance Test Script - Phase 11
# Fix: uses "message" field (not "question") to match ChatRequest Pydantic model

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "PHASE 11: PERFORMANCE TEST" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Setup: get auth token
Write-Host "`n[SETUP] Getting auth token..." -ForegroundColor Yellow
$loginResp = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/auth/login" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"email":"virajbane2004@gmail.com","password":"Viru@19762004"}' `
  -UseBasicParsing
$token = ($loginResp.Content | ConvertFrom-Json).access_token
Write-Host "✅ Got token: $($token.Substring(0, 20))..." -ForegroundColor Green

# Helper: call chat endpoint
function Invoke-Chat($msg) {
    $body = "{`"message`":`"$msg`"}"   # <-- "message", not "question"
    return Invoke-WebRequest -Uri "http://localhost:8000/api/v1/agents/chat" `
      -Method POST `
      -Headers @{
          "Content-Type"  = "application/json"
          "Authorization" = "Bearer $token"
      } `
      -Body $body `
      -UseBasicParsing
}

# Test 1: same query twice — 2nd should be a cache hit
Write-Host "`n[TEST 1] Query Caching" -ForegroundColor Yellow
$q = "What is machine learning?"

Write-Host "  Request 1 (cold — no cache)..." -ForegroundColor Gray
$t1 = Get-Date
$r1 = Invoke-Chat $q
$s1 = ((Get-Date) - $t1).TotalSeconds
$d1 = $r1.Content | ConvertFrom-Json
Write-Host "  Time   : $([Math]::Round($s1, 2))s" -ForegroundColor Green
Write-Host "  Answer : $($d1.answer.Substring(0, [Math]::Min(60, $d1.answer.Length)))..." -ForegroundColor Gray

Write-Host "  Request 2 (should be cache hit)..." -ForegroundColor Gray
$t2 = Get-Date
$r2 = Invoke-Chat $q
$s2 = ((Get-Date) - $t2).TotalSeconds
$d2 = $r2.Content | ConvertFrom-Json
Write-Host "  Time   : $([Math]::Round($s2, 2))s" -ForegroundColor Green

if ($s2 -lt ($s1 / 2)) {
    Write-Host "  ✅ Cache working! 2nd ($([Math]::Round($s2,2))s) << 1st ($([Math]::Round($s1,2))s)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Cache not kicking in (2nd: $([Math]::Round($s2,2))s, 1st: $([Math]::Round($s1,2))s)" -ForegroundColor Yellow
}

# Test 2: different query — no cache; checks parallel speedup
Write-Host "`n[TEST 2] New Query (parallel execution, no cache)" -ForegroundColor Yellow
$t3 = Get-Date
$r3 = Invoke-Chat "Tell me about artificial intelligence"
$s3 = ((Get-Date) - $t3).TotalSeconds
$d3 = $r3.Content | ConvertFrom-Json
Write-Host "  Time   : $([Math]::Round($s3, 2))s" -ForegroundColor Green
Write-Host "  Answer : $($d3.answer.Substring(0, [Math]::Min(60, $d3.answer.Length)))..." -ForegroundColor Gray

if ($s3 -lt 90) {
    Write-Host "  ✅ Performance improved — under 90s (was ~115s before Phase 11)" -ForegroundColor Green
} elseif ($s3 -lt 120) {
    Write-Host "  ⚠️  Partial improvement ($([Math]::Round($s3,2))s) — parallel agents helping" -ForegroundColor Yellow
} else {
    Write-Host "  ❌ Still slow ($([Math]::Round($s3,2))s) — check orchestrator parallel code" -ForegroundColor Red
}

# Summary
Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Cold query   : $([Math]::Round($s1,2))s" -ForegroundColor White
Write-Host "Cached query : $([Math]::Round($s2,2))s" -ForegroundColor White
Write-Host "New query    : $([Math]::Round($s3,2))s" -ForegroundColor White

if ($s2 -lt ($s1 / 3) -and $s3 -lt 90)  {
    Write-Host "`n🎉 ALL PHASE 11 PERFORMANCE TESTS PASSED" -ForegroundColor Green
} elseif ($s2 -lt 1) {
    Write-Host "`n✅ Caching works. New-query latency still optimising." -ForegroundColor Yellow
} else {
    Write-Host "`n⚠️  Check cache wiring in orchestrator.py" -ForegroundColor Yellow
}