# Deployment Guide

## Prerequisites
- Docker and Docker Compose installed
- All `.env` files configured
- MongoDB, Redis, Qdrant, Ollama properly configured

## Production Deployment

### 1. Prepare Environment
```bash
cp .env.example .env
# Edit .env with production values
```

### 2. Build and Start Services
```bash
docker-compose up -d
```

### 3. Verify Services
```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed
curl http://localhost:3000
```

### 4. Monitor Logs
```bash
docker-compose logs -f backend
docker-compose logs -f frontend
```

## Security Checklist
- [ ] All passwords changed from defaults
- [ ] JWT secret key is strong (32+ chars)
- [ ] API keys stored in .env (not in code)
- [ ] SSL/TLS configured (reverse proxy)
- [ ] Firewall rules configured
- [ ] Database backups enabled

## Scaling
- Increase replicas: `docker-compose up -d --scale backend=3`
- Use load balancer (nginx) in front
- Cache with CDN for frontend

## Troubleshooting
See TROUBLESHOOTING.md for common issues