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
