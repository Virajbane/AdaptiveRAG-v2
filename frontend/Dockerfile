# PATH: frontend/Dockerfile
# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# FIX: NEXT_PUBLIC_* variables are baked into the JS bundle at BUILD time,
# not read at container-run time. The old Dockerfile had no ARG for this,
# so setting NEXT_PUBLIC_API_URL in docker-compose `environment:` did
# nothing -- the frontend silently called whatever URL (or none) was
# present when `npm run build` ran. This ARG/ENV pair fixes that: pass it
# via `docker build --build-arg NEXT_PUBLIC_API_URL=...` or compose's
# `build.args`.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

# Copy files
COPY package*.json ./
RUN npm ci

COPY . .

# FIX: this project has no frontend/public folder, but the later
# `COPY --from=builder /app/public ./public` step assumes one exists and
# fails the whole build otherwise. Guarantee it exists (empty is fine --
# Next.js doesn't require a public folder to run).
RUN mkdir -p public

# Build application (NEXT_PUBLIC_API_URL above is baked in right here)
RUN npm run build

# Production stage
FROM node:20-alpine

WORKDIR /app

# Install serve to run production build
RUN npm install -g serve

# Copy built application from builder
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package*.json ./

RUN npm ci --only=production

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Run application
CMD ["npm", "start"]