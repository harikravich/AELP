# Node 20 for Next.js 14
FROM node:20 AS deps
WORKDIR /app
# Install dependencies using npm
COPY AELP2/apps/dashboard/package.json ./
RUN npm install --no-audit --no-fund

FROM node:20 AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY AELP2/apps/dashboard .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20 AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
# bash typically present on Debian-based node image; if not, fallback to sh
# public folder is optional; skip if not present
# COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
# Bundle python pipelines and scripts for control endpoints (best-effort)
COPY AELP2 /workspace/AELP2
# Install Python and minimal dependencies for pipelines
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/* \
 && pip3 install --no-cache-dir \
    google-cloud-bigquery \
    google-analytics-data \
    google-ads \
    numpy \
    scipy \
    google-auth google-api-core
# Ensure Python can import AELP2
ENV PYTHONPATH=/workspace
EXPOSE 3000
CMD ["npm","start"]
