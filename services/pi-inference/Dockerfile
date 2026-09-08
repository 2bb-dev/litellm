ARG NODE_IMAGE=node:22.23.2-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5

FROM ${NODE_IMAGE} AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM ${NODE_IMAGE} AS dependencies
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts --no-audit --no-fund

FROM ${NODE_IMAGE} AS runtime
ENV NODE_ENV=production \
    PI_INFERENCE_CONFIG=/config/models.json \
    PI_INFERENCE_HOST=0.0.0.0 \
    PI_INFERENCE_PORT=4001 \
    PI_INFERENCE_SLOT_ID=slot1
WORKDIR /app
RUN mkdir -p /data /config && chown node:node /data && chmod 0700 /data
COPY --from=dependencies /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY package.json THIRD_PARTY_NOTICES.md ./
USER node
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["node", "--input-type=module", "-e", "const r = await fetch('http://127.0.0.1:' + (process.env.PI_INFERENCE_PORT || '4001') + '/health/readiness', {signal: AbortSignal.timeout(4000)}); process.exit(r.ok ? 0 : 1)"]
ENTRYPOINT ["node", "dist/main.js"]
