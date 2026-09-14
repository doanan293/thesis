# syntax=docker/dockerfile:1

FROM node:24.21.0-alpine3.24 AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --ignore-scripts

FROM node:24.21.0-alpine3.24 AS build
WORKDIR /app
# Prerendered pages embed absolute canonical and hreflang URLs.
ARG PUBLIC_SITE_URL=http://localhost:8080
ENV PUBLIC_SITE_URL=${PUBLIC_SITE_URL}
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:24.21.0-alpine3.24 AS production-deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --omit=dev --ignore-scripts

FROM node:24.21.0-alpine3.24
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000
# 1000:1000 is the image's `node` user; numeric IDs let a runtime verify the non-root user.
COPY --chown=1000:1000 package.json package-lock.json ./
COPY --from=production-deps --chown=1000:1000 /app/node_modules ./node_modules
COPY --from=build --chown=1000:1000 /app/build ./build
USER 1000:1000
EXPOSE 3000
CMD ["node_modules/.bin/react-router-serve", "./build/server/index.js"]
