# syntax=docker/dockerfile:1
# The backend is a uv workspace member, so the build context is the repository root:
# docker build -f backend/Dockerfile .

FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src

# Dependencies first, so a code change does not reinstall them. The lock covers the whole
# workspace, so uv reads every member's pyproject.toml but installs only the backend.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=backend/pyproject.toml,target=backend/pyproject.toml \
    --mount=type=bind,source=seed-pipeline/pyproject.toml,target=seed-pipeline/pyproject.toml \
    uv sync --locked --no-dev --package pharma-agent --no-install-workspace

COPY pyproject.toml uv.lock ./
COPY seed-pipeline/pyproject.toml seed-pipeline/
COPY backend/pyproject.toml backend/README.md backend/
COPY backend/src backend/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --package pharma-agent --no-editable

FROM python:3.12-slim-bookworm
# Numeric IDs, so a runtime that enforces a non-root user can verify it without /etc/passwd.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app
WORKDIR /app
COPY --from=build --chown=10001:10001 /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1
USER 10001:10001
EXPOSE 8000
CMD ["pharma-agent", "serve"]
