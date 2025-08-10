ARG PYTHON_VERSION=3.12
ARG UV_VERSION=latest

# Stage 1: Fetch the specified version of `uv` and `uvx` executables
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_installer

# Stage 2: Builder - Prepare the application environment
FROM python:${PYTHON_VERSION}-slim AS builder

# Copy the `uv` and `uvx` executables from the installer stage to `/bin/`
COPY --from=uv_installer /uv /uvx /bin/

# Set the working directory inside the container to `/app`
WORKDIR /app

# Add all project files to the `/app` directory in the container
ADD . /app

# Install dependencies using `uv`, leveraging cache for efficiency
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-editable

# Stage 3: Serving - Prepare the runtime environment
FROM python:${PYTHON_VERSION}-slim AS serving
ENV IS_CONTAINER=true

# Copy the `uv` and `uvx` executables from the builder stage to `/bin/`
COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /bin/uvx /bin/uvx

# Copy the Python virtual environment with installed dependencies
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy the application code (including `ml_production_service`) from the builder stage
COPY --from=builder /app/ml_production_service /ml_production_service

# Copy production models into the image from registry
COPY registry/prd /app/registry/prd

# Set working directory to /app for consistent relative paths
WORKDIR /app

# Update the PATH environment variable to prioritize binaries from the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Add the root directory to PYTHONPATH so the application modules can be imported
ENV PYTHONPATH="/:$PYTHONPATH"

# Expose port 8000 to allow external connections to the FastAPI application
EXPOSE 8000

# Start the application using `gunicorn` managed by `uv`
CMD ["uv", "run", "gunicorn", "-c", "/ml_production_service/server/gunicorn_config.py", "ml_production_service.server.main:get_app"]