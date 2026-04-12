# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy dependency files first for better layer caching
COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the full project to repo root inside container
COPY . /app

# Make imports work from project root
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server/app.py", "--host", "0.0.0.0", "--port", "8000"]