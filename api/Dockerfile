# ── Build stage ────────────────────────────────────────────────────────────────
# Use a slim Python base to keep the image small. Python 3.12 matches the
# version used for local development.
FROM python:3.12-slim AS base

WORKDIR /app

# Install dependencies first so Docker caches this layer separately from the
# application code. Changing app code won't invalidate the pip install layer.
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application layer ──────────────────────────────────────────────────────────
COPY . /app/api

# Expose the port uvicorn listens on. AWS ECS task definitions should map
# the host port to this container port.
EXPOSE 8000

# Run as a non-root user — best practice for containers, required by some
# AWS security policies.
RUN adduser --disabled-password --gecos "" lumina
USER lumina

# Start uvicorn. --workers 1 is correct for a single ECS task; scale
# horizontally by running more tasks rather than more processes per task.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
