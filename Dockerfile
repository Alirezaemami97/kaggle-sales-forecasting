# --- Build stage: install dependencies ---
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install poetry==2.4.1

COPY pyproject.toml poetry.lock ./

# Install only production deps (no dev tools) into an in-project venv
RUN poetry config virtualenvs.in-project true \
    && poetry install --only main --no-interaction --no-root

# --- Runtime stage: lean final image ---
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the built venv from the builder stage
COPY --from=builder /app/.venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source and config
COPY src/ src/
COPY config/ config/

# Make the demand_forecasting package importable without a pip install step.
ENV PYTHONPATH=/app/src

# This image runs batch jobs, not a server. The default command is the data
# conversion step; training and inference jobs override it, e.g.:
#   docker run <image> python -m demand_forecasting.training.train --config config/config.yaml
CMD ["python", "-m", "demand_forecasting.data.convert", "--config", "config/config.yaml"]
