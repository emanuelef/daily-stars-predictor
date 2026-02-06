FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev group, no editable install)
RUN uv sync --no-dev --no-install-project --frozen

# Copy application code
COPY daily_stars_predictor/. .

EXPOSE 8082

CMD ["uv", "run", "--no-dev", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082"]
