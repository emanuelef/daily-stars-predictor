# daily-stars-predictor

A FastAPI service that predicts future GitHub star trends for any repository using [Prophet](https://facebook.github.io/prophet/) time series forecasting.

Given a GitHub repo, it fetches the historical daily star count, fits a Prophet model, and returns a 60-day forecast with confidence intervals and trend data.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.14+

## Getting started

```bash
# Install dependencies
uv sync

# Run the server (with hot reload)
uv run uvicorn daily_stars_predictor.main:app --host 0.0.0.0 --port 8082 --reload
```

## API

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8082/health
# {"message": "Ciao"}
```

### `GET /predict?repo={owner/repo}`

Returns a 60-day star forecast for the given GitHub repository.

```bash
curl "http://localhost:8082/predict?repo=astral-sh/uv"
```

**Response:**

| Field | Description |
|-------|-------------|
| `forecast_data` | 61 entries with `ds` (date), `yhat` (predicted stars), `yhat_lower` / `yhat_upper` (confidence interval) |
| `forecast_trend` | Full time series trend component |

Results are cached in-memory for 10 days.

## Testing

```bash
uv run pytest tests/ -v
```

## Docker

```bash
# Build locally
docker build -t daily-stars-predictor .

# Run
docker run -p 8082:8082 daily-stars-predictor

# Or pull from GHCR
docker run -p 8082:8082 ghcr.io/emanuelef/daily-stars-predictor:main

# Run as a background daemon with auto-restart
docker run --restart=always -d -p 8082:8082 ghcr.io/emanuelef/daily-stars-predictor:main
```
