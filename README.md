# daily-stars-predictor

A FastAPI service that predicts future GitHub star trends for any repository using time series forecasting.

Given a GitHub repo, it fetches the historical daily star count, fits a forecasting model, and returns a 60-day forecast with confidence intervals and trend data. Two engines are available: [Prophet](https://facebook.github.io/prophet/) and [statsmodels Holt-Winters](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html).

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

Returns a 60-day star forecast using **Prophet**.

```bash
curl "http://localhost:8082/predict?repo=astral-sh/uv"
```

### `GET /predict/statsmodels?repo={owner/repo}`

Returns a 60-day star forecast using **Holt-Winters Exponential Smoothing** (faster, no Stan dependency).

```bash
curl "http://localhost:8082/predict/statsmodels?repo=astral-sh/uv"
```

Both endpoints return the same response shape:

| Field | Description |
|-------|-------------|
| `forecast_data` | 61 entries with `ds` (date), `yhat` (predicted stars), `yhat_lower` / `yhat_upper` (confidence interval) |
| `forecast_trend` | Full time series trend component |

Results are cached in-memory for 2 days.

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

# Run as a background daemon with auto-restart (works with Watchtower)
docker run --restart=always -d -p 8082:8082 ghcr.io/emanuelef/daily-stars-predictor:main
```

### Watchtower

If you use [Watchtower](https://containrrr.dev/watchtower/) it will automatically detect new image pushes to GHCR and restart the container:

```bash
docker run -d --name watchtower -v /var/run/docker.sock:/var/run/docker.sock containrrr/watchtower
```
