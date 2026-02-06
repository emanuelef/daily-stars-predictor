from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from daily_stars_predictor.main import app, cache


@pytest.fixture(autouse=True)
def clear_cache():
    cache.clear()
    yield
    cache.clear()


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "Ciao"}


def _make_stars_data(days=120):
    """Generate fake stars data matching the external API format."""
    base = datetime(2024, 1, 1)
    return [
        [(base + timedelta(days=i)).strftime("%d-%m-%Y"), i, 0]
        for i in range(days)
    ]


def _mock_httpx_client(stars_data):
    """Create a properly mocked httpx.AsyncClient for async with usage."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"stars": stars_data}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    return mock_client


@patch("daily_stars_predictor.main.httpx.AsyncClient")
def test_predict_returns_forecast(mock_client_cls):
    mock_client = _mock_httpx_client(_make_stars_data())
    mock_client_cls.return_value = mock_client

    response = client.get("/predict?repo=test/repo")

    assert response.status_code == 200
    data = response.json()
    assert "forecast_data" in data
    assert "forecast_trend" in data
    assert len(data["forecast_data"]) == 61
    entry = data["forecast_data"][0]
    assert "ds" in entry
    assert "yhat" in entry
    assert "yhat_lower" in entry
    assert "yhat_upper" in entry


@patch("daily_stars_predictor.main.httpx.AsyncClient")
def test_predict_caches_result(mock_client_cls):
    mock_client = _mock_httpx_client(_make_stars_data())
    mock_client_cls.return_value = mock_client

    # First call
    resp1 = client.get("/predict?repo=cached/repo")
    assert resp1.status_code == 200

    # Second call should use cache
    mock_client_cls.reset_mock()
    resp2 = client.get("/predict?repo=cached/repo")
    assert resp2.status_code == 200
    assert resp1.json() == resp2.json()


def test_predict_missing_repo_param():
    response = client.get("/predict")
    assert response.status_code == 422


# --- /predict/statsmodels tests ---


@patch("daily_stars_predictor.main.httpx.AsyncClient")
def test_predict_statsmodels_returns_forecast(mock_client_cls):
    mock_client = _mock_httpx_client(_make_stars_data())
    mock_client_cls.return_value = mock_client

    response = client.get("/predict/statsmodels?repo=test/repo")

    assert response.status_code == 200
    data = response.json()
    assert "forecast_data" in data
    assert "forecast_trend" in data
    assert len(data["forecast_data"]) == 61
    entry = data["forecast_data"][0]
    assert "ds" in entry
    assert "yhat" in entry
    assert "yhat_lower" in entry
    assert "yhat_upper" in entry


@patch("daily_stars_predictor.main.httpx.AsyncClient")
def test_predict_statsmodels_caches_result(mock_client_cls):
    mock_client = _mock_httpx_client(_make_stars_data())
    mock_client_cls.return_value = mock_client

    resp1 = client.get("/predict/statsmodels?repo=cached/sm-repo")
    assert resp1.status_code == 200

    mock_client_cls.reset_mock()
    resp2 = client.get("/predict/statsmodels?repo=cached/sm-repo")
    assert resp2.status_code == 200
    assert resp1.json() == resp2.json()


def test_predict_statsmodels_missing_repo_param():
    response = client.get("/predict/statsmodels")
    assert response.status_code == 422
