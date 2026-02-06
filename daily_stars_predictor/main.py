from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
import httpx
import pandas as pd
from prophet import Prophet
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import math
import time
from cachetools import TTLCache

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=0)  # Compress all responses

# CORS (Cross-Origin Resource Sharing) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with the list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache with a TTL (Time To Live) of 3 days
cache = TTLCache(maxsize=1000, ttl=259200)


@app.get("/health")
async def root():
    return {"message": "Ciao"}


@app.get("/predict")
async def predict(repo: str):
    # Check if the data is already in the cache
    if repo in cache:
        cached_data = cache[repo]
        print("Returning cached data.")
        return JSONResponse(content=cached_data)

    api_url = f"http://143.47.226.125:8080/allStars?repo={repo}"
    async with httpx.AsyncClient() as client:
        api_response = await client.get(api_url)
    api_data = api_response.json()

    # Extracting stars data from the JSON
    stars_data = api_data["stars"]

    # Creating a DataFrame
    df = pd.DataFrame(stars_data, columns=["ds", "y", "y2"])

    # Converting the 'ds' column to datetime format
    df["ds"] = pd.to_datetime(df["ds"], format="%d-%m-%Y")

    print(df.tail())

    start_time = time.time()

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=60, freq="D")
    # print(future.tail())

    forecast = m.predict(future)

    forecast["ds"] = forecast["ds"].dt.strftime("%Y-%m-%d")

    # Extract relevant information from the forecast
    forecast_data = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast_trend = forecast[["ds", "trend"]]

    # Round up 'yhat' to the next integer
    forecast_data.loc[:, "yhat"] = forecast_data["yhat"].apply(math.ceil).astype(int)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Prediction took {elapsed_time:.2f} seconds")

    last_60_forecast = forecast_data.tail(61)
    # Combine the API data and forecast data
    result = {
        "forecast_data": last_60_forecast.to_dict(orient="records"),
        "forecast_trend": forecast_trend.to_dict(orient="records"),
    }

    # Cache the result
    cache[repo] = result

    return JSONResponse(content=result)


@app.get("/predict/statsmodels")
async def predict_statsmodels(repo: str):
    cache_key = f"sm:{repo}"
    if cache_key in cache:
        cached_data = cache[cache_key]
        print("Returning cached data.")
        return JSONResponse(content=cached_data)

    api_url = f"http://143.47.226.125:8080/allStars?repo={repo}"
    async with httpx.AsyncClient() as client:
        api_response = await client.get(api_url)
    api_data = api_response.json()

    stars_data = api_data["stars"]
    df = pd.DataFrame(stars_data, columns=["ds", "y", "y2"])
    df["ds"] = pd.to_datetime(df["ds"], format="%d-%m-%Y")
    df = df.set_index("ds").asfreq("D")
    df["y"] = df["y"].ffill()

    print(df.tail())

    start_time = time.time()

    model = ExponentialSmoothing(
        df["y"],
        trend="add",
        seasonal="add",
        seasonal_periods=7,
    ).fit()

    forecast_values = model.forecast(60)
    fitted_values = model.fittedvalues

    # Build confidence intervals using residual std
    residuals = df["y"] - fitted_values
    std = residuals.std()
    steps = np.arange(1, 61)
    margin = 1.645 * std * np.sqrt(steps)  # 90% CI

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1), periods=60, freq="D"
    )
    forecast_df = pd.DataFrame(
        {
            "ds": future_dates.strftime("%Y-%m-%d"),
            "yhat": forecast_values.apply(math.ceil).astype(int).values,
            "yhat_lower": (forecast_values.values - margin).round(2),
            "yhat_upper": (forecast_values.values + margin).round(2),
        }
    )

    # Include last historical day for continuity (61 entries like Prophet)
    last_day = pd.DataFrame(
        {
            "ds": [df.index[-1].strftime("%Y-%m-%d")],
            "yhat": [math.ceil(fitted_values.iloc[-1])],
            "yhat_lower": [round(fitted_values.iloc[-1] - 1.645 * std, 2)],
            "yhat_upper": [round(fitted_values.iloc[-1] + 1.645 * std, 2)],
        }
    )
    forecast_df = pd.concat([last_day, forecast_df], ignore_index=True)

    # Build trend from fitted + forecast
    all_dates = df.index.strftime("%Y-%m-%d").tolist() + future_dates.strftime("%Y-%m-%d").tolist()
    all_trend = fitted_values.tolist() + forecast_values.tolist()
    trend_df = pd.DataFrame({"ds": all_dates, "trend": all_trend})

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Prediction (statsmodels) took {elapsed_time:.2f} seconds")

    result = {
        "forecast_data": forecast_df.to_dict(orient="records"),
        "forecast_trend": trend_df.to_dict(orient="records"),
    }

    cache[cache_key] = result

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8082)
