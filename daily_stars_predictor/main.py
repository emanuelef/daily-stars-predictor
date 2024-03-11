from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
import httpx
import pandas as pd
from prophet import Prophet

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=0)  # Compress all responses


@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}


@app.get("/predict")
async def predict(repo: str):
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

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=365, freq="D")
    print(future.tail())

    forecast = m.predict(future)

    forecast["ds"] = forecast["ds"].dt.strftime("%Y-%m-%d")

    # Extract relevant information from the forecast
    forecast_data = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    print(forecast_data.tail())

    # Combine the API data and forecast data
    result = {"forecast_data": forecast_data.to_dict(orient="records")}

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080)
