# daily_stars_predictor

`poetry run uvicorn main:app --host 0.0.0.0 --port 8082 --reload`

`docker run -p 8082:8082 daily-stars-predictor`

`docker run -p 8082:8082 ghcr.io/emanuelef/daily-stars-predictor:main`

```bash
docker run --restart=always -d -p 8082:8082 ghcr.io/emanuelef/daily-stars-predictor:main
```
