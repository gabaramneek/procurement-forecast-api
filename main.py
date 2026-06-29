from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load(
    "model/procurement_forecast_model.pkl"
)

class ForecastInput(BaseModel):
    NumberOfPOs: float
    lag_1: float
    lag_2: float
    lag_3: float
    rolling_mean_3: float

@app.get("/")
def home():
    return {
        "message": "Procurement Forecast API is running"
    }

@app.post("/predict")
def predict(data: ForecastInput):

    input_df = pd.DataFrame([{
        "NumberOfPOs": data.NumberOfPOs,
        "lag_1": data.lag_1,
        "lag_2": data.lag_2,
        "lag_3": data.lag_3,
        "rolling_mean_3": data.rolling_mean_3
    }])

    prediction = model.predict(input_df)[0]

    return {
        "predicted_spend": float(prediction)
    }