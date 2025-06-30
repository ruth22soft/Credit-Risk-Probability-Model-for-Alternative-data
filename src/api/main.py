# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")  # Load model trained in Step 3

class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: int
    Monetary: float

@app.post("/predict")
def predict(features: CustomerFeatures):
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)[0]
    return {"risk_probability": float(prediction)}