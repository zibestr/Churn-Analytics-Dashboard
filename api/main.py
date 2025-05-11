from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import joblib
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')


@app.on_event("startup")
def load_models():
    global preprocessor, model
    try:
        preprocessor = joblib.load(os.path.join(MODEL_PATH, 'feature_transformer.pkl'))
        model = joblib.load(os.path.join(MODEL_PATH, 'survival_analysis.pkl'))
    except Exception as e:
        raise RuntimeError(f"Error loading models: {e}")


class PredictionRequest(BaseModel):
    data: List[Dict]
    days: List[int]


class PredictionResponse(BaseModel):
    customerID: str
    probabilities: Dict[int, float]


def predict_survival(input_data, days):
    try:
        features = pd.DataFrame(input_data).drop(columns=['customerID'])
        transformed = preprocessor.transform(features)

        surv_funcs = model.predict_survival_function(transformed)

        return [
            {day: round(float(func(day)), 3) for day in days}
            for func in surv_funcs
        ]
    except Exception as e:
        raise ValueError(str(e))


@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: PredictionRequest):
    required_cols = ["customerID","gender","SeniorCitizen","Partner","Dependents",
                    "PhoneService","MultipleLines","InternetService","OnlineSecurity",
                    "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
                    "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
                    "MonthlyCharges","TotalCharges"]
    
    if not all(col in request.data[0] for col in required_cols):
        raise HTTPException(status_code=400, detail="Missing required columns")

    try:
        preds = predict_survival(request.data, request.days)
        return [
            PredictionResponse(
                customerID=str(row['customerID']),
                probabilities={day: preds[i][day] for day in request.days}
            )
            for i, row in enumerate(request.data)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
