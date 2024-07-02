from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import requests

from function import preprocess_data, train_model, load_model, predict

app = FastAPI(
    title="Machine Learning API",
    description="API for training and using a machine learning model with endpoints categorized and well-documented."
)

class TrainingData(BaseModel):
    features: List[List[float]]
    labels: List[int]

class PredictionData(BaseModel):
    features: List[List[float]]

model = None
model_path = "model/model.joblib"

@app.post("/training", summary="Train the model", response_description="Model training response")
async def train_model_endpoint(data: TrainingData):
    global model
    X, y = preprocess_data(data.features, data.labels)
    model = train_model(X, y, model_path)
    return {"message": "Model trained successfully"}

@app.post("/predict", summary="Make a prediction", response_description="Prediction response")
async def make_prediction(
    data: PredictionData = Body(
        ...,
        example={
            "features": [[4, 180, 80, 40, 130, 40.0, 1.2, 50], [2, 120, 70, 25, 0, 30.0, 0.5, 35]]
        }
    )
):
    global model
    if model is None:
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    X, _ = preprocess_data(data.features)
    predictions = predict(model, X)
    return {"predictions": predictions}

@app.get("/model", summary="Get model info")
async def get_model_info():
    return {"model": "Logistic Regression", "library": "Scikit-learn"}

@app.get("/huggingface_model", summary="Get model info from HuggingFace")
async def get_huggingface_model_info():
    response = requests.get("https://api-inference.huggingface.co/models/distilbert-base-uncased", headers={"Authorization": "Bearer <your_huggingface_token>"})
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )

@app.exception_handler(TypeError)
async def type_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"message": "Invalid input type"}
    )

@app.exception_handler(HTTPException)
async def http_error_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )
