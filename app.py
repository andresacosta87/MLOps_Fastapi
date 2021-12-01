from fastapi import FastAPI, Depends
from joblib import load
import numpy as np
import uvicorn 
from pathlib import Path
from pydantic import BaseModel, validator 
from typing import List

from Model import Model, get_model, n_features

class PredictRequest(BaseModel):
    data: List[List[float]]

    @validator("data")
    def check_dimensionality(cls, v):
        for point in v:
            if len(point) != n_features:
                raise ValueError(f"Each data point must contain {n_features} features")

class PredictResponse(BaseModel):
    data:list[float]

app = FastAPI()

@app.post("/predict", response_model = PredictResponse)
def predict(input: PredictRequest, model: Model =Depends(get_model)):
    X = np.array(input.data)
    y = model.predict(X)
    result = PredictResponse(data = y_pred.tolist())

    return result

if __name__=='__main__':
    uvicorn.run(app, host ='127.0.0.1', port=8000)

