from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Bulldozer Price Prediction API")

MODEL_PATH = "bulldozer_price_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Model load failed: {e}")

class Features(BaseModel):
    YearMade: int
    MachineHoursCurrentMeter: float | None = None
    ModelID: int
    ProductSize: int | None = None

@app.get("/")
def root():
    return {"message": "Welcome to the Bulldozer Price Prediction API"}



@app.post("/predict")
def predict(payload: Features):
    if model is None:
        return {"error": "Model not loaded"}

    # Build a 56-length vector filled with zeros
    x = np.zeros(56)

    # Place known features in the first few slots
    x[0] = payload.YearMade
    x[1] = payload.MachineHoursCurrentMeter if payload.MachineHoursCurrentMeter else 0.0
    x[2] = payload.ModelID
    x[3] = payload.ProductSize if payload.ProductSize else 0

    try:
        y_pred = model.predict(x.reshape(1, -1))
        return {"prediction": float(y_pred[0])}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
