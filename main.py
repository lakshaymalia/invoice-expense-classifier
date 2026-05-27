from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Invoice Expense Classifier API")

# Load the trained model
MODEL_PATH = "model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class InvoiceRequest(BaseModel):
    text: str

class InvoiceResponse(BaseModel):
    category: str
    confidence: float

@app.post("/predict", response_model=InvoiceResponse)
def predict_category(request: InvoiceRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found. Run train.py first.")
    
    text = request.text
    
    # Get prediction and confidence score
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence_score = max(probabilities)
    
    return {
        "category": prediction,
        "confidence": round(float(confidence_score), 2)
    }