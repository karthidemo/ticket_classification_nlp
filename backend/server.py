from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from backend.service import predict_text, get_class_metadata
import logging

# Logger
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Text Classification API",
    description="API for classifying text using fine-tuned BERT model",
    version="1.0.0"
)

# Request/Response models
class PredictionRequest(BaseModel):
    text: str
    top_k: Optional[int] 

class PredictionResponse(BaseModel):
    original_text: str
    processed_text: str
    predicted_class: str
    confidence: float
    all_predictions: List[dict]

# API endpoints

@app.get("/classes")
async def classes():
    return get_class_metadata()

@app.post("/predict")
async def predict(request: dict):
    result = predict_text(request)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result