from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from backend.utils import TextClassifier
from contextlib import asynccontextmanager
import logging

# Logger
logger = logging.getLogger(__name__)

# Load model on startup
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    model_path = "./models/text_classification_bert_model"
    #start up
    classifier = TextClassifier(model_path)
    logger.info(f"Loaded model successfully from path {model_path}")
    yield
    logger.info("Shutting down API...")

# Initialize FastAPI
app = FastAPI(
    title="Text Classification API",
    description="API for classifying text using fine-tuned BERT model",
    version="1.0.0",
    lifespan=lifespan
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
async def get_classes():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        'classes': classifier.label_encoder.classes_.tolist(),
        'num_classes': len(classifier.label_encoder.classes_)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = classifier.predict(request.text, top_k=request.top_k)
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


