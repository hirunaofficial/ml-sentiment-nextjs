from fastapi import FastAPI, HTTPException
from typing import List
from app.schemas import TextIn, PredictionOut, DetailedPredictionOut
from app.model_loader import predict_sentiment_with_confidence
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logger = logging.getLogger("sentiment-api")

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text with enhanced confidence scores",
    version="1.0.0"
)

# Allow CORS (for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Sentiment Analysis API",
        "endpoints": [
            {"path": "/predict", "method": "POST", "description": "Simple sentiment prediction"},
            {"path": "/predict/detailed", "method": "POST", "description": "Detailed sentiment analysis with confidence"}
        ]
    }

@app.post("/predict", response_model=PredictionOut)
def predict_sentiment(data: TextIn):
    """
    Get a simple sentiment prediction (positive/negative)
    """
    if not data.text or len(data.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text must be at least 3 characters long")
    
    try:
        # Get prediction with confidence but only return the sentiment
        result = predict_sentiment_with_confidence(data.text)
        logger.info(f"Predicted sentiment: {result['sentiment']} for text: {data.text[:30]}...")
        return {"sentiment": result["sentiment"]}
    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing sentiment")

@app.post("/predict/detailed", response_model=DetailedPredictionOut)
def predict_sentiment_detailed(data: TextIn):
    """
    Get a detailed sentiment prediction with confidence and explanation
    """
    if not data.text or len(data.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text must be at least 3 characters long")
    
    try:
        # Get full prediction details
        result = predict_sentiment_with_confidence(data.text)
        
        # Format influential words for output
        influential_words = [
            {"word": word, "score": float(score), "sentiment": "positive" if score > 0 else "negative"}
            for word, score in result["influential_words"]
        ]
        
        logger.info(f"Detailed prediction - Sentiment: {result['sentiment']}, Confidence: {result['confidence']}")
        
        return {
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "confidence_score": result["confidence_score"],
            "cleaned_text": result["cleaned_text"],
            "influential_words": influential_words
        }
    except Exception as e:
        logger.error(f"Error predicting detailed sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing detailed sentiment")

@app.post("/predict/batch", response_model=List[DetailedPredictionOut])
def predict_batch(data: List[TextIn]):
    """
    Process multiple texts in a single request
    """
    if not data:
        raise HTTPException(status_code=400, detail="Empty batch request")
    
    if len(data) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 items")
    
    try:
        results = []
        for item in data:
            if not item.text or len(item.text.strip()) < 3:
                # Skip invalid items
                continue
                
            result = predict_sentiment_with_confidence(item.text)
            
            # Format influential words for output
            influential_words = [
                {"word": word, "score": float(score), "sentiment": "positive" if score > 0 else "negative"}
                for word, score in result["influential_words"]
            ]
            
            results.append({
                "sentiment": result["sentiment"],
                "confidence": result["confidence"],
                "confidence_score": result["confidence_score"],
                "cleaned_text": result["cleaned_text"],
                "influential_words": influential_words
            })
        
        logger.info(f"Batch prediction completed for {len(results)} items")
        return results
    except Exception as e:
        logger.error(f"Error processing batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch prediction")