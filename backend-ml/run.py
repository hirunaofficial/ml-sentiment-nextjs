import uvicorn
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("sentiment-api")

# Check if model files exist before starting
def check_model_files():
    model_path = os.path.join("app", "models", "sentiment_model.pkl")
    vectorizer_path = os.path.join("app", "models", "vectorizer.pkl")
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return False
    
    if not os.path.exists(vectorizer_path):
        logger.warning(f"Vectorizer file not found: {vectorizer_path}")
        return False
        
    return True

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs(os.path.join("app", "models"), exist_ok=True)
    
    # Check model files
    models_exist = check_model_files()
    if not models_exist:
        logger.warning("Model files missing. The API will use fallback models.")
    
    # Log startup message
    logger.info("Starting Sentiment Analysis API server")
    
    # Run the FastAPI application
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )