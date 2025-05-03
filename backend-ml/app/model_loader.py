import joblib
import os
import re
import numpy as np

# Adjust these paths as needed
model_path = os.path.join("app", "models", "sentiment_model.pkl")
vectorizer_path = os.path.join("app", "models", "vectorizer.pkl")

# Enhanced text cleaning function
def clean_text(text: str) -> str:
    """Clean and preprocess text for sentiment analysis"""
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    
    return text.strip()

# Enhanced sentiment lexicon for confidence boosting
positive_lexicon = {
    'amazing': 3.0, 'awesome': 3.0, 'excellent': 3.0, 'fantastic': 3.0, 'outstanding': 3.0,
    'perfect': 3.0, 'wonderful': 3.0, 'brilliant': 3.0, 'great': 2.5, 'love': 2.5,
    'best': 2.5, 'superb': 2.5, 'incredible': 2.5, 'terrific': 2.5, 'exceptional': 2.5,
    'good': 2.0, 'nice': 2.0, 'well': 2.0, 'like': 2.0, 'happy': 2.0,
    'enjoy': 2.0, 'impressive': 2.0, 'recommend': 2.0, 'favorite': 2.0, 'pleased': 2.0
}

negative_lexicon = {
    'terrible': 3.0, 'awful': 3.0, 'horrible': 3.0, 'worst': 3.0, 'disgusting': 3.0,
    'pathetic': 3.0, 'dreadful': 3.0, 'abysmal': 3.0, 'bad': 2.5, 'hate': 2.5,
    'poor': 2.5, 'disappointed': 2.5, 'frustrating': 2.5, 'disappointing': 2.5, 'useless': 2.5,
    'annoying': 2.0, 'mediocre': 2.0, 'problem': 2.0, 'issues': 2.0, 'fail': 2.0,
    'fails': 2.0, 'sucks': 2.0, 'waste': 2.0, 'boring': 2.0, 'broken': 2.0
}

# Load the trained model and vectorizer with error handling
try:
    # Try joblib loader first (our enhanced model uses joblib)
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully with joblib")
except Exception as e:
    # Fall back to pickle if joblib fails
    print(f"Joblib loading failed: {e}. Trying pickle...")
    import pickle
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        print("Model and vectorizer loaded successfully with pickle")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Create dummy model and vectorizer as fallback
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        
        print("Creating fallback model for demonstration")
        vectorizer = CountVectorizer()
        vectorizer.fit(["positive text", "negative text"])
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        model.coef_ = np.array([[0.5, -0.5]])
        model.intercept_ = np.array([0])

# Function to predict sentiment with confidence scores
def predict_sentiment_with_confidence(text):
    """
    Predict sentiment with confidence scores
    
    Returns:
        dict: Contains prediction, confidence, and explanation
    """
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize the text
    text_vector = vectorizer.transform([cleaned_text])
    
    # Get prediction
    try:
        # For calibrated model with predict_proba
        prediction = model.predict(text_vector)[0]
        
        # Try to get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vector)[0]
            confidence = max(probabilities)
            
            # Apply lexicon-based confidence boost
            lexicon_confidence = 0
            for word in cleaned_text.split():
                if word in positive_lexicon and prediction == 1:
                    lexicon_confidence += positive_lexicon[word] / 10
                elif word in negative_lexicon and prediction == 0:
                    lexicon_confidence += negative_lexicon[word] / 10
            
            # Cap and apply the boost
            lexicon_confidence = min(lexicon_confidence, 0.2)
            adjusted_confidence = min(confidence + lexicon_confidence, 0.99)
        else:
            # If model doesn't have predict_proba, use basic confidence
            adjusted_confidence = 0.7  # Default confidence level
    except:
        # Fallback for any errors
        prediction = 1 if "good" in cleaned_text or "great" in cleaned_text else 0
        adjusted_confidence = 0.6
    
    # Get influential words if possible
    influential_words = []
    try:
        if hasattr(model, 'coef_') or (hasattr(model, 'base_estimator') and hasattr(model.base_estimator, 'coef_')):
            # Get coefficients from base model if it's a calibrated model
            if hasattr(model, 'base_estimator'):
                coefficients = model.base_estimator.coef_[0]
            else:
                coefficients = model.coef_[0]
                
            # Get vocabulary
            vocabulary = vectorizer.vocabulary_
            
            # Find influential words in text
            for word in set(cleaned_text.split()):
                if word in vocabulary:
                    idx = vocabulary[word]
                    influence = coefficients[idx]
                    influential_words.append((word, float(influence)))
            
            # Sort by absolute influence
            influential_words.sort(key=lambda x: abs(x[1]), reverse=True)
            influential_words = influential_words[:5]  # Top 5
    except Exception as e:
        print(f"Could not extract influential words: {e}")
    
    # Format the sentiment prediction
    sentiment_label = "positive" if prediction == 1 else "negative"
    
    # Format the result
    result = {
        "sentiment": sentiment_label,
        "confidence": f"{adjusted_confidence:.2%}",
        "confidence_score": float(adjusted_confidence),
        "cleaned_text": cleaned_text,
        "influential_words": influential_words
    }
    
    return result