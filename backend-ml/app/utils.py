import re
import string
import nltk
from typing import Dict, Any, List, Tuple, Union

# Try to load nltk resources (with fallback if not available)
try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    STOPWORDS.update(['rt', 'via', 'amp', 'gt', 'lt'])  # Twitter-specific stopwords
except:
    # Basic stopwords if NLTK is not available
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'when', 'where', 'how', 'is', 'am', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'to', 'for', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
        'just', 'should', 'now', 'rt', 'via', 'amp', 'gt', 'lt'
    }

# Enhanced sentiment lexicons
POSITIVE_WORDS = {
    'amazing': 3.0, 'awesome': 3.0, 'excellent': 3.0, 'fantastic': 3.0, 'outstanding': 3.0,
    'perfect': 3.0, 'wonderful': 3.0, 'brilliant': 3.0, 'great': 2.5, 'love': 2.5,
    'best': 2.5, 'superb': 2.5, 'incredible': 2.5, 'terrific': 2.5, 'exceptional': 2.5,
    'good': 2.0, 'nice': 2.0, 'well': 2.0, 'like': 2.0, 'happy': 2.0,
    'enjoy': 2.0, 'impressive': 2.0, 'recommend': 2.0, 'favorite': 2.0, 'pleased': 2.0
}

NEGATIVE_WORDS = {
    'terrible': 3.0, 'awful': 3.0, 'horrible': 3.0, 'worst': 3.0, 'disgusting': 3.0,
    'pathetic': 3.0, 'dreadful': 3.0, 'abysmal': 3.0, 'bad': 2.5, 'hate': 2.5,
    'poor': 2.5, 'disappointed': 2.5, 'frustrating': 2.5, 'disappointing': 2.5, 'useless': 2.5,
    'annoying': 2.0, 'mediocre': 2.0, 'problem': 2.0, 'issues': 2.0, 'fail': 2.0,
    'fails': 2.0, 'sucks': 2.0, 'waste': 2.0, 'boring': 2.0, 'broken': 2.0
}

def clean_text(text: str) -> str:
    """
    Enhanced text cleaning function for sentiment analysis
    """
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

def advanced_clean_text(text: str) -> str:
    """
    More advanced text cleaning with stopword removal
    """
    # Basic cleaning first
    text = clean_text(text)
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    
    return " ".join(words)

def extract_text_features(text: str) -> Dict[str, Any]:
    """
    Extract features from text for sentiment analysis enhancement
    """
    # Clean text for feature extraction
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    
    features = {
        "word_count": len(words),
        "char_count": len(cleaned_text),
        "exclamation_count": text.count('!'),
        "question_count": text.count('?'),
        "uppercase_word_count": sum(1 for word in text.split() if word.isupper()),
        "positive_word_count": sum(1 for word in words if word in POSITIVE_WORDS),
        "negative_word_count": sum(1 for word in words if word in NEGATIVE_WORDS),
        "positive_score": sum(POSITIVE_WORDS.get(word, 0) for word in words),
        "negative_score": sum(NEGATIVE_WORDS.get(word, 0) for word in words),
    }
    
    # Add negation detection
    negations = {'not', 'no', 'never', "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't"}
    features["has_negation"] = any(neg in words for neg in negations)
    
    return features

def calculate_lexicon_confidence_boost(text: str, prediction: int) -> float:
    """
    Calculate confidence boost based on sentiment lexicons
    
    Args:
        text: Cleaned text
        prediction: 1 for positive, 0 for negative
        
    Returns:
        float: Confidence boost (0.0-0.2)
    """
    words = text.split()
    boost = 0.0
    
    for word in words:
        if prediction == 1 and word in POSITIVE_WORDS:
            boost += POSITIVE_WORDS[word] / 10  # Scale down the boost
        elif prediction == 0 and word in NEGATIVE_WORDS:
            boost += NEGATIVE_WORDS[word] / 10  # Scale down the boost
    
    # Cap the boost at 0.2 (20%)
    return min(boost, 0.2)

def format_prediction_result(
    text: str, 
    sentiment: str, 
    confidence: float, 
    influential_words: List[Tuple[str, float]] = None
) -> Dict[str, Any]:
    """
    Format prediction results for API output
    """
    if influential_words is None:
        influential_words = []
        
    return {
        "original_text": text,
        "cleaned_text": clean_text(text),
        "sentiment": sentiment,
        "confidence": f"{confidence:.2%}",
        "confidence_score": float(confidence),
        "influential_words": [
            {
                "word": word,
                "score": float(score),
                "sentiment": "positive" if score > 0 else "negative"
            }
            for word, score in influential_words
        ]
    }