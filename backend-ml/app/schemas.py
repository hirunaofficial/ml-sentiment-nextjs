from pydantic import BaseModel, Field
from typing import List, Optional

class TextIn(BaseModel):
    text: str = Field(..., description="Text to analyze for sentiment")

class PredictionOut(BaseModel):
    sentiment: str = Field(..., description="Sentiment prediction (positive or negative)")

class InfluentialWord(BaseModel):
    word: str = Field(..., description="Word from the text")
    score: float = Field(..., description="Influence score (positive or negative value)")
    sentiment: str = Field(..., description="Whether this word contributes to positive or negative sentiment")

class DetailedPredictionOut(BaseModel):
    sentiment: str = Field(..., description="Sentiment prediction (positive or negative)")
    confidence: str = Field(..., description="Confidence score as a percentage string (e.g., '85.4%')")
    confidence_score: float = Field(..., description="Confidence score as a float between 0 and 1")
    cleaned_text: str = Field(..., description="Preprocessed text used for prediction")
    influential_words: List[InfluentialWord] = Field([], description="List of words that influenced the prediction")

    class Config:
        schema_extra = {
            "example": {
                "sentiment": "positive",
                "confidence": "87.5%",
                "confidence_score": 0.875,
                "cleaned_text": "i love this product it is amazing",
                "influential_words": [
                    {"word": "love", "score": 2.34, "sentiment": "positive"},
                    {"word": "amazing", "score": 1.89, "sentiment": "positive"},
                    {"word": "product", "score": 0.42, "sentiment": "positive"}
                ]
            }
        }