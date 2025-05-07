// API endpoints for the application
export const API_URL = {
    basic: "http://localhost:8000/predict",
    detailed: "http://localhost:8000/predict/detailed",
    batch: "http://localhost:8000/predict/batch",
  };
  
  // Sample API request/response examples for documentation
  export const API_EXAMPLES = {
    basic: {
      request: `{
    "text": "Your text to analyze"
  }`,
      response: `{
    "sentiment": "positive",
    "confidence": "92.5%"
  }`
    },
    detailed: {
      request: `{
    "text": "Your text to analyze"
  }`,
      response: `{
    "sentiment": "positive",
    "confidence": "99.00%",
    "confidence_score": 0.99,
    "cleaned_text": "your text to analyze",
    "influential_words": [
      {"word": "text", "score": 0.2, "sentiment": "neutral"}
    ]
  }`
    },
    batch: {
      request: `[
    {"text": "I love this product!"},
    {"text": "This is terrible."}
  ]`,
      response: `[
    {
      "sentiment": "positive",
      "confidence": "92.5%"
    },
    {
      "sentiment": "negative",
      "confidence": "87.6%"
    }
  ]`
    }
  };