# Sentiment Analysis API

A FastAPI application for sentiment analysis with improved confidence scores and detailed predictions.

## Features

- High-confidence sentiment prediction (positive/negative)
- Detailed analysis with confidence scores and influential words
- Batch processing capabilities
- Robust error handling and fallbacks
- REST API with Swagger documentation

## API Endpoints

### Basic Sentiment Prediction

```
POST /predict
```

Request:
```json
{
  "text": "I love this product!"
}
```

Response:
```json
{
  "sentiment": "positive"
}
```

### Detailed Sentiment Analysis

```
POST /predict/detailed
```

Request:
```json
{
  "text": "I love this product!"
}
```

Response:
```json
{
  "sentiment": "positive",
  "confidence": "92.5%",
  "confidence_score": 0.925,
  "cleaned_text": "i love this product",
  "influential_words": [
    {"word": "love", "score": 2.34, "sentiment": "positive"},
    {"word": "product", "score": 0.42, "sentiment": "positive"}
  ]
}
```

### Batch Processing

```
POST /predict/batch
```

Process multiple texts in a single request.

Request:
```json
[
  {"text": "I love this product!"},
  {"text": "This is terrible."}
]
```

## Project Structure

```
app/
├── __init__.py
├── main.py             # FastAPI application logic
├── model_loader.py     # Model loading and prediction functions
├── schemas.py          # Pydantic models for request/response
├── utils.py            # Utility functions for text processing
├── models/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
main.py                 # Application entry point
requirements.txt        # Project dependencies
```

## Model Details

The sentiment analysis model uses:
- Calibrated Logistic Regression for better probability estimates
- TF-IDF vectorization with n-grams
- Sentiment lexicons for confidence boosting
- Feature extraction for improved accuracy

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

- Author: Hiruna Gallage
- Website: [hiruna.dev](https://hiruna.dev)
- Email: [hello@hiruna.dev](mailto:hello@hiruna.dev)