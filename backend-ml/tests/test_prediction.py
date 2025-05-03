import requests

# Base URL for the API (adjust if deployed)
BASE_URL = "http://localhost:8000"

def test_sentiment_prediction():
    sample_text = "I love this product! It's amazing."
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": sample_text}
    )

    assert response.status_code == 200
    result = response.json()
    
    assert "sentiment" in result
    print(f"Input: {sample_text}\nPredicted Sentiment: {result['sentiment']}")

if __name__ == "__main__":
    test_sentiment_prediction()