"""
Standalone CLI Prediction Script for Sentiment Analysis
Usage:
    python src/predict.py "The movie was fantastic and emotional!"
    python src/predict.py
"""
import os
import sys
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.preprocessor import clean_text

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

def predict_sentiment(text: str):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("[!] Model files not found. Please train the model first by running:\n    python src/train.py")
        return None, 0.0

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[1] if prediction == 1 else probabilities[0]

    return sentiment, confidence

def main():
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        sentiment, conf = predict_sentiment(input_text)
        if sentiment:
            print(f"\nReview:     \"{input_text}\"")
            print(f"Sentiment:  {sentiment} ({conf * 100:.2f}% confidence)\n")
    else:
        print("=" * 55)
        print("  Interactive Sentiment Analysis CLI (Type 'exit' to quit)")
        print("=" * 55)
        while True:
            try:
                user_input = input("\nEnter a review: ").strip()
                if not user_input or user_input.lower() in ['exit', 'quit']:
                    break
                sentiment, conf = predict_sentiment(user_input)
                if sentiment:
                    emoji = "😊" if sentiment == "Positive" else "😞"
                    print(f"Result: {emoji} {sentiment} (Confidence: {conf * 100:.2f}%)")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
