"""
Training Pipeline for Sentiment Analysis
Trains TF-IDF + Logistic Regression (and Naive Bayes), evaluates performance,
and saves the trained model artifacts.
"""
import os
import sys
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directory to sys.path so imports work properly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.preprocessor import clean_text
from data.download_dataset import get_dataset

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
CONF_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")

def train_and_evaluate():
    print("=" * 60)
    print("       SENTIMENT ANALYSIS TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Dataset
    df = get_dataset(prefer_full=True)
    print(f"[*] Total dataset records: {len(df)}")
    print(f"[*] Class distribution:\n{df['sentiment'].value_counts()}\n")

    # 2. Text Preprocessing
    print("[*] Cleaning text data...")
    df['cleaned_review'] = df['review'].apply(lambda x: clean_text(x, remove_stopwords=False))

    # Standardize label encoding: positive -> 1, negative -> 0
    df['label'] = df['sentiment'].apply(lambda x: 1 if str(x).lower().strip() == 'positive' else 0)

    X = df['cleaned_review']
    y = df['label']

    # 3. Train-Test Split (80% Train, 20% Test)
    # Stratified split to ensure balanced test representation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 4. Feature Extraction (TF-IDF with Unigrams and Bigrams)
    print("[*] Extracting TF-IDF Features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 5. Model Training (Logistic Regression)
    print("[*] Training Logistic Regression Classifier...")
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Optional: Benchmark against Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_preds = nb_model.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_preds)

    # 6. Evaluation
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f" Logistic Regression Accuracy: {acc * 100:.2f}%")
    print(f" Multinomial Naive Bayes Accuracy: {nb_acc * 100:.2f}%")
    print("=" * 60)

    print("\n[+] Classification Report (Logistic Regression):")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # 7. Generate & Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title("Sentiment Classification Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix visualization saved to: {CONF_MATRIX_PATH}")

    # 8. Save Serialized Artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[+] Model saved to: {MODEL_PATH}")
    print(f"[+] Vectorizer saved to: {VECTORIZER_PATH}")
    print("\n🎉 Training complete! You are ready to run inference or launch the web app.")

if __name__ == "__main__":
    train_and_evaluate()
