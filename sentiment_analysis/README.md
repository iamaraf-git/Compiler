# 🎬 IMDb Movie Reviews Sentiment Analysis

An end-to-end Natural Language Processing (NLP) and Supervised Machine Learning project for binary sentiment classification (Positive vs. Negative) on movie reviews.

---

## 📌 1. Project Overview & Motivation

Sentiment analysis is a subfield of Natural Language Processing (NLP) that aims to quantify and categorize affective states and subjective opinions from raw text. This project processes customer movie reviews, cleans the text, transforms sentences into high-dimensional numerical vectors using **TF-IDF**, and classifies them using **Logistic Regression** and **Multinomial Naive Bayes**.

---

## 🏗️ 2. Pipeline Architecture

```
Raw Review Text (HTML, URLs, Punctuation)
                    │
                    ▼
          [ 1. Text Preprocessing ]
  • Remove HTML tags & URLs
  • Convert to lowercase
  • Strip non-alphabet characters
                    │
                    ▼
          [ 2. Feature Extraction ]
  • TF-IDF Vectorization (Unigrams + Bigrams)
  • Fixed vocabulary size: 5,000 features
                    │
                    ▼
          [ 3. Model Classification ]
  • Logistic Regression (L2 Regularization)
  • Multinomial Naive Bayes benchmark
                    │
                    ▼
  [ 4. Sentiment Output + Confidence Score ]
```

---

## 📁 3. Directory Structure

```
Compiler/sentiment_analysis/
├── data/
│   ├── sample_reviews.csv          # Pre-packaged balanced reviews dataset
│   └── download_dataset.py         # 50k IMDb dataset fetcher
├── models/
│   ├── sentiment_model.pkl         # Trained model artifact
│   ├── tfidf_vectorizer.pkl        # Saved TF-IDF feature extractor
│   └── confusion_matrix.png        # Evaluation heatmap
├── src/
│   ├── __init__.py
│   ├── preprocessor.py             # Reusable text cleaning functions
│   ├── train.py                    # Model training & metrics generation
│   └── predict.py                  # Fast CLI inference tool
├── app.py                          # Interactive Streamlit Web Interface
├── Sentiment_Analysis_Project.ipynb# Jupyter Notebook for lab presentation
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🚀 4. How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python src/train.py
```
*This will train the model, output accuracy & classification metrics, generate `models/confusion_matrix.png`, and save the serialized model weights.*

### Step 3: Test with CLI
```bash
python src/predict.py "This movie was absolutely amazing and breathtaking!"
```
Or run interactively:
```bash
python src/predict.py
```

### Step 4: Launch the Web App
```bash
streamlit run app.py
```
Open the provided URL (`http://localhost:8501`) in your browser to interact with the GUI.

---

## 📊 5. Mathematical & Theoretical Background

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
TF-IDF reflects how important a word is to a document in a collection or corpus:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### 2. Logistic Regression
Calculates the posterior probability of the positive class using the sigmoid activation function:
$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
Optimized using Binary Cross-Entropy (Log-Loss):
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

---

## 📈 6. Evaluation Metrics
* **Accuracy:** Percentage of correctly classified sentiments on the unseen test set.
* **Precision:** Ratio of correctly predicted positive observations to total predicted positive observations.
* **Recall:** Ratio of correctly predicted positive observations to all observations in actual class.
* **F1-Score:** Weighted harmonic mean of Precision and Recall.
