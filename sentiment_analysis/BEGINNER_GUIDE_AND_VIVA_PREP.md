# 🎬 IMDb Movie Review Sentiment Analysis: Beginner's Guide & Viva Cheat Sheet

---

## 📌 1. What is this Project? (The Big Picture)

Imagine you have **50,000 movie reviews** written by users on the internet. 
- You want the computer to automatically read any review and tell you: **Is this person happy (Positive 😊) or angry (Negative 😞)?**

Since computers **cannot read English words** (they only understand numbers and math), this project builds an end-to-end Natural Language Processing (NLP) pipeline that translates English text into numerical features, trains a Machine Learning model, and makes live sentiment predictions.

---

## 🏗️ 2. The 5-Stage Project Pipeline

```
Raw Review Text ("The acting was brilliant and breathtaking!")
                         │
                         ▼
        [ Stage 1: Data Preprocessing & Stemming ]
     • Remove HTML tags (<br />) & punctuation
     • Convert to lowercase
     • Remove English stopwords ("is", "the", "and")
     • Reduce words to root form ("brilliant" -> "brilliant", "acting" -> "act")
                         │
                         ▼
        [ Stage 2: TF-IDF Feature Extraction ]
     • Turn cleaned words into numerical score vectors
     • High weights for emotional words, low weights for common words
                         │
                         ▼
        [ Stage 3: Train-Test Split (80% / 20%) ]
     • 40,000 reviews for Training
     • 10,000 unseen reviews for Testing
                         │
                         ▼
        [ Stage 4: Logistic Regression Model ]
     • The "Brain" learns patterns separating positive from negative reviews
     • Achieves ~89% Accuracy
                         │
                         ▼
        [ Stage 5: Live Inference & Prediction ]
     • 1 = Positive Sentiment 😊
     • 0 = Negative Sentiment 😞
```

---

## 🪜 3. Step-by-Step Breakdown of the Code

### **Step 1: Download & Extract Dataset**
- We download the **IMDb Dataset of 50K Movie Reviews** directly from Kaggle using the Kaggle API.
- We extract the `.zip` file using Python's built-in `ZipFile`.

### **Step 2: Load Data & Map Labels**
- The dataset has **50,000 reviews**: 25,000 positive and 25,000 negative.
- We map labels:
  - `"positive"` $\rightarrow$ **`1`**
  - `"negative"` $\rightarrow$ **`0`**
  *(Because machine learning algorithms require numeric targets).*

### **Step 3: Text Cleaning & Stemming (NLP Preprocessing)**
1. **Remove HTML tags**: Web-scraped IMDb reviews contain `<br />` line breaks.
2. **Strip non-alphabets**: Remove punctuation and numbers using `re.sub('[^a-zA-Z]', ' ', text)`.
3. **Lowercase**: Ensures `"Good"`, `"GOOD"`, and `"good"` are seen as identical.
4. **Remove Stopwords**: Eliminates filler words (*"in"*, *"at"*, *"the"*, *"is"*) using NLTK.
5. **Porter Stemming**: Reduces words to base root (*"loved"*, *"loving"*, *"loves"* $\rightarrow$ *"love"*).

### **Step 4: Train-Test Split**
- We divide the dataset:
  - **$80\%$ (40,000 reviews)**: Used to teach the model.
  - **$20\%$ (10,000 reviews)**: Kept secret to test the model's true accuracy.
- `stratify=Y` maintains a 50/50 balance in both sets.

### **Step 5: TF-IDF Vectorization**
- **TF (Term Frequency)**: How many times a word appears in a review.
- **IDF (Inverse Document Frequency)**: Gives higher importance to rare, emotionally charged words and lower weight to common words.

### **Step 6: Logistic Regression Training**
- We train `LogisticRegression(max_iter=1000)`.
- Reaches **$\approx 89\%$ test accuracy**.

### **Step 7: Model Serialization & Live Prediction**
- We save the trained model (`trainedmodel.sav`) and vectorizer (`vectorizer.pkl`) with **`pickle`**.
- When new reviews are entered by a user, the system cleans the text, vectorizes it, and outputs **1 (Positive)** or **0 (Negative)**.

---

## 🎓 4. Teacher / Viva Exam Cheat Sheet (Questions & Answers)

### **Q1: Why did you convert "positive" and "negative" to 1 and 0?**
> **Answer:** Machine learning models (like Logistic Regression) rely on mathematical optimization and linear algebra. They cannot compute equations on string text, so we map the binary classes to $1$ (Positive) and $0$ (Negative).

---

### **Q2: What is Stemming, and why did you use `PorterStemmer`?**
> **Answer:** Stemming is the process of trimming word suffixes to obtain the root base word (e.g., *"acting"*, *"acted"*, *"actor"* $\rightarrow$ *"act"*). It reduces vocabulary dimensionality, saves memory, and ensures the model treats different grammatical tenses of the same word consistently.

---

### **Q3: What are Stopwords, and why remove them?**
> **Answer:** Stopwords are high-frequency grammatical connector words (such as *"the"*, *"is"*, *"in"*, *"and"*). They contain almost zero emotional sentiment. Removing them cleans up noise and allows the model to focus on sentiment-bearing words.

---

### **Q4: What is TF-IDF in simple terms?**
> **Answer:** TF-IDF stands for *Term Frequency - Inverse Document Frequency*. It converts text into numerical scores:
> - **TF**: Measures how frequently a word occurs in a document.
> - **IDF**: Penalizes words that appear across all documents and rewards distinctive, informative words.

---

### **Q5: Why split data into Training (80%) and Testing (20%)?**
> **Answer:** If we test the model on the same data it learned from, it might just memorize answers (overfitting). Splitting data ensures we test the model on completely unseen reviews to measure real-world performance.

---

### **Q6: Why did you choose Logistic Regression for this project?**
> **Answer:** Logistic Regression is the industry-standard linear baseline for binary text classification with TF-IDF. It is fast, interpretable, computationally lightweight, and achieves high accuracy ($\approx 89\%$) on large sparse text matrices.

---

### **Q7: Why did you save the model using `pickle`?**
> **Answer:** `pickle` serializes Python objects into binary files on disk. This allows us to load the trained model instantly in production or demo web apps without waiting to re-train the model every time.
