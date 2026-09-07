# 🚀 NLP & Compiler Construction Lab Course

A repository containing complete NLP lab experiments, quick reference cheat sheets, and an end-to-end Sentiment Analysis machine learning project.

---

## 📂 Repository Structure

```
├── Lab_01_Text_Preprocessing.ipynb         # Lab 1: Lowercase, Digits, Punctuation, Stopwords
├── Lab_02_Regex_and_Tokenization.ipynb     # Lab 2: Regex Extraction & Custom Tokenizer
├── Lab_03_NLTK_and_Bengali_NLP.ipynb       # Lab 3: NLTK Tokenization, FreqDist & Bengali NLP
├── NLP_LAB_3MIN_CHEAT_SHEET.md             # ⚡ 3-Minute Fast Revision Card for Lab Tests
└── sentiment_analysis/                     # 🎬 End-to-End Sentiment Analysis ML Project
    ├── app.py                              # Streamlit Interactive Web Application
    ├── Sentiment_Analysis_Project.ipynb    # All-in-one Project Notebook
    ├── STUDENT_WORKFLOW_GUIDE.md           # Beginner's In-Depth Step-by-Step Guide
    ├── README.md                           # Detailed Project & Math Documentation
    ├── requirements.txt                    # Project Dependencies
    ├── data/                               # Dataset Loader & Sample Reviews
    ├── models/                             # Trained Logistic Regression & Vectorizer
    └── src/                                # Preprocessing, Training & CLI Predictor
```

---

## 🧪 Lab Experiments Summary

* **Lab 1:** Text preprocessing fundamentals: lowercasing, digit removal with `\d`, punctuation stripping with `[^\w\s]`, and stopword elimination using NLTK.
* **Lab 2:** Advanced pattern matching with Regular Expressions (`re.findall`) for email extraction, numbers, and custom token frequency counting using Python's `defaultdict` and `OrderedDict`.
* **Lab 3:** Standard NLTK tokenization (`word_tokenize`, `sent_tokenize`), frequency distributions (`FreqDist`), and Multilingual Bengali NLP processing using `indic-nlp-library`.

---

## 🎬 Sentiment Analysis Project Quickstart

```bash
cd sentiment_analysis
pip install -r requirements.txt
python src/train.py         # Train model (89.43% test accuracy)
streamlit run app.py        # Launch web application
```
