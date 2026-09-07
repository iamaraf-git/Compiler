"""
Streamlit Web Application for Sentiment Analysis
Run with:
    streamlit run app.py
"""
import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.preprocessor import clean_text

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
CONF_MATRIX_PATH = os.path.join(BASE_DIR, "models", "confusion_matrix.png")

# Page configuration
st.set_page_config(
    page_title="Movie Reviews Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .sentiment-box {
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Natural Language Processing (NLP) pipeline powered by TF-IDF & Logistic Regression</div>', unsafe_allow_html=True)

# Check model availability
models_ready = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)

if not models_ready:
    st.warning("⚠️ Model artifacts not found. Click the button below to train the model directly.")
    if st.button("🚀 Train Model Now"):
        with st.spinner("Training model on dataset... Please wait."):
            from src.train import train_and_evaluate
            train_and_evaluate()
            st.success("Model trained and saved successfully! Refreshing...")
            st.rerun()
else:
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Project Overview")
        st.write("""
        **Pipeline Steps:**
        1. **Preprocessing:** Regex cleaning, HTML/URL stripping, lowercasing.
        2. **Feature Extraction:** TF-IDF (Unigrams + Bigrams, 5000 features).
        3. **Classification:** Logistic Regression with L2 Regularization.
        """)
        st.markdown("---")
        st.write("**Dataset:** IMDb Movie Reviews Benchmark")
        st.write("**Target Classes:** Positive (1) vs Negative (0)")
        
        if os.path.exists(CONF_MATRIX_PATH):
            st.markdown("---")
            st.subheader("📊 Confusion Matrix")
            st.image(CONF_MATRIX_PATH, caption="Test Set Evaluation Matrix", use_container_width=True)

    # Main UI Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Live Prediction", "🧪 Batch Tester", "📖 Model Details & Theory"])

    with tab1:
        st.subheader("Analyze Single Review")

        # Preset test buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        sample_text = ""
        
        if col_btn1.button("✨ Load Positive Example"):
            sample_text = "This movie is a brilliant masterpiece! The acting was superb and the visuals were stunning."
        if col_btn2.button("⚠️ Load Negative Example"):
            sample_text = "Terrible script and awful directing. Total waste of time and complete bore."
        if col_btn3.button("🤔 Load Mixed Example"):
            sample_text = "While the cinematography was quite beautiful, the plot dragged on and felt incoherent."

        # Input text area
        user_input = st.text_area(
            "Enter your movie review:",
            value=sample_text if sample_text else "",
            height=130,
            placeholder="Type or paste any movie review here..."
        )

        if st.button("🚀 Analyze Sentiment", type="primary"):
            if not user_input.strip():
                st.warning("Please enter some review text to analyze.")
            else:
                model = joblib.load(MODEL_PATH)
                vectorizer = joblib.load(VECTORIZER_PATH)

                # Preprocess & predict
                cleaned = clean_text(user_input)
                features = vectorizer.transform([cleaned])
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]

                prob_neg = probabilities[0] * 100
                prob_pos = probabilities[1] * 100

                col_res1, col_res2 = st.columns([1, 1])

                with col_res1:
                    if prediction == 1:
                        st.success(f"### Result: Positive Sentiment 😊\n**Confidence:** `{prob_pos:.2f}%`")
                    else:
                        st.error(f"### Result: Negative Sentiment 😞\n**Confidence:** `{prob_neg:.2f}%`")

                with col_res2:
                    st.write("**Probability Breakdown:**")
                    prob_df = pd.DataFrame({
                        "Sentiment": ["Negative", "Positive"],
                        "Probability (%)": [prob_neg, prob_pos]
                    })
                    st.bar_chart(prob_df.set_index("Sentiment"), color="#2563EB")

                with st.expander("🛠️ View Preprocessed Text"):
                    st.code(cleaned, language="text")

    with tab2:
        st.subheader("🧪 Test Multiple Reviews at Once")
        batch_input = st.text_area(
            "Enter multiple reviews (one per line):",
            height=150,
            value="Great acting and lovely direction!\nAwful film, hated every second.\nA charming story with great emotional depth."
        )
        if st.button("Analyze Batch"):
            lines = [l.strip() for l in batch_input.split("\n") if l.strip()]
            if lines:
                model = joblib.load(MODEL_PATH)
                vectorizer = joblib.load(VECTORIZER_PATH)
                results = []
                for line in lines:
                    cl = clean_text(line)
                    ft = vectorizer.transform([cl])
                    pred = model.predict(ft)[0]
                    probs = model.predict_proba(ft)[0]
                    results.append({
                        "Review": line,
                        "Predicted Sentiment": "Positive 😊" if pred == 1 else "Negative 😞",
                        "Confidence": f"{max(probs)*100:.2f}%"
                    })
                st.dataframe(pd.DataFrame(results), use_container_width=True)

    with tab3:
        st.subheader("📖 NLP Pipeline Architecture & Theory")
        st.markdown("""
        ### 1. Term Frequency-Inverse Document Frequency (TF-IDF)
        TF-IDF computes the importance of words relative to the whole corpus:
        $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$
        $$\\text{IDF}(t, D) = \\ln\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$
        
        ### 2. Logistic Regression Classifier
        Predicts sentiment probability using the Sigmoid (logistic) function:
        $$P(y=1|x) = \\frac{1}{1 + e^{-(\\mathbf{w}^T \\mathbf{x} + b)}}$$
        
        ### 3. Key Advantages for this Project:
        * Fast training and low memory footprint.
        * Interpretable feature weights (which words contribute to positive vs negative ratings).
        * Robust generalization benchmark across NLP tasks.
        """)
