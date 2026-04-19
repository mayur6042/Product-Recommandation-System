import streamlit as st
import pandas as pd
import pickle
import re
from collections import Counter
import os
import matplotlib.pyplot as plt

# ==============================
# LOAD MODEL (ROBUST PATH)
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# ==============================
# FUNCTIONS
# ==============================

# 🔍 Predict Sentiment
def predict_sentiment(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# 🔑 Extract Keywords
def extract_keywords(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)
    common = Counter(words).most_common(10)
    return [word for word, count in common]

# 👍 👎 Extract Pros & Cons
def extract_pros_cons(reviews):
    pros = []
    cons = []

    for review in reviews:
        sentiment = predict_sentiment(review)
        if sentiment == "positive":
            pros.append(review)
        else:
            cons.append(review)

    return pros, cons

# 🤖 Recommendation Logic (Improved)
def recommend(sentiments):
    positive = sentiments.count("positive")
    negative = sentiments.count("negative")

    total = positive + negative

    if total == 0:
        return "⚠️ Not enough data"

    score = positive / total

    if score > 0.6:
        return "✅ Strongly Recommended"
    elif score > 0.4:
        return "⚖️ Average Product"
    else:
        return "❌ Not Recommended"

# ==============================
# STREAMLIT UI
# ==============================

st.title("🛍️ Product Review Analyzer + Recommendation System")

# 🔹 Single Review
st.subheader("Analyze Single Review")
user_input = st.text_area("Enter your review:")

if st.button("Analyze Review"):
    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:
        result = predict_sentiment(user_input)
        st.success(f"Sentiment: {result}")

# 🔹 CSV Upload
st.subheader("Upload Reviews CSV")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    try:
        df = pd.read_csv(file, quotechar='"', encoding='utf-8', on_bad_lines='skip')

        if "review" not in df.columns:
            st.error("CSV must contain 'review' column")
        else:
            # Predict sentiments
            df["Sentiment"] = df["review"].astype(str).apply(predict_sentiment)

            # Show data
            st.write("### 📊 Results")
            st.write(df)

            # Summary
            st.write("### 📈 Sentiment Summary")
            summary = df["Sentiment"].value_counts()
            st.write(summary)

            # 📊 PIE CHART
            st.write("### 🥧 Sentiment Distribution")
            fig1, ax1 = plt.subplots()
            ax1.pie(summary, labels=summary.index, autopct='%1.1f%%')
            st.pyplot(fig1)

            # 📊 BAR CHART
            st.write("### 📊 Sentiment Count")
            fig2, ax2 = plt.subplots()
            ax2.bar(summary.index, summary.values)
            st.pyplot(fig2)

            # 🔑 Keywords
            all_text = " ".join(df["review"].astype(str))
            keywords = extract_keywords(all_text)

            st.write("### 🔑 Top Keywords")
            st.info(", ".join(keywords))

            # 👍 👎 Pros & Cons
            pros, cons = extract_pros_cons(df["review"].tolist())

            col1, col2 = st.columns(2)

            with col1:
                st.write("### 👍 Pros")
                for p in pros[:5]:
                    st.success(p)

            with col2:
                st.write("### 👎 Cons")
                for c in cons[:5]:
                    st.error(c)

            # 🤖 Recommendation
            decision = recommend(df["Sentiment"].tolist())

            st.write("### 🤖 Final Recommendation")

            if "Recommended" in decision:
                st.success(decision)
            else:
                st.error(decision)

    except Exception as e:
        st.error(f"Error reading file: {e}")