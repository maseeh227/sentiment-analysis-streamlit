
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.data.path.append(
    r"C:\Users\MOHAMMED MASEEH\AppData\Roaming\nltk_data"
)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

st.title("🎬 Movie Review Sentiment Analysis")

review = st.text_input("Enter Movie Review")

if st.button("Predict"):
    clean = clean_text(review)
    vec = vectorizer.transform([clean])
    result = model.predict(vec)

    if result[0] == 1:
        st.success("✅ Positive Review 😊")
    else:
        st.error("❌ Negative Review 😡")
