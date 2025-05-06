import streamlit as st
import joblib

#Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

#App title
st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📧 Spam Message Detector")

#User input
user_input = st.text_area("Enter a message:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        #Transform input and predict
        X = tfidf.transform([user_input])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        #Output
        label = "🚫 Spam" if pred == 1 else "✅ Not Spam"
        st.subheader("Prediction:")
        st.success(label)
        st.metric("Spam Probability", f"{prob:.2%}")