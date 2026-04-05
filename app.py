import streamlit as st
import pickle

# Load the model and vectorizer
model = pickle.load(open(r'C:\Users\Nitin Patil\Documents\spam_model.pkl', 'rb'))
tfidf = pickle.load(open(r'C:\Users\Nitin Patil\Documents\tfidf_vectorizer.pkl', 'rb'))

# App title
st.title("📧 Email Spam Detector")
st.write("Type any email below and check if it is Spam or Not!")

# Text input box
email_input = st.text_area("Enter Email Text Here:", height=200)

# Button
if st.button("Check Email"):
    if email_input.strip() == "":
        st.warning("Please enter some email text!")
    else:
        vectorized = tfidf.transform([email_input])
        prediction = model.predict(vectorized)[0]
        if prediction == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT SPAM!")