import streamlit as st
import pickle

# Page config
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .title { color: #1a73e8; font-size: 42px; font-weight: bold; text-align: center; }
    .subtitle { color: #555; font-size: 16px; text-align: center; margin-bottom: 30px; }
    .result-spam { background-color: #fde8e8; border-left: 6px solid #e53935;
                   padding: 15px; border-radius: 8px; font-size: 20px; font-weight: bold; }
    .result-ham { background-color: #e8f5e9; border-left: 6px solid #43a047;
                  padding: 15px; border-radius: 8px; font-size: 20px; font-weight: bold; }
    .info-box { background-color: #e3f2fd; border-radius: 10px;
                padding: 15px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open('spam_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/email.png")
    st.title("About This App")
    st.write("This Email Spam Detector uses Machine Learning to classify emails as Spam or Not Spam.")
    st.markdown("---")
    st.write("**Model:** Naive Bayes")
    st.write("**Accuracy:** 97.48%")
    st.write("**Dataset:** 5,572 emails")
    st.markdown("---")
    st.write("Built by **Shreya Shukla**")

# Main title
st.markdown('<p class="title">📧 Email Spam Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Type any email below and instantly find out if it is Spam or Not!</p>', unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
💡 <b>How it works:</b> Our AI model analyzes the words in your email and predicts whether it is spam or not with 97.48% accuracy!
</div>
""", unsafe_allow_html=True)

# Example buttons
st.write("### Try an example:")
col1, col2 = st.columns(2)
with col1:
    if st.button("🚨 Spam Example"):
        st.session_state.email_text = "Congratulations! You won a free iPhone. Click now to claim your prize!"
with col2:
    if st.button("✅ Normal Example"):
        st.session_state.email_text = "Hey, are we still meeting for lunch tomorrow at 1pm?"

# Text area
email_input = st.text_area(
    "Enter Email Text Here:",
    value=st.session_state.get('email_text', ''),
    height=200,
    placeholder="Type or paste your email here..."
)

# Check button
if st.button("🔍 Check Email", use_container_width=True):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter some email text first!")
    else:
        vectorized = tfidf.transform([email_input])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        st.markdown("---")
        st.write("### Result:")

        if prediction == 1:
            confidence = round(proba[1] * 100, 2)
            st.markdown(f'<div class="result-spam">🚨 This email is SPAM! &nbsp;&nbsp; Confidence: {confidence}%</div>', unsafe_allow_html=True)
        else:
            confidence = round(proba[0] * 100, 2)
            st.markdown(f'<div class="result-ham">✅ This email is NOT SPAM! &nbsp;&nbsp; Confidence: {confidence}%</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤️ by Shreya Shukla | ML Project 2026</center>", unsafe_allow_html=True)