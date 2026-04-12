import streamlit as st
import pickle

# Page config
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# CSS
st.markdown("""
<style>
/* Header hero */
.hero {
    background: linear-gradient(135deg, #2F4156 0%, #567C8D 100%);
    padding: 2rem 1.5rem 1.5rem 1.5rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-tagline {
    font-size: 1rem;
    color: #C8D9E6;
    margin-top: 0.4rem;
    margin-bottom: 0;
}

/* Info box */
.info-box {
    background: linear-gradient(135deg, #567C8D, #C8D9E6);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 1.2rem;
    color: #FFFFFF;
    font-size: 0.95rem;
    font-weight: 500;
}

/* Example buttons */
.stButton > button {
    border-radius: 10px;
    border: 2px solid #567C8D;
    background-color: transparent;
    color: #567C8D;
    font-weight: 600;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background-color: #567C8D;
    color: white;
}

/* Check button */
div[data-testid="stButton"]:last-of-type > button {
    background: linear-gradient(135deg, #2F4156, #567C8D);
    color: white;
    border: none;
    font-size: 1.1rem;
    padding: 0.6rem;
    border-radius: 12px;
    font-weight: 700;
}

/* Result cards - visible in both dark and light mode */
.result-spam {
    background: #c0392b;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-top: 1rem;
    color: #FFFFFF !important;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: 0.3px;
}
.result-ham {
    background: #1e8449;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-top: 1rem;
    color: #FFFFFF !important;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: 0.3px;
}
.result-sub {
    font-size: 0.9rem;
    font-weight: 400;
    margin-top: 0.3rem;
    color: #f0f0f0;
}

/* Footer */
.footer {
    text-align: center;
    color: #567C8D;
    font-size: 0.85rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #C8D9E6;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open('spam_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# ── HERO HEADER ──
st.markdown("""
<div class="hero">
    <p class="hero-title">📧 Email Spam Detector</p>
    <p class="hero-tagline">Instantly detect if any email is spam or not</p>
</div>
""", unsafe_allow_html=True)

# ── INFO BOX ──
st.markdown("""
<div class="info-box">
    💡 Our AI model analyzes email content and predicts spam with <b>97.48% accuracy</b>
</div>
""", unsafe_allow_html=True)

# ── EXAMPLE BUTTONS ──
st.write("**Try an example:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("🚨 Spam Example"):
        st.session_state.email_text = "Congratulations! You won a free iPhone. Click now to claim your prize worth $1000!"
with col2:
    if st.button("✅ Normal Example"):
        st.session_state.email_text = "Hey, are we still meeting for lunch tomorrow at 1pm?"

# ── TEXT INPUT ──
st.write("")
email_input = st.text_area(
    "Enter Email Text Here:",
    value=st.session_state.get('email_text', ''),
    height=180,
    placeholder="Type or paste your email here..."
)

# ── CHECK BUTTON ──
st.write("")
if st.button("🔍 Check Email", use_container_width=True):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter some email text first!")
    else:
        vectorized = tfidf.transform([email_input])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        if prediction == 1:
            confidence = round(proba[1] * 100, 1)
            st.markdown(f"""
            <div class="result-spam">
                🚨 SPAM DETECTED
                <div class="result-sub">Confidence: {confidence}% — This email looks like spam</div>
            </div>""", unsafe_allow_html=True)
        else:
            confidence = round(proba[0] * 100, 1)
            st.markdown(f"""
            <div class="result-ham">
                ✅ NOT SPAM
                <div class="result-sub">Confidence: {confidence}% — This email looks safe</div>
            </div>""", unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div class="footer">
    © 2026 Shreya Shukla &nbsp;|&nbsp; Email Spam Detector &nbsp;|&nbsp; ML Project
</div>
""", unsafe_allow_html=True)
