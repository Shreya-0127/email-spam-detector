import streamlit as st
import pickle

# Page config
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# CSS - Professional, minimal, adaptive to both Light and Dark mode
st.markdown("""
<style>
/* Global Font & Reset */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Centered Layout Constraints */
.main .block-container {
    max-width: 680px !important;
    padding-top: 2rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Modern Clean Header */
.hero {
    border-bottom: 1px solid rgba(125, 140, 160, 0.2);
    padding-bottom: 1.25rem;
    margin-bottom: 1.25rem;
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero-tagline {
    font-size: 0.95rem;
    opacity: 0.7;
    margin-top: 0.25rem;
    margin-bottom: 0;
}

/* Subtle Info Badge */
.info-box {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: rgba(125, 140, 160, 0.08);
    border: 1px solid rgba(125, 140, 160, 0.18);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 1.25rem;
    font-size: 0.9rem;
    font-weight: 500;
}

/* Example Buttons Base */
.stButton > button {
    border-radius: 8px !important;
    border: 1px solid rgba(125, 140, 160, 0.25) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.15s ease !important;
    width: 100%;
}

/* Main Check Button Override */
div[data-testid="stButton"]:last-of-type > button {
    background-color: #1C2E4A !important;
    color: #FFFFFF !important;
    border: none !important;
    font-size: 1rem !important;
    padding: 0.75rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    margin-top: 0.5rem;
}

div[data-testid="stButton"]:last-of-type > button:hover {
    background-color: #0F1A2B !important;
    color: #FFFFFF !important;
}

/* Result Cards - Fully Compatible with Light & Dark Themes */
.result-spam {
    background-color: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-top: 1.25rem;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.2px;
}
.result-ham {
    background-color: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.35);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-top: 1.25rem;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.2px;
}
.result-sub {
    font-size: 0.875rem;
    font-weight: 400;
    margin-top: 0.35rem;
    opacity: 0.85;
}

/* Text Area Override */
.stTextArea textarea {
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    padding: 12px !important;
}

/* Footer Styling */
.footer {
    text-align: center;
    font-size: 0.825rem;
    opacity: 0.65;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(125, 140, 160, 0.2);
}

/* Mobile Responsiveness */
@media (max-width: 640px) {
    .main .block-container {
        padding-top: 1.25rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }
    .hero-title {
        font-size: 1.5rem !important;
    }
    .hero-tagline {
        font-size: 0.85rem !important;
    }
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
    <p class="hero-tagline">Instantly detect if any email is spam or legitimate</p>
</div>
""", unsafe_allow_html=True)

# ── INFO BOX ──
st.markdown("""
<div class="info-box">
    <span>💡 Model Classification Accuracy</span>
    <span style="color: #2563EB; font-weight: 700;">97.48%</span>
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
