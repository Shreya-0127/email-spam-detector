import pickle
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# DEEP STREAMLIT DOM CSS OVERRIDES (FORCED MOBILE OPTIMIZATION)
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
/* 1. FORCE REMOVE STREAMLIT DEFAULT MOBILE PADDING */
div[data-testid="stAppViewContainer"] > section {
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
}

div[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
    max-width: 600px !important;
}

/* 2. PREVENT IOS AUTO-ZOOM & MAKE INPUT TOUCH-FRIENDLY */
div[data-testid="stTextArea"] textarea {
    font-size: 16px !important; /* Mandatory to stop iOS safari zoom on focus */
    border-radius: 12px !important;
    padding: 14px !important;
    line-height: 1.4 !important;
}

/* 3. MOBILE-OPTIMIZED TOUCH BUTTONS (MIN 48PX HEIGHT) */
div[data-testid="stButton"] button {
    min-height: 48px !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    touch-action: manipulation;
}

/* 4. PRIMARY CHECK BUTTON - STYLED DIRECTLY */
div[data-testid="stVerticalBlock"] > div:has(button[key="check_btn"]) button,
div[data-testid="stButton"]:last-of-type button {
    background-color: #1C2E4A !important;
    color: #FFFFFF !important;
    border: none !important;
    font-size: 1.05rem !important;
    margin-top: 0.5rem !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12) !important;
}

/* 5. HEADER & INFO CARD MOBILE FIXES */
.hero-title {
    font-size: 1.6rem !important;
    font-weight: 700;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 0.875rem;
    opacity: 0.75;
    margin-top: 0.25rem;
    margin-bottom: 1rem;
}

.info-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: rgba(125, 140, 160, 0.08);
    border: 1px solid rgba(125, 140, 160, 0.18);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 1rem;
    font-size: 0.85rem;
}

/* 6. ADAPTIVE RESULT CARDS */
.result-box {
    border-radius: 12px;
    padding: 1.1rem;
    text-align: center;
    margin-top: 1.25rem;
}
.result-spam {
    background-color: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.35);
}
.result-ham {
    background-color: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.result-heading {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.result-details {
    font-size: 0.85rem;
    opacity: 0.85;
}

/* 7. FOOTER */
.footer {
    text-align: center;
    font-size: 0.78rem;
    opacity: 0.6;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(125, 140, 160, 0.2);
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# MODEL LOADING & ORIGINAL LOGIC
# -----------------------------------------------------------------------------
model = pickle.load(open("spam_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ── HEADER ──
st.markdown(
    """
<div>
    <div class="hero-title">📧 Email Spam Detector</div>
    <div class="hero-subtitle">Instantly detect if any email is spam or legitimate</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── METRIC BADGE ──
st.markdown(
    """
<div class="info-card">
    <span>💡 Model Accuracy</span>
    <span style="color: #2563EB; font-weight: 700;">97.48%</span>
</div>
""",
    unsafe_allow_html=True,
)

# ── EXAMPLE BUTTONS ──
st.write("**Try an example:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("🚨 Spam Example"):
        st.session_state.email_text = (
            "Congratulations! You won a free iPhone. Click now to claim your prize worth $1000!"
        )
with col2:
    if st.button("✅ Normal Example"):
        st.session_state.email_text = (
            "Hey, are we still meeting for lunch tomorrow at 1pm?"
        )

# ── TEXT INPUT ──
st.write("")
email_input = st.text_area(
    "Enter Email Text Here:",
    value=st.session_state.get("email_text", ""),
    height=160,
    placeholder="Type or paste your email here...",
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
            st.markdown(
                f"""
            <div class="result-box result-spam">
                <div class="result-heading">🚨 SPAM DETECTED</div>
                <div class="result-details">Confidence: {confidence}% — This email looks like spam</div>
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            confidence = round(proba[0] * 100, 1)
            st.markdown(
                f"""
            <div class="result-box result-ham">
                <div class="result-heading">✅ NOT SPAM</div>
                <div class="result-details">Confidence: {confidence}% — This email looks safe</div>
            </div>""",
                unsafe_allow_html=True,
            )

# ── FOOTER ──
st.markdown(
    """
<div class="footer">
    © 2026 Shreya Shukla &nbsp;|&nbsp; Email Spam Detector &nbsp;|&nbsp; ML Project
</div>
""",
    unsafe_allow_html=True,
)
