import streamlit as st
import pickle
import numpy as np

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
# MODEL LOADING & BACKEND LOGIC (UNTOUCHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        with open("spam_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

model, vectorizer = load_assets()

# Preset examples
SPAM_EXAMPLE = "CONGRATULATIONS! You've been selected to win a $1,000 Walmart Gift Card. Click here immediately to claim your reward before it expires: http://bit.ly/claim-now-free"
SAFE_EXAMPLE = "Hi Team,\n\nPlease find attached the agenda for tomorrow's strategy alignment call at 10:00 AM. Let me know if you have any questions or items to add to the deck.\n\nBest regards,\nAlex"

# State management for text input and auto-run trigger
if "email_input" not in st.session_state:
    st.session_state.email_input = ""
if "trigger_check" not in st.session_state:
    st.session_state.trigger_check = False

def set_spam_example():
    st.session_state.email_input = SPAM_EXAMPLE
    st.session_state.trigger_check = True

def set_safe_example():
    st.session_state.email_input = SAFE_EXAMPLE
    st.session_state.trigger_check = True

# -----------------------------------------------------------------------------
# ADAPTIVE CSS DESIGN SYSTEM (LIGHT & DARK MODE SUPPORT)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Main Container Constraint */
    .main .block-container {
        max-width: 700px !important;
        padding-top: 2rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Header Styling */
    .app-title {
        font-size: 30px !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 15px !important;
        font-weight: 400 !important;
        opacity: 0.7;
        margin-bottom: 20px !important;
    }

    /* Model Accuracy Display Card */
    .accuracy-card {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: rgba(125, 140, 160, 0.08);
        border: 1px solid rgba(125, 140, 160, 0.2);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 20px;
    }
    .accuracy-label {
        font-size: 13px;
        font-weight: 600;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .accuracy-value {
        font-size: 18px;
        font-weight: 700;
        color: #2563EB;
    }

    /* Text Area Tweaks */
    .stTextArea textarea {
        border-radius: 8px !important;
        font-size: 15px !important;
        padding: 12px !important;
    }

    /* General Button Tweaks */
    .stButton>button {
        border-radius: 8px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }

    /* Primary Check Button Override */
    div.element-container:has(button[key="check_btn"]) button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin-top: 4px;
    }
    div.element-container:has(button[key="check_btn"]) button:hover {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }

    /* Character Counter */
    .char-counter {
        font-size: 13px;
        opacity: 0.7;
        text-align: right;
        margin-top: -12px;
        margin-bottom: 16px;
    }

    /* Adaptive Result Cards */
    .result-card {
        border-radius: 10px;
        padding: 18px 20px;
        margin-top: 20px;
    }
    .result-spam {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .result-safe {
        background-color: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .result-header {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .confidence-badge {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .result-desc {
        font-size: 14px;
        line-height: 1.5;
        opacity: 0.9;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid rgba(125, 140, 160, 0.2);
        font-size: 13px;
        opacity: 0.6;
        line-height: 1.5;
    }

    /* Mobile Responsive Optimizations */
    @media (max-width: 640px) {
        .main .block-container {
            padding-top: 1rem !important;
        }
        .app-title {
            font-size: 24px !important;
        }
        .app-subtitle {
            font-size: 13px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# APPLICATION HEADER & MODEL CARD
# -----------------------------------------------------------------------------
st.markdown('<div class="app-title">📧 Email Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Machine Learning Based Email Classification</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="accuracy-card">
        <span class="accuracy-label">Model Accuracy</span>
        <span class="accuracy-value">97.48%</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# EXAMPLE BUTTONS (FILLS TEXT AND TRIGGERS CLASSIFICATION INSTANTLY)
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.button("Spam Example", on_click=set_spam_example, use_container_width=True)

with col2:
    st.button("Safe Example", on_click=set_safe_example, use_container_width=True)

# -----------------------------------------------------------------------------
# INPUT TEXTAREA & METRICS
# -----------------------------------------------------------------------------
email_text = st.text_area(
    label="Email Content Input",
    value=st.session_state.email_input,
    placeholder="Paste your email here...",
    height=180,
    label_visibility="collapsed",
    key="email_input_area",
)

# Sync state
st.session_state.email_input = email_text

char_count = len(email_text)
st.markdown(
    f'<div class="char-counter">Character Count: <strong>{char_count}</strong></div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# PREDICTION LOGIC & RESULT CARD
# -----------------------------------------------------------------------------
check_clicked = st.button("Check Email", key="check_btn", use_container_width=True)

# Run classification if 'Check Email' clicked OR an Example button was clicked
if check_clicked or st.session_state.trigger_check:
    # Reset auto-trigger flag
    st.session_state.trigger_check = False
    
    if not email_text.strip():
        st.warning("Please enter or paste an email message to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model or vectorizer assets are missing. Please check your .pkl files.")
    else:
        with st.spinner("Analyzing Email..."):
            transformed_input = vectorizer.transform([email_text])
            prediction = model.predict(transformed_input)[0]
            
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_input)[0]
                confidence = float(np.max(probabilities) * 100)
            else:
                confidence = 100.0

        if prediction == 1 or str(prediction).lower() == "spam":
            st.markdown(
                f"""
                <div class="result-card result-spam">
                    <div class="result-header">⚠ Spam Detected</div>
                    <div class="confidence-badge">Confidence: {confidence:.2f}%</div>
                    <div class="result-desc">
                        This email exhibits traits commonly associated with unsolicited or malicious communications, such as suspicious links, high-pressure language, or unverified offers.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card result-safe">
                    <div class="result-header">✓ Legitimate Email</div>
                    <div class="confidence-badge">Confidence: {confidence:.2f}%</div>
                    <div class="result-desc">
                        This email appears safe and resembles normal corporate or personal correspondence. No standard phishing or spam markers were detected.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Developed by Shreya Shukla<br>
        Machine Learning Project &bull; 2026
    </div>
    """,
    unsafe_allow_html=True,
)
