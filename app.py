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
# MODEL LOADING & BACKEND LOGIC
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("spam_model.pkl", "rb"))
        vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, tfidf = load_assets()

# Preset examples
SPAM_EXAMPLE = "Congratulations! You won a free iPhone. Click now to claim your prize worth $1000!"
SAFE_EXAMPLE = "Hey, are we still meeting for lunch tomorrow at 1pm?"

# Session state initialization
if "email_text" not in st.session_state:
    st.session_state.email_text = ""
if "trigger_check" not in st.session_state:
    st.session_state.trigger_check = False

def load_spam():
    st.session_state.email_text = SPAM_EXAMPLE
    st.session_state.trigger_check = True

def load_safe():
    st.session_state.email_text = SAFE_EXAMPLE
    st.session_state.trigger_check = True

# -----------------------------------------------------------------------------
# ADAPTIVE CSS DESIGN SYSTEM (LIGHT & DARK MODE + MOBILE RESPONSIVE)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Typography & Font Reset */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Centered Container Constraints */
    .main .block-container {
        max-width: 680px !important;
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Minimal Native Header */
    .app-header {
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-bottom: 0.25rem !important;
    }
    .app-subtitle {
        font-size: 0.95rem !important;
        opacity: 0.7;
        margin-bottom: 0 !important;
    }

    /* Professional Metric Card */
    .metric-card {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: rgba(125, 140, 160, 0.08);
        border: 1px solid rgba(125, 140, 160, 0.18);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 1.5rem;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.75;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2563EB;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        padding: 14px !important;
        border: 1px solid rgba(125, 140, 160, 0.25) !important;
    }

    /* Character Counter */
    .char-counter {
        font-size: 0.8rem;
        opacity: 0.65;
        text-align: right;
        margin-top: -0.75rem;
        margin-bottom: 1.25rem;
    }

    /* Example Buttons Styling */
    .stButton>button {
        border-radius: 8px !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border: 1px solid rgba(125, 140, 160, 0.25) !important;
        transition: all 0.15s ease !important;
    }

    /* Primary Submit Button Override */
    div.element-container:has(button[key="check_btn"]) button {
        background-color: #1C2E4A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.25rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
    }
    div.element-container:has(button[key="check_btn"]) button:hover {
        background-color: #0F1A2B !important;
        color: #FFFFFF !important;
    }

    /* Adaptive Result Alert Cards */
    .result-card {
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .result-spam {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .result-ham {
        background-color: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .result-header {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .result-badge {
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .result-desc {
        font-size: 0.875rem;
        line-height: 1.5;
        opacity: 0.85;
    }

    /* Clean Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.25rem;
        border-top: 1px solid rgba(125, 140, 160, 0.2);
        font-size: 0.825rem;
        opacity: 0.65;
        line-height: 1.6;
    }

    /* Mobile Adaptations */
    @media (max-width: 640px) {
        .main .block-container {
            padding-top: 1.5rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        .app-title {
            font-size: 1.5rem !important;
        }
        .app-subtitle {
            font-size: 0.875rem !important;
        }
        .result-card {
            padding: 1rem 1.15rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HEADER & METRIC CARD
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">📧 Email Spam Detector</div>
        <div class="app-subtitle">Machine learning tool for real-time email classification</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="metric-card">
        <span class="metric-label">Model Accuracy</span>
        <span class="metric-value">97.48%</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# EXAMPLE BUTTONS (CLICK TO AUTO-FILL & INSTANTLY EVALUATE)
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.button("🚨 Spam Example", on_click=load_spam, use_container_width=True)

with col2:
    st.button("✅ Safe Example", on_click=load_safe, use_container_width=True)

# -----------------------------------------------------------------------------
# TEXT INPUT AREA
# -----------------------------------------------------------------------------
email_input = st.text_area(
    label="Email Text Input",
    value=st.session_state.email_text,
    placeholder="Type or paste your email content here...",
    height=170,
    label_visibility="collapsed",
    key="email_textarea",
)

st.session_state.email_text = email_input

char_count = len(email_input)
st.markdown(
    f'<div class="char-counter">Character Count: <strong>{char_count}</strong></div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# PREDICTION LOGIC & RESULT DISPLAY
# -----------------------------------------------------------------------------
check_clicked = st.button("🔍 Check Email", key="check_btn", use_container_width=True)

# Trigger if user clicks Check Email OR selects an example button
if check_clicked or st.session_state.trigger_check:
    st.session_state.trigger_check = False

    if not email_input.strip():
        st.warning("⚠️ Please enter or paste some email text first.")
    elif model is None or tfidf is None:
        st.error("Model or vectorizer assets are missing.")
    else:
        with st.spinner("Analyzing email content..."):
            vectorized = tfidf.transform([email_input])
            prediction = model.predict(vectorized)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vectorized)[0]
                confidence = float(np.max(proba) * 100)
            else:
                confidence = 100.0

        if prediction == 1 or str(prediction).lower() == "spam":
            st.markdown(
                f"""
                <div class="result-card result-spam">
                    <div class="result-header">🚨 Spam Detected</div>
                    <div class="result-badge">Confidence: {confidence:.1f}%</div>
                    <div class="result-desc">
                        This email contains structural patterns and keywords frequently associated with phishing or unsolicited bulk emails.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card result-ham">
                    <div class="result-header">✅ Legitimate Email</div>
                    <div class="result-badge">Confidence: {confidence:.1f}%</div>
                    <div class="result-desc">
                        This email appears safe and aligns with normal personal or corporate correspondence.
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
    <div class="footer">
        Developed by Shreya Shukla &nbsp;|&nbsp; Machine Learning Project &nbsp;|&nbsp; 2026
    </div>
    """,
    unsafe_allow_html=True,
)
