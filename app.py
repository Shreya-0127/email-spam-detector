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

# Session state initialization for text area input
if "email_input" not in st.session_state:
    st.session_state.email_input = ""

# -----------------------------------------------------------------------------
# CUSTOM CSS DESIGN SYSTEM (DARK BLUE SLATE PALETTE)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* CSS Reset & Dark Mode Enforcement */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #0B1120 !important;
        color: #F8FAFC !important;
    }

    /* Hide Default Header & Menu Overlays */
    header, [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Main Container Constraint (Centered, Max ~700px) */
    .main .block-container {
        max-width: 720px !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Header Styling */
    .app-title {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        margin-bottom: 6px !important;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 15px !important;
        font-weight: 400 !important;
        color: #94A3B8 !important;
        margin-bottom: 24px !important;
    }

    /* Model Accuracy Card */
    .accuracy-card {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 24px;
    }
    .accuracy-label {
        font-size: 13px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .accuracy-value {
        font-size: 20px;
        font-weight: 700;
        color: #60A5FA;
    }

    /* Example Buttons */
    .stButton>button {
        width: 100%;
        background-color: #1E293B !important;
        color: #E2E8F0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #334155 !important;
        border-color: #64748B !important;
        color: #FFFFFF !important;
    }

    /* Streamlit Text Area Customization */
    .stTextArea textarea {
        background-color: #0F172A !important;
        border: 1.5px solid #334155 !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        font-size: 15px !important;
        padding: 14px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextArea textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    .stTextArea textarea::placeholder {
        color: #64748B !important;
    }

    /* Primary Check Button Override */
    div.element-container:has(button[key="check_btn"]) button {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 20px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3) !important;
        transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div.element-container:has(button[key="check_btn"]) button:hover {
        background-color: #2563EB !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.4) !important;
        color: #FFFFFF !important;
    }

    /* Character Counter */
    .char-counter {
        font-size: 13px;
        color: #94A3B8;
        text-align: right;
        margin-top: -12px;
        margin-bottom: 20px;
    }

    /* Result Cards */
    .result-card {
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    .result-spam {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .result-safe {
        background-color: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .result-header {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .result-spam .result-header {
        color: #FCA5A5;
    }
    .result-safe .result-header {
        color: #86EFAC;
    }
    .confidence-badge {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .result-spam .confidence-badge {
        color: #F87171;
    }
    .result-safe .confidence-badge {
        color: #4ADE80;
    }
    .result-desc {
        font-size: 14px;
        line-height: 1.5;
    }
    .result-spam .result-desc {
        color: #FECDD3;
    }
    .result-safe .result-desc {
        color: #BBF7D0;
    }

    /* Footer Styling */
    .app-footer {
        text-align: center;
        margin-top: 48px;
        padding-top: 20px;
        border-top: 1px solid #1E293B;
        font-size: 13px;
        color: #64748B;
        line-height: 1.6;
    }

    /* Mobile Responsiveness Rules */
    @media (max-width: 640px) {
        .main .block-container {
            padding-top: 1.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .app-title {
            font-size: 24px !important;
        }
        .app-subtitle {
            font-size: 14px !important;
            margin-bottom: 16px !important;
        }
        .accuracy-card {
            padding: 12px 16px;
        }
        .accuracy-value {
            font-size: 18px;
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

# Model Accuracy Display Card
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
# EXAMPLE BUTTONS
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Spam Example", use_container_width=True):
        st.session_state.email_input = SPAM_EXAMPLE

with col2:
    if st.button("Safe Example", use_container_width=True):
        st.session_state.email_input = SAFE_EXAMPLE

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

# Keep internal state synchronized
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

if check_clicked:
    if not email_text.strip():
        st.warning("Please enter or paste an email message to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model or vectorizer assets are missing. Please check your .pkl files.")
    else:
        with st.spinner("Analyzing Email..."):
            # Feature extraction
            transformed_input = vectorizer.transform([email_text])
            
            # Predict class and probability
            prediction = model.predict(transformed_input)[0]
            
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_input)[0]
                confidence = float(np.max(probabilities) * 100)
            else:
                confidence = 100.0

        # Display formatted output cards based on prediction outcome
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
