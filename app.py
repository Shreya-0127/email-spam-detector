import pickle
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
model = pickle.load(open("spam_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ── HEADER ──
st.title("📧 Email Spam Detector")
st.caption("Instantly detect if any email is spam or legitimate")

# ── METRIC ──
st.metric(label="Model Accuracy", value="97.48%")

st.divider()

# ── EXAMPLE BUTTONS ──
st.write("**Try an example:**")
col1, col2 = st.columns(2)

with col1:
    if st.button("🚨 Spam Example", use_container_width=True):
        st.session_state.email_text = (
            "Congratulations! You won a free iPhone. Click now to claim your prize worth $1000!"
        )

with col2:
    if st.button("✅ Normal Example", use_container_width=True):
        st.session_state.email_text = (
            "Hey, are we still meeting for lunch tomorrow at 1pm?"
        )

# ── TEXT INPUT ──
email_input = st.text_area(
    "Enter Email Text Here:",
    value=st.session_state.get("email_text", ""),
    height=160,
    placeholder="Type or paste your email here...",
)

# ── CHECK BUTTON ──
if st.button("🔍 Check Email", type="primary", use_container_width=True):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter some email text first!")
    else:
        vectorized = tfidf.transform([email_input])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        if prediction == 1:
            confidence = round(proba[1] * 100, 1)
            st.error(f"🚨 **SPAM DETECTED**\n\nConfidence: **{confidence}%** — This email looks like spam")
        else:
            confidence = round(proba[0] * 100, 1)
            st.success(f"✅ **NOT SPAM**\n\nConfidence: **{confidence}%** — This email looks safe")

# ── FOOTER ──
st.divider()
st.caption("© 2026 Shreya Shukla | Email Spam Detector | ML Project")
