import streamlit as st
import pickle

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ----------------- CSS -----------------

st.markdown("""
<style>

.main{
    background:#BDC4D4;
}

/* Hide Streamlit menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.title{
    text-align:center;
    color:#0F1A2B;
    font-size:2rem;
    font-weight:700;
    margin-bottom:0;
}

.subtitle{
    text-align:center;
    color:#52677D;
    margin-bottom:25px;
}

.block{
    background:#D1CFC9;
    padding:22px;
    border-radius:14px;
    border:1px solid #52677D30;
}

.stTextArea textarea{
    border-radius:12px;
    border:2px solid #52677D;
    font-size:16px;
}

.stButton>button{
    width:100%;
    background:#1C2E4A;
    color:white;
    border:none;
    border-radius:10px;
    padding:12px;
    font-size:17px;
    font-weight:600;
}

.stButton>button:hover{
    background:#0F1A2B;
    color:white;
}

.result-safe{
    background:#E8F5EC;
    border-left:6px solid #2E8B57;
    padding:18px;
    border-radius:10px;
    margin-top:15px;
}

.result-spam{
    background:#FCEAEA;
    border-left:6px solid #C0392B;
    padding:18px;
    border-radius:10px;
    margin-top:15px;
}

.footer{
    text-align:center;
    color:#52677D;
    font-size:13px;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# ----------------- Load Model -----------------

model = pickle.load(open("spam_model.pkl","rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl","rb"))

# ----------------- Header -----------------

st.markdown("<h1 class='title'>📧 Email Spam Detector</h1>",unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Machine Learning Based Email Classification</p>",unsafe_allow_html=True)

# ----------------- Card -----------------

st.markdown("<div class='block'>",unsafe_allow_html=True)

email = st.text_area(
    "Paste Email",
    height=180,
    placeholder="Paste your email content here..."
)

col1,col2 = st.columns(2)

with col1:
    if st.button("Spam Example"):
        st.session_state.example="Congratulations! You have won $1000. Click here to claim."

with col2:
    if st.button("Safe Example"):
        st.session_state.example="Hello, let's meet tomorrow at 10 AM regarding the project."

if "example" in st.session_state:
    email=st.session_state.example
    st.text_area("Example Email",value=email,height=180)

if st.button("Check Email"):

    if email.strip()=="":

        st.warning("Please enter an email.")

    else:

        vector=tfidf.transform([email])

        prediction=model.predict(vector)[0]

        probability=model.predict_proba(vector)[0]

        if prediction==1:

            confidence=round(probability[1]*100,2)

            st.markdown(f"""
            <div class="result-spam">
            <h3>🚫 Spam Email</h3>
            <p>This email appears to be spam.</p>
            <b>Confidence : {confidence}%</b>
            </div>
            """,unsafe_allow_html=True)

        else:

            confidence=round(probability[0]*100,2)

            st.markdown(f"""
            <div class="result-safe">
            <h3>✅ Legitimate Email</h3>
            <p>This email appears to be safe.</p>
            <b>Confidence : {confidence}%</b>
            </div>
            """,unsafe_allow_html=True)

st.markdown("</div>",unsafe_allow_html=True)

st.markdown(
"""
<div class='footer'>
Developed by <b>Shreya Shukla</b><br>
Email Spam Detector • Machine Learning Project • 2026
</div>
""",
unsafe_allow_html=True
)
