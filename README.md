# Email Spam Detector (Machine Learning)

A complete end-to-end Machine Learning project that classifies emails as **Spam** or **Not Spam (Ham)** using NLP and Naive Bayes.

🔗 Live Demo: https://email-spam-detector-ml.streamlit.app/  
🔗 GitHub Repo: https://github.com/Shreya-0127/email-spam-detector  

---

## 📌 Overview

This project demonstrates a full ML pipeline:
- Data collection and cleaning  
- Model training and evaluation  
- Building a web application  
- Deploying it live on the internet  

The model achieves an accuracy of **97.48%** and is accessible via a Streamlit web app.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **Scikit-learn**
- **Matplotlib & Seaborn**
- **Streamlit**
- **Git & GitHub**

---

## 📊 Dataset

- SMS Spam Collection Dataset (from Kaggle)  
- Total messages: **5,572**  
- Spam: **747**  
- Ham: **4,825**

---

## ⚙️ How It Works

1. Data Cleaning (remove unused columns)  
2. Label Encoding (ham = 0, spam = 1)  
3. Train-Test Split (80/20)  
4. Text Vectorization using **TF-IDF**  
5. Model Training using **Multinomial Naive Bayes**  
6. Model Evaluation  

---

## 📈 Model Performance

- **Accuracy:** 97.48%  
- **Spam Precision:** 100%  
- **Ham Precision:** 97%  

---

## 🚀 Features

- Real-time spam detection  
- Clean web interface using Streamlit  
- Confidence score for predictions  
- Works on custom user input  

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/Shreya-0127/email-spam-detector
cd email-spam-detector
pip install -r requirements.txt
streamlit run app.py
