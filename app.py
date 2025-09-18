import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ---------- Streamlit Config ----------
st.set_page_config(page_title="Bank Churn Predictor", page_icon="💠", layout="wide")

# ---------- Compact Layout CSS ----------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 95%;
}
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f0fdfa 100%);
    font-family: 'Poppins', sans-serif;
    color: #1e293b;
}
.title {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    color: #0e7490;
    text-shadow: 0px 1px 5px rgba(14, 116, 144, 0.15);
    margin-bottom: 0px;
}
.subtitle {
    text-align: center;
    font-size: 14px;
    color: #334155;
    margin-bottom: 15px;
}
.card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(8px);
    border-radius: 14px;
    box-shadow: 0 4px 15px rgba(14, 116, 144, 0.08);
    padding: 12px;
}
.result-card {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(8px);
    border-radius: 14px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
}
.metric {
    font-size: 28px;
    font-weight: 700;
}
.safe { color: #16a34a; }
.danger { color: #dc2626; }
.stButton button {
    background: linear-gradient(90deg, #0ea5e9, #06b6d4);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    font-size: 16px;
    padding: 6px;
    transition: 0.2s;
    border: none;
    box-shadow: 0px 2px 8px rgba(6,182,212,0.25);
}
.stButton button:hover { transform: scale(1.03); }
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown('<div class="title">💠 Bank Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI tool to forecast customer churn risk instantly</div>', unsafe_allow_html=True)

# ---------- Load Model & Scaler ----------
@st.cache_resource
def load_artifacts():
    if not os.path.exists("model.keras"):
        st.error("❌ model.keras not found!")
        st.stop()
    if not os.path.exists("scaler.pkl"):
        st.error("❌ scaler.pkl not found!")
        st.stop()
    model = load_model("model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ---------- Prediction Function ----------
def predict_single(data_dict):
    cols = ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard',
            'IsActiveMember','EstimatedSalary','Geography_Germany','Geography_Spain','Gender_Male']
    X_df = pd.DataFrame([data_dict], columns=cols, dtype=float)
    X_scaled = scaler.transform(X_df)
    prob = model.predict(X_scaled)[0][0]
    return float(prob)

# ---------- Layout ----------
left, right = st.columns([1.2, 1])

with left:
    with st.form("churn_form"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Customer Details")
        c1,c2,c3 = st.columns(3)
        credit_score = c1.number_input("Credit Score", 300, 1000, 650)
        age = c2.number_input("Age", 18, 100, 35)
        tenure = c3.number_input("Tenure", 0, 10, 3)

        c4,c5,c6 = st.columns(3)
        balance = c4.number_input("Balance", 0.0, 1e7, 50000.0, step=100.0)
        num_products = c5.slider("Products", 1, 4, 1)
        salary = c6.number_input("Estimated Salary", 0.0, 1e7, 50000.0, step=100.0)

        c7,c8,c9 = st.columns(3)
        has_card = int(c7.checkbox("Has Card", True))
        is_active = int(c8.checkbox("Active", True))
        gender = c9.radio("Gender", ["Male","Female"], horizontal=True)
        geo = st.selectbox("🌍 Geography", ["France","Germany","Spain"])
        threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05)

        submit = st.form_submit_button("🔮 Predict", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    if submit:
        row = {
            "CreditScore": int(credit_score),
            "Age": int(age),
            "Tenure": int(tenure),
            "Balance": float(balance),
            "NumOfProducts": int(num_products),
            "HasCrCard": int(has_card),
            "IsActiveMember": int(is_active),
            "EstimatedSalary": float(salary),
            "Geography_Germany": 1 if geo == "Germany" else 0,
            "Geography_Spain": 1 if geo == "Spain" else 0,
            "Gender_Male": 1 if gender == "Male" else 0,
        }
        prob = predict_single(row)
        churn = prob >= threshold

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"<div class='metric {'danger' if churn else 'safe'}'>{prob:.2%}</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:16px;'>{'🚨 High Risk' if churn else '✅ Likely Safe'}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Bigger & Clear Gauge Chart with % ---
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%", 'font': {'size': 38, 'color': "#0f172a"}},  # bigger & bold %
            title={'text': "Churn Probability", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#dc2626" if churn else "#16a34a"},
                'steps': [
                    {'range': [0, 50], 'color': "#d1fae5"},
                    {'range': [50, 100], 'color': "#fee2e2"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.7,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",  # transparent
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

