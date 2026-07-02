import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

# Simple, elegant CSS
st.markdown("""
    <style>
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
    }
    
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 30px;
    }
    
    .risk-low {
        background-color: #d5f4e6;
        color: #0b7285;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .risk-high {
        background-color: #fdeaed;
        color: #c92a2a;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .metric-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #3498db;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# Header
st.markdown("<h1>🫀 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered cardiac health assessment</div>", unsafe_allow_html=True)
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")
    age = st.number_input("Age (years)", min_value=25, max_value=85, value=50)
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True, label_visibility="collapsed")
    gender_val = 0 if gender == "Female" else 1
    trestbps = st.number_input("Blood Pressure (mm Hg)", min_value=90, max_value=200, value=125)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=212)

with col2:
    st.subheader("Heart Metrics")
    thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=168)
    cp = st.select_slider("Chest Pain Type", options=[0, 1, 2, 3, 4], value=3)
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    exang = st.radio("Exercise-Induced Angina", ["No", "Yes"], horizontal=True, label_visibility="collapsed")
    exang_val = 0 if exang == "No" else 1

# Advanced parameters
with st.expander("📊 Advanced Parameters"):
    col_adv1, col_adv2 = st.columns(2)
    
    with col_adv1:
        fbs = st.radio("High Fasting Blood Sugar", ["No", "Yes"], horizontal=True, label_visibility="collapsed")
        fbs_val = 0 if fbs == "No" else 1
        restecg = st.select_slider("Resting ECG", options=[0, 1, 2], value=1)
        slope = st.select_slider("ST Segment Slope", options=[0, 1, 2], value=2)
    
    with col_adv2:
        ca = st.number_input("Major Vessels", min_value=0, max_value=3, value=2)
        thal = st.select_slider("Thalassemia Type", options=[0, 1, 2, 3], value=3)

st.divider()

# Prediction button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

# Results
if predict:
    input_data = np.array([[age, gender_val, cp, trestbps, chol, fbs_val, restecg, 
                            thalach, exang_val, oldpeak, slope, ca, thal]])
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    
    st.divider()
    
    # Risk result
    col_res1, col_res2, col_res3 = st.columns([1, 1, 1])
    
    with col_res2:
        if prediction == 0:
            st.markdown("<div class='risk-low'>✅ LOW RISK</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='risk-high'>⚠️ HIGH RISK</div>", unsafe_allow_html=True)
    
    st.markdown("")
    
    if prediction == 0:
        st.markdown(
            "**Explanation:** This profile indicates a lower probability of heart disease based on the selected inputs. "
            "Maintain healthy habits, monitor symptoms, and follow routine checkups."
        )
    else:
        st.markdown(
            "**Explanation:** This profile suggests an elevated risk of heart disease. "
            "Consider consulting a healthcare professional and reviewing lifestyle factors such as diet, exercise, and stress."
        )
    
    st.markdown("")
    
    # Summary table
    st.subheader("Health Summary")
    
    summary = pd.DataFrame({
        "Metric": ["Age", "Heart Rate", "Cholesterol", "Blood Pressure"],
        "Value": [f"{age} yrs", f"{thalach} bpm", f"{chol} mg/dl", f"{trestbps} mm Hg"],
        "Assessment": [
            "✅ Normal" if 40 <= age <= 70 else "⚠️ Check",
            "✅ Healthy" if thalach >= 100 else "⚠️ Low",
            "✅ Good" if chol < 240 else "⚠️ Elevated",
            "✅ Normal" if trestbps < 140 else "⚠️ Elevated"
        ]
    })
    
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()
st.markdown("""
    **📌 Disclaimer:** This tool is for educational purposes only.  
    **Always consult a healthcare professional** for diagnosis and treatment.
    
    **Model Specs:** Random Forest | Accuracy: 88.3% | Dataset: UCI Heart Disease
""")
