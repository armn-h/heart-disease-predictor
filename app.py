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

    .risk-moderate {
        background-color: #fff4d6;
        color: #b97b00;
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

    .section-box {
        background-color: #ffffff;
        border: 1px solid #dfe3e8;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.04);
        margin-bottom: 24px;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 16px;
        color: #283747;
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
    with st.expander("Patient Information", expanded=True):
        age = st.number_input("Age (years)", min_value=25, max_value=85, value=50)
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True, label_visibility="collapsed")
        gender_val = 0 if gender == "Female" else 1
        trestbps = st.number_input("Blood Pressure (mm Hg)", min_value=90, max_value=200, value=125)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=212)

with col2:
    with st.expander("Heart Metrics", expanded=True):
        thalach = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=220, value=168)
        cp_options = {
            "Typical Angina (0)": 0,
            "Atypical Angina (1)": 1,
            "Non-anginal Pain (2)": 2,
            "Asymptomatic (3)": 3,
            "Unknown / Other (4)": 4
        }
        cp_label = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), index=3)
        cp = cp_options[cp_label]
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        exang = st.radio("Exercise-Induced Angina", ["No", "Yes"], horizontal=True)
        exang_val = 0 if exang == "No" else 1

# Advanced parameters
with st.expander("📊 Advanced Parameters", expanded=False):
    col_adv1, col_adv2 = st.columns(2)
    
    with col_adv1:
        fbs = st.radio("High Fasting Blood Sugar", ["No", "Yes"], horizontal=True)
        fbs_val = 0 if fbs == "No" else 1
        restecg_options = {
            "Normal (0)": 0,
            "ST-T Wave Abnormality (1)": 1,
            "Left Ventricular Hypertrophy (2)": 2
        }
        restecg_label = st.selectbox("Resting ECG", options=list(restecg_options.keys()), index=1)
        restecg = restecg_options[restecg_label]
        slope_options = {
            "Upsloping (0)": 0,
            "Flat (1)": 1,
            "Downsloping (2)": 2
        }
        slope_label = st.selectbox("ST Segment Slope", options=list(slope_options.keys()), index=1)
        slope = slope_options[slope_label]
    
    with col_adv2:
        ca = st.number_input("Major Vessels", min_value=0, max_value=3, value=2)
        thal_options = {
            "Normal (0)": 0,
            "Fixed Defect (1)": 1,
            "Reversible Defect (2)": 2,
            "Unknown / Other (3)": 3
        }
        thal_label = st.selectbox("Thalassemia Type", options=list(thal_options.keys()), index=0)
        thal = thal_options[thal_label]
    st.markdown("</div>", unsafe_allow_html=True)

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
    
    # Risk score / probability bands
    prob = model.predict_proba(scaled_input)[0][1]
    if prob < 0.4:
        band = "LOW RISK"
        band_class = "risk-low"
        advice = "This profile indicates a lower probability of heart disease. Maintain healthy habits, monitor symptoms, and follow routine checkups."
    elif prob < 0.7:
        band = "MODERATE RISK"
        band_class = "risk-moderate"
        advice = "This profile indicates a moderate probability of heart disease. Consider following up with a healthcare professional and reviewing lifestyle factors."
    else:
        band = "HIGH RISK"
        band_class = "risk-high"
        advice = "This profile suggests an elevated probability of heart disease. Consider consulting a healthcare professional and reviewing diet, exercise, and symptoms."
    
    col_res1, col_res2, col_res3 = st.columns([1, 1, 1])
    
    with col_res2:
        icon = "✅" if band == "LOW RISK" else "⚠️"
        st.markdown(f"<div class='{band_class}'>{icon} {band}</div>", unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown(f"**Estimated probability of disease:** {prob*100:.1f}%")
    st.markdown("")
    st.markdown(advice)
    st.markdown("")
    st.markdown("**Risk band guide:** Low = < 40%, Moderate = 40–70%, High = > 70%")
    
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
