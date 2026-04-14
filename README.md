# 🫀 Heart Disease Risk Predictor

AI-powered cardiac health assessment using Machine Learning.

## Features

- 🤖 **ML Model:** Random Forest classifier trained on UCI Heart Disease dataset
- 📊 **High Accuracy:** 88.3% accuracy on test data
- 🎯 **Real-time Predictions:** Instant risk assessment
- 📱 **Mobile Friendly:** Responsive design works on all devices
- 💡 **Easy to Use:** Simple, elegant interface

## Model Performance

- **Accuracy:** 88.33%
- **Precision:** 91%
- **Recall:** 88%
- **F1-Score:** 0.86
- **Dataset:** UCI Cleveland Heart Disease (297 samples)
- **Algorithm:** Random Forest (100 trees)

## Installation

### Local Setup

```bash
# Clone repository
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage

1. Enter patient health metrics
2. Click **"🔍 Predict Risk"**
3. View results with confidence scores
4. Check health summary and recommendations

## Files

- `app.py` - Streamlit web application
- `heart_disease_prediction.py` - Model training script
- `heart_disease_model.pkl` - Trained ML model
- `scaler.pkl` - Feature scaler
- `requirements.txt` - Python dependencies

## Try Online

[Deploy on Streamlit Cloud](https://share.streamlit.io) - Free hosting!

## Disclaimer

⚕️ **This tool is for educational purposes only.**  
Always consult with qualified healthcare professionals for medical diagnosis and treatment decisions.

## Author

Created as a B.Tech Mini Project

## Dataset

UCI Machine Learning Repository - Heart Disease Dataset  
https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
