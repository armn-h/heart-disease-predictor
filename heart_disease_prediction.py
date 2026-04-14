# ============================================================
# HEART DISEASE PREDICTION USING RANDOM FOREST
# B.Tech Mini Project
# ============================================================

# ============================================================
# STEP 1: INSTALL & IMPORT LIBRARIES
# ------------------------------------------------------------
# Libraries are pre-built tools that help us do tasks easily.
# We don't need to install most of them in Colab — they're
# already available!
# ============================================================

import pandas as pd           # For loading and handling data (like Excel but in Python)
import numpy as np            # For numerical calculations
import matplotlib.pyplot as plt  # For drawing charts/graphs
import seaborn as sns         # For beautiful statistical charts
import joblib                 # For saving/loading trained models

from sklearn.model_selection import train_test_split   # To split data into train & test
from sklearn.ensemble import RandomForestClassifier    # Our ML model: Random Forest
from sklearn.metrics import (
    accuracy_score,           # Measures how accurate our model is
    confusion_matrix,         # Shows correct vs wrong predictions
    classification_report     # Detailed performance report
)
from sklearn.preprocessing import StandardScaler       # To normalize/scale our data

import warnings
warnings.filterwarnings('ignore')  # Suppress unnecessary warnings

print("✅ All libraries imported successfully!")


# ============================================================
# STEP 2: LOAD THE DATASET
# ------------------------------------------------------------
# We use the famous UCI Heart Disease dataset.
# It contains patient medical data and whether they have
# heart disease (1) or not (0).
#
# COLUMNS EXPLAINED:
# age       - Age of the patient
# gender    - 1 = Male, 0 = Female
# cp        - Chest pain type (0-3)
# trestbps  - Resting blood pressure (mm Hg)
# chol      - Serum cholesterol (mg/dl)
# fbs       - Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
# restecg   - Resting ECG results (0-2)
# thalach   - Maximum heart rate achieved
# exang     - Exercise induced angina (1 = Yes, 0 = No)
# oldpeak   - ST depression induced by exercise
# slope     - Slope of peak exercise ST segment
# ca        - Number of major vessels (0-3)
# thal      - Thalassemia type (0-3)
# target    - 1 = Heart Disease, 0 = No Heart Disease  ← This is what we PREDICT
# ============================================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
# Load with proper column names
column_names = ['age', 'gender', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, names=column_names)

# Replace '?' with NaN and drop those rows
df = df.replace('?', np.nan)
df = df.dropna()

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Convert target to binary: 0 = No Disease, 1 = Disease present
df['target'] = (df['target'] > 0).astype(int)

print("✅ Dataset loaded successfully!")
print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n🔍 First 5 rows of the dataset:")
print(df.head())


# ============================================================
# STEP 3: EXPLORE THE DATA (EDA)
# ------------------------------------------------------------
# Before building the model, we understand our data.
# This is called Exploratory Data Analysis (EDA).
# ============================================================

print("\n📋 Dataset Info:")
print(df.info())

print("\n📈 Basic Statistics:")
print(df.describe())

print("\n❓ Missing Values:")
print(df.isnull().sum())

print(f"\n🎯 Target Distribution:")
print(df['target'].value_counts())
print(f"   0 = No Heart Disease | 1 = Heart Disease")


# ============================================================
# STEP 4: VISUALIZATIONS
# ------------------------------------------------------------
# Charts help us understand patterns in the data visually.
# ============================================================

plt.figure(figsize=(18, 14))

# --- Chart 1: Target distribution (how many have heart disease?) ---
plt.subplot(3, 3, 1)
df['target'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'], edgecolor='black')
plt.title('Heart Disease Distribution', fontsize=13, fontweight='bold')
plt.xlabel('0 = No Disease | 1 = Disease')
plt.ylabel('Count')
plt.xticks(rotation=0)

# --- Chart 2: Age vs Heart Disease ---
plt.subplot(3, 3, 2)
sns.histplot(data=df, x='age', hue='target', kde=True, palette=['steelblue', 'tomato'])
plt.title('Age vs Heart Disease', fontsize=13, fontweight='bold')
plt.xlabel('Age')

# --- Chart 3: Gender vs Heart Disease ---
plt.subplot(3, 3, 3)
sns.countplot(data=df, x='gender', hue='target', palette=['steelblue', 'tomato'])
plt.title('Gender vs Heart Disease', fontsize=13, fontweight='bold')
plt.xlabel('0 = Female | 1 = Male')

# --- Chart 4: Chest Pain type vs Heart Disease ---
plt.subplot(3, 3, 4)
sns.countplot(data=df, x='cp', hue='target', palette=['steelblue', 'tomato'])
plt.title('Chest Pain Type vs Heart Disease', fontsize=13, fontweight='bold')
plt.xlabel('Chest Pain Type (0-3)')

# --- Chart 5: Max Heart Rate vs Heart Disease ---
plt.subplot(3, 3, 5)
sns.boxplot(data=df, x='target', y='thalach', palette=['steelblue', 'tomato'])
plt.title('Max Heart Rate vs Heart Disease', fontsize=13, fontweight='bold')
plt.xlabel('0 = No Disease | 1 = Disease')

# --- Chart 6: Cholesterol vs Heart Disease ---
plt.subplot(3, 3, 6)
sns.boxplot(data=df, x='target', y='chol', palette=['steelblue', 'tomato'])
plt.title('Cholesterol vs Heart Disease', fontsize=13, fontweight='bold')
plt.xlabel('0 = No Disease | 1 = Disease')

# --- Chart 7: Correlation Heatmap ---
plt.subplot(3, 3, 7)
corr = df.corr()
sns.heatmap(corr[['target']].sort_values('target', ascending=False),
            annot=True, cmap='RdYlGn', linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation with Target', fontsize=13, fontweight='bold')

# --- Chart 8: Blood Pressure vs Heart Disease ---
plt.subplot(3, 3, 8)
sns.boxplot(data=df, x='target', y='trestbps', palette=['steelblue', 'tomato'])
plt.title('Blood Pressure vs Heart Disease', fontsize=13, fontweight='bold')
plt.xlabel('0 = No Disease | 1 = Disease')

# --- Chart 9: Age Distribution ---
plt.subplot(3, 3, 9)
sns.histplot(df['age'], bins=20, color='steelblue', kde=True)
plt.title('Age Distribution of Patients', fontsize=13, fontweight='bold')
plt.xlabel('Age')

plt.suptitle('Heart Disease Prediction - Data Exploration', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Charts saved as 'eda_charts.png'")


# ============================================================
# STEP 5: PREPARE DATA FOR MACHINE LEARNING
# ------------------------------------------------------------
# X = Input features (all columns except target)
# y = Output/label (target column — what we want to predict)
#
# We also split data into:
#   Training set (80%) — model LEARNS from this
#   Testing set  (20%) — model is EVALUATED on this
# ============================================================

X = df.drop('target', axis=1)   # All columns except 'target'
y = df['target']                 # Only the 'target' column

print(f"📥 Input features (X) shape: {X.shape}")
print(f"🎯 Output labels (y) shape:  {y.shape}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% data for testing
    random_state=42     # Fixed seed so results are reproducible
)

print(f"\n✅ Data Split Done!")
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")


# ============================================================
# STEP 6: FEATURE SCALING
# ------------------------------------------------------------
# Different features have different ranges (e.g., age: 20-80,
# cholesterol: 100-600). Scaling brings them to the same range
# so the model treats all features fairly.
# ============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Learn scale from training data, then apply
X_test  = scaler.transform(X_test)        # Only apply (don't re-learn) on test data

print("✅ Feature scaling done!")


# ============================================================
# STEP 7: BUILD & TRAIN THE RANDOM FOREST MODEL
# ------------------------------------------------------------
# Random Forest = a collection of many Decision Trees.
# Each tree votes, and the majority vote wins.
# It's like asking 100 doctors and taking the most common answer!
#
# n_estimators = number of trees in the forest
# random_state = ensures same results every time you run
# ============================================================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)   # TRAINING the model

# Save the trained model
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Random Forest model trained successfully!")
print(f"   Trees in the forest: {model.n_estimators}")
print("✅ Model saved as 'heart_disease_model.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")


# ============================================================
# STEP 8: MAKE PREDICTIONS & EVALUATE THE MODEL
# ------------------------------------------------------------
# Now we use the trained model to predict on TEST data
# (data the model has NEVER seen before).
# ============================================================

y_pred = model.predict(X_test)

# --- Accuracy Score ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# --- Classification Report ---
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Heart Disease'],
            yticklabels=['No Disease', 'Heart Disease'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("✅ Confusion matrix saved as 'confusion_matrix.png'")

# Confusion matrix explanation
tn, fp, fn, tp = cm.ravel()
print(f"\n🔎 Confusion Matrix Breakdown:")
print(f"   ✅ True Negatives  (Correctly predicted No Disease): {tn}")
print(f"   ✅ True Positives  (Correctly predicted Disease)   : {tp}")
print(f"   ❌ False Positives (Predicted Disease, was healthy): {fp}")
print(f"   ❌ False Negatives (Predicted healthy, had Disease): {fn}")


# ============================================================
# STEP 9: FEATURE IMPORTANCE
# ------------------------------------------------------------
# Random Forest tells us which features (columns) are most
# important in making predictions.
# ============================================================

feature_names = df.drop('target', axis=1).columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_names)), importances[indices], color='steelblue', edgecolor='black')
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.title('Feature Importance - Which factors matter most?', fontsize=14, fontweight='bold')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("✅ Feature importance chart saved as 'feature_importance.png'")

print("\n📊 Top 5 Most Important Features:")
for i in range(5):
    print(f"   {i+1}. {feature_names[indices[i]]:12s} → {importances[indices[i]]:.4f}")


# ============================================================
# STEP 10: PREDICT FOR A NEW PATIENT
# ------------------------------------------------------------
# Now let's use the model to predict for a brand new patient!
# Enter patient details below and get a prediction.
# ============================================================

print("\n" + "="*55)
print("🏥 PREDICTING FOR A NEW PATIENT")
print("="*55)

# Sample patient data — you can change these values!
new_patient = pd.DataFrame({
    'age':      [52],   # Age: 52 years
    'gender':   [1],    # Gender: Male (1)
    'cp':       [0],    # Chest Pain: Type 0
    'trestbps': [125],  # Blood Pressure: 125
    'chol':     [212],  # Cholesterol: 212
    'fbs':      [0],    # Fasting Blood Sugar <= 120 (0)
    'restecg':  [1],    # Resting ECG: 1
    'thalach':  [168],  # Max Heart Rate: 168
    'exang':    [0],    # Exercise Angina: No (0)
    'oldpeak':  [1.0],  # ST Depression: 1.0
    'slope':    [2],    # Slope: 2
    'ca':       [2],    # Major Vessels: 2
    'thal':     [3],    # Thalassemia: 3
})

# Scale the new patient data the same way we scaled training data
new_patient_scaled = scaler.transform(new_patient)

# Make prediction
prediction = model.predict(new_patient_scaled)
probability = model.predict_proba(new_patient_scaled)

print(f"\n👤 Patient Details:")
for col in new_patient.columns:
    print(f"   {col:12s}: {new_patient[col].values[0]}")

print(f"\n🔮 Prediction Result:")
if prediction[0] == 1:
    print(f"   ⚠️  HIGH RISK — The patient is likely to have Heart Disease")
else:
    print(f"   ✅  LOW RISK  — The patient is unlikely to have Heart Disease")

print(f"\n📊 Prediction Confidence:")
print(f"   No Disease  : {probability[0][0]*100:.1f}%")
print(f"   Heart Disease: {probability[0][1]*100:.1f}%")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*55)
print("🎓 PROJECT SUMMARY")
print("="*55)
print(f"  Dataset        : UCI Heart Disease Dataset")
print(f"  Total Samples  : {df.shape[0]}")
print(f"  Features Used  : {df.shape[1] - 1}")
print(f"  Algorithm      : Random Forest Classifier")
print(f"  Trees          : 100")
print(f"  Train/Test     : 80% / 20%")
print(f"  Model Accuracy : {accuracy * 100:.2f}%")
print("="*55)
print("✅ Project Complete! Well done! 🎉")
