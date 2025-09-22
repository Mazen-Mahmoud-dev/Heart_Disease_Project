import streamlit as st
import pandas as pd
import joblib   # or pickle

# -------------------------
# Load the trained pipeline
# -------------------------
model = joblib.load("../models/final_model.pkl")

# The model was trained on these exact columns, in this order:
FEATURES = [
    "exang", "slope", "thal", "cp",
    "age", "chol", "restecg", "thalach",
    "oldpeak", "ca"
]

st.title("Heart Disease Prediction")

st.write("Provide the following inputs:")

# -------------------------
# Collect user inputs
# -------------------------
# Categorical or numeric values should match the training preprocessing
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
slope = st.selectbox("Slope of ST segment (0,1,2)", [0, 1, 2])
thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])

age = st.number_input("Age", min_value=18, max_value=100, value=50)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
restecg = st.selectbox("Resting ECG (0,1,2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])

# Create DataFrame with **exact same columns and order**
input_data = pd.DataFrame([[
    exang, slope, thal, cp,
    age, chol, restecg, thalach,
    oldpeak, ca
]], columns=FEATURES)

st.write("Input preview:", input_data)

# -------------------------
# Predict
# -------------------------
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    st.write("Note: 0 means you don't suffer and 1 means you suffer")
    st.success(f"Predicted class: {pred}")
