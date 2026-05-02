import streamlit as st
import joblib
import numpy as np

# Load model & scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Employee Attrition Predictor")

st.title("💼 Employee Attrition Prediction")
st.write("Enter employee details to predict attrition")

# --- Input fields ---
age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 100000, 30000)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
years_at_company = st.slider("Years at Company", 0, 40, 5)
work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)

# --- Convert to array ---
input_data = np.array([[age, monthly_income, job_satisfaction, years_at_company, work_life_balance]])

# Scale input
input_scaled = scaler.transform(input_data)

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Employee is likely to leave")
    else:
        st.success("✅ Employee is likely to stay")