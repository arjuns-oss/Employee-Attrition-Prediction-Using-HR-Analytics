import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("attrition_pipeline.pkl")

st.set_page_config(page_title="Employee Attrition Predictor")

st.title("💼 Employee Attrition Prediction")
st.write("Enter employee details to predict attrition")

# --- Inputs ---
age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 100000, 30000)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
years_at_company = st.slider("Years at Company", 0, 40, 5)
work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)

# Convert to DataFrame (🔥 IMPORTANT)
input_df = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "JobSatisfaction": [job_satisfaction],
    "YearsAtCompany": [years_at_company],
    "WorkLifeBalance": [work_life_balance]
})

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Employee is likely to leave")
    else:
        st.success("✅ Employee is likely to stay")

    st.write(f"📈 Probability of Attrition: **{probability:.2f}**")
