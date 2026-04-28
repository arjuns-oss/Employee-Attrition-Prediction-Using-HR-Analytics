# Initial deployment version - dependencies and model integration to be refined
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("attrition_model.pkl", "rb"))

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

st.title("Employee Attrition Prediction App")

age = st.number_input("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000)
overtime = st.selectbox("OverTime", ["No", "Yes"])

overtime_value = 1 if overtime == "Yes" else 0

if st.button("Predict Attrition"):

    input_data = np.zeros(30)

    input_data[0] = age
    input_data[15] = monthly_income
    input_data[18] = overtime_value

    prediction = model.predict([input_data])

    if prediction[0] == 1:
        st.error("High Risk: Employee may leave")
    else:
        st.success("Low Risk: Employee likely to stay")
