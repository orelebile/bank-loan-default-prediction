
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model and scaler
model = joblib.load("loan_risk_logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")  # saved training columns

st.title("Loan Risk Prediction App")
st.write("Enter customer details to predict loan risk")

# Numeric inputs
income = st.number_input("Income")
age = st.number_input("Age")
experience = st.number_input("Experience (Years)")
current_job_yrs = st.number_input("Current Job Years")
current_house_yrs = st.number_input("Current House Years")

# Categorical inputs
married = st.selectbox("Marital Status", ["single","married"])
house = st.selectbox("House Ownership", ["rented","owned","norent_noown"])
car = st.selectbox("Car Ownership", ["no","yes"])
profession = st.text_input("Profession")
city = st.text_input("City")
state = st.text_input("State")

# Prediction button
if st.button("Predict Loan Risk"):

    # Create dataframe from inputs
    input_data = pd.DataFrame({
        'Income':[income],
        'Age':[age],
        'Experience':[experience],
        'Married/Single':[married],
        'House_Ownership':[house],
        'Car_Ownership':[car],
        'Profession':[profession],
        'City':[city],
        'State':[state],
        'Current_Job_Yrs':[current_job_yrs],
        'Current_House_Yrs':[current_house_yrs]
    })

    # One-hot encode categorical features
    input_data = pd.get_dummies(input_data)

    # Align input with training columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Loan Risk")
    else:
        st.success("✅ Low Loan Risk")
