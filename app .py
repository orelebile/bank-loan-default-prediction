
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model and scaler
model = joblib.load("loan_risk_logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Risk Prediction")

st.write("Enter customer information to predict loan risk")

# Numeric Inputs
income = st.number_input("Income")
age = st.number_input("Age")
experience = st.number_input("Work Experience (Years)")
current_job_yrs = st.number_input("Current Job Years")
current_house_yrs = st.number_input("Current House Years")

# Categorical Inputs
married = st.selectbox("Marital Status", ["single","married"])
house = st.selectbox("House Ownership", ["rented","owned","norent_noown"])
car = st.selectbox("Car Ownership", ["no","yes"])

profession = st.text_input("Profession")
city = st.text_input("City")
state = st.text_input("State")

if st.button("Predict Loan Risk"):

    # Convert categorical variables to numbers
    married = 1 if married=="married" else 0
    car = 1 if car=="yes" else 0

    if house == "owned":
        house_val = 2
    elif house == "rented":
        house_val = 1
    else:
        house_val = 0

    # Feature array
    features = np.array([[income, age, experience,
                          married,
                          house_val,
                          car,
                          current_job_yrs,
                          current_house_yrs]])

    # Scale the data
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Loan Risk")
    else:
        st.success("✅ Low Loan Risk")
