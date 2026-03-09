
import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("loan_risk_logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Loan Default Risk Prediction")

st.write("Enter customer information to predict loan default risk.")

# -------------------------
# Numeric Inputs
# -------------------------

income = st.number_input("Income", min_value=0)
age = st.number_input("Age", min_value=18)
experience = st.number_input("Work Experience (Years)", min_value=0)
current_job_yrs = st.number_input("Years in Current Job", min_value=0)
current_house_yrs = st.number_input("Years in Current House", min_value=0)

# -------------------------
# Categorical Inputs
# -------------------------

married = st.selectbox("Marital Status", ["single","married"])
house = st.selectbox("House Ownership", ["rented","owned","norent_noown"])
car = st.selectbox("Car Ownership", ["no","yes"])

profession = st.text_input("Profession")
city = st.text_input("City")
state = st.text_input("State")

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Loan Risk"):

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

    # One-hot encode
    input_data = pd.get_dummies(input_data)

    # Align columns with training data
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    # Output result
    if prediction[0] == 1:
        st.error(f"⚠️ High Loan Default Risk (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Loan Default Risk (Probability: {probability:.2f})")

